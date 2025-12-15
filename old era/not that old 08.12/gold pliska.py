import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# ==============================================================================
# A. KONFIGURACJA
# ==============================================================================
NEURONS       = 64
DROPOUT       = 0.3
LEARNING_RATE = 0.001
REG_STRENGTH  = 0.0001

# --- NOWE USTAWIENIE: KWANTYLE ---
# Określa, ile % najsilniejszych sygnałów bierzemy.
# 20 oznacza: Top 20% to Long, Bottom 20% to Short. Środkowe 60% to brak pozycji.
PERCENTILE_THRESHOLD = 20  

USE_CNN       = True
FILTERS       = 32
OPTIMIZER     = 'Adam'

# Stałe systemowe
TIMESTEPS     = 30
N_MODELS      = 5
EPOCHS        = 50
BATCH_SIZE    = 64
DATA_FILE     = 'data_master_v1.csv'

# ==============================================================================
# B. ŁADOWANIE DANYCH
# ==============================================================================
print("--- [FINAL EXECUTOR: QUANTILE EDITION] ---")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Brak pliku {DATA_FILE}! Uruchom najpierw skrypt 'data_downloader.py'.")

print(f"Wczytuję dane z {DATA_FILE}...")
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

# Definicja cech
features = [c for c in data.columns if c not in ['target', 'BTC_price']]
print(f"Cechy wejściowe ({len(features)}): {features}")

# Podział Chronologiczny
TRAIN_END = '2022-12-31'
train_df = data.loc[:TRAIN_END].copy()
val_df   = data.loc['2023-01-01':].copy()

# Skalowanie
scaler = MinMaxScaler()
scaler.fit(train_df[features])

def create_dataset(df):
    if len(df) < TIMESTEPS: return np.array([]), np.array([]), np.array([])
    
    X_sc = scaler.transform(df[features])
    X, y = [], []
    
    # Zwroty do Sharpe Ratio (szukamy kolumny ze zwrotem, przesuwamy o -1)
    ret_col = [c for c in df.columns if 'BTC' in c and 'ret' in c][0]
    rets = df[ret_col].shift(-1).fillna(0).values
    
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
        
    return np.array(X), np.array(y), rets[TIMESTEPS : len(X_sc)]

X_train, y_train, r_train = create_dataset(train_df)
X_val, y_val, r_val       = create_dataset(val_df)

print(f"Zbiór Treningowy: {X_train.shape}")
print(f"Zbiór Walidacyjny: {X_val.shape}")

# ==============================================================================
# C. TRENING ENSEMBLE
# ==============================================================================
def build_model():
    input_layer = Input(shape=(TIMESTEPS, len(features)))
    x = input_layer
    
    if USE_CNN:
        x = Conv1D(filters=FILTERS, kernel_size=2, activation='relu', padding='same', 
                   kernel_regularizer=l2(REG_STRENGTH))(x)
    
    lstm_out = Bidirectional(LSTM(NEURONS, return_sequences=True, 
                                  kernel_regularizer=l2(REG_STRENGTH)))(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(DROPOUT)(lstm_out)
    
    attention_out = Attention()([lstm_out, lstm_out])
    context = GlobalAveragePooling1D()(attention_out)
    
    dense = Dense(16, activation='relu', kernel_regularizer=l2(REG_STRENGTH))(context)
    output = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=input_layer, outputs=output)
    
    opt = Adam(learning_rate=LEARNING_RATE) if OPTIMIZER == 'Adam' else Adamax(learning_rate=LEARNING_RATE)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

models = []
print(f"\nRozpoczynam trening zespołu {N_MODELS} modeli...")

for i in range(N_MODELS):
    model = build_model()
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0)
    
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              callbacks=[es],
              verbose=0, 
              shuffle=True)
    models.append(model)
    print(f" -> Model {i+1} gotowy.")

# ==============================================================================
# D. ZAAWANSOWANA ANALIZA (METODA KWANTYLOWA)
# ==============================================================================
def calculate_safe_metrics(X, y_true, returns, name):
    print(f"   > Obliczam metryki (Metoda Kwantylowa) dla: {name}...")
    
    try:
        # 1. Predykcja Ensemble
        preds_probs = np.mean([m.predict(X, verbose=0).flatten() for m in models], axis=0)
        
        # 2. Obliczanie dynamicznych progów (Kwantyle)
        # Dolny próg (dla Short): np. 20-ty percentyl
        lower_cut = np.percentile(preds_probs, PERCENTILE_THRESHOLD)
        # Górny próg (dla Long): np. 80-ty percentyl (100 - 20)
        upper_cut = np.percentile(preds_probs, 100 - PERCENTILE_THRESHOLD)
        
        print(f"     [INFO] {name} Progi -> Short < {lower_cut:.4f} | Long > {upper_cut:.4f}")
        
        # 3. Generowanie sygnałów
        # 1 = Long, -1 = Short, 0 = Brak pozycji
        signals = np.where(preds_probs > upper_cut, 1, np.where(preds_probs < lower_cut, -1, 0))
        active_mask = (signals != 0)
        
        n_trades = int(np.sum(active_mask))
        total_days = len(y_true)
        action_rate = n_trades / total_days if total_days > 0 else 0
        
        # 4. WIN RATE (Poprawiony dla Shortów)
        if n_trades == 0:
            win_rate = 0.0
        else:
            # Filtrujemy tylko aktywne dni
            signals_active = signals[active_mask]
            y_true_active = y_true[active_mask]
            
            # Konwertujemy sygnały na oczekiwaną klasę:
            # Jeśli Sygnał = 1 (Long), oczekujemy y=1
            # Jeśli Sygnał = -1 (Short), oczekujemy y=0
            expected_class = np.where(signals_active == 1, 1, 0)
            
            win_rate = accuracy_score(y_true_active, expected_class) * 100
            
        # 5. SHARPE RATIO
        try:
            strat_rets = signals * returns
            if np.std(strat_rets) == 0:
                sharpe = 0.0
            else:
                sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(365)
        except:
            sharpe = 0.0
            
        # 6. GINI
        try:
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, preds_probs)
                gini = 2 * auc - 1
            else:
                gini = 0.0
        except:
            gini = 0.0

        return {
            "Period": name,
            "Win Rate": win_rate,
            "Action Rate": action_rate * 100,
            "Trades": n_trades,
            "Sharpe": sharpe,
            "Gini": gini
        }
        
    except Exception as e:
        print(f"!!! Błąd przy obliczaniu {name}: {e}")
        return {"Period": name, "Win Rate": 0, "Action Rate": 0, "Trades": 0, "Sharpe": 0, "Gini": 0}

print("\n" + "="*80)
print(f"   RAPORT KOŃCOWY (METRYKI KWANTYLOWE)   ")
print("="*80)

stats_train = calculate_safe_metrics(X_train, y_train, r_train, "TRAINING")
stats_val   = calculate_safe_metrics(X_val, y_val, r_val, "VALIDATION")

results_df = pd.DataFrame([stats_train, stats_val])
print_df = results_df.copy()
print_df['Win Rate'] = print_df['Win Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Action Rate'] = print_df['Action Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Sharpe'] = print_df['Sharpe'].apply(lambda x: f"{x:.2f}")
print_df['Gini'] = print_df['Gini'].apply(lambda x: f"{x:.3f}")

print("\n" + print_df.to_string(index=False))
print("-" * 80)

# ==============================================================================
# E. DIAGNOZA
# ==============================================================================
val_sharpe = stats_val['Sharpe']
val_wr = stats_val['Win Rate']

print("\n🤖 [DIAGNOZA MODELU]:")
if val_wr > 52.0 and val_sharpe > 0.5:
    print(f"🚀 SUKCES: Strategia działa! Win Rate {val_wr:.1f}%, Sharpe {val_sharpe:.2f}.")
elif val_wr > 50.0:
    print(f"👍 POTENCJAŁ: Win Rate {val_wr:.1f}%. Jest lekka przewaga.")
else:
    print(f"❌ SŁABO: Win Rate {val_wr:.1f}%. Model traci lub gra losowo.")

# ==============================================================================
# F. GENEROWANIE MATERIAŁÓW DO PREZENTACJI
# ==============================================================================
def generate_presentation_material():
    print("\n--- [GENEROWANIE WYKRESÓW Z KWANTYLAMI] ---")
    
    dates = val_df.index[TIMESTEPS:]
    raw_preds = np.mean([m.predict(X_val, verbose=0).flatten() for m in models], axis=0)
    
    # --- LOGIKA KWANTYLOWA DLA WYKRESÓW ---
    lower_cut = np.percentile(raw_preds, PERCENTILE_THRESHOLD)
    upper_cut = np.percentile(raw_preds, 100 - PERCENTILE_THRESHOLD)
    
    signals = np.where(raw_preds > upper_cut, 1, np.where(raw_preds < lower_cut, -1, 0))
    # --------------------------------------

    price_col = 'BTC_price' if 'BTC_price' in val_df.columns else 'BTC'
    close_prices = val_df[price_col].iloc[TIMESTEPS:].values
    
    # 1. Equity Curve
    strat_log_returns = signals * r_val
    cum_market = np.cumsum(r_val)
    cum_strategy = np.cumsum(strat_log_returns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cum_strategy, label='Model AI (Long/Short)', color='green', linewidth=2)
    plt.plot(dates, cum_market, label='Rynek (Buy&Hold)', color='gray', linestyle='--', alpha=0.6)
    plt.title(f'Symulacja: Model (WinRate {val_wr:.1f}%) vs Rynek', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Wykres_1_Equity_Curve.png')
    print("✅ Zapisano: Wykres_1_Equity_Curve.png")

    # 2. Momenty Wejścia
    plt.figure(figsize=(14, 7))
    plt.plot(dates, close_prices, label='Cena BTC', color='black', alpha=0.5)
    
    long_mask = signals == 1
    short_mask = signals == -1
    
    if np.sum(long_mask) > 0:
        plt.scatter(dates[long_mask], close_prices[long_mask], marker='^', color='green', s=100, label='LONG', zorder=5)
    if np.sum(short_mask) > 0:
        plt.scatter(dates[short_mask], close_prices[short_mask], marker='v', color='red', s=100, label='SHORT', zorder=5)
    
    plt.title('Sygnały Modelu (Top/Bottom 20%)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Wykres_2_Sygnaly.png')
    print("✅ Zapisano: Wykres_2_Sygnaly.png")

    # 3. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_preds, bins=50, kde=True, color='purple')
    plt.axvline(x=upper_cut, color='green', linestyle='--', label=f'Próg Long ({upper_cut:.3f})')
    plt.axvline(x=lower_cut, color='red', linestyle='--', label=f'Próg Short ({lower_cut:.3f})')
    plt.title('Rozkład Pewności Modelu', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Wykres_3_Histogram.png')
    print("✅ Zapisano: Wykres_3_Histogram.png")

    # 4. Confusion Matrix (Active Only)
    active_mask = signals != 0
    if np.sum(active_mask) > 0:
        # Prawdziwe klasy (0=Spadek, 1=Wzrost)
        y_true_active = y_val[active_mask]
        # Predykcje (0=Zagraliśmy Short, 1=Zagraliśmy Long) -> mapujemy -1 na 0 dla macierzy
        y_pred_mapped = np.where(signals[active_mask] == 1, 1, 0)
        
        cm = confusion_matrix(y_true_active, y_pred_mapped, labels=[0, 1])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Short (gra na spadek)', 'Long (gra na wzrost)'], 
                    yticklabels=['Faktyczny Spadek', 'Faktyczny Wzrost'])
        plt.title('Macierz Trafień (Tylko zawarte transakcje)', fontsize=14)
        plt.xlabel('Decyzja Modelu')
        plt.ylabel('Rzeczywistość')
        plt.tight_layout()
        plt.savefig('Wykres_4_Macierz.png')
        print("✅ Zapisano: Wykres_4_Macierz.png")

generate_presentation_material()