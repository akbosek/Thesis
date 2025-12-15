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
# A. KONFIGURACJA (TREND AWARE)
# ==============================================================================
NEURONS       = 64
DROPOUT       = 0.3
LEARNING_RATE = 0.001
REG_STRENGTH  = 0.0001

# --- KONFIGURACJA PROGÓW ---
# Nadal używamy asymetrii, ale teraz mamy bezpiecznik w postaci SMA
P_LONG  = 25   # Top 25% -> Long
P_SHORT = 10   # Bottom 10% -> Short (ale tylko gdy cena jest pod SMA!)

USE_CNN       = True
FILTERS       = 32
OPTIMIZER     = 'Adam'
TIMESTEPS     = 30
N_MODELS      = 5
EPOCHS        = 50
BATCH_SIZE    = 64
DATA_FILE     = 'data_master_v1.csv'

# ==============================================================================
# B. ŁADOWANIE DANYCH I WSKAŹNIK TRENDU
# ==============================================================================
print("--- [FINAL EXECUTOR: TREND FILTER EDITION] ---")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Brak pliku {DATA_FILE}!")

print(f"Wczytuję dane z {DATA_FILE}...")
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

# 1. Sprawdzamy kolumnę z ceną
price_col = 'BTC_price' if 'BTC_price' in data.columns else 'BTC'

# 2. OBLICZAMY FILTR TRENDU (SMA 50)
# Średnia z 50 dni. Jeśli cena jest nad nią - mamy Hossę.
data['SMA_FILTER'] = data[price_col].rolling(window=50).mean()
# Uzupełniamy braki na początku (bfill)
data['SMA_FILTER'] = data['SMA_FILTER'].fillna(method='bfill')

print("✅ Obliczono filtr trendu (SMA 50).")

features = [c for c in data.columns if c not in ['target', price_col, 'SMA_FILTER']]
print(f"Cechy wejściowe ({len(features)}): {features}")

# Podział Chronologiczny
TRAIN_END = '2022-12-31'
train_df = data.loc[:TRAIN_END].copy()
val_df   = data.loc['2023-01-01':].copy()

scaler = MinMaxScaler()
scaler.fit(train_df[features])

def create_dataset_with_filter(df):
    if len(df) < TIMESTEPS: return np.array([]), np.array([]), np.array([]), np.array([])
    
    X_sc = scaler.transform(df[features])
    X, y = [], []
    
    # Pobieramy zwroty
    ret_col = [c for c in df.columns if 'BTC' in c and 'ret' in c][0]
    rets = df[ret_col].shift(-1).fillna(0).values
    
    # Pobieramy surowe ceny i SMA do filtrowania
    prices = df[price_col].values
    smas   = df['SMA_FILTER'].values
    
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
        
    # Zwracamy: X, y, Zwroty, (Ceny, SMA)
    # Ceny i SMA muszą być wyrównane do predykcji (czyli od TIMESTEPS do końca)
    return (np.array(X), 
            np.array(y), 
            rets[TIMESTEPS : len(X_sc)], 
            prices[TIMESTEPS : len(X_sc)], 
            smas[TIMESTEPS : len(X_sc)])

# Rozpakowujemy dane
X_train, y_train, r_train, p_train, sma_train = create_dataset_with_filter(train_df)
X_val, y_val, r_val, p_val, sma_val           = create_dataset_with_filter(val_df)

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
# D. LOGIKA FILTROWANIA TRENDU
# ==============================================================================
def apply_trend_filter(raw_signals, prices, smas):
    """
    Nakłada filtr: Jeśli Cena > SMA, zamień wszystkie Shorty (-1) na Neutral (0).
    """
    filtered_signals = raw_signals.copy()
    
    # Tworzymy maskę: Gdzie Cena > SMA (Trend Wzrostowy)
    bull_market_mask = prices > smas
    
    # Znajdujemy indeksy, gdzie mamy SHORT (-1) ORAZ jest Bull Market
    problematic_shorts = (filtered_signals == -1) & bull_market_mask
    
    # Zerujemy te shorty (wymuszamy wyjście do gotówki zamiast grania na spadek)
    filtered_signals[problematic_shorts] = 0
    
    n_removed = np.sum(problematic_shorts)
    print(f"     🛡️ [SMA FILTER] Zablokowano {n_removed} ryzykownych pozycji SHORT (Trend Wzrostowy).")
    
    return filtered_signals

def calculate_smart_metrics(X, y_true, returns, prices, smas, name):
    print(f"   > Obliczam Smart Metryki dla: {name}...")
    
    try:
        preds_probs = np.mean([m.predict(X, verbose=0).flatten() for m in models], axis=0)
        
        # Progi Percentylowe
        lower_cut = np.percentile(preds_probs, P_SHORT)
        upper_cut = np.percentile(preds_probs, 100 - P_LONG)
        
        # 1. Sygnały Surowe (z modelu AI)
        raw_signals = np.where(preds_probs > upper_cut, 1, np.where(preds_probs < lower_cut, -1, 0))
        
        # 2. ZASTOSOWANIE FILTRA TRENDU
        final_signals = apply_trend_filter(raw_signals, prices, smas)
        
        active_mask = (final_signals != 0)
        n_trades = int(np.sum(active_mask))
        total_days = len(y_true)
        action_rate = n_trades / total_days if total_days > 0 else 0
        
        if n_trades == 0:
            win_rate = 0.0
        else:
            signals_active = final_signals[active_mask]
            y_true_active = y_true[active_mask]
            expected_class = np.where(signals_active == 1, 1, 0)
            win_rate = accuracy_score(y_true_active, expected_class) * 100
            
        try:
            strat_rets = final_signals * returns
            if np.std(strat_rets) == 0: sharpe = 0.0
            else: sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(365)
        except: sharpe = 0.0
        
        try:
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, preds_probs)
                gini = 2 * auc - 1
            else: gini = 0.0
        except: gini = 0.0

        return {
            "Period": name,
            "Win Rate": win_rate,
            "Action Rate": action_rate * 100,
            "Trades": n_trades,
            "Sharpe": sharpe,
            "Gini": gini
        }
    except Exception as e:
        print(f"!!! Błąd: {e}")
        return {"Period": name, "Win Rate": 0, "Action Rate": 0, "Trades": 0, "Sharpe": 0, "Gini": 0}

print("\n" + "="*80)
print(f"   RAPORT KOŃCOWY (SMA TREND FILTERED)   ")
print("="*80)

stats_train = calculate_smart_metrics(X_train, y_train, r_train, p_train, sma_train, "TRAINING")
stats_val   = calculate_smart_metrics(X_val, y_val, r_val, p_val, sma_val, "VALIDATION")

results_df = pd.DataFrame([stats_train, stats_val])
print_df = results_df.copy()
print_df['Win Rate'] = print_df['Win Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Action Rate'] = print_df['Action Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Sharpe'] = print_df['Sharpe'].apply(lambda x: f"{x:.2f}")

print("\n" + print_df.to_string(index=False))
print("-" * 80)

# ==============================================================================
# F. WYKRESY Z FILTREM
# ==============================================================================
def generate_presentation_material():
    print("\n--- [GENEROWANIE WYKRESÓW Z FILTREM SMA] ---")
    
    dates = val_df.index[TIMESTEPS:]
    raw_preds = np.mean([m.predict(X_val, verbose=0).flatten() for m in models], axis=0)
    
    # Logika progów
    lower_cut = np.percentile(raw_preds, P_SHORT)
    upper_cut = np.percentile(raw_preds, 100 - P_LONG)
    
    raw_signals = np.where(raw_preds > upper_cut, 1, np.where(raw_preds < lower_cut, -1, 0))
    
    # --- APLIKACJA FILTRA DO WYKRESÓW ---
    final_signals = apply_trend_filter(raw_signals, p_val, sma_val)
    # ------------------------------------

    # 1. Equity Curve
    strat_log_returns = final_signals * r_val
    cum_market = np.cumsum(r_val)
    cum_strategy = np.cumsum(strat_log_returns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cum_strategy, label='AI + SMA Filter', color='#00C805', linewidth=2)
    plt.plot(dates, cum_market, label='Rynek (Buy&Hold)', color='gray', linestyle='--', alpha=0.5)
    plt.title(f'Wynik Strategii (WinRate: {stats_val["Win Rate"]:.2f}%)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Wykres_1_Equity_Curve.png')
    
    # 2. Momenty Wejścia (+ linia SMA)
    plt.figure(figsize=(14, 7))
    plt.plot(dates, p_val, label='Cena BTC', color='black', alpha=0.6)
    plt.plot(dates, sma_val, label='Filtr SMA 50', color='blue', linestyle='--', alpha=0.7)
    
    long_mask = final_signals == 1
    short_mask = final_signals == -1
    
    if np.sum(long_mask) > 0:
        plt.scatter(dates[long_mask], p_val[long_mask], marker='^', color='green', s=80, label='LONG', zorder=5)
    if np.sum(short_mask) > 0:
        plt.scatter(dates[short_mask], p_val[short_mask], marker='v', color='red', s=100, edgecolors='black', label='SHORT', zorder=5)
    
    plt.title('Transakcje Modelu na tle Trendu', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Wykres_2_Sygnaly.png')
    
    # 3. Macierz
    active_mask = final_signals != 0
    if np.sum(active_mask) > 0:
        y_true_active = y_val[active_mask]
        y_pred_mapped = np.where(final_signals[active_mask] == 1, 1, 0)
        cm = confusion_matrix(y_true_active, y_pred_mapped, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Short', 'Long'], yticklabels=['Spadek', 'Wzrost'])
        plt.title('Macierz Trafień (Po filtracji)', fontsize=14)
        plt.tight_layout()
        plt.savefig('Wykres_4_Macierz.png')

    print("✅ Gotowe.")

generate_presentation_material()