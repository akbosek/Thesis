import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


# ==============================================================================
# A. KONFIGURACJA ZWYCIĘZCY (TUTAJ WKLEJ WYNIKI Z OPTUNY)
# ==============================================================================
# Przykładowe wartości (Zastąp je swoimi!):
NEURONS       = 64
DROPOUT       = 0.3
LEARNING_RATE = 0.001
REG_STRENGTH  = 0.0001
THRESH_DIST   = 0.02    # Margines pewności (np. 0.02 oznacza progi 0.48 i 0.52)
USE_CNN       = True    # Czy Optuna wybrała True?
FILTERS       = 32      # Jeśli USE_CNN=True
OPTIMIZER     = 'Adam'  # 'Adam' lub 'Adamax'

# Stałe systemowe
TIMESTEPS     = 30      # Okno pamięci (musi być takie samo jak w Optunie!)
N_MODELS      = 5       # Liczba modeli w zespole (Ensemble)
EPOCHS        = 50
BATCH_SIZE    = 64

# ==============================================================================
# B. ŁADOWANIE DANYCH (Z PLIKU MASTER)
# ==============================================================================
DATA_FILE = 'data_master_v1.csv'

print("--- [FINAL EXECUTOR: RESEARCH GRADE] ---")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Brak pliku {DATA_FILE}! Uruchom najpierw skrypt 'data_downloader.py'.")

print(f"Wczytuję dane z {DATA_FILE}...")
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

# Definicja cech (wszystko co nie jest targetem ani ceną surową)
# Zakładamy, że data_master_v1.csv ma już policzone wskaźniki (RSI, VIX, Returns)
features = [c for c in data.columns if c not in ['target', 'BTC_price']]
print(f"Cechy wejściowe ({len(features)}): {features}")

# Podział Chronologiczny
TRAIN_END = '2022-12-31'
train_df = data.loc[:TRAIN_END].copy()
val_df   = data.loc['2023-01-01':].copy()

# Skalowanie (Fit tylko na treningu!)
scaler = MinMaxScaler()
scaler.fit(train_df[features])

def create_dataset(df):
    if len(df) < TIMESTEPS: return np.array([]), np.array([]), np.array([])
    
    X_sc = scaler.transform(df[features])
    X, y = [], []
    # Zwroty do Sharpe Ratio (przesunięte o 1 dzień w przyszłość względem inputu)
    # Szukamy kolumny ze zwrotem BTC (zwykle BTC_ret lub BTC_USD_ret)
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
    
    # 1. Opcjonalny CNN
    if USE_CNN:
        x = Conv1D(filters=FILTERS, kernel_size=2, activation='relu', padding='same', 
                   kernel_regularizer=l2(REG_STRENGTH))(x)
    
    # 2. Bi-LSTM
    lstm_out = Bidirectional(LSTM(NEURONS, return_sequences=True, 
                                  kernel_regularizer=l2(REG_STRENGTH)))(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(DROPOUT)(lstm_out)
    
    # 3. Attention
    attention_out = Attention()([lstm_out, lstm_out])
    context = GlobalAveragePooling1D()(attention_out)
    
    # 4. Głowica
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
    # Early Stopping dla każdego modelu
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
# D. ZAAWANSOWANA ANALIZA (WERSJA BEZPIECZNA)
# ==============================================================================
def calculate_safe_metrics(X, y_true, returns, name):
    print(f"   > Obliczam metryki dla: {name}...")
    
    try:
        # Predykcja Ensemble (Uśrednianie)
        # verbose=0 blokuje śmieci w konsoli
        preds_probs = np.mean([m.predict(X, verbose=0).flatten() for m in models], axis=0)
        
        # Logika Snajpera
        upper, lower = 0.5 + THRESH_DIST, 0.5 - THRESH_DIST
        signals = np.where(preds_probs > upper, 1, np.where(preds_probs < lower, -1, 0))
        active_mask = (signals != 0)
        
        n_trades = int(np.sum(active_mask))
        total_days = len(y_true)
        action_rate = n_trades / total_days if total_days > 0 else 0
        
        # 1. WIN RATE
        if n_trades == 0:
            win_rate = 0.0
        else:
            active_preds = (preds_probs[active_mask] > 0.5).astype(int)
            win_rate = accuracy_score(y_true[active_mask], active_preds) * 100
            
        # 2. SHARPE RATIO
        try:
            strat_rets = signals * returns
            if np.std(strat_rets) == 0:
                sharpe = 0.0
            else:
                sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(365)
        except:
            sharpe = 0.0
            
        # 3. GINI
        try:
            # Gini wymaga zróżnicowanych klas. Jeśli y_true ma same 0 lub same 1, to wybuchnie.
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, preds_probs)
                gini = 2 * auc - 1
            else:
                gini = 0.0
        except:
            gini = 0.0

        return {
            "Period": name,
            "Win Rate": win_rate,     # Zwracamy jako float
            "Action Rate": action_rate * 100, # Zwracamy jako float
            "Trades": n_trades,
            "Sharpe": sharpe,
            "Gini": gini
        }
        
    except Exception as e:
        print(f"!!! Błąd przy obliczaniu {name}: {e}")
        return {"Period": name, "Win Rate": 0, "Action Rate": 0, "Trades": 0, "Sharpe": 0, "Gini": 0}

print("\n" + "="*80)
print(f"   RAPORT KOŃCOWY (METRYKI PARYSKIE)   ")
print("="*80)

# Obliczamy
stats_train = calculate_safe_metrics(X_train, y_train, r_train, "TRAINING")
stats_val   = calculate_safe_metrics(X_val, y_val, r_val, "VALIDATION")

# Tworzymy DataFrame
results_df = pd.DataFrame([stats_train, stats_val])

# Ręczne, bezpieczne formatowanie do druku (żeby nie zepsuć liczb w zmiennych)
print_df = results_df.copy()
print_df['Win Rate'] = print_df['Win Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Action Rate'] = print_df['Action Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Sharpe'] = print_df['Sharpe'].apply(lambda x: f"{x:.2f}")
print_df['Gini'] = print_df['Gini'].apply(lambda x: f"{x:.3f}")

print("\n" + print_df.to_string(index=False))
print("-" * 80)

# ==============================================================================
# E. DIAGNOZA AUTOMATYCZNA
# ==============================================================================
# Korzystamy z liczb (stats_val), a nie napisów (print_df)
tr_gini = stats_train['Gini']
val_gini = stats_val['Gini']
val_sharpe = stats_val['Sharpe']
val_wr = stats_val['Win Rate']

print("\n🤖 [DIAGNOZA MODELU]:")

# 1. Sprawdzenie Stabilności
gini_drop = tr_gini - val_gini
if gini_drop > 0.2:
    print(f"⚠️  UWAGA: Duży spadek Gini ({gini_drop:.2f}). Model przeuczony (Overfitting).")
elif gini_drop < -0.1:
    print("❓ DZIWNE: Walidacja lepsza niż trening. Sprawdź, czy dane nie są pomieszane.")
else:
    print("✅  STABILNOŚĆ: Model dobrze generalizuje (Train ~ Val).")
    
# 2. Ocena Jakości
if val_wr > 54.0 and val_sharpe > 1.0:
    print(f"🚀  SUKCES: Win Rate {val_wr:.1f}% i Sharpe {val_sharpe:.2f}. Model gotowy do testów demo.")
elif val_wr > 51.0:
    print(f"👍  POTENCJAŁ: Win Rate {val_wr:.1f}%. Jest lekka przewaga, ale ryzykowna.")
else:
    print(f"❌  SŁABO: Win Rate {val_wr:.1f}% to rzut monetą. Zmień parametry lub dane.")

# ==============================================================================
# F. GENEROWANIE MATERIAŁÓW DO PREZENTACJI (POPRAWIONE)
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_presentation_material():
    print("\n--- [GENEROWANIE WYKRESÓW DO PREZENTACJI] ---")
    
    # 1. Przygotowanie danych
    # Ucinamy pierwsze dni (TIMESTEPS), bo dla nich model nie ma predykcji
    dates = val_df.index[TIMESTEPS:]
    
    # Średnia predykcja Zespołu (Ensemble)
    raw_preds = np.mean([m.predict(X_val, verbose=0).flatten() for m in models], axis=0)
    
    # Sygnały (Snajper)
    upper, lower = 0.5 + THRESH_DIST, 0.5 - THRESH_DIST
    signals = np.where(raw_preds > upper, 1, np.where(raw_preds < lower, -1, 0))
    
    # --- POPRAWKA TUTAJ: Używamy 'BTC_price' zamiast 'BTC' ---
    # Sprawdzamy dostępną nazwę kolumny z ceną
    price_col = 'BTC_price' if 'BTC_price' in val_df.columns else 'BTC'
    close_prices = val_df[price_col].iloc[TIMESTEPS:].values
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # WYKRES 1: EQUITY CURVE (Symulacja Portfela)
    # ---------------------------------------------------------
    # r_val to zwroty logarytmiczne z rynku (przesunięte). 
    # Uwaga: r_val ma długość zgodną z X_val, więc pasuje do signals.
    strat_log_returns = signals * r_val
    
    # Kumulatywny zwrot (Cumulative Returns)
    cum_market = np.cumsum(r_val)
    cum_strategy = np.cumsum(strat_log_returns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cum_strategy, label='Model AI (Twoja Strategia)', color='green', linewidth=2)
    plt.plot(dates, cum_market, label='Buy & Hold (Rynek BTC)', color='gray', linestyle='--', alpha=0.6)
    plt.title(f'Symulacja Wyniku Finansowego (Validation WinRate: {val_wr:.2f}%)', fontsize=14)
    plt.ylabel('Skumulowany Zwrot (Log)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Wykres_1_Equity_Curve.png')
    print("✅ Zapisano: Wykres_1_Equity_Curve.png")

    # ---------------------------------------------------------
    # WYKRES 2: MOMENTY WEJŚCIA (Gdzie model grał?)
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 7))
    plt.plot(dates, close_prices, label='Cena BTC', color='black', alpha=0.5)
    
    # Filtrujemy tylko momenty zagrania
    long_mask = signals == 1
    short_mask = signals == -1
    
    if np.sum(long_mask) > 0:
        plt.scatter(dates[long_mask], close_prices[long_mask], marker='^', color='green', s=100, label='Sygnał LONG', zorder=5)
    if np.sum(short_mask) > 0:
        plt.scatter(dates[short_mask], close_prices[short_mask], marker='v', color='red', s=100, label='Sygnał SHORT', zorder=5)
    
    plt.title('Decyzje Modelu na tle Ceny BTC (Walidacja)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Wykres_2_Sygnaly.png')
    print("✅ Zapisano: Wykres_2_Sygnaly.png")

    # ---------------------------------------------------------
    # WYKRES 3: ROZKŁAD PEWNOŚCI (Histogram)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_preds, bins=50, kde=True, color='purple')
    
    # Rysujemy linie progów
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.axvline(x=upper, color='green', linestyle=':', label=f'Próg Long ({upper})')
    plt.axvline(x=lower, color='red', linestyle=':', label=f'Próg Short ({lower})')
    
    plt.title('Pewność Modelu (Histogram Predykcji)', fontsize=14)
    plt.xlabel('Prawdopodobieństwo (0 = Spadek, 1 = Wzrost)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Wykres_3_Histogram.png')
    print("✅ Zapisano: Wykres_3_Histogram.png")

    # ---------------------------------------------------------
    # WYKRES 4: CONFUSION MATRIX (Tylko dla aktywnych zagrań)
    # ---------------------------------------------------------
    active_mask = signals != 0
    if np.sum(active_mask) > 0:
        y_true_active = y_val[active_mask]
        y_pred_active = (raw_preds[active_mask] > 0.5).astype(int)
        
        # Sprawdź czy mamy obie klasy w y_true, żeby macierz nie wybuchła
        labels = [0, 1]
        cm = confusion_matrix(y_true_active, y_pred_active, labels=labels)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Spadek', 'Wzrost'], 
                    yticklabels=['Spadek', 'Wzrost'])
        plt.title('Macierz Trafień (Tylko zawarte transakcje)', fontsize=14)
        plt.xlabel('Przewidywanie Modelu')
        plt.ylabel('Faktyczny Kierunek Rynku')
        plt.tight_layout()
        plt.savefig('Wykres_4_Macierz.png')
        print("✅ Zapisano: Wykres_4_Macierz.png")
    else:
        print("⚠️ Brak transakcji - Macierz pomyłek nie została wygenerowana.")

# Uruchamiamy generator
generate_presentation_material()