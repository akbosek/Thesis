import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

# ==============================================================================
# A. KONFIGURACJA ZWYCIĘZCY (TUTAJ WKLEJ WYNIKI Z OPTUNY)
# ==============================================================================
# Przykładowe wartości (podmień je na te z logów Optymalizatora):
NEURONS       = 64
FILTERS       = 32
KERNEL_SIZE   = 3
DROPOUT       = 0.4152077232431495
LEARNING_RATE = 0.0006278716148885617
THRESH_DIST   = 0.017777720171110845
# ==============================================================================
# B. USTAWIENIA SYSTEMOWE
# ==============================================================================
INPUT_FILE = 'BTC_USD_1d_2014_2024.csv'
TIMESTEPS  = 21            # Okno pamięci (ile dni wstecz patrzymy)
PREDICTION_HORIZON = 1     # 1 = Przewidujemy zwrot na jutro (lub 3, jeśli tak szkoliłeś)
N_MODELS   = 5             # Ensemble: Trenujemy 5 modeli i uśredniamy wynik
EPOCHS     = 50            # Liczba epok na model
BATCH_SIZE = 32

# Zakresy Dat (Chronologiczne)
TRAIN_START = '2014-09-17'
TRAIN_END   = '2022-12-31'
VAL_START   = '2023-01-01'
VAL_END     = '2024-12-31'

# ==============================================================================
# 1. PRZYGOTOWANIE DANYCH (SILNIK HYBRYDOWY)
# ==============================================================================
print("--- [1. ŁADOWANIE I PRZETWARZANIE DANYCH] ---")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Brak pliku {INPUT_FILE}! Upewnij się, że jest w folderze.")

data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)

# 1. Target: Czy cena za X dni będzie wyższa? (1=Tak, 0=Nie)
data['target'] = np.where(data['Close'].shift(-PREDICTION_HORIZON) > data['Close'], 1, 0)

# 2. Wskaźniki Techniczne (Feature Engineering)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# RSI (Siła Trendu)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['rsi_norm'] = (100 - (100 / (1 + rs))) / 100.0

# Odległość od średniej (SMA 50)
data['sma_50'] = data['Close'].rolling(50).mean()
data['dist_sma50'] = (data['Close'] - data['sma_50']) / data['sma_50']

# Zmienność (Volatility)
data['volatility_14'] = data['log_return'].rolling(14).std()

data.dropna(inplace=True)

# Lista cech wejściowych
features = ['log_return', 'rsi_norm', 'dist_sma50', 'volatility_14']
print(f"Cechy wejściowe: {features}")

# 3. Podział na zbiory (Chronologicznie)
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

print(f"Dane Treningowe: {len(train_df)} dni")
print(f"Dane Walidacyjne: {len(val_df)} dni")

# 4. Skalowanie (Fitujemy TYLKO na treningu, żeby uniknąć Data Leakage)
scaler = MinMaxScaler()
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    # Zwroty przesunięte o horyzont (do obliczania Sharpe Ratio)
    future_returns = df['log_return'].shift(-PREDICTION_HORIZON).fillna(0).values
    
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
        
    # Przycinamy returns do długości X
    active_returns = future_returns[TIMESTEPS : len(X_sc)]
    return np.array(X), np.array(y), active_returns

X_train, y_train, r_train = create_dataset(train_df)
X_val, y_val, r_val       = create_dataset(val_df)

print(f"Gotowe sekwencje (X_train): {X_train.shape}")

# ==============================================================================
# 2. TRENING ZESPOŁU MODELI (ENSEMBLE CNN-GRU)
# ==============================================================================
print(f"\n--- [2. START TRENINGU ENSEMBLE ({N_MODELS} MODELI)] ---")
models = []

for i in range(N_MODELS):
    print(f"   -> Trenowanie modelu {i+1}/{N_MODELS}...")
    
    # Budowa modelu Hybrydowego
    model = Sequential([
        Input(shape=(TIMESTEPS, len(features))),
        
        # A. Część wizualna (CNN) - widzi kształty
        Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        
        # B. Część pamięciowa (GRU) - lżejsza i szybsza niż LSTM
        GRU(NEURONS, return_sequences=False),
        
        BatchNormalization(),
        Dropout(DROPOUT),
        
        # C. Decyzja
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # Trenujemy (bez verbose, żeby nie śmiecić w konsoli)
    model.fit(X_train, y_train, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              verbose=0, 
              shuffle=True)
    
    models.append(model)

print("Trening zakończony pomyślnie.")

# ==============================================================================
# 3. EWALUACJA I RAPORT (TRAIN VS VAL)
# ==============================================================================
def evaluate_ensemble(X, y_true, returns, period_name):
    # Pobieramy predykcje z każdego modelu
    all_preds = [m.predict(X, verbose=0).flatten() for m in models]
    # Uśredniamy (Ensemble averaging)
    avg_preds = np.mean(all_preds, axis=0)
    
    # Progi Snajpera
    upper = 0.5 + THRESH_DIST
    lower = 0.5 - THRESH_DIST
    
    # Logika sygnałów (1: Long, -1: Short, 0: Flat)
    signals = np.where(avg_preds > upper, 1, np.where(avg_preds < lower, -1, 0))
    
    # Statystyki aktywności
    active_mask = (signals != 0)
    n_trades = np.sum(active_mask)
    total = len(y_true)
    action_rate = (n_trades / total) * 100
    
    # Win Rate (tylko dla aktywnych zagrań)
    if n_trades == 0:
        win_rate = 0.0
        sharpe = 0.0
    else:
        # Porównanie: Czy Long(1) trafił Wzrost(1) lub Short(-1) trafił Spadek(0)
        # Konwersja y_true na 1/-1 nie jest konieczna, jeśli zrobimy tak:
        # Predykcja binarna dla aktywnych: >0.5 to 1, <=0.5 to 0
        pred_class_active = (avg_preds[active_mask] > 0.5).astype(int)
        true_class_active = y_true[active_mask]
        win_rate = accuracy_score(true_class_active, pred_class_active) * 100
        
        # Sharpe Ratio
        strat_returns = signals * returns
        if np.std(strat_returns) == 0:
            sharpe = 0
        else:
            sharpe = (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(365)
            
    # Gini
    try: gini = 2 * roc_auc_score(y_true, avg_preds) - 1
    except: gini = 0.0
    
    return {
        "Period": period_name,
        "Win Rate": f"{win_rate:.2f}%",
        "Action Rate": f"{action_rate:.2f}%",
        "Trades": n_trades,
        "Sharpe": f"{sharpe:.2f}",
        "Gini": f"{gini:.3f}"
    }

print("\n--- [3. GENEROWANIE RAPORTU] ---")
stats_train = evaluate_ensemble(X_train, y_train, r_train, "TRAINING (2014-2022)")
stats_val   = evaluate_ensemble(X_val, y_val, r_val, "VALIDATION (2023-2024)")

# Tworzenie tabeli
results_df = pd.DataFrame([stats_train, stats_val])

print("\n" + "="*80)
print(f"   FINALNY RAPORT SKUTECZNOŚCI (HYBRYDA CNN+GRU)   ")
print("="*80)
# Reorder kolumn dla czytelności
cols = ["Period", "Win Rate", "Sharpe", "Action Rate", "Trades", "Gini"]
print(results_df[cols].to_string(index=False))
print("-" * 80)

# Interpretacja AI
wr_val = float(stats_val["Win Rate"].strip('%'))
act_val = float(stats_val["Action Rate"].strip('%'))
tr_wr = float(stats_train["Win Rate"].strip('%'))

print("\n🤖 [DIAGNOZA AI]:")
if wr_val > 55.0:
    print("🚀 ŚWIETNY WYNIK! Model ma solidną przewagę (Edge).")
    if abs(tr_wr - wr_val) < 5.0:
        print("   ✅ Brak przetrenowania (różnica Train-Val jest mała).")
    else:
        print("   ⚠️ Uwaga: Wynik na treningu jest znacznie wyższy. Monitoruj na żywo.")
elif wr_val > 52.0:
    print("👍 DOBRY WYNIK. Model zarabia, ale wymaga dyscypliny (Action Rate > 10%).")
else:
    print("❌ WYNIK SŁABY. Model hybrydowy nie znalazł wzorca. Sprawdź dane wejściowe.")

if act_val < 10.0:
    print("⚠️ BARDZO NISKA AKTYWNOŚĆ. Zmniejsz 'THRESH_DIST', aby grać częściej.")