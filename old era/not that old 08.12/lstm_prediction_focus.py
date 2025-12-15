import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ==========================================
#      KONFIGURACJA PREDYKCYJNA
# ==========================================
INPUT_FILE = 'BTC_USD_1d_2014_2024.csv' # Plik z Yahoo Finance

# Nowy podział (Dłuższa historia)
TRAIN_START = '2014-09-17' # Start danych Yahoo
TRAIN_END   = '2022-12-31' # Koniec bessy 2022
VAL_START   = '2023-01-01' # Start 2023
VAL_END     = '2024-12-31' # Koniec 2024

# Parametry (Sprawdzone V12 + Trial 394 Mix)
N_MODELS      = 5         # Ensemble (Stabilizacja)
TIMESTEPS     = 21        # Dłuższa pamięć (bo mamy więcej historii)
NEURONS       = 64
DROPOUT       = 0.3
LEARNING_RATE = 0.001
EPOCHS        = 40
BATCH_SIZE    = 32

# Progi Decyzyjne (Dla Win Rate)
THRESH_SHORT  = 0.49
THRESH_LONG   = 0.51
# ==========================================

print("--- [LSTM: PURE PREDICTION FOCUS] ---")
if not os.path.exists(INPUT_FILE): 
    raise FileNotFoundError(f"Brak pliku {INPUT_FILE}! Uruchom najpierw skrypt pobierania.")

# 1. DANE
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
# Target: 1 jeśli cena wzrosła, 0 jeśli spadła
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Feature Engineering
data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

# Filtr Trendu (Pomocniczy do metryk Sharpe, ale nie blokujący predykcji w raporcie)
data['SMA_200'] = data['Close'].rolling(window=200).mean()

data.dropna(inplace=True)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']

# 2. PODZIAŁ
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

print(f"Trening:   {len(train_df)} dni ({TRAIN_START} - {TRAIN_END})")
print(f"Walidacja: {len(val_df)} dni ({VAL_START} - {VAL_END})")

# 3. SKALOWANIE
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    
    # Returns do Sharpe'a
    ret = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    return np.array(X), np.array(y), ret, df.index[TIMESTEPS:]

X_train, y_train, ret_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, idx_val = create_dataset(val_df)

# 4. TRENING ENSEMBLE
models = []
print(f"\n--- Trenowanie {N_MODELS} modeli na {len(X_train)} próbkach... ---")

for i in range(N_MODELS):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(NEURONS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=False)
    models.append(model)
    print(f"   -> Model {i+1} gotowy.")

# 5. AGREGACJA I METRYKI
p_train = np.mean([m.predict(X_train, verbose=0).flatten() for m in models], axis=0)
p_val   = np.mean([m.predict(X_val, verbose=0).flatten() for m in models], axis=0)

def calculate_pure_metrics(probs, y_true, ret, name):
    # Decyzja (Long/Short/Flat)
    pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    
    # Liczniki
    total_days = len(probs)
    active_days = np.sum(pos != 0)
    active_pct = (active_days / total_days) * 100
    
    # Win Rate (Tylko aktywne)
    if active_days > 0:
        # Win = Znak pozycji zgodny ze znakiem zwrotu
        # (ret > 0 i pos=1) LUB (ret < 0 i pos=-1)
        correct_dir = np.sign(pos[pos!=0]) == np.sign(ret[pos!=0])
        win_rate = np.mean(correct_dir) * 100
    else:
        win_rate = 0
        
    # Gini
    try: gini = 2 * roc_auc_score(y_true, probs) - 1
    except: gini = 0
    
    # Sharpe (Jako miara jakości sygnału risk-adjusted)
    strat_ret = pos * ret
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    return {
        "Period": name,
        "Win Rate %": round(win_rate, 2),
        "Gini": round(gini, 3),
        "Sharpe": round(sharpe, 2),
        "Trades Taken": active_days,
        "Total Days": total_days,
        "Activity %": round(active_pct, 1)
    }

print("\n--- Obliczanie Wyników ---")
m_train = calculate_pure_metrics(p_train, y_train, ret_train, "TRAINING (2014-2022)")
m_val   = calculate_pure_metrics(p_val, y_val, ret_val, "VALIDATION (2023-2024)")

# 6. RAPORT KOŃCOWY
results = pd.DataFrame([m_train, m_val])

print("\n" + "="*85)
print("   PREDICTIVE PERFORMANCE REPORT (NO FINANCIAL NOISE)   ")
print("="*85)
# Reorder columns for readability
cols = ["Period", "Win Rate %", "Gini", "Sharpe", "Trades Taken", "Total Days", "Activity %"]
print(results[cols].to_string(index=False))
print("-" * 85)

# Interpretacja
wr = m_val['Win Rate %']
act = m_val['Activity %']

print("\n🤖 [AI ADVISOR - PREDICTION QUALITY]:")
if wr > 55.0:
    print(f"🌟 WYBITNY WYNIK: Win Rate {wr}% na długim okresie walidacji (2 lata)!")
    print("   Ten model ma realną 'kryształową kulę'.")
elif wr > 52.0:
    print(f"✅ DOBRY WYNIK: Win Rate {wr}% daje statystyczną przewagę (Edge).")
    print("   Przy odpowiednim zarządzaniu ryzykiem to zarabia.")
else:
    print(f"⚠️ NA GRANICY: Win Rate {wr}% jest blisko losowości.")
    print("   Wymaga poprawy progów lub filtrów.")

if act < 15.0:
    print(f"ℹ️ UWAGA: Model jest bardzo wybredny (aktywny tylko w {act}% dni).")
    print("   To dobrze dla precyzji, ale wymaga cierpliwości.")

print("="*85)