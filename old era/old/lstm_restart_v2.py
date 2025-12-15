import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ==========================================
#      HIPERPARAMETRY (DO TESTÓW)
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'
TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' 
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31' 

# Model
TIMESTEPS = 14       # Zwiększamy nieco okno
NEURONS   = 64       # Więcej neuronów, żeby "złapał" sygnał
DROPOUT   = 0.2      # Mniejszy dropout
LEARNING_RATE = 0.001 
EPOCHS    = 50       
BATCH_SIZE = 32      

# Strategia
THRESH_SHORT = 0.48
THRESH_LONG  = 0.52
# ==========================================

print("--- [LSTM RESTART V2: DIAGNOSTIC] Inicjalizacja ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# 1. DANE
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)

# Feature Engineering
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Inputy (Twoje wybrane + RSI dla momentum)
data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
# Dodaję prostą średnią momentum, żeby model miał punkt odniesienia
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

data.dropna(inplace=True)

features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']
print(f"Features: {features}")

# 2. PODZIAŁ
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

# 3. SKALOWANIE
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    
    # Do symulacji equity musimy wziąć zwrot z PRZYSZŁEGO dnia (shift -1)
    # Ponieważ target[t] dotyczy zmiany t -> t+1
    future_returns = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    
    return np.array(X), np.array(y), future_returns, df.index[TIMESTEPS:]

X_train, y_train, ret_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, idx_val = create_dataset(val_df)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# 4. MODELOWANIE (Bez L2, z BatchNorm)
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(NEURONS, return_sequences=False), # Czysty LSTM
    BatchNormalization(), # Stabilizacja (pomaga na "martwe" wyjścia)
    Dropout(DROPOUT),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
estop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("\n--- Trening ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[estop],
    verbose=0
)

# 5. DIAGNOSTYKA PREDYKCJI
p_train = model.predict(X_train).flatten()
p_val   = model.predict(X_val).flatten()

print("\n--- STATYSTYKI PREDYKCJI (Czy model żyje?) ---")
print(f"TRAIN -> Min: {p_train.min():.4f}, Max: {p_train.max():.4f}, Mean: {p_train.mean():.4f}, Std: {p_train.std():.4f}")
print(f"VAL   -> Min: {p_val.min():.4f}, Max: {p_val.max():.4f}, Mean: {p_val.mean():.4f}, Std: {p_val.std():.4f}")

if p_val.std() < 0.005:
    print("⚠️  UWAGA: Model jest 'martwy' (zwraca stałą wartość). Zmień architekturę lub dane.")

# 6. METRYKI I STRATEGIA
def calculate_metrics(probs, returns, y_true, name, use_thresholds=True):
    # Logika pozycji
    if use_thresholds:
        pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    else:
        # Wersja "Forced" - zawsze zajmij pozycję
        pos = np.where(probs > 0.5, 1, -1)
        
    # Wynik strategii
    strat_ret = pos * returns
    
    # Win Rate (tylko aktywne)
    active_mask = pos != 0
    if np.sum(active_mask) > 0:
        # Win = Znak pozycji zgodny ze znakiem zwrotu
        # Uwaga: returns to log_return. sign(log_return) == sign(price_change)
        wins = np.sign(pos[active_mask]) == np.sign(returns[active_mask])
        win_rate = np.mean(wins) * 100
    else:
        win_rate = 0.0
        
    # Gini
    try:
        auc = roc_auc_score(y_true, probs)
        gini = 2 * auc - 1
    except: gini = 0.0
    
    # Sharpe & Volatility (Annualized)
    # Zakładamy 365 dni handlowych dla krypto
    if np.std(strat_ret) == 0:
        sharpe = 0
        volatility = 0
    else:
        sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
        volatility = np.std(strat_ret) * np.sqrt(365) * 100
        
    # Max Drawdown
    cum_ret = np.exp(np.cumsum(strat_ret))
    peak = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - peak) / peak
    max_dd = np.min(dd) * 100
    
    return {
        "Dataset": name,
        "Win Rate %": round(win_rate, 2),
        "Gini": round(gini, 4),
        "Sharpe": round(sharpe, 2),
        "Volatility %": round(volatility, 2),
        "Max DD %": round(max_dd, 2),
        "Trades": np.sum(active_mask)
    }

# Raport
m_train = calculate_metrics(p_train, ret_train, y_train, "Train (Thresh)")
m_val   = calculate_metrics(p_val, ret_val, y_val, "Val (Thresh)")
m_val_force = calculate_metrics(p_val, ret_val, y_val, "Val (Forced >0.5)", use_thresholds=False)

res_df = pd.DataFrame([m_train, m_val, m_val_force])
print("\n=== RAPORT KOŃCOWY ===")
print(res_df.to_string(index=False))

# Zapis
res_df.to_csv('restart_v2_metrics.csv', index=False)

# Wykres Equity (Dla wersji z progami i Forced)
cum_bh = np.exp(np.cumsum(ret_val))
cum_strat = np.exp(np.cumsum(np.where(p_val > THRESH_LONG, 1, np.where(p_val < THRESH_SHORT, -1, 0)) * ret_val))
cum_force = np.exp(np.cumsum(np.where(p_val > 0.5, 1, -1) * ret_val))

plt.figure(figsize=(12, 6))
plt.plot(idx_val, cum_bh, label='Buy & Hold', color='gray', alpha=0.5)
plt.plot(idx_val, cum_strat, label='Strategy (Thresholds)', color='blue')
plt.plot(idx_val, cum_force, label='Strategy (Forced)', color='orange', linestyle='--')
plt.title(f"Porównanie Strategii (Walidacja {VAL_START[:4]})")
plt.legend()
plt.savefig('restart_v2_chart.png')
print("\nZapisano: restart_v2_metrics.csv, restart_v2_chart.png")