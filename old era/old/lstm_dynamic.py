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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================================
#      HIPERPARAMETRY (RATUNKOWE)
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# DATY
TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31'
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31'

# MODEL (Powrót do stabilności)
TIMESTEPS = 14
NEURONS   = 64
DROPOUT   = 0.2
LEARNING_RATE = 0.001 # Spokojniejsza nauka
EPOCHS    = 50
BATCH_SIZE = 32

# STRATEGIA DYNAMICZNA
# "Graj tylko na X% najsilniejszych sygnałów"
PERCENTILE_THRESHOLD = 80 # Górne 20% na Long
# ==========================================

print("--- [LSTM V8: DYNAMIC DEFIBRILLATOR] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# 1. DANE
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

data.dropna(inplace=True)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']

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
    future_returns = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    return np.array(X), np.array(y), future_returns, df.index[TIMESTEPS:]

X_train, y_train, ret_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, idx_val = create_dataset(val_df)

# 4. TRENING (STABILNY)
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(NEURONS, return_sequences=False),
    BatchNormalization(), # <--- PRZYWRÓCONE (To ratuje przed Std=0)
    Dropout(DROPOUT),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
estop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("\n--- Reanimacja Modelu (Trening) ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[estop],
    verbose=0
)

# 5. OBLICZANIE DYNAMICZNYCH PROGÓW
p_train = model.predict(X_train, verbose=0).flatten()
p_val   = model.predict(X_val, verbose=0).flatten()

print(f"\n--- Kalibracja Progów (Na podstawie Treningu) ---")
print(f"Train Std Dev: {p_train.std():.5f} (Musi być > 0)")

# Obliczamy percentyle na zbiorze treningowym
thresh_long_dynamic = np.percentile(p_train, PERCENTILE_THRESHOLD)      # np. 80-ty percentyl (Top 20%)
thresh_short_dynamic = np.percentile(p_train, 100 - PERCENTILE_THRESHOLD) # np. 20-ty percentyl (Bottom 20%)

print(f"Wyliczone Progi Dynamiczne:")
print(f"SHORT poniżej: {thresh_short_dynamic:.5f}")
print(f"LONG  powyżej: {thresh_long_dynamic:.5f}")

# 6. DIAGNOSTYKA
def get_metrics(probs, y_true, ret, name):
    # Używamy wyliczonych progów dynamicznych
    pos = np.where(probs > thresh_long_dynamic, 1, np.where(probs < thresh_short_dynamic, -1, 0))
    strat_ret = pos * ret
    
    active = pos != 0
    trades = np.sum(active)
    if trades > 0:
        wr = (np.sign(pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    # Sharpe
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    # Returns
    cum_strat = np.exp(np.cumsum(strat_ret))
    cum_bh = np.exp(np.cumsum(ret))
    
    total_ret = (cum_strat[-1] - 1) * 100
    bh_ret = (cum_bh[-1] - 1) * 100
    
    # Max DD
    dd = (cum_strat / np.maximum.accumulate(cum_strat) - 1).min() * 100
    
    return {
        "Dataset": name, 
        "WinRate %": round(wr, 2), 
        "Sharpe": round(sharpe, 2), 
        "Return %": round(total_ret, 1), 
        "BH Return %": round(bh_ret, 1),
        "MaxDD %": round(dd, 1), 
        "Trades": trades,
        "StdDev": round(np.std(probs), 4)
    }

m_train = get_metrics(p_train, y_train, ret_train, "TRAIN (17-22)")
m_val   = get_metrics(p_val, y_val, ret_val, "VAL (2023)")

# 7. RAPORT
results = pd.DataFrame([m_train, m_val])
print("\n" + "="*90)
print(f"   RAPORT V8 (Dynamiczne Progi: <{thresh_short_dynamic:.3f} | >{thresh_long_dynamic:.3f})   ")
print("="*90)
print(results.to_string(index=False))
print("-" * 90)

# Wykres
plt.figure(figsize=(10, 6))
cum_bh = np.exp(np.cumsum(ret_val))
cum_strat = np.exp(np.cumsum(np.where(p_val > thresh_long_dynamic, 1, np.where(p_val < thresh_short_dynamic, -1, 0)) * ret_val))

plt.plot(idx_val, cum_bh, label=f'Buy & Hold (+{m_val["BH Return %"]}%)', color='gray', alpha=0.5)
plt.plot(idx_val, cum_strat, label=f'Dynamic Strategy (+{m_val["Return %"]}%)', color='purple', linewidth=2)
plt.title(f'WALIDACJA 2023 (Auto-Dostrojona)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v8_dynamic_result.png')
print("Zapisano: v8_dynamic_result.png")