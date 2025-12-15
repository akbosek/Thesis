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
from tensorflow.keras.regularizers import l2

# ==========================================
#      HIPERPARAMETRY (GAP CLOSER)
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# ZMIANA 1: Ocinamy prehistorię (Start 2019)
TRAIN_START = '2019-01-01' 
TRAIN_END   = '2022-12-31'
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31'

# ZMIANA 2: Mniejszy model, większy rygor
TIMESTEPS = 14
NEURONS   = 32       # Było 64 -> Zmniejszamy pojemność
DROPOUT   = 0.5      # Było 0.2 -> Agresywne zapominanie
L2_REG    = 0.001    # Lekka kara za wagi
LEARNING_RATE = 0.001
EPOCHS    = 60
BATCH_SIZE = 32

THRESH_SHORT = 0.48
THRESH_LONG  = 0.52
# ==========================================

print("--- [LSTM V3.7: GAP CLOSER] ---")
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

print(f"Train (2019-2022): {len(train_df)} dni")
print(f"Val (2023): {len(val_df)} dni")

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

# 4. TRENING (ODCHUDZONY MODEL)
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    # Dodajemy L2 Regularization do LSTM
    LSTM(NEURONS, return_sequences=False, kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(DROPOUT), # 50% neuronów wyłączane losowo
    Dense(16, activation='relu', kernel_regularizer=l2(L2_REG)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
estop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[estop],
    verbose=0
)

# 5. DIAGNOSTYKA
def get_metrics(X, y_true, ret, name):
    probs = model.predict(X, verbose=0).flatten()
    pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    strat_ret = pos * ret
    
    active = pos != 0
    trades = np.sum(active)
    if trades > 0:
        wr = (np.sign(pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    try: gini = 2 * roc_auc_score(y_true, probs) - 1
    except: gini = 0
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    cum = np.exp(np.cumsum(strat_ret))
    total_ret = (cum[-1] - 1) * 100
    
    # Buy & Hold return
    bh_cum = np.exp(np.cumsum(ret))
    bh_ret = (bh_cum[-1] - 1) * 100
    
    dd = (cum / np.maximum.accumulate(cum) - 1).min() * 100
    
    return {
        "Period": name,
        "WinRate %": round(wr, 2),
        "Gini": round(gini, 3),
        "Sharpe": round(sharpe, 2),
        "Return %": round(total_ret, 2),
        "BH Return %": round(bh_ret, 2),
        "Max DD %": round(dd, 2),
        "Trades": trades
    }

print("\n--- Generowanie Raportu V3.7 ---")
m_train = get_metrics(X_train, y_train, ret_train, "TRAIN (19-22)")
m_val   = get_metrics(X_val, y_val, ret_val, "VALIDATION (23)")

# 6. WYNIKI
results = pd.DataFrame([m_train, m_val])
print("\n" + "="*95)
print("   STRICT PERFORMANCE REPORT (GAP CLOSER)   ")
print("="*95)
print(results.to_string(index=False))
print("-" * 95)

# Sprawdzenie Rozstrzału
diff = m_train['WinRate %'] - m_val['WinRate %']
print(f"Różnica WinRate (Train - Val): {diff:.2f}%")
if diff < 5:
    print("✅ SUKCES: Model jest stabilny i nie przeuczony.")
else:
    print("⚠️ NADAL ROZSTRZAŁ: Model wciąż wkuwa na pamięć.")

# Wykres
plt.figure(figsize=(10, 6))
cum_bh = np.exp(np.cumsum(ret_val))
probs_val = model.predict(X_val, verbose=0).flatten()
cum_strat = np.exp(np.cumsum(np.where(probs_val > THRESH_LONG, 1, np.where(probs_val < THRESH_SHORT, -1, 0)) * ret_val))

plt.plot(idx_val, cum_bh, label=f'Buy & Hold (+{m_val["BH Return %"]}%)', color='gray', alpha=0.5)
plt.plot(idx_val, cum_strat, label=f'Strategy (+{m_val["Return %"]}%)', color='blue', linewidth=2)
plt.title(f'WALIDACJA 2023 (Mniejszy Model, Większy Dropout)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gap_closer_val.png')
print("Zapisano wykres: gap_closer_val.png")