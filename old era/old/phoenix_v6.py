import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# --- 1. KONFIGURACJA ---
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 
TIMESTEPS = 10 
EPOCHS = 100
BATCH_SIZE = 16

# LEWAR (To jest klucz do V6)
MAX_LEVERAGE = 3.0 # Pozwalamy na pozycję 300%

print(f"--- [PHOENIX V6: LEVERAGE BOOSTER] Wczytywanie... ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Brak pliku CSV!")

data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]
data = data_4h.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

# --- 3. FEATURES ---
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['RSI'] = ta.rsi(data['Close'], length=14)
data['ATR_Pct'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) / data['Close']
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=30).mean()
data['MACD'] = ta.macd(data['Close']).iloc[:, 0]
data['SMA_50'] = data['Close'].rolling(window=50).mean()

for i in range(1, 4):
    data[f'lag_{i}'] = data['log_return'].shift(i)

features = ['log_return', 'RSI', 'ATR_Pct', 'Rel_Vol', 'MACD', 'lag_1', 'lag_2', 'lag_3']

data['target'] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
data['sample_weight'] = data['log_return'].shift(-1).abs()
data['actual_return'] = data['log_return'].shift(-1)
data = data.dropna()

# --- 4. PODZIAŁ ---
train_df = data.loc[:'2021-12-31'].copy()
val_df   = data.loc['2022-01-01':'2022-12-31'].copy()
test_df  = data.loc['2023-01-01':].copy()

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_df[features])

# Helper function
def prepare_dataset(df):
    X_scaled = scaler.transform(df[features])
    X, y, w = [], [], []
    for i in range(len(X_scaled) - TIMESTEPS):
        X.append(X_scaled[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
        w.append(df['sample_weight'].iloc[i + TIMESTEPS])
    # Zwracamy też Close i SMA do strategii
    return (np.array(X), np.array(y), np.array(w), 
            df.index[TIMESTEPS:], 
            df['actual_return'].iloc[TIMESTEPS:].values, 
            df['ATR_Pct'].iloc[TIMESTEPS:].values,
            df['Close'].iloc[TIMESTEPS:].values,
            df['SMA_50'].iloc[TIMESTEPS:].values)

X_train, y_train, w_train, _, _, _, _, _ = prepare_dataset(train_df)
X_val, y_val, w_val, _, _, _, _, _ = prepare_dataset(val_df)
X_test, y_test, w_test, idx_test, ret_test, atr_test, close_test, sma_test = prepare_dataset(test_df)

# --- 5. MODEL (Sprawdzony V5) ---
print("\n--- Training LSTM V6 ---")
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)]

history = model.fit(X_train, y_train, sample_weight=w_train, validation_data=(X_val, y_val, w_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, shuffle=False, verbose=1)

# --- 6. STRATEGIA V6 (LEVERAGED) ---
probs = model.predict(X_test).flatten()
res = pd.DataFrame(index=idx_test)
res['Actual_Ret'] = ret_test
res['Prob'] = probs
res['SMA_50'] = sma_test
res['Close'] = close_test
res['ATR'] = atr_test

ATR_THRESHOLD = train_df['ATR_Pct'].median() * 0.40 

def get_position_v6(row):
    p = row['Prob']
    atr = row['ATR']
    trend_up = row['Close'] > row['SMA_50']
    
    if atr < ATR_THRESHOLD: return 0.0

    # LONG STRATEGY
    if p > 0.50:
        if not trend_up: return 0.0 # Filtr trendu nadal aktywny (bezpieczeństwo)
        
        # Obliczamy bazową pewność (0.0 do 1.0)
        certainty = (p - 0.5) * 2
        
        # APLIKUJEMY LEWAR:
        # Jeśli pewność > 50% (czyli p > 0.75), wchodzimy z lewarem x3
        # Jeśli pewność mała, wchodzimy np. x1
        size = certainty * MAX_LEVERAGE 
        
        # Cap na max lewar
        return min(size, MAX_LEVERAGE)
        
    # SHORT STRATEGY
    else:
        if trend_up: return 0.0 
        
        certainty = (0.5 - p) * 2
        # Shorty z mniejszym lewarem (max x1.5 dla bezpieczeństwa)
        size = certainty * 2.0 * 0.8 
        if certainty < 0.05: return 0.0
        
        return -1 * min(size, 1.5)

res['Position'] = res.apply(get_position_v6, axis=1)
res['Strat_Ret'] = res['Position'] * res['Actual_Ret']

# --- 7. WYNIKI ---
def calc_metrics(returns):
    cum = (1 + returns).cumprod()
    total = (cum.iloc[-1] - 1) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() != 0 else 0
    dd = (cum / cum.cummax() - 1).min() * 100
    return total, sharpe, dd

bh_tot, bh_sh, bh_dd = calc_metrics(res['Actual_Ret'])
st_tot, st_sh, st_dd = calc_metrics(res['Strat_Ret'])

active = res[res['Position'] != 0]
if len(active) > 0:
    wr = (np.sign(active['Position']) == np.sign(active['Actual_Ret'])).mean() * 100
else: wr = 0

print(f"\n=== WYNIKI PHOENIX V6 (Leveraged x{MAX_LEVERAGE}) ===")
print(f"Buy & Hold: {bh_tot:.2f}% | Sharpe: {bh_sh:.2f} | DD: {bh_dd:.2f}%")
print(f"Strategy:   {st_tot:.2f}%   | Sharpe: {st_sh:.2f} | DD: {st_dd:.2f}%")
print(f"Win Rate:   {wr:.2f}% (Trades: {len(active)})")

res.to_csv('phoenix_v6_trades.csv')

plt.figure(figsize=(12, 6))
plt.plot(res.index, (1 + res['Actual_Ret']).cumprod(), label='Buy & Hold', color='grey', alpha=0.5)
plt.plot(res.index, (1 + res['Strat_Ret']).cumprod(), label=f'Phoenix V6 (Lev x{MAX_LEVERAGE})', color='orange')
plt.title(f'Phoenix V6 Performance (Leverage Max x{MAX_LEVERAGE})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('phoenix_v6_equity.png')