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
#      KONFIGURACJA OSTATECZNA
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# Nowy podział (Włączamy 2023 do treningu!)
TRAIN_START = '2017-09-01'
TRAIN_END   = '2023-12-31' # Uczymy się aż do końca 2023
TEST_START  = '2024-01-01' # Testujemy na 2024 (Świeżynka)

# Parametry z V2 (Sprawdzone)
TIMESTEPS = 14
NEURONS   = 64
DROPOUT   = 0.2
LEARNING_RATE = 0.001
EPOCHS    = 60 # Trochę więcej epok, bo więcej danych
BATCH_SIZE = 32

THRESH_SHORT = 0.48
THRESH_LONG  = 0.52
# ==========================================

print("--- [LSTM FINAL RUN: TEST 2024] ---")
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
# Teraz Train to wszystko do końca 2023
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
test_df  = data.loc[TEST_START:].copy()

print(f"Trening (2017-2023): {len(train_df)} dni")
print(f"Test (2024): {len(test_df)} dni")

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

X_train, y_train, _, _ = create_dataset(train_df)
X_test, y_test, ret_test, idx_test = create_dataset(test_df)

# 4. TRENING
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(NEURONS, return_sequences=False),
    BatchNormalization(),
    Dropout(DROPOUT),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
# Early stopping na treningu (bo nie mamy walidacji w tym trybie, ufamy parametrom z V2)
# Używamy loss treningowego jako progu, ale ostrożnie
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# 5. PREDYKCJA NA 2024
print("\n--- TESTOWANIE NA ROKU 2024 ---")
p_test = model.predict(X_test).flatten()

# Strategia
pos_test = np.where(p_test > THRESH_LONG, 1, np.where(p_test < THRESH_SHORT, -1, 0))
strat_ret = pos_test * ret_test

# Win Rate
active_mask = pos_test != 0
if np.sum(active_mask) > 0:
    wins = np.sign(pos_test[active_mask]) == np.sign(ret_test[active_mask])
    win_rate = np.mean(wins) * 100
else: win_rate = 0

# Sharpe
if np.std(strat_ret) == 0: sharpe = 0
else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)

# Equity
cum_bh = np.exp(np.cumsum(ret_test))
cum_strat = np.exp(np.cumsum(strat_ret))
total_ret = (cum_strat[-1] - 1) * 100
bh_ret = (cum_bh[-1] - 1) * 100

# Max DD
peak = np.maximum.accumulate(cum_strat)
dd = (cum_strat - peak) / peak
max_dd = np.min(dd) * 100

# 6. RAPORT
print("\n" + "="*30)
print("   WYNIKI KOŃCOWE (ROK 2024)   ")
print("="*30)
print(f"Strategia Return: {total_ret:.2f}%")
print(f"Buy & Hold Return: {bh_ret:.2f}%")
print("-" * 30)
print(f"Win Rate: {win_rate:.2f}% (Trades: {np.sum(active_mask)})")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2f}%")
print("="*30)

# CSV
df_res = pd.DataFrame(index=idx_test)
df_res['Prob'] = p_test
df_res['Position'] = pos_test
df_res['Strat_Ret'] = strat_ret
df_res.to_csv('final_2024_results.csv')

# Wykres
plt.figure(figsize=(12, 6))
plt.plot(idx_test, cum_bh, label='Buy & Hold (BTC)', color='gray', alpha=0.5)
plt.plot(idx_test, cum_strat, label=f'LSTM Strategy (Sharpe {sharpe:.2f})', color='green', linewidth=2)
plt.title('Wynik Końcowy Strategii (Rok 2024)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('final_chart_2024.png')
print("Zapisano: final_chart_2024.png, final_2024_results.csv")