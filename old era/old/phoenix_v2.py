import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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

# NOWE PROGI (Agresywniejsze filtrowanie)
# Gramy tylko, gdy model jest pewien na >53% lub <47%
THRESH_LONG = 0.53
THRESH_SHORT = 0.47

TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE   = '2022-12-31'
TEST_START_DATE = '2023-01-01'

# --- 2. PRZYGOTOWANIE ---
print(f"--- [PHOENIX V2: HIGH PRECISION] Wczytywanie... ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Brak pliku CSV!")

data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]
data = data_4h.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

# --- 3. FEATURE ENGINEERING (Z ADX) ---
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# ADX - Klucz do filtrowania trendu bocznego
adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
data['ADX'] = adx['ADX_14']
data['DMP'] = adx['DMP_14'] # Plus Directional Movement
data['DMN'] = adx['DMN_14'] # Minus Directional Movement

data['RSI'] = ta.rsi(data['Close'], length=14)
data['ATR_Pct'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) / data['Close']
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=30).mean()
macd = ta.macd(data['Close'])
data['MACD'] = macd.iloc[:, 0]

for i in range(1, 4):
    data[f'lag_{i}'] = data['log_return'].shift(i)

# Nowa lista cech (z ADX)
features = ['log_return', 'RSI', 'ATR_Pct', 'Rel_Vol', 'MACD', 'ADX', 'DMP', 'DMN', 'lag_1', 'lag_2', 'lag_3']

data['target'] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
# Wagi: Jeszcze mocniej karzemy za błędy na dużych ruchach (kwadrat zwrotu)
data['sample_weight'] = (data['log_return'].shift(-1).abs()) * 100 
data['actual_return'] = data['log_return'].shift(-1)

data = data.dropna()

# --- 4. PODZIAŁ I SKALOWANIE ---
train_df = data.loc[:TRAIN_END_DATE].copy()
val_df   = data.loc[VAL_START_DATE:VAL_END_DATE].copy()
test_df  = data.loc[TEST_START_DATE:].copy()

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_df[features])

def prepare_dataset(df):
    X_scaled = scaler.transform(df[features])
    X, y, w = [], [], []
    for i in range(len(X_scaled) - TIMESTEPS):
        X.append(X_scaled[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
        w.append(df['sample_weight'].iloc[i + TIMESTEPS])
    return np.array(X), np.array(y), np.array(w), df.index[TIMESTEPS:], df['actual_return'].iloc[TIMESTEPS:].values

X_train, y_train, w_train, _, _ = prepare_dataset(train_df)
X_val, y_val, w_val, _, _ = prepare_dataset(val_df)
X_test, y_test, w_test, idx_test, ret_test = prepare_dataset(test_df)

# --- 5. MODELOWANIE (LSTM Tuned) ---
print("\n--- Training LSTM V2 ---")
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    # Zwiększamy nieco sieć, bo dodaliśmy ADX (więcej informacji do przetworzenia)
    Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001))),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train, sample_weight=w_train,
    validation_data=(X_val, y_val, w_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, shuffle=False, verbose=1
)

# --- 6. PREDYKCJA I STRATEGIA ---
probs = model.predict(X_test).flatten()
res = pd.DataFrame(index=idx_test)
res['Actual_Ret'] = ret_test
res['Prob'] = probs

# STRATEGIA "SNAJPER"
def get_position(p):
    # Jeśli pewność jest niska (pomiędzy 0.47 a 0.53), nie graj.
    if p > THRESH_SHORT and p < THRESH_LONG:
        return 0.0
    
    # Kierunek
    direction = 1 if p > 0.5 else -1
    
    # Skalowanie pozycją (im pewniej tym mocniej)
    # Dla Long: skalujemy od 0.53 do 1.0
    if direction == 1:
        certainty = (p - THRESH_LONG) / (1 - THRESH_LONG)
    else:
        certainty = (THRESH_SHORT - p) / THRESH_SHORT
        
    size = np.clip(certainty * 1.5, 0.1, 1.0) # Boost mnożnik x1.5
    return direction * size

res['Position'] = res['Prob'].apply(get_position)
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

# Win Rate (Tylko aktywne dni)
active_trades = res[res['Position'] != 0]
if len(active_trades) > 0:
    # Win = (Long i wzrost) LUB (Short i spadek)
    wins = np.sign(active_trades['Position']) == np.sign(active_trades['Actual_Ret'])
    win_rate = wins.mean() * 100
    trade_count = len(active_trades)
else:
    win_rate = 0
    trade_count = 0

print(f"\n=== WYNIKI PHOENIX V2 ===")
print(f"Buy & Hold: {bh_tot:.2f}% | Sharpe: {bh_sh:.2f} | DD: {bh_dd:.2f}%")
print(f"Strategy:   {st_tot:.2f}%   | Sharpe: {st_sh:.2f} | DD: {st_dd:.2f}%")
print(f"Win Rate:   {win_rate:.2f}% (Liczba transakcji: {trade_count} z {len(res)})")

# --- Exporty ---
res.to_csv('phoenix_v2_trades.csv')

plt.figure(figsize=(12, 6))
plt.plot(res.index, (1 + res['Actual_Ret']).cumprod(), label='Buy & Hold', color='grey', alpha=0.5)
plt.plot(res.index, (1 + res['Strat_Ret']).cumprod(), label=f'Phoenix V2 (WR {win_rate:.1f}%)', color='green')
plt.title(f'Phoenix V2 Performance (Thresholds {THRESH_SHORT}-{THRESH_LONG})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('phoenix_v2_equity.png')