import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# --- 1. KONFIGURACJA ---
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 

# Patrzymy 10 dni wstecz (krótka pamięć, szybka reakcja)
TIMESTEPS = 10 
EPOCHS = 100
BATCH_SIZE = 16

# Daty (Twoje sztywne ramy)
TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE   = '2022-12-31'
TEST_START_DATE = '2023-01-01'

# --- 2. PRZYGOTOWANIE DANYCH (4h -> Daily) ---
print(f"--- [PHOENIX V1] Wczytywanie: {INPUT_FILE} ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Brak pliku CSV!")

data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]

# Agregacja do Dnia (Decyzja raz dziennie)
data = data_4h.resample('1D').agg({
    'Open': 'first', 
    'High': 'max', 
    'Low': 'min', 
    'Close': 'last', 
    'Volume': 'sum'
}).dropna()

# --- 3. FEATURE ENGINEERING ---
# Log Return (To co przewidujemy)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# Wskaźniki
data['RSI'] = ta.rsi(data['Close'], length=14)
data['ATR_Pct'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) / data['Close']
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=30).mean()

# MACD
macd = ta.macd(data['Close'])
data['MACD'] = macd.iloc[:, 0]
data['MACD_Signal'] = macd.iloc[:, 2] # Histogram

# Lags (Kluczowe dla momentum)
for i in range(1, 4):
    data[f'lag_{i}'] = data['log_return'].shift(i)

features = ['log_return', 'RSI', 'ATR_Pct', 'Rel_Vol', 'MACD', 'MACD_Signal', 'lag_1', 'lag_2', 'lag_3']

# --- 4. TARGET I WAGI (SERCE SYSTEMU) ---

# Target: 1 jeśli jutro cena wzrośnie, 0 jeśli spadnie
data['target'] = np.where(data['log_return'].shift(-1) > 0, 1, 0)

# Wagi: Absolutna wartość zwrotu jutrzejszego.
# Model ma się uczyć mocno na dużych świecach, a olewać małe "doji".
data['sample_weight'] = data['log_return'].shift(-1).abs()

# Actual Return (do backtestu)
data['actual_return'] = data['log_return'].shift(-1)

data = data.dropna()

# --- 5. PODZIAŁ I SKALOWANIE ---
train_df = data.loc[:TRAIN_END_DATE].copy()
val_df   = data.loc[VAL_START_DATE:VAL_END_DATE].copy()
test_df  = data.loc[TEST_START_DATE:].copy()

# Skalowanie (MinMax jest bezpieczniejszy dla LSTM)
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

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# --- 6. MODELOWANIE (LSTM) ---
# Prosta, ale solidna architektura
print("\n--- Trenowanie LSTM (Weighted Loss) ---")

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    # Bidirectional pozwala widzieć kontekst w obu kierunkach
    Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# KLUCZOWE: sample_weight=w_train
# To tutaj dzieje się magia z maila od promotora
history = model.fit(
    X_train, y_train,
    sample_weight=w_train, # <--- WAGI
    validation_data=(X_val, y_val, w_val), # Walidacja też ważona!
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=False, # Time series - nie tasujemy!
    verbose=1
)

# --- 7. PREDYKCJA I BACKTEST ---
print("\n--- Backtest ---")

probs = model.predict(X_test).flatten()

# DataFrame Wynikowy
res = pd.DataFrame(index=idx_test)
res['Actual_Ret'] = ret_test
res['Prob'] = probs

# STRATEGIA PHOENIX
# Skalujemy pozycję w zależności od pewności (distance from 0.5)
# 0.50 -> Position 0
# 0.60 -> Position 0.2
# 0.90 -> Position 0.8
# Wzór: (Prob - 0.5) * 2 -> Skaluje zakres 0.5-1.0 na 0.0-1.0
# Możemy dodać mnożnik (np. * 2), żeby być bardziej agresywnym

def get_position(p):
    # Kierunek
    direction = 1 if p > 0.5 else -1
    # Pewność (od 0 do 0.5)
    certainty = abs(p - 0.5)
    # Rozmiar pozycji (skalowany liniowo)
    size = certainty * 2 
    # Cap na 100%
    size = min(size, 1.0)
    
    # Opcjonalnie: Filtr szumu (jeśli pewność < 2%, nie wchodź)
    if size < 0.04: return 0.0
    
    return direction * size

res['Position'] = res['Prob'].apply(get_position)
res['Strat_Ret'] = res['Position'] * res['Actual_Ret']

# --- 8. METRYKI ---
def calc_metrics(returns):
    cum = (1 + returns).cumprod()
    total = (cum.iloc[-1] - 1) * 100
    if returns.std() == 0: sharpe = 0
    else: sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
    dd = (cum / cum.cummax() - 1).min() * 100
    return total, sharpe, dd

bh_tot, bh_sh, bh_dd = calc_metrics(res['Actual_Ret'])
st_tot, st_sh, st_dd = calc_metrics(res['Strat_Ret'])

print(f"\n=== WYNIKI (2023-2024) ===")
print(f"Buy & Hold: {bh_tot:.2f}% | Sharpe: {bh_sh:.2f} | DD: {bh_dd:.2f}%")
print(f"Phoenix V1: {st_tot:.2f}% | Sharpe: {st_sh:.2f} | DD: {st_dd:.2f}%")

# Win Rate (Weighted by magnitude implicitly via results, but let's count simple wins)
wins = np.sign(res['Position']) == np.sign(res['Actual_Ret'])
# Wykluczamy dni bez pozycji
real_trades = res[res['Position'] != 0]
if len(real_trades) > 0:
    win_rate = (np.sign(real_trades['Position']) == np.sign(real_trades['Actual_Ret'])).mean() * 100
else:
    win_rate = 0
print(f"Win Rate (Active Days): {win_rate:.2f}%")

# --- 9. EXPORTY ---
# 1. CSV z każdą decyzją
res.to_csv('phoenix_trades.csv')
print("Zapisano: phoenix_trades.csv")

# 2. Wykres Equity
plt.figure(figsize=(12, 6))
plt.plot(res.index, (1 + res['Actual_Ret']).cumprod(), label='Buy & Hold', color='grey', alpha=0.5)
plt.plot(res.index, (1 + res['Strat_Ret']).cumprod(), label=f'Phoenix LSTM (Sharpe {st_sh:.2f})', color='blue')
plt.title('Phoenix V1 Strategy Performance')
plt.ylabel('Equity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('phoenix_equity.png')

# 3. Wykres Rozkładu Pozycji (Czy model jest aktywny?)
plt.figure(figsize=(12, 4))
plt.plot(res.index, res['Position'], label='Position Size (-1 to 1)', color='orange', linewidth=0.5)
plt.title('Zaangażowanie Modelu w Czasie')
plt.ylabel('Position')
plt.axhline(0, color='black')
plt.savefig('phoenix_positions.png')

# 4. Learning Curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Weighted Loss')
plt.plot(history.history['val_loss'], label='Val Weighted Loss')
plt.title('Weighted Learning Process')
plt.legend()
plt.savefig('phoenix_learning.png')

print("Wykresy wygenerowane.")