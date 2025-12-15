import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- 1. KONFIGURACJA ---
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 
TIMESTEPS = 10 
EPOCHS = 60 # Wystarczy dla ensemble
BATCH_SIZE = 32

# Progi dla poszczególnych modeli
THRESH = 0.50 

# DŹWIGNIA (Tym razem "Sztywna")
LEVERAGE_BOOST = 2.0 # Mnożnik gdy modele są zgodne

# --- 2. DANE ---
print(f"--- [PHOENIX V7: ENSEMBLE LEVERAGE] Wczytywanie... ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Brak pliku CSV!")

data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]
data = data_4h.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

# --- 3. FEATURE ENGINEERING ---
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['RSI'] = ta.rsi(data['Close'], length=14)
data['ATR_Pct'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) / data['Close']
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=30).mean()
data['MACD'] = ta.macd(data['Close']).iloc[:, 0]
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Lags
for i in range(1, 4):
    data[f'lag_{i}'] = data['log_return'].shift(i)

features = ['log_return', 'RSI', 'ATR_Pct', 'Rel_Vol', 'MACD', 'lag_1', 'lag_2', 'lag_3']

data['target'] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
data['actual_return'] = data['log_return'].shift(-1)
data = data.dropna()

# --- 4. PODZIAŁ ---
train_df = data.loc[:'2021-12-31'].copy()
val_df   = data.loc['2022-01-01':'2022-12-31'].copy()
test_df  = data.loc['2023-01-01':].copy()

# Skalowanie
scaler = StandardScaler()
scaler.fit(train_df[features])

def prepare_dataset(df):
    X_scaled = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_scaled) - TIMESTEPS):
        X.append(X_scaled[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    return np.array(X), np.array(y), df.index[TIMESTEPS:], df['actual_return'].iloc[TIMESTEPS:].values, df['Close'].iloc[TIMESTEPS:].values, df['SMA_50'].iloc[TIMESTEPS:].values

X_train, y_train, _, _, _, _ = prepare_dataset(train_df)
X_val, y_val, _, _, _, _ = prepare_dataset(val_df)
X_test, y_test, idx_test, ret_test, close_test, sma_test = prepare_dataset(test_df)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# --- 5. TRENOWANIE ENSEMBLE ---

# A. LSTM
print("\n--- Training LSTM ---")
lstm_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(class_weights))

lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, 
               class_weight=cw_dict, callbacks=[callback], verbose=0)

# B. Random Forest
print("--- Training Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train_flat, y_train)

# C. XGBoost
print("--- Training XGBoost ---")
xgb_model = XGBClassifier(n_estimators=200, max_depth=4, scale_pos_weight=1.5, learning_rate=0.03, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_flat, y_train)

# --- 6. STRATEGIA ENSEMBLE (Consensus) ---
print("\n--- Generowanie Sygnałów ---")

# Pobieramy predykcje (0-1)
p_lstm = lstm_model.predict(X_test).flatten()
p_rf = rf_model.predict_proba(X_test_flat)[:, 1]
p_xgb = xgb_model.predict_proba(X_test_flat)[:, 1]

# DataFrame Roboczy
df_strat = pd.DataFrame(index=idx_test)
df_strat['Actual'] = ret_test
df_strat['SMA'] = sma_test
df_strat['Close'] = close_test

# Sygnały binarne (1 = Long, -1 = Short)
# Używamy prostego progu 0.50, bo ensemble sam odfiltruje błędy
s_lstm = np.where(p_lstm > 0.50, 1, -1)
s_rf = np.where(p_rf > 0.50, 1, -1)
s_xgb = np.where(p_xgb > 0.50, 1, -1)

# Sumujemy głosy (od -3 do +3)
df_strat['Vote_Score'] = s_lstm + s_rf + s_xgb 

def get_position_v7(row):
    score = row['Vote_Score']
    trend_up = row['Close'] > row['SMA']
    
    # 1. STRONG LONG (Zgoda 3 modeli + Trend Wzrostowy)
    if score == 3 and trend_up:
        return LEVERAGE_BOOST # Wchodzimy x2.0
    
    # 2. NORMAL LONG (Zgoda 2 lub 3 modeli, bez wymogu trendu)
    # Ryzykowne, ale potrzebujemy zasięgu
    if score >= 1:
        return 1.0 # Wchodzimy x1.0
        
    # 3. SHORT (Tylko jeśli wszyscy 3 mówią spadki)
    # Shorty w krypto są groźne, więc wymagamy pełnej zgody
    if score == -3:
        return -1.0
        
    # 4. FLAT (Brak zgody lub słabe sygnały spadkowe)
    return 0.0

df_strat['Position'] = df_strat.apply(get_position_v7, axis=1)
df_strat['Strat_Ret'] = df_strat['Position'] * df_strat['Actual']

# --- 7. WYNIKI ---
bh_cum = (1 + df_strat['Actual']).cumprod()
st_cum = (1 + df_strat['Strat_Ret']).cumprod()

bh_tot = (bh_cum.iloc[-1] - 1) * 100
st_tot = (st_cum.iloc[-1] - 1) * 100

def get_sharpe(ret):
    if ret.std() == 0: return 0
    return (ret.mean() / ret.std()) * np.sqrt(365)

def get_dd(cum):
    return (cum / cum.cummax() - 1).min() * 100

print(f"\n=== WYNIKI PHOENIX V7 (Ensemble x{LEVERAGE_BOOST}) ===")
print(f"Buy & Hold: {bh_tot:.2f}% | Sharpe: {get_sharpe(df_strat['Actual']):.2f} | DD: {get_dd(bh_cum):.2f}%")
print(f"Strategy:   {st_tot:.2f}% | Sharpe: {get_sharpe(df_strat['Strat_Ret']):.2f} | DD: {get_dd(st_cum):.2f}%")

# Win Rate (tylko dni z lewarem i bez)
active = df_strat[df_strat['Position'] != 0]
if len(active) > 0:
    wr = (np.sign(active['Position']) == np.sign(active['Actual'])).mean() * 100
    print(f"Win Rate:   {wr:.2f}% (Trades: {len(active)})")
    
    # Win Rate na lewarze (Strong Signals)
    lev_trades = df_strat[abs(df_strat['Position']) > 1.0]
    if len(lev_trades) > 0:
        lev_wr = (np.sign(lev_trades['Position']) == np.sign(lev_trades['Actual'])).mean() * 100
        print(f"Lev WinRate:{lev_wr:.2f}% (Strong Trades: {len(lev_trades)})")

df_strat.to_csv('phoenix_v7_trades.csv')

plt.figure(figsize=(12, 6))
plt.plot(df_strat.index, bh_cum, label='Buy & Hold', color='grey', alpha=0.5)
plt.plot(df_strat.index, st_cum, label=f'Phoenix V7 (Lev x{LEVERAGE_BOOST})', color='darkblue', linewidth=1.5)
plt.title(f'Phoenix V7 Ensemble Performance')
plt.ylabel('Equity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('phoenix_v7_equity.png')