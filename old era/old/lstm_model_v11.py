import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler # Dodano MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- 1. KONFIGURACJA ---
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 
TIMESTEPS = 5  
THRESH_SHORT = 0.45
THRESH_LONG = 0.55
TRANSACTION_FEE = 0.0 # No fee for research phase

TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE   = '2022-12-31'
TEST_START_DATE = '2023-01-01'

# --- 2. WCZYTANIE ---
print(f"--- [V11 HYBRID SCALING] Wczytywanie: {INPUT_FILE} ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Brak pliku CSV!")

data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]

# Resampling to Daily
data = data_4h.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

# --- 3. FEATURES ---
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=30).mean()
data['RSI'] = ta.rsi(data['Close'], length=14)
data['MACD'] = ta.macd(data['Close']).iloc[:, 0]
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) / data['Close']
data['Is_Weekend'] = np.where(data.index.dayofweek >= 5, 1, 0)

lags = 3
for lag in range(1, lags + 1):
    data[f'log_return_lag_{lag}'] = data['log_return'].shift(lag)

features = ['log_return', 'Rel_Vol', 'RSI', 'MACD', 'ATR', 'Is_Weekend'] + [f'log_return_lag_{i}' for i in range(1, lags + 1)]

data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data['actual_return'] = data['log_return'].shift(-1)
data = data.dropna()

# --- 4. PODZIAŁ ---
train_df = data.loc[:TRAIN_END_DATE].copy()
val_df   = data.loc[VAL_START_DATE:VAL_END_DATE].copy()
test_df  = data.loc[TEST_START_DATE:].copy()

class_weights = compute_class_weight('balanced', classes=np.unique(train_df['target']), y=train_df['target'])
class_weights_dict = dict(enumerate(class_weights))

# --- 5. SKALOWANIE (HYBRYDOWE) ---
# Dla Drzew (RF/XGB) używamy StandardScalera (lepszy do outlierów)
scaler_std = StandardScaler()
scaler_std.fit(train_df[features])

# Dla LSTM używamy MinMaxScalera (lepszy do sieci neuronowych, zakres 0-1)
scaler_mm = MinMaxScaler(feature_range=(-1, 1))
scaler_mm.fit(train_df[features])

def prepare_data(df, scaler):
    X_scaled = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_scaled) - TIMESTEPS):
        X.append(X_scaled[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    return np.array(X), np.array(y), df.index[TIMESTEPS:], df['actual_return'].iloc[TIMESTEPS:].values

# Dane dla Drzew (Standard)
X_train_tree, y_train_tree, _, _ = prepare_data(train_df, scaler_std)
X_test_tree, _, _, _ = prepare_data(test_df, scaler_std)
X_train_flat = X_train_tree.reshape(X_train_tree.shape[0], -1)
X_test_flat = X_test_tree.reshape(X_test_tree.shape[0], -1)

# Dane dla LSTM (MinMax)
X_train_lstm, y_train_lstm, _, _ = prepare_data(train_df, scaler_mm)
X_val_lstm, y_val_lstm, _, _ = prepare_data(val_df, scaler_mm)
X_test_lstm, y_test_lstm, idx_test, ret_test = prepare_data(test_df, scaler_mm)

# --- 6. MODELOWANIE ---

# LSTM (Lżejszy, szybszy, MinMax)
print("\n--- Training LSTM V11 (Unleashed) ---")
lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    # Zmniejszamy regularyzację (l2=0.0001 zamiast 0.01) - niech się uczy!
    LSTM(32, return_sequences=False, kernel_regularizer=l2(0.0001)), 
    Dropout(0.2), # Mniejszy dropout
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

history = lstm_model.fit(
    X_train_lstm, y_train_lstm, epochs=40, batch_size=32, 
    validation_data=(X_val_lstm, y_val_lstm), class_weight=class_weights_dict, 
    shuffle=False, callbacks=callbacks, verbose=1
)

# RF & XGB (Te same co były dobre)
print("\n--- Training Trees ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train_flat, y_train_tree)
xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.02, scale_pos_weight=1.2, n_jobs=-1, random_state=42).fit(X_train_flat, y_train_tree)

# --- 7. WYNIKI ---
probs_lstm = lstm_model.predict(X_test_lstm).flatten()
probs_rf = rf_model.predict_proba(X_test_flat)[:, 1]
probs_xgb = xgb_model.predict_proba(X_test_flat)[:, 1]

df_res = pd.DataFrame(index=idx_test)
df_res['Actual'] = ret_test

# Funkcja strategii
def get_strat(probs, returns, t_short, t_long):
    pos = np.where(probs > t_long, 1, np.where(probs < t_short, -1, 0))
    # Variable sizing
    size = np.where(pos==1, (probs-t_long)/(1-t_long), (t_short-probs)/t_short)
    return pos * np.clip(size, 0, 1) * returns

# DLA LSTM: Zdejmujemy blokadę (Próg 0.50)
df_res['LSTM'] = get_strat(probs_lstm, ret_test, 0.50, 0.50) 
# Dla Drzew: Zostawiamy strefę neutralną (bo działały dobrze)
df_res['RF'] = get_strat(probs_rf, ret_test, THRESH_SHORT, THRESH_LONG)
df_res['XGB'] = get_strat(probs_xgb, ret_test, THRESH_SHORT, THRESH_LONG)

# Metryki
metrics = []
for m in ['LSTM', 'RF', 'XGB']:
    cum = (1 + df_res[m]).cumprod()
    tot = (cum.iloc[-1] - 1) * 100
    if df_res[m].std() == 0: sh = 0
    else: sh = (df_res[m].mean() / df_res[m].std()) * np.sqrt(365)
    metrics.append({'Model': m, 'Return': tot, 'Sharpe': sh})

print(pd.DataFrame(metrics))

# Wykres
plt.figure(figsize=(12, 6))
plt.plot(df_res.index, (1 + df_res['Actual']).cumprod(), label='Buy & Hold', color='grey', alpha=0.3)
plt.plot(df_res.index, (1 + df_res['LSTM']).cumprod(), label='LSTM (No Threshold)')
plt.plot(df_res.index, (1 + df_res['RF']).cumprod(), label='RF')
plt.plot(df_res.index, (1 + df_res['XGB']).cumprod(), label='XGB')
plt.title('Backtest V11 (LSTM MinMax + No Threshold)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v11_results.png')
print("Zapisano: v11_results.png")