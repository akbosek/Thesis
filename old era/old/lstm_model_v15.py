import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score
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

TIMESTEPS = 30  
THRESH_SHORT = 0.45
THRESH_LONG = 0.55
TRANSACTION_FEE = 0.0

# Daty
TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE   = '2022-12-31'
TEST_START_DATE = '2023-01-01'

# --- 2. WCZYTANIE ---
print(f"--- [V15: FIXED & EXPANDED] Wczytywanie: {INPUT_FILE} ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Brak pliku CSV!")

data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]

# Daily Resampling
data = data_4h.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

# --- 3. FEATURES ---
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=60).mean()
data['RSI'] = ta.rsi(data['Close'], length=14)
data['MACD'] = ta.macd(data['Close']).iloc[:, 0]
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) / data['Close']

# Bollinger Bands (Fixed iloc)
bb = ta.bbands(data['Close'], length=20, std=2)
data['BB_Pct'] = bb.iloc[:, 4]   
data['BB_Width'] = bb.iloc[:, 3] 

lags = 5
for lag in range(1, lags + 1):
    data[f'log_return_lag_{lag}'] = data['log_return'].shift(lag)

features = ['log_return', 'Rel_Vol', 'RSI', 'MACD', 'ATR', 'BB_Pct', 'BB_Width'] + [f'log_return_lag_{i}' for i in range(1, lags + 1)]

data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data['actual_return_pct'] = np.exp(data['log_return'].shift(-1)) - 1 

data = data.dropna()

# --- 4. PODZIAŁ ---
train_df = data.loc[:TRAIN_END_DATE].copy()
val_df   = data.loc[VAL_START_DATE:VAL_END_DATE].copy()
test_df  = data.loc[TEST_START_DATE:].copy()

class_weights = compute_class_weight('balanced', classes=np.unique(train_df['target']), y=train_df['target'])
class_weights_dict = dict(enumerate(class_weights))

# --- 5. SKALOWANIE ---
scaler_std = StandardScaler().fit(train_df[features]) 
scaler_mm = MinMaxScaler((-1, 1)).fit(train_df[features]) 

def prepare_data(df, scaler):
    X_scaled = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_scaled) - TIMESTEPS):
        X.append(X_scaled[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    return np.array(X), np.array(y), df.index[TIMESTEPS:], df['actual_return_pct'].iloc[TIMESTEPS:].values, df['target'].iloc[TIMESTEPS:].values

X_train_tree, y_train_tree, _, _, _ = prepare_data(train_df, scaler_std)
X_test_tree, _, _, _, _ = prepare_data(test_df, scaler_std)
X_train_flat = X_train_tree.reshape(X_train_tree.shape[0], -1)
X_test_flat = X_test_tree.reshape(X_test_tree.shape[0], -1)

X_train_lstm, y_train_lstm, _, _, _ = prepare_data(train_df, scaler_mm)
X_val_lstm, y_val_lstm, _, _, _ = prepare_data(val_df, scaler_mm)
X_test_lstm, _, idx_test, ret_test, target_test = prepare_data(test_df, scaler_mm)

# --- 6. MODELOWANIE ---

print("\n--- Training LSTM V15 ---")
lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
    Dropout(0.3),
    LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=60, batch_size=32, 
                         validation_data=(X_val_lstm, y_val_lstm), class_weight=class_weights_dict, 
                         shuffle=False, callbacks=callbacks, verbose=1)

print("\n--- Training Trees ---")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train_flat, y_train_tree)
xgb_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03, scale_pos_weight=1.5, n_jobs=-1, random_state=42).fit(X_train_flat, y_train_tree)

# --- 7. RAPORTY & EXPORT ---
print("\n--- Generowanie Raportu V15 ---")

probs_lstm = lstm_model.predict(X_test_lstm).flatten()
probs_rf = rf_model.predict_proba(X_test_flat)[:, 1]
probs_xgb = xgb_model.predict_proba(X_test_flat)[:, 1]

# DataFrame do exportu
df_details = pd.DataFrame(index=idx_test)
df_details['Actual_Return'] = ret_test
df_details['Target'] = target_test

def analyze_model(name, probs, returns, t_short, t_long):
    if name == 'LSTM':
        pos = np.where(probs > 0.50, 1, -1)
        size = np.ones_like(pos)
    else:
        pos = np.where(probs > t_long, 1, np.where(probs < t_short, -1, 0))
        size = np.where(pos==1, (probs-t_long)/(1-t_long), (t_short-probs)/t_short)
        size = np.clip(size, 0, 1)

    strat_ret = pos * size * returns
    
    active = strat_ret[pos != 0]
    if len(active) > 0: win_rate = (active > 0).mean() * 100
    else: win_rate = 0
    
    cum_ret = (1 + strat_ret).cumprod()
    total_ret = (cum_ret[-1] - 1) * 100
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    # NAPRAWA BŁĘDU CUMMAX: Używamy np.maximum.accumulate dla numpy array
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    return {'Model': name, 'Win Rate': win_rate, 'Return %': total_ret, 'Sharpe': sharpe, 'Max DD': max_dd, 'Equity': cum_ret, 'Drawdown': drawdown}

res_bh = {'Model': 'Buy & Hold', 'Return %': ((1+ret_test).cumprod()[-1]-1)*100, 'Sharpe':0, 'Max DD':0, 'Win Rate':0, 'Equity': (1+ret_test).cumprod(), 'Drawdown': (1+ret_test).cumprod()/np.maximum.accumulate((1+ret_test).cumprod())-1}
res_lstm = analyze_model('LSTM', probs_lstm, ret_test, THRESH_SHORT, THRESH_LONG)
res_rf = analyze_model('RF', probs_rf, ret_test, THRESH_SHORT, THRESH_LONG)
res_xgb = analyze_model('XGB', probs_xgb, ret_test, THRESH_SHORT, THRESH_LONG)

# Tabela wyników
results_table = pd.DataFrame([
    {k:v for k,v in res_bh.items() if k not in ['Equity', 'Drawdown']},
    {k:v for k,v in res_lstm.items() if k not in ['Equity', 'Drawdown']},
    {k:v for k,v in res_rf.items() if k not in ['Equity', 'Drawdown']},
    {k:v for k,v in res_xgb.items() if k not in ['Equity', 'Drawdown']}
])

print(results_table.round(2).to_string(index=False))

# --- WYKRES 1: EQUITY CURVE ---
plt.figure(figsize=(14, 8))
plt.plot(res_bh['Equity'], label='Buy & Hold', color='grey', alpha=0.4)
plt.plot(res_lstm['Equity'], label='LSTM (Stacked)', linewidth=1.5)
plt.plot(res_rf['Equity'], label='RF', linewidth=1.5)
plt.plot(res_xgb['Equity'], label='XGB', linewidth=1.5)
plt.title(f'V15: Strategy Equity (Fixed)')
plt.ylabel('Normalized Equity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v15_equity_curve.png')
print("Wykres zapisano: v15_equity_curve.png")

# --- WYKRES 2: DRAWDOWN ---
plt.figure(figsize=(14, 6))
plt.plot(res_lstm['Drawdown'], label='LSTM Drawdown', linewidth=1)
plt.plot(res_rf['Drawdown'], label='RF Drawdown', linewidth=1)
plt.title('Drawdown Analysis')
plt.ylabel('Drawdown %')
plt.legend()
plt.grid(True, alpha=0.3)
plt.fill_between(res_lstm['Drawdown'].index, res_lstm['Drawdown'], 0, alpha=0.1)
plt.savefig('v15_drawdown.png')
print("Wykres zapisano: v15_drawdown.png")

# --- WYKRES 3: LSTM LEARNING ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training Process')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v15_lstm_learning.png')
print("Wykres zapisano: v15_lstm_learning.png")

# Export szczegółowy
df_details['LSTM_Prob'] = probs_lstm
df_details['LSTM_Pos'] = np.where(probs_lstm > 0.5, 1, -1)
df_details['LSTM_PnL'] = df_details['LSTM_Pos'] * ret_test
df_details.to_csv('v15_detailed_trades.csv')
print("Zapisano CSV: v15_detailed_trades.csv")