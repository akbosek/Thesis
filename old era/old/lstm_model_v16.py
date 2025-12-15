import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, BatchNormalization # Zmiana na GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- 1. KONFIGURACJA ---
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 

TIMESTEPS = 14  
THRESH_SHORT = 0.45
THRESH_LONG = 0.55
TRANSACTION_FEE = 0.0

TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE   = '2022-12-31'
TEST_START_DATE = '2023-01-01'

# --- 2. WCZYTANIE ---
print(f"--- [V16: GRU & FIXES] Wczytywanie: {INPUT_FILE} ---")
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
    return np.array(X), np.array(y), df.index[TIMESTEPS:], df['actual_return_pct'].iloc[TIMESTEPS:].values

X_train_tree, y_train_tree, _, _ = prepare_data(train_df, scaler_std)
X_test_tree, _, idx_test, ret_test = prepare_data(test_df, scaler_std)
X_train_flat = X_train_tree.reshape(X_train_tree.shape[0], -1)
X_test_flat = X_test_tree.reshape(X_test_tree.shape[0], -1)

X_train_lstm, y_train_lstm, _, _ = prepare_data(train_df, scaler_mm)
X_val_lstm, y_val_lstm, _, _ = prepare_data(val_df, scaler_mm)
X_test_lstm, _, _, _ = prepare_data(test_df, scaler_mm)

# --- 6. MODELOWANIE ---

print("\n--- Training GRU V16 (Simpler Architecture) ---")
# Zmiana na GRU - często lepsze dla mniejszych zbiorów danych finansowych
lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    # Pojedyncza warstwa GRU zamiast Stacked LSTM - wymuszamy generalizację
    GRU(32, return_sequences=False, kernel_regularizer=l2(0.001)), 
    Dropout(0.2),
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
print("\n--- Generowanie Raportu V16 ---")

probs_lstm = lstm_model.predict(X_test_lstm).flatten()
probs_rf = rf_model.predict_proba(X_test_flat)[:, 1]
probs_xgb = xgb_model.predict_proba(X_test_flat)[:, 1]

def analyze_model(name, probs, returns, t_short, t_long):
    if name == 'GRU': # GRU zamiast LSTM
        # Znowu agresywnie, żeby zobaczyć czy działa
        pos = np.where(probs > 0.50, 1, -1)
        size = np.ones_like(pos)
    else:
        pos = np.where(probs > t_long, 1, np.where(probs < t_short, -1, 0))
        size = np.where(pos==1, (probs-t_long)/(1-t_long), (t_short-probs)/t_short)
        size = np.clip(size, 0, 1)

    strat_ret = pos * size * returns
    active = strat_ret[pos != 0]
    win_rate = (active > 0).mean() * 100 if len(active) > 0 else 0
    
    cum_ret = (1 + strat_ret).cumprod()
    total_ret = (cum_ret[-1] - 1) * 100
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    # Drawdown Calculation (Poprawione)
    # Konwertujemy do Pandas Series dla bezpieczeństwa
    s_cum_ret = pd.Series(cum_ret)
    running_max = s_cum_ret.cummax()
    drawdown = (s_cum_ret - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    return {'Model': name, 'Win Rate': win_rate, 'Return %': total_ret, 'Sharpe': sharpe, 'Max DD': max_dd, 'Equity': s_cum_ret, 'Drawdown': drawdown}

res_bh = {'Model': 'Buy & Hold', 'Return %': ((1+ret_test).cumprod()[-1]-1)*100, 'Sharpe':0, 'Max DD':0, 'Win Rate':0, 'Equity': pd.Series((1+ret_test).cumprod())}
res_lstm = analyze_model('GRU', probs_lstm, ret_test, THRESH_SHORT, THRESH_LONG)
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
# Używamy idx_test dla osi X (naprawa błędu)
plt.plot(idx_test, res_bh['Equity'], label='Buy & Hold', color='grey', alpha=0.4)
plt.plot(idx_test, res_lstm['Equity'], label=f'GRU (Sharpe {res_lstm["Sharpe"]:.2f})', linewidth=1.5)
plt.plot(idx_test, res_rf['Equity'], label=f'RF (Sharpe {res_rf["Sharpe"]:.2f})', linewidth=1.5)
plt.plot(idx_test, res_xgb['Equity'], label=f'XGB (Sharpe {res_xgb["Sharpe"]:.2f})', linewidth=1.5)
plt.title(f'V16 Strategy Equity ({TEST_START_DATE} - Present)')
plt.ylabel('Normalized Equity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v16_equity_curve.png')
print("Wykres zapisano: v16_equity_curve.png")

# --- WYKRES 2: DRAWDOWN ---
plt.figure(figsize=(14, 6))
# Używamy idx_test dla osi X (naprawa błędu)
plt.plot(idx_test, res_lstm['Drawdown'], label='GRU Drawdown', linewidth=1)
plt.plot(idx_test, res_rf['Drawdown'], label='RF Drawdown', linewidth=1)
plt.fill_between(idx_test, res_lstm['Drawdown'], 0, alpha=0.1)
plt.title('Drawdown Analysis')
plt.ylabel('Drawdown %')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v16_drawdown.png')
print("Wykres zapisano: v16_drawdown.png")

# --- WYKRES 3: LEARNING ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('GRU Training Process')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v16_learning.png')
print("Wykres zapisano: v16_learning.png")

# --- EXPORT SZCZEGÓŁOWY ---
df_details = pd.DataFrame(index=idx_test)
df_details['Actual_Return'] = ret_test
df_details['GRU_Prob'] = probs_lstm
df_details['GRU_Result'] = np.where(probs_lstm > 0.5, 1, -1) * ret_test
df_details['RF_Prob'] = probs_rf
df_details['XGB_Prob'] = probs_xgb
df_details.to_csv('v16_detailed_metrics.csv')
print("Zapisano CSV: v16_detailed_metrics.csv")