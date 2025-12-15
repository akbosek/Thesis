import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
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

TIMESTEPS = 5  # Patrzymy 5 dni wstecz
THRESH_SHORT = 0.45
THRESH_LONG = 0.55

# --- ZMIANA: WYŁĄCZONE FEE ---
TRANSACTION_FEE = 0.0 
# -----------------------------

# Daty
TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE   = '2022-12-31'
TEST_START_DATE = '2023-01-01'

# --- 2. WCZYTANIE I KONWERSJA NA 1D ---
print(f"--- [DAILY V10.1 - NO FEE] Wczytywanie danych: {INPUT_FILE} ---")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("Brak pliku CSV!")

data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]

# RESAMPLING: Zamiana 4h na 1D (Daily)
print("--- Konwersja danych 4h -> 1D (Daily) ---")
data = data_4h.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})
data = data.dropna()

print(f"Liczba dni handlowych: {len(data)}")

# --- 3. FEATURE ENGINEERING (Daily) ---
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

# Target: Czy jutro cena wyższa niż dziś?
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data['actual_return'] = data['log_return'].shift(-1)

data = data.dropna()

# --- 4. PODZIAŁ DANYCH ---
train_df = data.loc[:TRAIN_END_DATE].copy()
val_df   = data.loc[VAL_START_DATE:VAL_END_DATE].copy()
test_df  = data.loc[TEST_START_DATE:].copy()

print(f"Train Days: {len(train_df)} | Val Days: {len(val_df)} | Test Days: {len(test_df)}")

class_weights = compute_class_weight('balanced', classes=np.unique(train_df['target']), y=train_df['target'])
class_weights_dict = dict(enumerate(class_weights))

# --- 5. SKALOWANIE ---
scaler = StandardScaler()
scaler.fit(train_df[features])

def prepare_set(df):
    X_scaled = scaler.transform(df[features])
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - TIMESTEPS):
        X_seq.append(X_scaled[i:(i + TIMESTEPS)])
        y_seq.append(df['target'].iloc[i + TIMESTEPS])
    return np.array(X_seq), np.array(y_seq), df.index[TIMESTEPS:], df['actual_return'].iloc[TIMESTEPS:].values

X_train, y_train, idx_train, ret_train = prepare_set(train_df)
X_val, y_val, idx_val, ret_val = prepare_set(val_df)
X_test, y_test, idx_test, ret_test = prepare_set(test_df)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

# --- 6. MODELOWANIE ---

# LSTM
print("\n--- Training LSTM ---")
lstm_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(16, return_sequences=False, kernel_regularizer=l2(0.01)), 
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

history = lstm_model.fit(
    X_train, y_train, epochs=50, batch_size=32, 
    validation_data=(X_val, y_val), class_weight=class_weights_dict, 
    shuffle=False, callbacks=callbacks, verbose=1
)

# RF & XGB
print("\n--- Training Trees ---")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_leaf=10, 
    class_weight='balanced', n_jobs=-1, random_state=42
).fit(X_train_flat, y_train)

xgb_model = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.02, 
    scale_pos_weight=1.2, n_jobs=-1, random_state=42
).fit(X_train_flat, y_train)

# --- 7. DIAGNOSTYKA I EKSPORT ---
print("\n--- Generowanie Raportu V10.1 (No Fee) ---")

def get_probs(model, X, deep=False):
    if deep: return model.predict(X).flatten()
    return model.predict_proba(X)[:, 1]

probs = {
    'LSTM': get_probs(lstm_model, X_test, True),
    'RF': get_probs(rf_model, X_test_flat),
    'XGB': get_probs(xgb_model, X_test_flat)
}

df_export = pd.DataFrame(index=idx_test)
df_export['Actual_Return_Daily'] = ret_test
df_export['Target'] = y_test

summary_metrics = []

for m in ['LSTM', 'RF', 'XGB']:
    p = probs[m]
    df_export[f'{m}_Prob'] = p
    
    # STRATEGIA DZIENNA
    pos = np.where(p > THRESH_LONG, 1, np.where(p < THRESH_SHORT, -1, 0))
    
    # Variable Sizing
    size = np.where(pos==1, (p-THRESH_LONG)/(1-THRESH_LONG), (THRESH_SHORT-p)/THRESH_SHORT)
    size = np.clip(size, 0, 1)
    
    # Fee = 0.0 (Wyłączone)
    fee = np.abs(pos) * size * TRANSACTION_FEE 
    
    strat_ret = (pos * size * ret_test) - fee
    df_export[f'{m}_Return'] = strat_ret
    
    # Metryki
    cum_ret = (1 + strat_ret).cumprod()
    total_ret = (cum_ret[-1] - 1) * 100
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    try: gini = 2 * roc_auc_score(y_test, p) - 1
    except: gini = 0
    
    summary_metrics.append({'Model': m, 'Return %': total_ret, 'Sharpe': sharpe, 'Gini': gini})

df_export.to_csv('v10_nofee_diagnostic.csv')
print(pd.DataFrame(summary_metrics))

# Wykres
plt.figure(figsize=(12, 6))
plt.plot(df_export.index, (1 + df_export['Actual_Return_Daily']).cumprod(), label='Buy & Hold', color='grey', alpha=0.4)
plt.plot(df_export.index, (1 + df_export['LSTM_Return']).cumprod(), label='LSTM Daily')
plt.plot(df_export.index, (1 + df_export['RF_Return']).cumprod(), label='RF Daily')
plt.plot(df_export.index, (1 + df_export['XGB_Return']).cumprod(), label='XGB Daily')
plt.title('Backtest V10.1 (Daily Strategy | NO FEES)')
plt.ylabel('Equity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v10_nofee_results.png')
print("Zapisano: v10_nofee_results.png")