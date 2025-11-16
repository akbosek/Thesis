import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Imports for Stationarity & Metrics
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Imports for Models & Tuning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# --- Step 1: Parameters ---
TICKER = 'BTC-USD'
TIMESTEPS = 20 
PROB_THRESHOLD = 0.505 

# --- Step 2: Dates ---
TRAIN_START = '2019-01-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2024-12-31'
DOWNLOAD_START = '2018-06-01'
DOWNLOAD_END = datetime.now().strftime('%Y-%m-%d') 

# --- Step 3: Data ---
print(f"Downloading data for {TICKER}...")
data = yf.download(TICKER, start=DOWNLOAD_START, end=DOWNLOAD_END)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
if 'Close' not in data.columns:
    data['Close'] = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

# --- FEATURE ENGINEERING ---
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['RSI'] = ta.rsi(data['Close'], length=14)
data['Volatility'] = data['log_return'].rolling(window=TIMESTEPS).std()
macd = ta.macd(data['Close'])
data['MACD'] = macd.iloc[:, 0]
data['IsWeekend'] = np.where(data.index.dayofweek >= 5, 1, 0)

# Lags (Pamięć dla drzew)
lags = 5
for lag in range(1, lags + 1):
    data[f'log_return_lag_{lag}'] = data['log_return'].shift(lag)

features = ['log_return', 'RSI', 'Volatility', 'MACD', 'IsWeekend'] + [f'log_return_lag_{i}' for i in range(1, lags + 1)]

# Targets
target_col = 'target'
actual_return_col = 'actual_return'
data[target_col] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
data[actual_return_col] = data['log_return'].shift(-1)

data = data.dropna()
print(f"Data points: {len(data)}")

# --- Step 4: Split ---
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df = data.loc[VAL_START:VAL_END].copy()
test_df = data.loc[TEST_START:TEST_END].copy()

# Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_df[target_col]), y=train_df[target_col])
class_weights_dict = dict(enumerate(class_weights))

# --- Step 5: Scaling ---
scaler = MinMaxScaler()
scaler.fit(train_df[features])
train_df[features] = scaler.transform(train_df[features])
val_df[features] = scaler.transform(val_df[features])
test_df[features] = scaler.transform(test_df[features])

# --- Step 6: Sequences ---
def create_sequences(data_df, feature_cols, target_col, time_steps=TIMESTEPS):
    X, y = [], []
    data_features = data_df[feature_cols].values
    data_target = data_df[target_col].values
    for i in range(len(data_features) - time_steps):
        X.append(data_features[i:(i + time_steps)])
        y.append(data_target[i + time_steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_df, features, target_col)
X_val, y_val = create_sequences(val_df, features, target_col)
X_test, y_test = create_sequences(test_df, features, target_col)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# --- NAPRAWA 1: Tworzenie nazw cech dla wykresu ---
flat_feature_names = []
for i in range(TIMESTEPS):
    for feature in features:
        flat_feature_names.append(f"{feature}_t-{TIMESTEPS-1-i}")

# --- Step 7: MODELING V4 ---

# == 1. LSTM ==
print("\n--- Training LSTM (V4 Resuscitation) ---")
lstm_model = Sequential()
lstm_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(LSTM(units=64, return_sequences=False))
lstm_model.add(Dropout(0.4))
lstm_model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.005) 
lstm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
]

history = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weights_dict,
    shuffle=False,
    callbacks=callbacks,
    verbose=1
)

# == 2. Random Forest ==
print("\n--- Tuning Random Forest (Please wait...) ---")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

tscv = TimeSeriesSplit(n_splits=3)

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=5, # Zmniejszone dla szybkości
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_search.fit(X_train_flat, y_train)
print(f"Best RF Params: {rf_search.best_params_}")
rf_best = rf_search.best_estimator_


# == 3. XGBoost ==
print("\n--- Tuning XGBoost (Please wait...) ---")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [1, 1.5]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_distributions=xgb_param_grid,
    n_iter=5, # Zmniejszone dla szybkości
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
xgb_search.fit(X_train_flat, y_train)
print(f"Best XGB Params: {xgb_search.best_params_}")
xgb_best = xgb_search.best_estimator_

# --- Evaluation ---
y_pred_prob_lstm = lstm_model.predict(X_test).flatten()
y_pred_prob_rf = rf_best.predict_proba(X_test_flat)[:, 1]
y_pred_prob_xgb = xgb_best.predict_proba(X_test_flat)[:, 1]

# --- Visualization: LSTM Learning ---
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('LSTM V4: Loss (with LR Decay)')
plt.legend()
plt.savefig('v4_lstm_learning.png')
plt.close()

# --- Feature Importance (RF Best) ---
plt.figure(figsize=(10, 6))
importances = rf_best.feature_importances_
indices = np.argsort(importances)[-15:]
plt.barh(range(len(indices)), importances[indices], color='green')
plt.yticks(range(len(indices)), [flat_feature_names[i] for i in indices])
plt.title('Top Features (Optimized RF)')
plt.tight_layout()
plt.savefig('v4_rf_importance.png')
plt.close()

# --- Backtest & Metrics ---

# NAPRAWA 2: Definiujemy test_dates
test_dates = test_df.index[TIMESTEPS:]

results_df = pd.DataFrame(index=test_dates)
results_df['actual_return'] = test_df[actual_return_col].values[TIMESTEPS:]

def strategy(probs, returns, threshold):
    return np.where(probs > threshold, 1, 0) * returns

results_df['buy_hold_equity'] = results_df['actual_return'].cumsum().apply(np.exp)
results_df['lstm_equity'] = strategy(y_pred_prob_lstm, results_df['actual_return'], PROB_THRESHOLD).cumsum().apply(np.exp)
results_df['rf_equity'] = strategy(y_pred_prob_rf, results_df['actual_return'], PROB_THRESHOLD).cumsum().apply(np.exp)
results_df['xgb_equity'] = strategy(y_pred_prob_xgb, results_df['actual_return'], PROB_THRESHOLD).cumsum().apply(np.exp)

# Metrics Calculation
def get_metrics(equity, name):
    total_ret = (equity.iloc[-1] - 1) * 100
    dd = (equity - equity.cummax()) / equity.cummax() * 100
    mdd = dd.min()
    return {"Strategy": name, "Return %": round(total_ret, 2), "Max DD %": round(mdd, 2)}

metrics = [
    get_metrics(results_df['buy_hold_equity'], "Buy & Hold"),
    get_metrics(results_df['lstm_equity'], "LSTM"),
    get_metrics(results_df['rf_equity'], "Random Forest (Opt)"),
    get_metrics(results_df['xgb_equity'], "XGBoost (Opt)")
]

print("\n" + "="*40)
print("FINAL V4 METRICS")
print("="*40)
print(pd.DataFrame(metrics).to_string(index=False))
print("="*40)

# Plot Backtest
plt.figure(figsize=(14, 8))
plt.plot(results_df['buy_hold_equity'], label='Buy & Hold', alpha=0.5)
plt.plot(results_df['lstm_equity'], label='LSTM V4', linewidth=2)
plt.plot(results_df['rf_equity'], label='RF Optimized', linewidth=2)
plt.plot(results_df['xgb_equity'], label='XGB Optimized', linewidth=2)
plt.title(f'Backtest V4 (Optimized) - Threshold {PROB_THRESHOLD}')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.savefig('v4_backtest_results.png')
plt.close()

print("Done. Check v4_backtest_results.png")