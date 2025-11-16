import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Imports for Stationarity
from statsmodels.tsa.stattools import adfuller

# Imports for Scaling and Metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# Imports for Models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # <-- NOWE IMPORTY
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Step 1: Define Parameters ---
TICKER = 'BTC-USD'
TIMESTEPS = 20 

# --- Step 2: Define Strict Date Ranges ---
TRAIN_START = '2019-01-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2024-12-31'

DOWNLOAD_START = '2018-06-01' # Dłuższy bufor dla MACD
DOWNLOAD_END = datetime.now().strftime('%Y-%m-%d') 

# --- Step 3: Download and Preprocess Data ---

print(f"Downloading data for {TICKER}...")
data = yf.download(TICKER, start=DOWNLOAD_START, end=DOWNLOAD_END)
if data.empty:
    raise ValueError(f"No data downloaded for {TICKER}.")

# YFinance MultiIndex Fix
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Ensure Close column
if 'Close' not in data.columns:
    if 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']
    else:
        raise ValueError(f"Critical: Column 'Close' not found.")

# --- FEATURE ENGINEERING (NOWOŚCI) ---

# 1. Podstawowe
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['RSI'] = ta.rsi(data['Close'], length=14)
data['Volatility'] = data['log_return'].rolling(window=TIMESTEPS).std()

# 2. (NOWOŚĆ) MACD - Wskaźnik Trendu
# pandas_ta zwraca 3 kolumny dla MACD, bierzemy główną linię (iloc[:, 0])
macd = ta.macd(data['Close'])
data['MACD'] = macd.iloc[:, 0]

# 3. (NOWOŚĆ) IsWeekend - Dzień tygodnia
# dayofweek: 0=Mon, 1=Tue, ..., 4=Fri, 5=Sat, 6=Sun
# Jeśli >= 5 to weekend (1), inaczej dzień roboczy (0)
data['IsWeekend'] = np.where(data.index.dayofweek >= 5, 1, 0)

# Lista wszystkich cech
features = ['log_return', 'RSI', 'Volatility', 'MACD', 'IsWeekend']
print(f"Używane cechy: {features}")

# --- Step 3c: Create Target and Weights ---
target_col = 'target'
weight_col = 'weight'
actual_return_col = 'actual_return'

data[target_col] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
data[weight_col] = data['log_return'].shift(-1).abs()
data[actual_return_col] = data['log_return'].shift(-1)

# --- CLEANING ---
data = data.dropna()
print(f"Total data points after cleaning: {len(data)}")

if len(data) == 0:
    raise ValueError("Data cleaning removed all rows! Check calculation logic.")

# --- Step 3b: Stationarity Check ---
print("\n--- Stationarity Check (ADF Test) ---")
for feature in features:
    # IsWeekend jest binarny, nie ma sensu robić ADF, pomijamy
    if feature == 'IsWeekend': continue
    
    adf_result = adfuller(data[feature])
    print(f"Feature: {feature}, P-value: {adf_result[1]:.4f}")
    if adf_result[1] > 0.05:
        print(f"Warning: Feature '{feature}' may be non-stationary")
print("----------------------------------------\n")

# --- Step 4: Split Data ---
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df = data.loc[VAL_START:VAL_END].copy()
test_df = data.loc[TEST_START:TEST_END].copy()

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if train_df.empty or val_df.empty or test_df.empty:
     raise ValueError("One of the data splits is empty.")

# --- Step 5: Scaling ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[features])

train_df[features] = scaler.transform(train_df[features])
val_df[features] = scaler.transform(val_df[features])
test_df[features] = scaler.transform(test_df[features])

# --- Step 6: Create Sequences ---
def create_sequences(data_df, feature_cols, target_col, weight_col, time_steps=TIMESTEPS):
    X, y, w = [], [], []
    data_features = data_df[feature_cols].values
    data_target = data_df[target_col].values
    data_weights = data_df[weight_col].values
    for i in range(len(data_features) - time_steps):
        X.append(data_features[i:(i + time_steps)])
        y.append(data_target[i + time_steps])
        w.append(data_weights[i + time_steps])
    return np.array(X), np.array(y), np.array(w)

X_train, y_train, w_train = create_sequences(train_df, features, target_col, weight_col, TIMESTEPS)
X_val, y_val, w_val = create_sequences(val_df, features, target_col, weight_col, TIMESTEPS)
X_test, y_test, w_test = create_sequences(test_df, features, target_col, weight_col, TIMESTEPS)

# Flatten for RF/XGB
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Names for feature importance
flat_feature_names = []
for i in range(TIMESTEPS):
    for feature in features:
        flat_feature_names.append(f"{feature}_t-{TIMESTEPS-1-i}")

# --- Step 7: Model Training ---

# == Model 1: LSTM (ULEPSZONY) ==
print("\n--- Training LSTM Model (with Early Stopping) ---")
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dropout(0.3)) # Zwiększony Dropout dla redukcji overfittingu
lstm_model.add(Dense(units=1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# (NOWOŚĆ) Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_lstm_model.keras', monitor='val_loss', save_best_only=True, verbose=0)
]

history = lstm_model.fit(
    X_train, y_train,
    epochs=50, # Możemy dać więcej, bo EarlyStopping i tak zatrzyma
    batch_size=32,
    validation_data=(X_val, y_val, w_val),
    sample_weight=w_train,
    shuffle=False,
    callbacks=callbacks, # Dodajemy callbacks
    verbose=1
)

# == Model 2: Random Forest ==
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_flat, y_train, sample_weight=w_train)

# == Model 3: XGBoost ==
print("\n--- Training XGBoost Model ---")
xgb_model = XGBClassifier(n_estimators=150, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_flat, y_train, sample_weight=w_train)

print("\n--- Model Training Finished ---")

# --- VISUALIZATION 1: LSTM Learning Curves ---
print("Saving LSTM learning curves...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM: Krzywa Straty')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM: Krzywa Trafności')
plt.legend()
plt.tight_layout()
plt.savefig('v2_1_lstm_learning_curves.png')
plt.close()

# --- Step 8: Evaluation ---
print("\n--- Model Evaluation ---")
y_pred_lstm = (lstm_model.predict(X_test) > 0.5).astype(int).flatten()
y_pred_rf = rf_model.predict(X_test_flat)
y_pred_xgb = xgb_model.predict(X_test_flat)

print("\nLSTM Report:")
print(classification_report(y_test, y_pred_lstm, target_names=['Down', 'Up'], zero_division=0))
print("\nXGBoost Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Down', 'Up'], zero_division=0))

# --- VISUALIZATION 2: Confusion Matrices ---
print("Saving confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
class_names = ['Down', 'Up']
sns.heatmap(confusion_matrix(y_test, y_pred_lstm), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('LSTM')
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest')
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Oranges', ax=axes[2])
axes[2].set_title('XGBoost')
plt.savefig('v2_2_confusion_matrices.png')
plt.close()

# --- VISUALIZATION 3: Feature Importance ---
print("Saving feature importances...")
def plot_feature_importance(model, names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.title(f'Top 20 Features - {model_name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [names[i] for i in indices])
    plt.tight_layout()
    filename = f'v2_3_features_{model_name.replace(" ", "_")}.png'
    plt.savefig(filename)
    plt.close()

plot_feature_importance(rf_model, flat_feature_names, 'Random Forest')
plot_feature_importance(xgb_model, flat_feature_names, 'XGBoost')

# --- Step 9: Backtest ---
actual_returns_test = test_df[actual_return_col].values[TIMESTEPS:]
test_dates = test_df.index[TIMESTEPS:]

results_df = pd.DataFrame(index=test_dates)
results_df['actual_return'] = actual_returns_test
results_df['lstm_pred'] = y_pred_lstm
results_df['rf_pred'] = y_pred_rf
results_df['xgb_pred'] = y_pred_xgb

results_df['buy_hold_equity'] = results_df['actual_return'].cumsum().apply(np.exp)
results_df['lstm_equity'] = np.where(results_df['lstm_pred']==1, results_df['actual_return'], 0).cumsum().apply(np.exp)
results_df['rf_equity'] = np.where(results_df['rf_pred']==1, results_df['actual_return'], 0).cumsum().apply(np.exp)
results_df['xgb_equity'] = np.where(results_df['xgb_pred']==1, results_df['actual_return'], 0).cumsum().apply(np.exp)

# --- VISUALIZATION 4: Backtest Results ---
print("Saving backtest results...")
plt.figure(figsize=(14, 8))
plt.plot(results_df['buy_hold_equity'], label='Buy & Hold', linewidth=2, alpha=0.7)
plt.plot(results_df['lstm_equity'], label='LSTM (EarlyStop)', linestyle='--')
plt.plot(results_df['rf_equity'], label='Random Forest (Tuned)', linestyle=':')
plt.plot(results_df['xgb_equity'], label='XGBoost (Tuned)', linestyle='-.')
plt.title(f'{TICKER} Backtest - Improved V2')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.savefig('v2_4_backtest_results.png')
plt.close()

print("\n--- Pipeline V2 Finished Successfully ---")