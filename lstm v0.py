import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns # <-- (NOWA BIBLIOTEKA)

# Imports for Stationarity
from statsmodels.tsa.stattools import adfuller

# Imports for Scaling and Metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix # <-- (NOWY IMPORT)

# Imports for Models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Step 1: Define Parameters ---
TICKER = 'BTC-USD'
TIMESTEPS = 20 # "last 20 observations"

# --- Step 2: Define Strict Date Ranges ---
TRAIN_START = '2019-01-01'
TRAIN_END = '2022-12-31' # 4 years

VAL_START = '2023-01-01'
VAL_END = '2023-12-31' # 1 year

TEST_START = '2024-01-01'
TEST_END = '2024-12-31' # 1 year

DOWNLOAD_START = '2018-10-01' # Buffer for indicators
DOWNLOAD_END = datetime.now().strftime('%Y-%m-%d') 

# --- Step 3: Download and Preprocess Data ---

print(f"Downloading data for {TICKER}...")
data = yf.download(TICKER, start=DOWNLOAD_START, end=DOWNLOAD_END)
if data.empty:
    raise ValueError(f"No data downloaded for {TICKER}.")

# --- (FIX FOR YFINANCE ISSUE) ---
# Jeśli yfinance zwrócił MultiIndex (np. Price, Ticker), spłaszczamy go do zwykłych nazw
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
# --- (END FIX) ---

print("Columns after download:", data.columns) # Diagnostyka

# Calculate Features
features = ['log_return', 'RSI', 'Volatility']
# Upewniamy się, że mamy kolumnę Close
if 'Close' not in data.columns:
    # Czasami yfinance zwraca 'adj close' jako główną
    if 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']
    else:
        raise ValueError(f"Critical: Column 'Close' not found. Columns are: {data.columns}")

data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['RSI'] = ta.rsi(data['Close'], length=14)
data['Volatility'] = data['log_return'].rolling(window=TIMESTEPS).std()

# --- Step 3c: (MOVED UP) Create Target and Weights ---
target_col = 'target'
weight_col = 'weight'
actual_return_col = 'actual_return'

data[target_col] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
data[weight_col] = data['log_return'].shift(-1).abs()
data[actual_return_col] = data['log_return'].shift(-1)

# Diagnostyka przed czyszczeniem
print(f"Data before cleaning: {len(data)} rows")
print("Sample data (head):\n", data[['Close', 'RSI', 'log_return']].head())

# --- (FIX) ---
# CRITICAL: Drop NaNs NOW
data = data.dropna()
print(f"Total data points after cleaning: {len(data)}")

# Jeśli nadal jest 0, rzuć błąd, żebyśmy wiedzieli dlaczego
if len(data) == 0:
    raise ValueError("Data cleaning removed all rows! Check if indicators (RSI) are calculating correctly.")
# --- (END FIX) ---


# --- Step 4: Split Data by Date ---
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df = data.loc[VAL_START:VAL_END].copy()
test_df = data.loc[TEST_START:TEST_END].copy()

print(f"Training set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")

if train_df.empty or val_df.empty or test_df.empty:
     raise ValueError("One of the data splits is empty. Check your dates.")

# --- Step 5: Scaling (Fit ONLY on Training Data) ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[features])

train_df[features] = scaler.transform(train_df[features])
val_df[features] = scaler.transform(val_df[features])
test_df[features] = scaler.transform(test_df[features])

# --- Step 6: Create Sequences Function ---
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

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, w_train shape: {w_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, w_test shape: {w_test.shape}")

# --- Step 6b: Flatten Data for RF/XGB ---
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
print(f"X_train_flat shape: {X_train_flat.shape}")

# Create feature names for the flattened data (for feature importance plot)
flat_feature_names = []
for i in range(TIMESTEPS):
    for feature in features:
        flat_feature_names.append(f"{feature}_t-{TIMESTEPS-1-i}")

# --- Step 7: Model Training ---

# == Model 1: LSTM ==
print("\n--- Training LSTM Model ---")
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Zapisujemy historię treningu do zmiennej 'history'
history = lstm_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val, w_val),
    sample_weight=w_train,
    shuffle=False,
    verbose=1
)

# == Model 2: Random Forest ==
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_flat, y_train, sample_weight=w_train)

# == Model 3: XGBoost ==
print("\n--- Training XGBoost Model ---")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_flat, y_train, sample_weight=w_train)

print("\n--- Model Training Finished ---")

# --- (NEW VISUALIZATION 1: LSTM Learning Curves) ---
print("Plotting LSTM learning curves...")

plt.figure(figsize=(12, 5))
# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM: Krzywa Straty (Loss)')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM: Krzywa Trafności (Accuracy)')
plt.xlabel('Epoka')
plt.ylabel('Trafność')
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 8: Model Evaluation (Metrics) ---
print("\n--- Model Evaluation Reports (Test Set) ---")

# Get predictions
y_pred_lstm_prob = lstm_model.predict(X_test)
y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int).flatten()
y_pred_rf = rf_model.predict(X_test_flat)
y_pred_xgb = rf_model.predict(X_test_flat) # Poprawka: używamy rf_model, bo XGB ma te same predykcje w tym przykładzie. Popraw na xgb_model.predict(X_test_flat)
y_pred_xgb = xgb_model.predict(X_test_flat) # Poprawna linia

# Print reports
print("\nLSTM Classification Report:")
print(classification_report(y_test, y_pred_lstm, target_names=['Down (0)', 'Up (1)'], zero_division=0))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Down (0)', 'Up (1)'], zero_division=0))
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Down (0)', 'Up (1)'], zero_division=0))


# --- (NEW VISUALIZATION 2: Confusion Matrices) ---
print("Plotting confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Macierze Pomyłek (Test Set)', fontsize=16)
class_names = ['Down (0)', 'Up (1)']

# LSTM
cm_lstm = confusion_matrix(y_test, y_pred_lstm)
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=class_names, yticklabels=class_names)
axes[0].set_title('LSTM')
axes[0].set_ylabel('Rzeczywistość')
axes[0].set_xlabel('Predykcja')

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=class_names, yticklabels=class_names)
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Predykcja')

# XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[2], xticklabels=class_names, yticklabels=class_names)
axes[2].set_title('XGBoost')
axes[2].set_xlabel('Predykcja')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- (NEW VISUALIZATION 3: Feature Importance) ---
print("Plotting feature importances...")

def plot_feature_importance(model, names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:] # Weź 20 najważniejszych cech
    
    plt.figure(figsize=(10, 7))
    plt.title(f'Istotność Cech (Top 20) - {model_name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [names[i] for i in indices])
    plt.xlabel('Względna Istotność')
    plt.show()

# Tworzymy wykresy istotności dla RF i XGB
plot_feature_importance(rf_model, flat_feature_names, 'Random Forest')
plot_feature_importance(xgb_model, flat_feature_names, 'XGBoost')


# --- Step 9: Backtesting Strategy ---
print("\n--- Running Backtest ---")

actual_returns_test = test_df[actual_return_col].values[TIMESTEPS:]
test_dates = test_df.index[TIMESTEPS:]

results_df = pd.DataFrame(index=test_dates)
results_df['actual_return'] = actual_returns_test
results_df['lstm_pred'] = y_pred_lstm
results_df['rf_pred'] = y_pred_rf
results_df['xgb_pred'] = y_pred_xgb

results_df['buy_hold_return'] = results_df['actual_return']
results_df['lstm_strategy_return'] = np.where(results_df['lstm_pred'] == 1, results_df['actual_return'], 0)
results_df['rf_strategy_return'] = np.where(results_df['rf_pred'] == 1, results_df['actual_return'], 0)
results_df['xgb_strategy_return'] = np.where(results_df['xgb_pred'] == 1, results_df['actual_return'], 0)

results_df['buy_hold_equity'] = results_df['buy_hold_return'].cumsum().apply(np.exp)
results_df['lstm_strategy_equity'] = results_df['lstm_strategy_return'].cumsum().apply(np.exp)
results_df['rf_strategy_equity'] = results_df['rf_strategy_return'].cumsum().apply(np.exp)
results_df['xgb_strategy_equity'] = results_df['xgb_strategy_return'].cumsum().apply(np.exp)


# --- Step 10: (EXISTING VISUALIZATION) Plot Backtest Results ---
print("Plotting backtest results...")
plt.figure(figsize=(14, 8))
plt.plot(results_df['buy_hold_equity'], label='Buy & Hold')
plt.plot(results_df['lstm_strategy_equity'], label='LSTM Strategy', linestyle='--')
plt.plot(results_df['rf_strategy_equity'], label='Random Forest Strategy', linestyle=':')
plt.plot(results_df['xgb_strategy_equity'], label='XGBoost Strategy', linestyle='-.')

plt.title(f'{TICKER} Backtest ({TEST_START} to {TEST_END})')
plt.ylabel('Equity (Cumulative Returns)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Pipeline Finished ---")