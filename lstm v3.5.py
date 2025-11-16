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
from sklearn.utils.class_weight import compute_class_weight # <-- NOWOŚĆ

# Imports for Models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Step 1: Define Parameters ---
TICKER = 'BTC-USD'
TIMESTEPS = 20 
PROB_THRESHOLD = 0.55 # <-- NOWOŚĆ: Krok 4 (Filtr Pewności) - gramy tylko jak jesteśmy pewni na 55%

# --- Step 2: Define Strict Date Ranges ---
TRAIN_START = '2019-01-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2024-12-31'

DOWNLOAD_START = '2018-06-01'
DOWNLOAD_END = datetime.now().strftime('%Y-%m-%d') 

# --- Step 3: Download and Preprocess Data ---
print(f"Downloading data for {TICKER}...")
data = yf.download(TICKER, start=DOWNLOAD_START, end=DOWNLOAD_END)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if 'Close' not in data.columns:
    if 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']
    else:
        raise ValueError(f"Critical: Column 'Close' not found.")

# --- FEATURE ENGINEERING ---

# 1. Podstawowe
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['RSI'] = ta.rsi(data['Close'], length=14)
data['Volatility'] = data['log_return'].rolling(window=TIMESTEPS).std()
macd = ta.macd(data['Close'])
data['MACD'] = macd.iloc[:, 0]
data['IsWeekend'] = np.where(data.index.dayofweek >= 5, 1, 0)

# 2. (NOWOŚĆ: KROK 1) Lagged Features - Pamięć dla drzew
# Dodajemy opóźnione zwroty z ostatnich 5 dni jako osobne kolumny
lags = 5
for lag in range(1, lags + 1):
    data[f'log_return_lag_{lag}'] = data['log_return'].shift(lag)

# Aktualizacja listy cech
features = ['log_return', 'RSI', 'Volatility', 'MACD', 'IsWeekend'] + [f'log_return_lag_{i}' for i in range(1, lags + 1)]
print(f"Cechy modelu: {features}")

# --- Create Target and Weights ---
target_col = 'target'
weight_col = 'weight' # To są wagi oparte na "wielkości ruchu" (Algograding thesis)
actual_return_col = 'actual_return'

data[target_col] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
data[weight_col] = data['log_return'].shift(-1).abs()
data[actual_return_col] = data['log_return'].shift(-1)

data = data.dropna()
print(f"Total data points after cleaning: {len(data)}")

# --- Step 4: Split Data ---
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df = data.loc[VAL_START:VAL_END].copy()
test_df = data.loc[TEST_START:TEST_END].copy()

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# --- (NOWOŚĆ: KROK 2) Class Weights ---
# Obliczamy wagi, aby zrównoważyć klasy (Up vs Down)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df[target_col]),
    y=train_df[target_col]
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Wagi klas (0: Down, 1: Up): {class_weights_dict}") 
# Jeśli np. {0: 1.2, 1: 0.8}, to znaczy że spadki są rzadsze i ważniejsze.

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
    data_weights = data_df[weight_col].values # To są sample_weights (wielkość ruchu)
    
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

# == Model 1: LSTM (KROK 3: Uproszczony) ==
print("\n--- Training LSTM Model ---")
lstm_model = Sequential()
# Używamy Input() zgodnie z nowymi zaleceniami Keras
lstm_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
# Zmniejszamy liczbę jednostek do 32 (mniej overfittingu)
lstm_model.add(LSTM(units=32)) 
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(units=1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_lstm_v3.keras', monitor='val_loss', save_best_only=True, verbose=0)
]

# UWAGA: Łączymy sample_weight (wielkość ruchu) z class_weight (balans klas)
# W Keras fit() używamy sample_weight ALBO class_weight. 
# Żeby użyć obu, trzeba by je pomnożyć ręcznie. 
# Dla uproszczenia w tym kroku użyjemy class_weight, aby naprawić problem "Permabull".
history = lstm_model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_val, y_val), # wagi walidacyjne są opcjonalne
    class_weight=class_weights_dict, # <-- Używamy Class Weights zamiast Sample Weights tym razem
    shuffle=False,
    callbacks=callbacks,
    verbose=1
)

# == Model 2: Random Forest (Z Lagami) ==
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train_flat, y_train) # RF ma wbudowane class_weight='balanced'

# == Model 3: XGBoost (Z Lagami) ==
print("\n--- Training XGBoost Model ---")
# Obliczamy scale_pos_weight dla XGBoost (to ich odpowiednik class_weight)
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
xgb_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, 
                          scale_pos_weight=scale_pos_weight, # <-- Balansowanie klas w XGB
                          random_state=42, n_jobs=-1)
xgb_model.fit(X_train_flat, y_train)

print("\n--- Model Training Finished ---")

# --- VISUALIZATION 1: LSTM Learning Curves ---
print("Saving LSTM learning curves...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM V3: Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM V3: Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('v3_1_lstm_learning.png')
plt.close()

# --- Step 8: Evaluation ---
# Pobieramy PRAWDOPODOBIEŃSTWA (probabilities), nie same klasy 0/1
y_pred_prob_lstm = lstm_model.predict(X_test).flatten()
y_pred_prob_rf = rf_model.predict_proba(X_test_flat)[:, 1] # Prawdopodobieństwo klasy 1
y_pred_prob_xgb = xgb_model.predict_proba(X_test_flat)[:, 1]

# Konwersja na klasy przy standardowym progu 0.5 do raportów
y_pred_lstm_cls = (y_pred_prob_lstm > 0.5).astype(int)
y_pred_rf_cls = (y_pred_prob_rf > 0.5).astype(int)
y_pred_xgb_cls = (y_pred_prob_xgb > 0.5).astype(int)

print("\nLSTM Report:")
print(classification_report(y_test, y_pred_lstm_cls, target_names=['Down', 'Up'], zero_division=0))

# --- VISUALIZATION 2: Confusion Matrices ---
print("Saving confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_lstm_cls), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('LSTM')
sns.heatmap(confusion_matrix(y_test, y_pred_rf_cls), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest')
sns.heatmap(confusion_matrix(y_test, y_pred_xgb_cls), annot=True, fmt='d', cmap='Oranges', ax=axes[2])
axes[2].set_title('XGBoost')
plt.savefig('v3_2_confusion_matrices.png')
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
    filename = f'v3_3_features_{model_name.replace(" ", "_")}.png'
    plt.savefig(filename)
    plt.close()

plot_feature_importance(rf_model, flat_feature_names, 'Random Forest')
plot_feature_importance(xgb_model, flat_feature_names, 'XGBoost')

# --- Step 9: Backtest with THRESHOLD & METRICS (KROK 4 + KROK 5) ---

# ZMIANA: Zmniejszamy próg, bo modele były zbyt "nieśmiałe" (płaskie linie)
PROB_THRESHOLD = 0.51 

actual_returns_test = test_df[actual_return_col].values[TIMESTEPS:]
test_dates = test_df.index[TIMESTEPS:]
results_df = pd.DataFrame(index=test_dates)
results_df['actual_return'] = actual_returns_test

# Funkcja strategii
def strategy_with_threshold(probs, returns, threshold=0.5):
    # Jeśli pewność > threshold -> 1 (Long)
    # W przeciwnym razie -> 0 (Cash)
    positions = np.where(probs > threshold, 1, 0)
    return positions * returns

# Obliczanie zwrotów
results_df['buy_hold_return'] = results_df['actual_return']
results_df['lstm_return'] = strategy_with_threshold(y_pred_prob_lstm, results_df['actual_return'], PROB_THRESHOLD)
results_df['rf_return'] = strategy_with_threshold(y_pred_prob_rf, results_df['actual_return'], PROB_THRESHOLD)
results_df['xgb_return'] = strategy_with_threshold(y_pred_prob_xgb, results_df['actual_return'], PROB_THRESHOLD)

# Krzywe kapitału (Equity Curves)
results_df['buy_hold_equity'] = results_df['buy_hold_return'].cumsum().apply(np.exp)
results_df['lstm_equity'] = results_df['lstm_return'].cumsum().apply(np.exp)
results_df['rf_equity'] = results_df['rf_return'].cumsum().apply(np.exp)
results_df['xgb_equity'] = results_df['xgb_return'].cumsum().apply(np.exp)

# --- (NOWOŚĆ: KROK 5) RISK METRICS FUNCTION ---
def calculate_metrics(equity_curve, strategy_name):
    """Oblicza profesjonalne metryki finansowe."""
    # Total Return
    total_return = (equity_curve.iloc[-1] - 1) * 100
    
    # Daily Returns (procentowe)
    daily_returns = equity_curve.pct_change().dropna()
    
    # Annualized Volatility (zakładamy 365 dni dla krypto)
    volatility = daily_returns.std() * np.sqrt(365) * 100
    
    # Sharpe Ratio (zakładamy Risk Free Rate = 0 dla uproszczenia)
    if volatility == 0:
        sharpe = 0
    else:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        
    # Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    return {
        "Strategy": strategy_name,
        "Total Return (%)": round(total_return, 2),
        "Volatility (%)": round(volatility, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_drawdown, 2)
    }

# Obliczanie metryk dla każdego modelu
metrics_list = []
metrics_list.append(calculate_metrics(results_df['buy_hold_equity'], "Buy & Hold"))
metrics_list.append(calculate_metrics(results_df['lstm_equity'], "LSTM"))
metrics_list.append(calculate_metrics(results_df['rf_equity'], "Random Forest"))
metrics_list.append(calculate_metrics(results_df['xgb_equity'], "XGBoost"))

# Tworzenie tabeli wyników
metrics_df = pd.DataFrame(metrics_list)
print("\n" + "="*50)
print("FINAL BACKTEST METRICS (2024)")
print("="*50)
print(metrics_df.to_string(index=False))
print("="*50 + "\n")

# --- VISUALIZATION 4: Backtest Results ---
print("Saving backtest results...")
plt.figure(figsize=(14, 8))
plt.plot(results_df['buy_hold_equity'], label='Buy & Hold', linewidth=2, alpha=0.6)
plt.plot(results_df['lstm_equity'], label=f'LSTM (Th {PROB_THRESHOLD})', linestyle='--')
plt.plot(results_df['rf_equity'], label=f'RF (Th {PROB_THRESHOLD})', linestyle=':')
plt.plot(results_df['xgb_equity'], label=f'XGB (Th {PROB_THRESHOLD})', linestyle='-.')
plt.title(f'Backtest V3.5: Equity Curve (Threshold {PROB_THRESHOLD})')
plt.ylabel('Equity (Normalized 1.0)')
plt.legend()
plt.grid(True)
plt.savefig('v3_5_backtest_results.png')
plt.close()

# Zapisz metryki do pliku CSV (dla promotora)
metrics_df.to_csv("v3_5_metrics.csv", index=False)
print("Saved metrics to: v3_5_metrics.csv")

print("\n--- Pipeline V3.5 Finished ---")