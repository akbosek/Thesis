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

# --- 1. KONFIGURACJA ---
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 

# Parametry LSTM
TIMESTEPS = 14  # 14 dni historii
BATCH_SIZE = 16 # Mniejszy batch = częstsze aktualizacje wag
EPOCHS = 100    # Dajemy mu czas

# Daty
TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE   = '2022-12-31'
TEST_START_DATE = '2023-01-01'

# --- 2. WCZYTANIE ---
print(f"--- [V18: LSTM PRECISION] Wczytywanie: {INPUT_FILE} ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Brak pliku CSV!")

data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]

# Daily Resampling
data = data_4h.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

# --- 3. FEATURE ENGINEERING ---
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=30).mean()
data['RSI'] = ta.rsi(data['Close'], length=14)
data['MACD'] = ta.macd(data['Close']).iloc[:, 0]
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) / data['Close']

# Bollinger Bands %B
bb = ta.bbands(data['Close'], length=20, std=2)
data['BB_Pct'] = bb.iloc[:, 4]

# Lags
lags = 5
for lag in range(1, lags + 1):
    data[f'log_return_lag_{lag}'] = data['log_return'].shift(lag)

features = ['log_return', 'Rel_Vol', 'RSI', 'MACD', 'ATR', 'BB_Pct'] + [f'log_return_lag_{i}' for i in range(1, lags + 1)]

# Target
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data['actual_return_pct'] = np.exp(data['log_return'].shift(-1)) - 1 

data = data.dropna()

# --- 4. PODZIAŁ ---
train_df = data.loc[:TRAIN_END_DATE].copy()
val_df   = data.loc[VAL_START_DATE:VAL_END_DATE].copy()
test_df  = data.loc[TEST_START_DATE:].copy()

class_weights = compute_class_weight('balanced', classes=np.unique(train_df['target']), y=train_df['target'])
class_weights_dict = dict(enumerate(class_weights))

# --- 5. SKALOWANIE (MinMax dla LSTM jest zazwyczaj lepszy) ---
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_df[features])

def prepare_data(df, scaler):
    X_scaled = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_scaled) - TIMESTEPS):
        X.append(X_scaled[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    return np.array(X), np.array(y), df.index[TIMESTEPS:], df['actual_return_pct'].iloc[TIMESTEPS:].values

X_train, y_train, _, _ = prepare_data(train_df, scaler)
X_val, y_val, _, _ = prepare_data(val_df, scaler)
X_test, y_test, idx_test, ret_test = prepare_data(test_df, scaler)

# --- 6. MODELOWANIE (LSTM V18 Optimized) ---

print("\n--- Training LSTM V18 ---")
lstm_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # WARSTWA 1: Bidirectional LSTM
    # 64 jednostki - wystarczająco dużo by złapać wzorce, ale nie za dużo by przeuczyć
    # l2(0.001) - lekka regularyzacja
    Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001))),
    
    BatchNormalization(), # Stabilizuje uczenie
    Dropout(0.4),         # Zapobiega poleganiu na pojedynczych neuronach
    
    # WARSTWA GĘSTA
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    
    # WYJŚCIE
    Dense(1, activation='sigmoid')
])

# Bardzo wolny learning rate dla precyzji
optimizer = Adam(learning_rate=0.0001) 

lstm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    # Cierpliwe Early Stopping
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    # Redukcja LR gdy utknie
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

history = lstm_model.fit(
    X_train, y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, # Ważna zmiana
    validation_data=(X_val, y_val), 
    class_weight=class_weights_dict, 
    shuffle=False, 
    callbacks=callbacks, 
    verbose=1
)

# --- 7. RAPORT I WYKRESY ---
print("\n--- Generowanie Wyników V18 ---")

probs_lstm = lstm_model.predict(X_test).flatten()

# DataFrame
df_res = pd.DataFrame(index=idx_test)
df_res['Actual'] = ret_test
df_res['LSTM_Prob'] = probs_lstm

# STRATEGIA (Agresywna - Próg 0.50)
df_res['LSTM_Pos'] = np.where(probs_lstm > 0.50, 1, -1)
df_res['LSTM_Ret'] = df_res['LSTM_Pos'] * df_res['Actual']

# Equity Curves
eq_bh = (1 + df_res['Actual']).cumprod()
eq_lstm = (1 + df_res['LSTM_Ret']).cumprod()

# Metryki
def get_metrics(equity, name):
    total = (equity.iloc[-1] - 1) * 100
    daily = equity.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() != 0 else 0
    dd = (equity / equity.cummax() - 1).min() * 100
    return {'Model': name, 'Return %': total, 'Sharpe': sharpe, 'Max DD %': dd}

m_bh = get_metrics(eq_bh, "Buy & Hold")
m_lstm = get_metrics(eq_lstm, "LSTM V18")

print(pd.DataFrame([m_bh, m_lstm]).round(2).to_string(index=False))

# --- WYKRES 1: EQUITY ---
plt.figure(figsize=(14, 7))
plt.plot(df_res.index, eq_bh, label='Buy & Hold', color='gray', alpha=0.5)
plt.plot(df_res.index, eq_lstm, label='LSTM V18 Strategy', color='orange', linewidth=2)
plt.title('V18 LSTM Performance (No Threshold)')
plt.ylabel('Equity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v18_lstm_result.png')
print("Zapisano: v18_lstm_result.png")

# --- WYKRES 2: LEARNING ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM V18 Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v18_learning.png')
print("Zapisano: v18_learning.png")

# --- EXPORT SZCZEGÓŁOWY ---
df_res.to_csv('v18_lstm_trades.csv')
print("Zapisano: v18_lstm_trades.csv")