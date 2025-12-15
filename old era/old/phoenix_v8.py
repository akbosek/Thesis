import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# --- 1. KONFIGURACJA ---
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 
TIMESTEPS = 10 
EPOCHS = 100
BATCH_SIZE = 16

# Definicja okresów (Sztywne ramy dla bezpieczeństwa)
TRAIN_START = '2018-01-01'
TRAIN_END   = '2021-12-31' # 4 lata nauki
VAL_START   = '2022-01-01'
VAL_END     = '2022-12-31' # 1 rok sprawdzianu (bessa 2022!)
TEST_START  = '2023-01-01' # Ostatnie 2 lata (To zostawiamy na koniec)

print(f"--- [PHOENIX V8: AUDIT & OPTIMIZE] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Brak pliku CSV!")

# --- 2. PRZYGOTOWANIE DANYCH ---
data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]
data = data_4h.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

# --- 3. FEATURE ENGINEERING (Bez wycieków) ---
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['RSI'] = ta.rsi(data['Close'], length=14)
data['ATR_Pct'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) / data['Close']
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=30).mean()
data['MACD'] = ta.macd(data['Close']).iloc[:, 0]

# Lags
for i in range(1, 4):
    data[f'lag_{i}'] = data['log_return'].shift(i)

features = ['log_return', 'RSI', 'ATR_Pct', 'Rel_Vol', 'MACD', 'lag_1', 'lag_2', 'lag_3']

# Target (Następny dzień)
data['target'] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
# Wagi: Uczymy się mocniej na dużych ruchach
data['sample_weight'] = data['log_return'].shift(-1).abs() * 100
data['actual_return'] = data['log_return'].shift(-1)

data = data.dropna()

# --- 4. PODZIAŁ I SKALOWANIE (Kluczowy moment dla Data Leakage) ---
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()
test_df  = data.loc[TEST_START:].copy()

# UWAGA: Skaler uczy się TYLKO na treningu.
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_df[features])

def prepare_dataset(df):
    X_scaled = scaler.transform(df[features])
    X, y, w = [], [], []
    for i in range(len(X_scaled) - TIMESTEPS):
        X.append(X_scaled[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
        w.append(df['sample_weight'].iloc[i + TIMESTEPS])
    # Zwracamy też oryginalne zwroty do analizy
    return np.array(X), np.array(y), np.array(w), df['actual_return'].iloc[TIMESTEPS:].values

X_train, y_train, w_train, ret_train = prepare_dataset(train_df)
X_val, y_val, w_val, ret_val = prepare_dataset(val_df)
X_test, y_test, w_test, ret_test = prepare_dataset(test_df)

print(f"Dane gotowe. Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# --- 5. MODELOWANIE (LSTM) ---
print("\n--- Trenowanie Modelu ---")
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001))),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

history = model.fit(
    X_train, y_train, sample_weight=w_train,
    validation_data=(X_val, y_val, w_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0
)

# --- 6. AUDYT I OPTYMALIZACJA ---
print("\n--- Rozpoczynam Audyt Skuteczności ---")

# Pobieramy predykcje dla WSZYSTKICH zbiorów
p_train = model.predict(X_train).flatten()
p_val = model.predict(X_val).flatten()
p_test = model.predict(X_test).flatten()

# Funkcja licząca pełny zestaw metryk
def calculate_metrics(probs, y_true, returns, threshold=0.5):
    # Decyzje (1 = Long, -1 = Short)
    # Prosta logika: powyżej progu Long, poniżej Short
    # Można tu dodać strefę neutralną, ale do audytu bierzemy pure binary
    preds = np.where(probs > threshold, 1, -1)
    
    # Skuteczność (Win Rate)
    # Win = (Long i wzrost) lub (Short i spadek)
    wins = np.sign(preds) == np.sign(returns)
    win_rate = np.mean(wins) * 100
    
    # Gini Ratio (2*AUC - 1)
    try:
        auc = roc_auc_score(y_true, probs)
        gini = 2 * auc - 1
    except: gini = 0
    
    # Wynik finansowy (bez lewaru, czysta strategia)
    strat_ret = preds * returns
    cum_ret = (1 + strat_ret).cumprod()
    total_ret = (cum_ret[-1] - 1) * 100
    
    # Sharpe
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    # Max Drawdown
    peak = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - peak) / peak
    max_dd = np.min(dd) * 100
    
    # Volatility (Annualized)
    vol = np.std(strat_ret) * np.sqrt(365) * 100
    
    return {
        "Win Rate %": round(win_rate, 2),
        "Gini Ratio": round(gini, 3),
        "Sharpe": round(sharpe, 2),
        "Max DD %": round(max_dd, 2),
        "Return %": round(total_ret, 2),
        "Volatility %": round(vol, 2)
    }

# --- SZUKANIE OPTYMALNEGO PROGU (Na zbiorze Walidacyjnym!) ---
# Nie patrzymy na Test, żeby nie oszukiwać.
print("Szukanie optymalnego progu na zbiorze WALIDACYJNYM...")
best_thresh = 0.5
best_val_wr = 0

thresholds = np.arange(0.45, 0.55, 0.005) # Sprawdzamy co 0.005
for t in thresholds:
    # Symulacja na Val
    m = calculate_metrics(p_val, y_val, ret_val, t)
    if m["Win Rate %"] > best_val_wr:
        best_val_wr = m["Win Rate %"]
        best_thresh = t

print(f"Znaleziono optymalny próg: {best_thresh:.3f} (Val Win Rate: {best_val_wr}%)")

# --- RAPORT KOŃCOWY ---
metrics_train = calculate_metrics(p_train, y_train, ret_train, best_thresh)
metrics_val   = calculate_metrics(p_val, y_val, ret_val, best_thresh)
metrics_test  = calculate_metrics(p_test, y_test, ret_test, best_thresh) # Dopiero teraz sprawdzamy Test

# Tworzenie Tabeli Porównawczej
df_audit = pd.DataFrame([metrics_train, metrics_val, metrics_test], index=['TRAIN (2018-21)', 'VAL (2022)', 'TEST (2023-24)'])

print("\n=== RAPORT AUDYTU MODELU (Threshold: {best_thresh:.3f}) ===")
print(df_audit.to_string())

# --- WYKRESY DIAGNOSTYCZNE ---
plt.figure(figsize=(15, 5))

# 1. Krzywe Kapitału (Znormalizowane do 1)
plt.subplot(1, 3, 1)
plt.plot(np.cumprod(1 + np.where(p_train > best_thresh, 1, -1) * ret_train), label='Train')
plt.plot(np.cumprod(1 + np.where(p_val > best_thresh, 1, -1) * ret_val), label='Val')
plt.title('Equity Curves (Train vs Val)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Histogram Predykcji (Czy model jest pewny?)
plt.subplot(1, 3, 2)
sns.histplot(p_train, color='blue', alpha=0.3, label='Train', kde=True)
sns.histplot(p_val, color='orange', alpha=0.3, label='Val', kde=True)
plt.axvline(best_thresh, color='red', linestyle='--', label='Threshold')
plt.title('Rozkład Pewności (Probabilities)')
plt.legend()

# 3. Learning Curve
plt.subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Krzywa Uczenia (Loss)')
plt.legend()

plt.tight_layout()
plt.savefig('v8_audit_report.png')
print("\nZapisano raport graficzny: v8_audit_report.png")

# CSV z pełnymi metrykami
df_audit.to_csv('v8_audit_metrics.csv')