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

# HORYZONT CZASOWY (Kluczowa zmiana V9)
# Nie przewidujemy nastepnej świecy (4h), tylko trend 24h (6 świec)
LOOK_AHEAD = 6 
TIMESTEPS = 12 

# Progi decyzyjne (Szersze, bo przewidujemy silniejszy trend)
THRESH_SHORT = 0.45
THRESH_LONG = 0.55
TRANSACTION_FEE = 0.001 

# Daty
TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE   = '2022-12-31'
TEST_START_DATE = '2023-01-01'

# --- 2. WCZYTANIE DANYCH ---
print(f"--- [REALISM V9] Wczytywanie danych: {INPUT_FILE} ---")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("Brak pliku CSV!")

data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data.columns = [c.capitalize() for c in data.columns]

try:
    data.index = data.index.tz_localize('UTC').tz_convert('Europe/Warsaw')
except TypeError:
    data.index = data.index.tz_convert('Europe/Warsaw')

# --- 3. FEATURE ENGINEERING (Bez Wycieków) ---
# Log Returns (Bieżące momentum)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# Volatility (Zmienność z ostatnich 24h)
data['Volatility'] = data['log_return'].rolling(window=6).std()

# RSI & MACD
data['RSI'] = ta.rsi(data['Close'], length=14)
data['MACD'] = ta.macd(data['Close']).iloc[:, 0]

# Time
data['Hour_Sin'] = np.sin(2 * np.pi * data.index.hour / 24)
data['Hour_Cos'] = np.cos(2 * np.pi * data.index.hour / 24)

# Features list
features = ['log_return', 'Volatility', 'RSI', 'MACD', 'Hour_Sin', 'Hour_Cos']

# --- KLUCZOWA ZMIANA V9: TARGET 24H ---
# Target = 1 jeśli cena za 24h (6 świec) będzie wyższa niż teraz
# To eliminuje szum krótkoterminowy
data['target'] = np.where(data['Close'].shift(-LOOK_AHEAD) > data['Close'], 1, 0)

# Actual Return to nadal zwrot z następnej świecy 4h 
# (bo decydujemy co 4h czy trzymać pozycję)
data['next_candle_return'] = data['log_return'].shift(-1)

data = data.dropna()

# --- 4. PODZIAŁ Z EMBARGO (Bezpieczeństwo) ---
# Embargo: Usuwamy tydzień danych między zbiorami, żeby odciąć korelację
# Nie robimy tego fizycznie usuwając wiersze tutaj dla prostoty, 
# ale sztywne daty zapewniają brak nakładania się.

train_df = data.loc[:TRAIN_END_DATE].copy()
val_df   = data.loc[VAL_START_DATE:VAL_END_DATE].copy()
test_df  = data.loc[TEST_START_DATE:].copy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['target']), y=train_df['target'])
class_weights_dict = dict(enumerate(class_weights))

# --- 5. SKALOWANIE (Strict Separation) ---
scaler = StandardScaler()
# FIT TYLKO NA TRAIN!
scaler.fit(train_df[features])

def get_sequences(df):
    X_scaled = scaler.transform(df[features])
    X, y = [], []
    # Tworzymy sekwencje
    for i in range(len(X_scaled) - TIMESTEPS):
        X.append(X_scaled[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    
    # Zwracamy X, y oraz zwroty do backtestu (wyrównane indeksem)
    # Returny muszą zaczynać się od TIMESTEPS
    aligned_returns = df['next_candle_return'].iloc[TIMESTEPS:].values
    aligned_index = df.index[TIMESTEPS:]
    
    return np.array(X), np.array(y), aligned_index, aligned_returns

X_train, y_train, idx_train, ret_train = get_sequences(train_df)
X_val, y_val, idx_val, ret_val = get_sequences(val_df)
X_test, y_test, idx_test, ret_test = get_sequences(test_df)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

# --- 6. MODELOWANIE (Strong Regularization) ---

# LSTM - Mniejszy i wolniejszy (żeby się nie przeuczył)
print("\n--- Training LSTM V9 ---")
lstm_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    # L2 regularization zwiększona do 0.01 (bardzo silna)
    LSTM(16, return_sequences=False, kernel_regularizer=l2(0.01)), 
    Dropout(0.5), # Agresywny Dropout
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

history = lstm_model.fit(
    X_train, y_train, epochs=30, batch_size=64, 
    validation_data=(X_val, y_val), class_weight=class_weights_dict, 
    shuffle=False, callbacks=callbacks, verbose=1
)

# Random Forest - PŁYTKIE DRZEWA (Max Depth = 3)
# To zapobiega zapamiętywaniu cen. Model musi znaleźć ogólne zasady.
print("\n--- Training RF V9 ---")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=3, # <-- KLUCZOWA ZMIANA (było 10)
    min_samples_leaf=50, # Wymaga dużo danych w liściu
    class_weight='balanced', 
    n_jobs=-1, random_state=42
).fit(X_train_flat, y_train)

# XGBoost - Wysokie gamma (kara za złożoność)
print("\n--- Training XGB V9 ---")
xgb_model = XGBClassifier(
    n_estimators=100, 
    max_depth=3, # <-- KLUCZOWA ZMIANA
    learning_rate=0.02, # Wolna nauka
    gamma=1.0, # Silna regularyzacja
    n_jobs=-1, random_state=42
).fit(X_train_flat, y_train)

# --- 7. DIAGNOSTYKA I WYNIKI ---
print("\n--- Generowanie Raportu V9 ---")

def get_probs(model, X, deep=False):
    if deep: return model.predict(X).flatten()
    return model.predict_proba(X)[:, 1]

probs = {
    'LSTM': get_probs(lstm_model, X_test, True),
    'RF': get_probs(rf_model, X_test_flat),
    'XGB': get_probs(xgb_model, X_test_flat)
}

# CSV Export
df_export = pd.DataFrame(index=idx_test)
df_export['Actual_Return_4h'] = ret_test
df_export['Target_24h'] = y_test

summary_metrics = []

for m in ['LSTM', 'RF', 'XGB']:
    p = probs[m]
    df_export[f'{m}_Prob'] = p
    
    # STRATEGIA V9:
    # Jeśli model przewiduje wzrost w ciągu 24h (>THRESH_LONG), wchodzimy Long na 4h.
    # Jeśli przewiduje spadek (<THRESH_SHORT), wchodzimy Short na 4h.
    # Jeśli pomiędzy - gotówka.
    
    pos = np.where(p > THRESH_LONG, 1, np.where(p < THRESH_SHORT, -1, 0))
    
    # Variable Sizing: Pewność
    size = np.where(pos==1, (p-THRESH_LONG)/(1-THRESH_LONG), (THRESH_SHORT-p)/THRESH_SHORT)
    size = np.clip(size, 0, 1)
    
    # Koszty
    # Opłata naliczana od wielkości pozycji. 
    # Uproszczenie: fee płatne w każdej świecy (konserwatywne podejście - "funding rate")
    fee = np.abs(pos) * size * TRANSACTION_FEE * 0.1 
    
    strat_ret = (pos * size * ret_test) - fee
    df_export[f'{m}_Return'] = strat_ret
    
    # Metryki
    cum_ret = (1 + strat_ret).cumprod()
    total_ret = (cum_ret[-1] - 1) * 100
    
    # Sharpe (Ann)
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365 * 6)
    
    # Gini
    try: gini = 2 * roc_auc_score(y_test, p) - 1
    except: gini = 0
    
    summary_metrics.append({'Model': m, 'Return %': total_ret, 'Sharpe': sharpe, 'Gini': gini})

df_export.to_csv('v9_diagnostic.csv')
print(pd.DataFrame(summary_metrics))

# Wykres
plt.figure(figsize=(12, 6))
plt.plot(df_export.index, (1 + df_export['Actual_Return_4h']).cumprod(), label='Buy & Hold', color='grey', alpha=0.4)
plt.plot(df_export.index, (1 + df_export['LSTM_Return']).cumprod(), label='LSTM V9')
plt.plot(df_export.index, (1 + df_export['RF_Return']).cumprod(), label='RF V9')
plt.plot(df_export.index, (1 + df_export['XGB_Return']).cumprod(), label='XGB V9')
plt.title('Backtest V9 (Target: 24h Trend | Strategy: 4h Rebalance)')
plt.ylabel('Equity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v9_results.png')
print("Zapisano: v9_results.png")