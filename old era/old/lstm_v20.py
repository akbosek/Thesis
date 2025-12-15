import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. KONFIGURACJA ---
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 
TIMESTEPS = 10 
BATCH_SIZE = 32
EPOCHS = 20 # Szybkie epoki w pętli

# PARAMETRY WALK-FORWARD (W dniach)
TRAIN_WINDOW = 365 * 2  # 2 lata historii do nauki
VAL_WINDOW = 90         # 3 miesiące walidacji (do sprawdzenia modelu przed testem)
TEST_WINDOW = 30        # 1 miesiąc testu (rzeczywista gra)

print(f"--- [V20: DIAGNOSTIC WALK-FORWARD] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Brak pliku!")

# --- 2. DANE ---
data_4h = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data_4h.columns = [c.capitalize() for c in data_4h.columns]
data = data_4h.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

# --- 3. FEATURES ---
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['RSI'] = ta.rsi(data['Close'], length=14)
data['ATR_Pct'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) / data['Close']
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=30).mean()
data['MACD'] = ta.macd(data['Close']).iloc[:, 0]

features = ['log_return', 'RSI', 'ATR_Pct', 'Rel_Vol', 'MACD']

data['target'] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
data['actual_return'] = data['log_return'].shift(-1)
data = data.dropna()

# --- 4. FUNKCJE POMOCNICZE ---
def calculate_step_metrics(probs, y_true, returns):
    # Proste metryki dla wycinka danych
    if len(y_true) < 2: return 0, 0, 0
    
    # Win Rate
    preds = np.where(probs > 0.5, 1, -1)
    actual_dir = np.sign(returns)
    # Fix for 0 returns
    actual_dir[actual_dir == 0] = 1 
    wins = (preds == actual_dir)
    win_rate = np.mean(wins) * 100
    
    # Gini
    try:
        auc = roc_auc_score(y_true, probs)
        gini = 2 * auc - 1
    except: gini = 0
    
    # Sharpe (uproszczony dla wycinka)
    strat_ret = preds * returns
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    return win_rate, gini, sharpe

def create_seq(df, scaler):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    return np.array(X), np.array(y)

# --- 5. PĘTLA WALK-FORWARD ---
# Startujemy tak, żeby mieć miejsce na Train + Val
start_index = TRAIN_WINDOW + VAL_WINDOW 
total_rows = len(data)

results_history = [] # Tu zbieramy diagnostykę z każdego kroku
equity_curve = []    # Tu zbieramy wynik finansowy

print(f"Start symulacji od indeksu: {start_index} ({data.index[start_index]})")

current_idx = start_index

while current_idx < total_rows - TEST_WINDOW:
    # A. Definiowanie Okien
    # Indices: [Train Start ... Train End] [Val Start ... Val End] [Test Start ... Test End]
    
    val_start = current_idx - VAL_WINDOW
    train_start = val_start - TRAIN_WINDOW
    
    train_data = data.iloc[train_start : val_start].copy()
    val_data   = data.iloc[val_start : current_idx].copy()
    test_data  = data.iloc[current_idx : current_idx + TEST_WINDOW].copy()
    
    # B. Skalowanie (Tylko na Train!)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data[features])
    
    # C. Przygotowanie Sekwencji
    X_train, y_train = create_seq(train_data, scaler)
    # Val i Test potrzebują kawałka danych poprzedzających
    X_val, y_val = create_seq(pd.concat([train_data.tail(TIMESTEPS), val_data]), scaler)
    X_test, y_test = create_seq(pd.concat([val_data.tail(TIMESTEPS), test_data]), scaler)
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("Brak danych w oknie, pomijam...")
        current_idx += TEST_WINDOW
        continue

    # D. Trening Modelu (Reset co krok!)
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Cichy trening
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0, shuffle=False)
    
    # E. Diagnostyka (Train vs Val vs Test)
    p_train = model.predict(X_train, verbose=0).flatten()
    p_val   = model.predict(X_val, verbose=0).flatten()
    p_test  = model.predict(X_test, verbose=0).flatten()
    
    # Pobieramy zwroty do obliczeń
    # Uwaga: create_seq ucina pierwsze TIMESTEPS wierszy z przekazanego DF
    # Musimy dopasować length
    
    # Helper do zwrotów
    def get_rets(df_source, preds_len):
        return df_source['actual_return'].iloc[-preds_len:].values

    r_train = get_rets(train_data, len(p_train))
    r_val   = get_rets(val_data, len(p_val))
    r_test  = get_rets(test_data, len(p_test))
    
    # Obliczamy metryki dla każdego zbioru
    wr_tr, gini_tr, sh_tr = calculate_step_metrics(p_train, y_train, r_train)
    wr_val, gini_val, sh_val = calculate_step_metrics(p_val, y_val, r_val)
    wr_test, gini_test, sh_test = calculate_step_metrics(p_test, y_test, r_test)
    
    # Zapisujemy diagnostykę
    step_date = data.index[current_idx]
    results_history.append({
        'Date': step_date,
        'Train_Gini': gini_tr, 'Val_Gini': gini_val, 'Test_Gini': gini_test,
        'Train_Sharpe': sh_tr, 'Val_Sharpe': sh_val, 'Test_Sharpe': sh_test,
        'Train_WR': wr_tr,     'Val_WR': wr_val,     'Test_WR': wr_test
    })
    
    # Zbieramy equity curve (Test)
    for i in range(len(p_test)):
        pos = 1 if p_test[i] > 0.5 else -1
        ret = r_test[i]
        equity_curve.append({
            'Date': test_data.index[i],
            'Actual': ret,
            'Strat_Ret': pos * ret,
            'Prob': p_test[i]
        })
        
    print(f"[{step_date.date()}] Val Gini: {gini_val:.3f} | Test Gini: {gini_test:.3f}")
    
    current_idx += TEST_WINDOW

# --- 6. RAPORTY ---
df_diag = pd.DataFrame(results_history).set_index('Date')
df_eq = pd.DataFrame(equity_curve).set_index('Date')

# Obliczanie Equity
df_eq['Equity'] = (1 + df_eq['Strat_Ret']).cumprod()
df_eq['BH_Equity'] = (1 + df_eq['Actual']).cumprod()

print("\n=== PODSUMOWANIE DIAGNOSTYCZNE ===")
print("Średnie wyniki z wszystkich okresów Walk-Forward:")
print(df_diag.mean().round(3))

# Zapis plików
df_diag.to_csv('v20_audit_steps.csv')
df_eq.to_csv('v20_equity_data.csv')

# --- WYKRES 1: GINI OVER TIME ---
plt.figure(figsize=(12, 6))
plt.plot(df_diag['Train_Gini'], label='Train Gini', color='blue', alpha=0.3)
plt.plot(df_diag['Val_Gini'], label='Val Gini', color='orange', alpha=0.8)
plt.plot(df_diag['Test_Gini'], label='Test Gini (Real)', color='green', linewidth=2)
plt.axhline(0, color='black', linestyle='--')
plt.title('V20: Evolution of Model Intelligence (Gini Score)')
plt.legend()
plt.savefig('v20_gini_evolution.png')

# --- WYKRES 2: EQUITY ---
plt.figure(figsize=(12, 6))
plt.plot(df_eq['BH_Equity'], label='Buy & Hold', color='grey', alpha=0.5)
plt.plot(df_eq['Equity'], label='Walk-Forward Strategy', color='red')
plt.title('V20: Realistic Equity Curve (No Leakage)')
plt.legend()
plt.savefig('v20_equity_curve.png')

print("Zapisano: v20_audit_steps.csv, v20_equity_data.csv, v20_gini_evolution.png, v20_equity_curve.png")