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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. KONFIGURACJA ---
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 
TIMESTEPS = 10 
BATCH_SIZE = 16
EPOCHS = 50 # Mniej epok, bo będziemy douczać wiele razy

# PARAMETRY WALK-FORWARD
TRAIN_WINDOW = 365 * 2  # Uczymy się na ostatnich 2 latach (730 dni)
TEST_WINDOW = 30        # Przewidujemy na 30 dni do przodu (re-trening co miesiąc)

print(f"--- [V19: WALK-FORWARD VALIDATION] ---")
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

# Features list
features = ['log_return', 'RSI', 'ATR_Pct', 'Rel_Vol', 'MACD']

# Target (Next Day)
data['target'] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
data['actual_return'] = data['log_return'].shift(-1)
data = data.dropna()

# --- 4. MECHANIZM WALK-FORWARD ---
# Startujemy symulację od 2022 roku (żeby mieć 2018-2021 na pierwszy trening)
start_index = data.index.get_loc('2022-01-01')
total_rows = len(data)

results = []
equity_curve = [1.0] # Start z 100% kapitału

print(f"Start symulacji: {data.index[start_index]} (Index: {start_index})")
print(f"Krok walidacji: {TEST_WINDOW} dni. To może chwilę potrwać...")

# Główna pętla czasowa
current_idx = start_index

while current_idx < total_rows - TEST_WINDOW:
    # 1. Definiujemy okno treningowe (przesuwne)
    train_start_idx = current_idx - TRAIN_WINDOW
    if train_start_idx < 0: train_start_idx = 0
    
    train_data = data.iloc[train_start_idx : current_idx].copy()
    test_data  = data.iloc[current_idx : current_idx + TEST_WINDOW].copy()
    
    # 2. Skalowanie (Izolowane! Uczymy scaler tylko na obecnym oknie train)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data[features])
    
    # Funkcja pomocnicza do sekwencji
    def create_seq(df):
        X_sc = scaler.transform(df[features])
        X, y = [], []
        for i in range(len(X_sc) - TIMESTEPS):
            X.append(X_sc[i:(i + TIMESTEPS)])
            y.append(df['target'].iloc[i + TIMESTEPS])
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_data)
    # Dla testu musimy wziąć trochę danych wstecz z train, żeby zbudować sekwencje dla pierwszych dni testu
    combined_test = pd.concat([train_data.tail(TIMESTEPS), test_data])
    X_test, y_test = create_seq(combined_test)
    
    # Jeśli coś poszło nie tak z wymiarami (np. koniec danych), przerwij
    if len(X_test) == 0: break

    # 3. Budowa i Trening Modelu (Od zera co miesiąc!)
    # Prosty LSTM, żeby było szybciej
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Szybki trening
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, shuffle=False)
    
    # 4. Predykcja na miesiąc do przodu
    probs = model.predict(X_test, verbose=0).flatten()
    
    # Zbieranie wyników
    # Odcinamy nadmiarowe predykcje jeśli create_seq wygenerował ich za mało/za dużo względem test_data
    # create_seq na combined_test powinien dać dokładnie len(test_data)
    
    actual_returns = test_data['actual_return'].values
    dates = test_data.index
    
    # Dopasowanie długości (safety check)
    min_len = min(len(probs), len(actual_returns))
    
    for i in range(min_len):
        p = probs[i]
        ret = actual_returns[i]
        date = dates[i]
        
        # STRATEGIA (Prosta: >0.5 Long, <0.5 Short)
        pos = 1 if p > 0.5 else -1
        strat_ret = pos * ret
        
        results.append({
            'Date': date,
            'Actual': ret,
            'Prob': p,
            'Position': pos,
            'Strat_Ret': strat_ret
        })
    
    print(f"Przetworzono okres: {data.index[current_idx].date()} -> {data.index[current_idx + TEST_WINDOW].date()}")
    
    # Przesuwamy okno
    current_idx += TEST_WINDOW

# --- 5. ANALIZA WYNIKÓW ---
df_res = pd.DataFrame(results).set_index('Date')

# Equity Curve
df_res['Equity'] = (1 + df_res['Strat_Ret']).cumprod()
df_res['BH_Equity'] = (1 + df_res['Actual']).cumprod()

# Metryki
total_ret = (df_res['Equity'].iloc[-1] - 1) * 100
bh_total = (df_res['BH_Equity'].iloc[-1] - 1) * 100

# Sharpe
daily_rets = df_res['Strat_Ret']
sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(365) if daily_rets.std() != 0 else 0

# Win Rate
wins = np.sign(df_res['Position']) == np.sign(df_res['Actual'])
win_rate = wins.mean() * 100

# Gini
try:
    auc = roc_auc_score(np.where(df_res['Actual'] > 0, 1, 0), df_res['Prob'])
    gini = 2 * auc - 1
except: gini = 0

print("\n=== WYNIKI WALK-FORWARD (2022-2024) ===")
print(f"Liczba dni handlowych: {len(df_res)}")
print(f"Strategia Return: {total_ret:.2f}%")
print(f"Buy & Hold Return: {bh_total:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Gini Ratio (Cały okres): {gini:.3f}")

# Wykres
plt.figure(figsize=(12, 6))
plt.plot(df_res['BH_Equity'], label='Buy & Hold', color='grey', alpha=0.5)
plt.plot(df_res['Equity'], label=f'Walk-Forward LSTM (Sharpe {sharpe:.2f})', color='blue')
plt.title('Walk-Forward Validation Results (No Leakage Guaranteed)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v19_walk_forward.png')
df_res.to_csv('v19_results.csv')