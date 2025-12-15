import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================================
#      KONFIGURACJA (ZWYCIĘSKA - NIE DOTYKAĆ)
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# Podział: Uczymy się na wszystkim do końca 2023, Testujemy 2024
TRAIN_START = '2017-09-01'
TRAIN_END   = '2023-12-31' 
TEST_START  = '2024-01-01' 

# Parametry (Zwycięski zestaw)
TIMESTEPS = 14
NEURONS   = 64
DROPOUT   = 0.2
LEARNING_RATE = 0.001
EPOCHS    = 60
BATCH_SIZE = 32

THRESH_SHORT = 0.48
THRESH_LONG  = 0.52
# ==========================================

print("--- [LSTM FINAL ENHANCED: REPORTING MODE] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# 1. DANE
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

data.dropna(inplace=True)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']

# 2. PODZIAŁ
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
test_df  = data.loc[TEST_START:].copy()

print(f"Okres Treningowy (In-Sample): {len(train_df)} dni")
print(f"Okres Testowy (Out-of-Sample): {len(test_df)} dni")

# 3. SKALOWANIE
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    # Future Returns do symulacji equity
    future_returns = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    return np.array(X), np.array(y), future_returns, df.index[TIMESTEPS:]

X_train, y_train, ret_train, idx_train = create_dataset(train_df)
X_test, y_test, ret_test, idx_test = create_dataset(test_df)

# 4. TRENING (Identyczny jak w wersji finalnej)
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(NEURONS, return_sequences=False),
    BatchNormalization(),
    Dropout(DROPOUT),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

# Uwaga: Nie używamy EarlyStopping na val_loss tutaj, bo trenujemy na całym zbiorze historycznym
# żeby odtworzyć wynik "final_test". Używamy stałej liczby epok lub loss treningowego.
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)
print("Trening zakończony.")

# 5. GENEROWANIE PEŁNEGO RAPORTU
def calculate_full_metrics(X, y_true, ret, name):
    # Predykcja
    probs = model.predict(X, verbose=0).flatten()
    
    # Strategia
    pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    strat_ret = pos * ret
    
    # Win Rate
    active = pos != 0
    trades = np.sum(active)
    if trades > 0:
        wr = (np.sign(pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    # GINI INDICATOR
    try:
        auc = roc_auc_score(y_true, probs)
        gini = 2 * auc - 1
    except: gini = 0.0
    
    # Sharpe
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    # Returns & DD
    cum_strat = np.exp(np.cumsum(strat_ret))
    cum_bh = np.exp(np.cumsum(ret))
    
    total_ret = (cum_strat[-1] - 1) * 100
    bh_ret = (cum_bh[-1] - 1) * 100
    
    dd = (cum_strat / np.maximum.accumulate(cum_strat) - 1).min() * 100
    
    return {
        "Period": name,
        "WinRate %": round(wr, 2),
        "Gini": round(gini, 3), # <--- NOWOŚĆ
        "Sharpe": round(sharpe, 2),
        "Return %": round(total_ret, 2),
        "BH Return %": round(bh_ret, 2),
        "Max DD %": round(dd, 2),
        "Trades": trades
    }

print("\n--- Obliczanie metryk dla Treningu i Testu ---")
m_train = calculate_full_metrics(X_train, y_train, ret_train, "TRAIN (2017-2023)")
m_test  = calculate_full_metrics(X_test, y_test, ret_test, "TEST (2024)")

# 6. WYŚWIETLANIE TABELI
results = pd.DataFrame([m_train, m_test])

print("\n" + "="*95)
print("   FINAL PERFORMANCE REPORT (WITH GINI)   ")
print("="*95)
print(results.to_string(index=False))
print("-" * 95)

# Interpretacja Giniego
print("\nℹ️ [INFO] Interpretacja Gini Ratio:")
print(f"   TRAIN Gini ({m_train['Gini']}): Jak dobrze model dopasował się do historii.")
print(f"   TEST Gini  ({m_test['Gini']}):  Jak dobrze model generalizuje na nowe dane.")
if m_test['Gini'] > 0.05:
    print("   -> Wynik > 0.05 na Teście oznacza realną przewagę statystyczną (Edge).")
elif m_test['Gini'] > 0:
    print("   -> Wynik dodatni oznacza, że model jest lepszy od rzutu monetą.")
else:
    print("   -> Wynik ujemny lub 0 oznacza brak zdolności predykcyjnych.")
print("="*95)

# Wykresy
plt.figure(figsize=(12, 10))

# 1. Equity Test (2024)
plt.subplot(2, 1, 1)
cum_bh_test = np.exp(np.cumsum(ret_test))
cum_strat_test = np.exp(np.cumsum(np.where(model.predict(X_test, verbose=0).flatten() > THRESH_LONG, 1, np.where(model.predict(X_test, verbose=0).flatten() < THRESH_SHORT, -1, 0)) * ret_test))
plt.plot(idx_test, cum_bh_test, label='Buy & Hold (2024)', color='gray', alpha=0.5)
plt.plot(idx_test, cum_strat_test, label=f'LSTM Strategy (Sharpe {m_test["Sharpe"]})', color='green', linewidth=2)
plt.title('TEST 2024: Equity Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Equity Train (2017-2023) - Żeby zobaczyć historię
plt.subplot(2, 1, 2)
cum_bh_train = np.exp(np.cumsum(ret_train))
cum_strat_train = np.exp(np.cumsum(np.where(model.predict(X_train, verbose=0).flatten() > THRESH_LONG, 1, np.where(model.predict(X_train, verbose=0).flatten() < THRESH_SHORT, -1, 0)) * ret_train))
plt.plot(idx_train, cum_bh_train, label='Buy & Hold (History)', color='gray', alpha=0.5)
plt.plot(idx_train, cum_strat_train, label='LSTM Strategy (History)', color='blue')
plt.title('TRAIN (2017-2023): Historical Fit')
plt.legend()
plt.yscale('log') # Logarytmiczna skala dla długiego okresu
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_enhanced_report.png')
print("Zapisano wykres: final_enhanced_report.png")