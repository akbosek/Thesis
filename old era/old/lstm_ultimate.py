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
#      HIPERPARAMETRY (TWOJE SUWAKI)
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# Daty (Pełen podział)
TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' # 5 lat nauki
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31' # 1 rok walidacji
TEST_START  = '2024-01-01' # 1 rok testu

# Parametry Modelu (Zwycięskie z V3)
TIMESTEPS = 14
NEURONS   = 64
DROPOUT   = 0.2
LEARNING_RATE = 0.001
EPOCHS    = 60
BATCH_SIZE = 32

# Strategia
THRESH_SHORT = 0.48
THRESH_LONG  = 0.52
# ==========================================

print("--- [LSTM ULTIMATE: DIAGNOSTIC & ADVISOR] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# 1. DANE & FEATURES
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

data.dropna(inplace=True)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']

# 2. PODZIAŁ NA 3 CZĘŚCI
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()
test_df  = data.loc[TEST_START:].copy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# 3. SKALOWANIE (Fit tylko na Train)
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
X_val, y_val, ret_val, idx_val = create_dataset(val_df)
X_test, y_test, ret_test, idx_test = create_dataset(test_df)

# 4. TRENING
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(NEURONS, return_sequences=False),
    BatchNormalization(),
    Dropout(DROPOUT),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
estop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[estop],
    verbose=0
)

# 5. DIAGNOSTYKA (Dla każdego zbioru)
def get_metrics(X, y_true, ret, name):
    probs = model.predict(X, verbose=0).flatten()
    pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    strat_ret = pos * ret
    
    # Win Rate
    active = pos != 0
    if np.sum(active) > 0:
        wr = (np.sign(pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    # Gini
    try: gini = 2 * roc_auc_score(y_true, probs) - 1
    except: gini = 0
    
    # Sharpe
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    # Return & DD
    cum = np.exp(np.cumsum(strat_ret))
    total_ret = (cum[-1] - 1) * 100
    dd = (cum / np.maximum.accumulate(cum) - 1).min() * 100
    
    return {
        "Dataset": name, "WinRate": round(wr, 2), "Gini": round(gini, 3), 
        "Sharpe": round(sharpe, 2), "Return%": round(total_ret, 1), 
        "MaxDD%": round(dd, 1), "Trades": np.sum(active)
    }

print("\n--- Obliczanie Metryk ---")
m_train = get_metrics(X_train, y_train, ret_train, "TRAIN (17-22)")
m_val   = get_metrics(X_val, y_val, ret_val, "VAL (2023)")
m_test  = get_metrics(X_test, y_test, ret_test, "TEST (2024)")

# 6. RAPORT KOŃCOWY
results = pd.DataFrame([m_train, m_val, m_test])
print("\n" + "="*60)
print("   ULTIMATE DIAGNOSTIC REPORT   ")
print("="*60)
print(results.to_string(index=False))
print("-" * 60)

# 7. INTELIGENTNY DORADCA (SMART ADVISOR)
print("\n🤖 [AI ADVISOR] Analiza wyników i sugestie:")

suggestions = []
priorities = []

# Analiza Overfittingu (Train vs Val)
wr_drop = m_train['WinRate'] - m_val['WinRate']
gini_drop = m_train['Gini'] - m_val['Gini']

if wr_drop > 8.0:
    p = "WYSOKI"
    s = f"Wykryto OVERFITTING! WinRate spada o {wr_drop:.1f}% na walidacji."
    a = "-> Zwiększ DROPOUT do 0.3 lub 0.4.\n-> Zmniejsz NEURONS do 32.\n-> Zwiększ TIMESTEPS (więcej kontekstu)."
    suggestions.append((p, s, a))
elif wr_drop < -5.0:
    suggestions.append(("NISKI", "Ciekawostka: Model radzi sobie lepiej na nowych danych (Underfitting/Luck?).", "-> Możesz spróbować zwiększyć NEURONS do 128, żeby model nauczył się więcej z historii."))

# Analiza Decyzyjności (Trades)
if m_val['Trades'] < 20:
    p = "ŚREDNI"
    s = "Model jest zbyt pasywny (mało transakcji na Val)."
    a = f"-> Zawęź progi (np. {THRESH_SHORT+0.01}/{THRESH_LONG-0.01}).\n-> Sprawdź czy zmienna 'volatility_30' nie ma outlierów."
    suggestions.append((p, s, a))

# Analiza Jakości (Gini)
if m_val['Gini'] < 0.02:
    p = "WYSOKI"
    s = "Model na walidacji zgaduje losowo (Gini bliskie 0)."
    a = "-> Twoje cechy (Features) mogą być za słabe.\n-> Dodaj wskaźnik trendu (np. SMA Distance) lub oscylator (StochRSI)."
    suggestions.append((p, s, a))

# Analiza Sharpe (Ryzyko)
if m_test['Sharpe'] < 1.0 and m_test['Return%'] > 50:
    p = "ŚREDNI"
    s = "Wysoki zwrot, ale niska stabilność (Sharpe < 1)."
    a = "-> Rozważ dodanie filtru zmienności (nie graj, gdy ATR jest bardzo wysoki)."
    suggestions.append((p, s, a))

if not suggestions:
    print("✅ Model wygląda na STABILNY i ZBALANSOWANY. Brak krytycznych uwag.")
    print("   Możesz śmiało użyć tych wyników w pracy.")
else:
    for i, (pri, sug, act) in enumerate(suggestions):
        print(f"\n#{i+1}. PRIORYTET: [{pri}]")
        print(f"   Problem: {sug}")
        print(f"   Sugestia: \n   {act}")

print("="*60)

# Wykres
plt.figure(figsize=(12, 6))
cum_bh = np.exp(np.cumsum(ret_test))
cum_strat = np.exp(np.cumsum(np.where(model.predict(X_test, verbose=0).flatten() > THRESH_LONG, 1, np.where(model.predict(X_test, verbose=0).flatten() < THRESH_SHORT, -1, 0)) * ret_test))

plt.plot(idx_test, cum_bh, label='Buy & Hold (2024)', color='gray', alpha=0.5)
plt.plot(idx_test, cum_strat, label=f'LSTM Strategy (+{m_test["Return%"]}%)', color='green', linewidth=2)
plt.title(f'FINAL PERFORMANCE 2024 (Sharpe {m_test["Sharpe"]})')
plt.legend()
plt.savefig('ultimate_result.png')