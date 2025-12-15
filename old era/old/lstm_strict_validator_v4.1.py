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
#      HIPERPARAMETRY (ZATWIERDZONE)
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# DATY (BEZ OKRESU TESTOWEGO!)
TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' # Nauka na hossie i bessie
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31' # Egzamin na "trudnym" rynku bocznym
# Rok 2024 jest zakryty.

# Parametry Modelu
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

print("--- [LSTM V4.1: STRICT VALIDATOR] ---")
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

# 2. PODZIAŁ (Tylko Train i Val)
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

print(f"Train (Nauka): {len(train_df)} dni")
print(f"Val (Sprawdzian): {len(val_df)} dni")

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

# 5. DIAGNOSTYKA (Tylko Train vs Val)
def get_metrics(X, y_true, ret, name):
    probs = model.predict(X, verbose=0).flatten()
    pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    strat_ret = pos * ret
    
    # Win Rate (tylko aktywne)
    active = pos != 0
    if np.sum(active) > 0:
        wr = (np.sign(pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    # Gini
    try: gini = 2 * roc_auc_score(y_true, probs) - 1
    except: gini = 0
    
    # Sharpe & Volatility
    if np.std(strat_ret) == 0: 
        sharpe = 0
        vol = 0
    else: 
        sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
        vol = np.std(strat_ret) * np.sqrt(365) * 100
    
    # Return & DD
    cum = np.exp(np.cumsum(strat_ret))
    total_ret = (cum[-1] - 1) * 100
    dd = (cum / np.maximum.accumulate(cum) - 1).min() * 100
    
    return {
        "Dataset": name, 
        "WinRate %": round(wr, 2), 
        "Gini": round(gini, 3), 
        "Sharpe": round(sharpe, 2), 
        "Volatility %": round(vol, 2),
        "MaxDD %": round(dd, 1), 
        "Trades": np.sum(active)
    }

print("\n--- Obliczanie Metryk Diagnostycznych ---")
m_train = get_metrics(X_train, y_train, ret_train, "TRAIN (2017-2022)")
m_val   = get_metrics(X_val, y_val, ret_val, "VAL (2023)")

# 6. RAPORT PORÓWNAWCZY
results = pd.DataFrame([m_train, m_val])
print("\n" + "="*80)
print("   STRICT VALIDATION REPORT (NO TEST DATA)   ")
print("="*80)
print(results.to_string(index=False))
print("-" * 80)

# 7. AI ADVISOR (Ocena Przetrenowania)
print("\n🤖 [AI ADVISOR] Diagnoza Stabilności:")

wr_diff = m_train['WinRate %'] - m_val['WinRate %']
gini_diff = m_train['Gini'] - m_val['Gini']
sharpe_val = m_val['Sharpe']

# Ocena Win Rate
if abs(wr_diff) < 5.0:
    print(f"✅ Win Rate Stabilny (Różnica {wr_diff:.1f}%). Model dobrze generalizuje.")
elif wr_diff > 10.0:
    print(f"⚠️ PRZETRENOWANIE (Overfitting). Wynik na Train o {wr_diff:.1f}% lepszy niż na Val.")
    print("   Sugestia: Zwiększ Dropout lub zmniejsz liczbę neuronów.")
else:
    print(f"ℹ️ Lekka rozbieżność Win Rate ({wr_diff:.1f}%). Akceptowalne.")

# Ocena Sharpe (Opłacalność)
if sharpe_val > 1.0:
    print(f"✅ Sharpe na Walidacji ({sharpe_val}) jest BARDZO DOBRY. Strategia zarabia stabilnie.")
elif sharpe_val > 0.5:
    print(f"✅ Sharpe na Walidacji ({sharpe_val}) jest OK. Strategia jest zyskowna.")
else:
    print(f"⚠️ Sharpe na Walidacji niski ({sharpe_val}). Ryzyko jest zbyt duże względem zysku.")

# Ocena Gini (Inteligencja)
if m_val['Gini'] > 0.05:
    print(f"✅ Gini dodatni ({m_val['Gini']}). Model widzi przewagę statystyczną.")
elif m_val['Gini'] < 0:
    print(f"⚠️ Gini ujemny ({m_val['Gini']}). Model myli się częściej niż losowo (odwróć sygnał?).")
else:
    print(f"⚠️ Gini bliski zera. Model zgaduje.")

print("="*80)

# Wykresy Diagnostyczne
plt.figure(figsize=(12, 10))

# 1. Equity Val
plt.subplot(2, 1, 1)
cum_bh_val = np.exp(np.cumsum(ret_val))
cum_strat_val = np.exp(np.cumsum(np.where(model.predict(X_val, verbose=0).flatten() > THRESH_LONG, 1, np.where(model.predict(X_val, verbose=0).flatten() < THRESH_SHORT, -1, 0)) * ret_val))
plt.plot(idx_val, cum_bh_val, label='Buy & Hold (2023)', color='gray', alpha=0.5)
plt.plot(idx_val, cum_strat_val, label='Strategy (2023)', color='blue')
plt.title(f'WALIDACJA: Equity Curve 2023 (Sharpe {m_val["Sharpe"]})')
plt.legend()

# 2. Rozkład Predykcji (Train vs Val)
plt.subplot(2, 1, 2)
sns.kdeplot(model.predict(X_train, verbose=0).flatten(), label='Train Dist', fill=True, alpha=0.3)
sns.kdeplot(model.predict(X_val, verbose=0).flatten(), label='Val Dist', fill=True, alpha=0.3)
plt.axvline(0.5, color='red', linestyle='--')
plt.title('Czy model zachowuje się tak samo na Val jak na Train?')
plt.legend()

plt.tight_layout()
plt.savefig('strict_validation_check.png')
print("Zapisano: strict_validation_check.png")