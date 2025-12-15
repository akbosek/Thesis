import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==============================================================================
#      KONFIGURACJA "TRIAL 394" (ZWYCIĘSKA KOMBINACJA)
# ==============================================================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# OKRESY (Diagnostyka: Train vs Val)
TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' 
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31'

# --- PARAMETRY Z OPTYMALIZACJI ---
TIMESTEPS     = 23        # 'lookback': 23
NEURONS       = 80        # 'neurons': 80
DROPOUT       = 0.1358    # 'dropout': 0.1357...
LEARNING_RATE = 0.00088   # 'lr': 0.00088...
BATCH_SIZE    = 16        # 'batch_size': 16 (Precyzyjne uczenie)
EPOCHS        = 50        # 'epochs': 50

# --- PROGI (Obliczone z 'threshold_dist') ---
# Center = 0.5. Dist = 0.0426
DIST = 0.0427
THRESH_SHORT = 0.5 - DIST  # ok. 0.457
THRESH_LONG  = 0.5 + DIST  # ok. 0.543
# ==============================================================================

print(f"--- [LSTM TRIAL 394 CONFIG] ---")
print(f"Params: Lookback={TIMESTEPS}, Neurons={NEURONS}, Drop={DROPOUT}, LR={LEARNING_RATE}")
print(f"Thresholds: < {THRESH_SHORT:.4f} | > {THRESH_LONG:.4f}")

if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# 1. DANE
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Features (Standardowy zestaw, który działał)
data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

data.dropna(inplace=True)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']

# 2. PODZIAŁ
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

# 3. SKALOWANIE
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    future_returns = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    return np.array(X), np.array(y), future_returns, df.index[TIMESTEPS:]

X_train, y_train, ret_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, idx_val = create_dataset(val_df)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# 4. TRENING (Czysty model, bez udziwnień)
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(NEURONS, return_sequences=False),
    BatchNormalization(), # Zostawiamy dla stabilności przy małym batchu
    Dropout(DROPOUT),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
estop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("\n--- Rozpoczynam Trening (Trial 394) ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[estop],
    verbose=0
)

# 5. DIAGNOSTYKA
def get_metrics(X, y_true, ret, name):
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
    
    # Gini
    try: gini = 2 * roc_auc_score(y_true, probs) - 1
    except: gini = 0
    
    # Sharpe & Profit
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    cum_strat = np.exp(np.cumsum(strat_ret))
    cum_bh = np.exp(np.cumsum(ret))
    
    total_ret = (cum_strat[-1] - 1) * 100
    bh_ret = (cum_bh[-1] - 1) * 100
    dd = (cum_strat / np.maximum.accumulate(cum_strat) - 1).min() * 100
    
    return {
        "Dataset": name,
        "WinRate %": round(wr, 2),
        "Gini": round(gini, 3),
        "Sharpe": round(sharpe, 2),
        "Return %": round(total_ret, 2),
        "BH Return %": round(bh_ret, 2),
        "Max DD %": round(dd, 2),
        "Trades": trades,
        "Probs Std": round(np.std(probs), 4)
    }, probs

print("\n--- Obliczanie Wyników ---")
m_train, p_train = get_metrics(X_train, y_train, ret_train, "TRAIN (17-22)")
m_val, p_val = get_metrics(X_val, y_val, ret_val, "VAL (2023)")

# 6. RAPORT
results = pd.DataFrame([m_train, m_val])
print("\n" + "="*95)
print(f"   RAPORT OPTYMALIZACJI (TRIAL 394)   ")
print("="*95)
print(results.to_string(index=False))
print("-" * 95)

# Interpretacja
val_wr = m_val['WinRate %']
val_sharpe = m_val['Sharpe']
val_trades = m_val['Trades']

print("\n🤖 [AI ADVISOR - OCENA KONFIGURACJI 394]:")
if val_trades == 0:
    print("❌ SZEROKIE PROGI: Przy Threshold Dist 0.043 model nie gra. Model jest zbyt 'płaski'.")
    print("   -> Sugestia: Jeśli to nie zadziała, wróć do węższych progów (np. 0.01).")
elif val_wr > 52:
    print("✅ POTWIERDZONE: Parametry z Triala 394 działają! Win Rate jest solidny.")
    if val_sharpe > 1.0:
        print("🚀 I mamy wysoki Sharpe! To może być wersja ostateczna.")
else:
    print(f"ℹ️ WYNIK PRZECIĘTNY: Win Rate {val_wr}%. Parametry optymalne 'teoretycznie' mogą potrzebować tuningu progów.")

print("="*95)

# Wykres
plt.figure(figsize=(10, 6))
cum_bh = np.exp(np.cumsum(ret_val))
cum_strat = np.exp(np.cumsum(np.where(p_val > THRESH_LONG, 1, np.where(p_val < THRESH_SHORT, -1, 0)) * ret_val))

plt.plot(idx_val, cum_bh, label='Buy & Hold', color='gray', alpha=0.5)
plt.plot(idx_val, cum_strat, label='Optimized Strategy (Trial 394)', color='blue', linewidth=2)
plt.title(f'WALIDACJA 2023 (Parametry Trial 394)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('trial394_result.png')
print("Zapisano: trial394_result.png")