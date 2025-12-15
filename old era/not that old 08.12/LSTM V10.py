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
from tensorflow.keras.optimizers import Adam

# ==========================================
#      KONFIGURACJA V10 (STABILIZACJA)
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' 
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31'

# ENSEMBLE
N_MODELS = 5

# MODEL (Mniej agresywny)
TIMESTEPS = 23
NEURONS   = 80
DROPOUT   = 0.13578811194918147  # Wysoki dropout, żeby zbić wynik na Train
BATCH_SIZE = 16
EPOCHS    = 50

# STRATEGIA
THRESH_SHORT = 0.5-0.04266374486481164
THRESH_LONG  = 0.5+0.04266374486481164
# ==========================================

print("--- [LSTM V10: PRUDENT HYBRID] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# 1. DANE
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Features
data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

# FILTR TRENDU (SMA 200)
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['Trend_Up'] = (data['Close'] > data['SMA_200']).astype(int)

data.dropna(inplace=True)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']

# 2. PODZIAŁ
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

# 3. SKALOWANIE (Powrót do MinMax - bezpieczniej)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    
    ret = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    trend = df['Trend_Up'].iloc[TIMESTEPS:].values
    
    return np.array(X), np.array(y), ret, trend, df.index[TIMESTEPS:]

X_train, y_train, ret_train, trend_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, trend_val, idx_val = create_dataset(val_df)

# 4. TRENING ENSEMBLE
models = []
print(f"\n--- Trening {N_MODELS} modeli (MinMax + Dropout {DROPOUT}) ---")

for i in range(N_MODELS):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(NEURONS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0008821542812160525), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=False)
    models.append(model)

print("--- Ensemble Gotowy ---")

# 5. AGREGACJA
p_train = np.mean([m.predict(X_train, verbose=0).flatten() for m in models], axis=0)
p_val   = np.mean([m.predict(X_val, verbose=0).flatten() for m in models], axis=0)

# 6. ANALIZA Z FILTREM I PORÓWNANIEM
def analyze_strategy(probs, ret, trend, name):
    # Podstawowa decyzja AI
    raw_pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    
    # *** LOGIKA FILTRA (HYBRYDA) ***
    final_pos = []
    blocked = 0
    
    for i in range(len(raw_pos)):
        decision = raw_pos[i]
        is_bull_market = (trend[i] == 1)
        
        if decision == 1 and not is_bull_market:
            # AI chce Long, ale jest Bessa (Cena < SMA200) -> BLOKUJEMY
            final_pos.append(0)
            blocked += 1
        elif decision == -1 and is_bull_market:
            # AI chce Short, ale jest Hossa (Cena > SMA200) -> BLOKUJEMY
            final_pos.append(0)
            blocked += 1
        else:
            # Zgoda AI i Trendu -> GRAJ
            final_pos.append(decision)
            
    final_pos = np.array(final_pos)
    strat_ret = final_pos * ret
    
    # Metryki
    active = final_pos != 0
    trades = np.sum(active)
    if trades > 0:
        wr = (np.sign(final_pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    cum_strat = np.exp(np.cumsum(strat_ret))
    cum_bh = np.exp(np.cumsum(ret))
    
    total_ret = (cum_strat[-1] - 1) * 100
    bh_ret = (cum_bh[-1] - 1) * 100
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    try: gini = 2 * roc_auc_score(np.where(ret>0, 1, 0), probs) - 1
    except: gini = 0
    
    return {
        "Name": name, "WinRate %": wr, "Gini": gini, "Sharpe": sharpe, 
        "Return %": total_ret, "BH Return %": bh_ret, # <--- JEST!
        "Trades": trades, "Blocked": blocked
    }

m_train = analyze_strategy(p_train, ret_train, trend_train, "TRAIN")
m_val   = analyze_strategy(p_val, ret_val, trend_val, "VAL")

# 7. RAPORT
res_df = pd.DataFrame([m_train, m_val])
print("\n" + "="*90)
print("   RAPORT HYBRYDOWY V10 (Z PORÓWNANIEM RYNKU)   ")
print("="*90)
print(res_df.round(2).to_string(index=False))
print("-" * 90)

# AI Suggestion
val_ret = m_val['Return %']
val_bh = m_val['BH Return %']
if val_ret > val_bh:
    print("✅ SUKCES: Strategia pobiła rynek na Walidacji!")
elif val_ret > 0:
    print("ℹ️ STABILNOŚĆ: Strategia zarabia, choć mniej niż rynek (bezpieczeństwo kosztuje).")
else:
    print("⚠️ PROBLEM: Strategia traci. Sprawdź, czy filtr trendu nie jest zbyt restrykcyjny.")

# Wykres
plt.figure(figsize=(10, 6))
cum_bh = np.exp(np.cumsum(ret_val))
# Odtwarzamy logikę do wykresu
pos_val_plot = []
for i in range(len(p_val)):
    d = 1 if p_val[i] > THRESH_LONG else (-1 if p_val[i] < THRESH_SHORT else 0)
    t = trend_val[i]
    if (d==1 and t==0) or (d==-1 and t==1): pos_val_plot.append(0)
    else: pos_val_plot.append(d)
    
cum_strat = np.exp(np.cumsum(np.array(pos_val_plot) * ret_val))

plt.plot(idx_val, cum_bh, label=f'Buy & Hold ({m_val["BH Return %"]:.1f}%)', color='gray', alpha=0.5)
plt.plot(idx_val, cum_strat, label=f'Hybrid Strategy ({m_val["Return %"]:.1f}%)', color='green', linewidth=2)
plt.title(f'WALIDACJA 2023: Strategy vs Market')
plt.legend()
plt.savefig('v10_hybrid_result.png')
print("Zapisano: v10_hybrid_result.png")