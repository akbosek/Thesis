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
#      KONFIGURACJA V12 (TREND BIAS)
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' 
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31'

# ENSEMBLE
N_MODELS = 5

# MODEL
TIMESTEPS = 14
NEURONS   = 64
DROPOUT   = 0.4
BATCH_SIZE = 32
EPOCHS    = 40

# STRATEGIA
THRESH_SHORT = 0.49
THRESH_LONG  = 0.51

# NOWOŚĆ: TREND BIAS
# Ile % pewności dodajemy, gdy trend jest zgodny?
TREND_BIAS = 0.02 # Dodajemy 2% do pewności Longa w hossie
# ==========================================

print("--- [LSTM V12: TREND ALIGNMENT] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# 1. DANE
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['Trend_Up'] = (data['Close'] > data['SMA_200']).astype(int)

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
    
    ret = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    trend = df['Trend_Up'].iloc[TIMESTEPS:].values
    
    return np.array(X), np.array(y), ret, trend, df.index[TIMESTEPS:]

X_train, y_train, ret_train, trend_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, trend_val, idx_val = create_dataset(val_df)

# 4. TRENING ENSEMBLE
models = []
print(f"\n--- Trening {N_MODELS} modeli... ---")

for i in range(N_MODELS):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(NEURONS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=False)
    models.append(model)

print("--- Modele Gotowe ---")

# 5. AGREGACJA
p_train = np.mean([m.predict(X_train, verbose=0).flatten() for m in models], axis=0)
p_val   = np.mean([m.predict(X_val, verbose=0).flatten() for m in models], axis=0)

# 6. ANALIZA Z BIAS'EM
def analyze_with_bias(probs, ret, trend, name):
    # KROK 1: Modyfikacja prawdopodobieństwa (Bias)
    # Jeśli Trend UP -> Dodaj bias do prob (zachęta do Longa)
    # Jeśli Trend DOWN -> Odejmij bias od prob (zachęta do Shorta)
    
    adjusted_probs = []
    for i in range(len(probs)):
        p = probs[i]
        t = trend[i]
        
        if t == 1: # Hossa
            new_p = p + TREND_BIAS
        else:      # Bessa
            new_p = p - TREND_BIAS
            
        # Clip żeby nie wyjść poza 0-1
        adjusted_probs.append(max(0.0, min(1.0, new_p)))
        
    adjusted_probs = np.array(adjusted_probs)
    
    # KROK 2: Standardowe Progi na ZMODYFIKOWANYCH danych
    pos = np.where(adjusted_probs > THRESH_LONG, 1, np.where(adjusted_probs < THRESH_SHORT, -1, 0))
    
    # KROK 3: Hard Filter (Bezpiecznik) - nadal blokujemy głupie ruchy
    final_pos = []
    for i in range(len(pos)):
        d = pos[i]
        t = trend[i]
        
        # Nadal nie pozwalamy shortować w hossie, nawet z biasem
        if d == -1 and t == 1: final_pos.append(0)
        elif d == 1 and t == 0: final_pos.append(0)
        else: final_pos.append(d)
        
    final_pos = np.array(final_pos)
    strat_ret = final_pos * ret
    
    # Metryki
    active = final_pos != 0
    trades = np.sum(active)
    if trades > 0:
        wr = (np.sign(final_pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    cum = np.exp(np.cumsum(strat_ret))
    total_ret = (cum[-1] - 1) * 100
    bh_ret = (np.exp(np.cumsum(ret))[-1] - 1) * 100
    
    return {
        "Name": name, "WinRate %": wr, "Sharpe": sharpe, 
        "Return %": total_ret, "BH Return %": bh_ret, "Trades": trades
    }

m_train = analyze_with_bias(p_train, ret_train, trend_train, "TRAIN")
m_val   = analyze_with_bias(p_val, ret_val, trend_val, "VAL")

# 7. RAPORT
res_df = pd.DataFrame([m_train, m_val])
print("\n" + "="*80)
print(f"   RAPORT V12 (TREND BIAS = {TREND_BIAS})   ")
print("="*80)
print(res_df.round(2).to_string(index=False))
print("-" * 80)

# Impact Check
print(f"\nEfekt Biasu na Walidacji:")
print(f"Zysk: {m_val['Return %']:.2f}% (vs Benchmark {m_val['BH Return %']:.2f}%)")
if m_val['Return %'] > 20:
    print("✅ SUKCES: Bias 'obudził' model w trendzie!")
elif m_val['Trades'] > 150:
    print("⚠️ UWAŻAJ: Bias wymusza zbyt wiele transakcji (Overtrading).")
else:
    print("ℹ️ BEZ ZMIAN: Model jest bardzo oporny.")

# Wykres
plt.figure(figsize=(10, 6))
cum_bh = np.exp(np.cumsum(ret_val))
# Odtwarzamy logikę dla wykresu
adj_probs_val = []
for i in range(len(p_val)):
    if trend_val[i] == 1: adj_probs_val.append(p_val[i] + TREND_BIAS)
    else: adj_probs_val.append(p_val[i] - TREND_BIAS)
adj_probs_val = np.array(adj_probs_val)

pos_val = np.where(adj_probs_val > THRESH_LONG, 1, np.where(adj_probs_val < THRESH_SHORT, -1, 0))
final_pos_val = []
for i in range(len(pos_val)):
    if pos_val[i] == -1 and trend_val[i] == 1: final_pos_val.append(0)
    elif pos_val[i] == 1 and trend_val[i] == 0: final_pos_val.append(0)
    else: final_pos_val.append(pos_val[i])

cum_strat = np.exp(np.cumsum(np.array(final_pos_val) * ret_val))

plt.plot(idx_val, cum_bh, label='Buy & Hold', color='gray', alpha=0.5)
plt.plot(idx_val, cum_strat, label=f'Trend Aligned (+{m_val["Return %"]:.1f}%)', color='orange', linewidth=2)
plt.title(f'V12: Strategy with Trend Bias (+{TREND_BIAS})')
plt.legend()
plt.savefig('v12_bias_result.png')
print("Zapisano: v12_bias_result.png")