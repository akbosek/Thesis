import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================================
#      KONFIGURACJA HYBRYDOWA
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' 
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31'

# Ustawienia z wersji Balanced (żeby model widział spadki)
TIMESTEPS = 14
NEURONS   = 64
DROPOUT   = 0.4  # Zwiększony dropout (walka z wynikiem 33mln%)
BATCH_SIZE = 32
EPOCHS    = 40
N_MODELS  = 5

# Progi (Standard)
THRESH_SHORT = 0.49
THRESH_LONG  = 0.51
# ==========================================

print("--- [LSTM HYBRID V2: BALANCED + TREND FILTER] ---")
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

# *** KLUCZOWY FILTR TRENDU ***
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['Trend_Up'] = (data['Close'] > data['SMA_200']).astype(int)

data.dropna(inplace=True)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']

# 2. PODZIAŁ
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

# 3. SKALOWANIE (StandardScaler dla Balansu)
scaler = StandardScaler()
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    # Zwracamy też filtr trendu
    trend = df['Trend_Up'].iloc[TIMESTEPS:].values
    ret = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    return np.array(X), np.array(y), ret, trend, df.index[TIMESTEPS:]

X_train, y_train, ret_train, trend_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, trend_val, idx_val = create_dataset(val_df)

# Wagi klas (To zostawiamy, żeby model widział spadki)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(class_weights))

# 4. TRENING ENSEMBLE
models = []
print(f"\n--- Trening {N_MODELS} modeli (Balanced + Dropout 0.4) ---")

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
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=False, class_weight=cw_dict)
    models.append(model)

print("--- Modele Gotowe ---")

# 5. AGREGACJA I STRATEGIA Z FILTREM
p_train = np.mean([m.predict(X_train, verbose=0).flatten() for m in models], axis=0)
p_val   = np.mean([m.predict(X_val, verbose=0).flatten() for m in models], axis=0)

def analyze_strategy(probs, ret, trend, name):
    # Podstawowa decyzja modelu
    raw_pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    
    # *** FILTRACJA TRENDEM ***
    # Jeśli model chce LONG, a trend jest spadkowy (trend=0) -> ZABROŃ (0)
    # Jeśli model chce SHORT, a trend jest wzrostowy (trend=1) -> ZABROŃ (0)
    
    final_pos = []
    blocked_count = 0
    
    for i in range(len(raw_pos)):
        decision = raw_pos[i]
        market_trend = trend[i]
        
        if decision == 1 and market_trend == 0:
            final_pos.append(0) # Blokada Longa w Bessie
            blocked_count += 1
        elif decision == -1 and market_trend == 1:
            final_pos.append(0) # Blokada Shorta w Hossie
            blocked_count += 1
        else:
            final_pos.append(decision)
            
    final_pos = np.array(final_pos)
    strat_ret = final_pos * ret
    
    # Metryki
    active = final_pos != 0
    if np.sum(active) > 0:
        wr = (np.sign(final_pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    cum = np.exp(np.cumsum(strat_ret))
    total_ret = (cum[-1] - 1) * 100
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    return {
        "Name": name, "WinRate %": wr, "Sharpe": sharpe, 
        "Return %": total_ret, "Trades": np.sum(active),
        "Blocked Trades": blocked_count # Ile głupich decyzji zablokowaliśmy?
    }

m_train = analyze_strategy(p_train, ret_train, trend_train, "TRAIN")
m_val   = analyze_strategy(p_val, ret_val, trend_val, "VAL")

# 6. RAPORT
res_df = pd.DataFrame([m_train, m_val])
print("\n" + "="*80)
print("   RAPORT HYBRYDOWY (BALANCED AI + TREND FILTER)   ")
print("="*80)
print(res_df.round(2).to_string(index=False))
print("-" * 80)

# Porównanie z Benchmarkiem (dla Val)
bh_ret = (np.exp(np.cumsum(ret_val))[-1] - 1) * 100
print(f"Benchmark (Buy & Hold) Val Return: {bh_ret:.2f}%")

if m_val['Return %'] > 0 and m_val['Sharpe'] > 1.0:
    print("\n✅ SUKCES: Strategia zarabia i jest stabilna! Filtr trendu działa.")
elif m_val['Trades'] < 20:
    print("\n⚠️ ZBYT RESTRYKCYJNIE: Filtr wyciął prawie wszystko. Poluzuj progi.")
else:
    print("\nℹ️ WALKA TRWA: Model nadal szuka przewagi.")

# Wykres
plt.figure(figsize=(10, 6))
cum_bh = np.exp(np.cumsum(ret_val))
cum_strat = np.exp(np.cumsum(np.where(p_val > THRESH_LONG, 1, np.where(p_val < THRESH_SHORT, -1, 0)) * ret_val)) # Bez filtra (dla porównania)


# Odtwarzamy logikę filtra do wykresu
final_pos_val = []
for i in range(len(p_val)):
    d = 1 if p_val[i] > THRESH_LONG else (-1 if p_val[i] < THRESH_SHORT else 0)
    t = trend_val[i]
    if (d==1 and t==0) or (d==-1 and t==1): final_pos_val.append(0)
    else: final_pos_val.append(d)
    
cum_strat_filter = np.exp(np.cumsum(np.array(final_pos_val) * ret_val))

plt.plot(idx_val, cum_bh, label='Buy & Hold', color='gray', alpha=0.3)
plt.plot(idx_val, cum_strat, label='AI Only (Balanced)', color='red', linestyle='--')
plt.plot(idx_val, cum_strat_filter, label='AI + Trend Filter', color='green', linewidth=2)
plt.title('Wpływ Filtra Trendu na Wyniki')
plt.legend()
plt.savefig('hybrid_v2_result.png')
print("Zapisano: hybrid_v2_result.png")