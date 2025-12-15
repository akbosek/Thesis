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
from tensorflow.keras.optimizers import Adam

# ==============================================================================
#      KONFIGURACJA V-FINAL (ZWYCIĘSKA V12)
# ==============================================================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# PODZIAŁ OSTATECZNY
# Train: Uczymy się na WSZYSTKIM do końca 2023 (Włącznie z Walidacją!)
TRAIN_START = '2017-09-01'
TRAIN_END   = '2023-12-31' 
# Test: Rok 2024 (ETF-y, Halving - zupełnie nowe wyzwanie)
TEST_START  = '2024-01-01' 

# PARAMETRY (Zatwierdzone)
N_MODELS  = 5        # Ensemble dla stabilności
TIMESTEPS = 14       
NEURONS   = 64       
DROPOUT   = 0.4      
BATCH_SIZE = 32
EPOCHS    = 40       
LEARNING_RATE = 0.001

# STRATEGIA + BIAS
THRESH_SHORT = 0.49
THRESH_LONG  = 0.51
TREND_BIAS   = 0.02  # "Dopalacz" trendowy
# ==============================================================================

print("--- [LSTM GRAND FINALE: 2024 UNLOCKED] ---")
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

# 2. PODZIAŁ (TRAIN obejmuje teraz też 2023!)
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
test_df  = data.loc[TEST_START:].copy()

print(f"Trening (2017-2023): {len(train_df)} dni")
print(f"TEST FINALNY (2024): {len(test_df)} dni")

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
X_test, y_test, ret_test, trend_test, idx_test = create_dataset(test_df)

# 4. TRENING ENSEMBLE
models = []
print(f"\n--- Trenowanie 5 Ekspertów (To potrwa chwilę)... ---")

for i in range(N_MODELS):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(NEURONS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    # Trenujemy na pełnym zbiorze historycznym (bez walidacji, bo walidację już zaliczyliśmy w V12)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=False)
    models.append(model)
    print(f"   -> Model {i+1}/{N_MODELS} gotowy.")

# 5. AGREGACJA
print("\n--- Generowanie Predykcji na 2024... ---")
p_test = np.mean([m.predict(X_test, verbose=0).flatten() for m in models], axis=0)

# 6. ANALIZA Z BIASEM (OSTATECZNA)
def run_final_strategy(probs, ret, trend):
    # Bias Injection
    adj_probs = []
    for i in range(len(probs)):
        p = probs[i]
        t = trend[i]
        if t == 1: new_p = p + TREND_BIAS # Boost w Hossie
        else:      new_p = p - TREND_BIAS # Kara w Bessie
        adj_probs.append(new_p)
    adj_probs = np.array(adj_probs)
    
    # Progi
    pos = np.where(adj_probs > THRESH_LONG, 1, np.where(adj_probs < THRESH_SHORT, -1, 0))
    
    # Filtr ostateczny (Safety Net)
    final_pos = []
    for i in range(len(pos)):
        d = pos[i]
        t = trend[i]
        # Nie shortujemy w hossie, nie longujemy w bessie (chyba że bias był tak silny, że przebił)
        # Ale dla bezpieczeństwa w wersji Final - trzymajmy się trendu mocno.
        if d == -1 and t == 1: final_pos.append(0)
        elif d == 1 and t == 0: final_pos.append(0)
        else: final_pos.append(d)
        
    final_pos = np.array(final_pos)
    strat_ret = final_pos * ret
    
    # Statystyki
    active = final_pos != 0
    if np.sum(active) > 0:
        wr = (np.sign(final_pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    cum_strat = np.exp(np.cumsum(strat_ret))
    cum_bh = np.exp(np.cumsum(ret))
    
    total_ret = (cum_strat[-1] - 1) * 100
    bh_ret = (cum_bh[-1] - 1) * 100
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    max_dd = (cum_strat / np.maximum.accumulate(cum_strat) - 1).min() * 100
    
    return total_ret, bh_ret, wr, sharpe, max_dd, np.sum(active), cum_strat, cum_bh

res_strat, res_bh, res_wr, res_sharpe, res_dd, res_trades, eq_strat, eq_bh = run_final_strategy(p_test, ret_test, trend_test)

# 7. RAPORT KOŃCOWY
print("\n" + "#"*60)
print("   WYNIKI OFICJALNE: ROK 2024 (TEST SET)   ")
print("#"*60)
print(f"Strategia Zysk:      {res_strat:.2f}%")
print(f"Bitcoin (B&H) Zysk:  {res_bh:.2f}%")
print("-" * 40)
print(f"Win Rate:            {res_wr:.2f}%")
print(f"Sharpe Ratio:        {res_sharpe:.2f}")
print(f"Max Drawdown:        {res_dd:.2f}%")
print(f"Liczba Transakcji:   {res_trades}")
print("#"*60)

# Interpretacja
if res_strat > res_bh:
    print("\n🏆 GRATULACJE! Pobiłeś rynek w roku hossy (2024). To wybitne osiągnięcie.")
elif res_strat > 0:
    print("\n✅ SUKCES: Strategia jest zyskowna na nieznanych danych. Cel osiągnięty.")
else:
    print("\n⚠️ WYNIK UJEMNY: Rynek 2024 był zbyt trudny (zmienność ETF).")

# Wykres
plt.figure(figsize=(12, 7))
plt.plot(idx_test, eq_bh, label='Bitcoin (Buy & Hold)', color='grey', alpha=0.5, linewidth=1.5)
plt.plot(idx_test, eq_strat, label=f'LSTM Strategy (+{res_strat:.1f}%)', color='green', linewidth=2.5)
plt.title(f'FINAL PERFORMANCE 2024 (Sharpe: {res_sharpe:.2f})')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylabel('Kapitał (znormalizowany)')
plt.savefig('GRAND_FINALE_2024.png')
print("Wykres zapisany: GRAND_FINALE_2024.png")