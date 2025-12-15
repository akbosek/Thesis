import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ==========================================
#      TUTAJ ZMIENIASZ PARAMETRY (TUNING)
# ==========================================
# 1. Dane
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'
TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' # Do końca bessy 2022
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31' # Rok 2023 jako walidacja
# Test zostawiamy w spokoju (2024)

# 2. Model (LSTM Architecture)
TIMESTEPS = 30       # Ile dni wstecz widzi model (okno pamięci)
NEURONS   = 8       # Ilość neuronów (mniej = mniejsze ryzyko overfittingu)
DROPOUT   = 0.4      # Jak dużo zapominać (0.2 - 0.5)
LEARNING_RATE = 0.0005 # Szybkość nauki (mniejsza = dokładniejsza)
L2_REG    = 0.001    # Kara za skomplikowanie wag (regularyzacja)
EPOCHS    = 50       # Ilość epok
BATCH_SIZE = 16      # Wielkość paczki danych

# 3. Strategia (Progi)
THRESH_SHORT = 0.48
THRESH_LONG  = 0.52

# ==========================================

print("--- [LSTM RESTART V1] Inicjalizacja ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError("Uruchom najpierw download_daily.py!")

# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)

# Feature Engineering (Minimalizm)
# Target: Czy jutro Close > Dzisiaj Close?
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0) # 1=Wzrost, 0=Spadek

# Input 1: Zmienność z 30 dni (Volatility Deviation)
data['volatility_30'] = data['log_return'].rolling(window=30).std()

# Input 2: Relative High/Low (Gdzie jesteśmy w dziennym zasięgu)
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']

# Czyścimy NaN
data.dropna(inplace=True)

# Wybór cech (BEZ WOLUMENU)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low']
print(f"Użyte zmienne: {features}")

# 2. PODZIAŁ NA ZBIORY
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

print(f"Trening: {len(train_df)} dni | Walidacja: {len(val_df)} dni")

# 3. SKALOWANIE (Tylko na Train!)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    # Zwracamy też oryginalne zwroty do symulacji equity
    return np.array(X), np.array(y), df['log_return'].iloc[TIMESTEPS:].values, df.index[TIMESTEPS:]

X_train, y_train, ret_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, idx_val = create_dataset(val_df)

# 4. BUDOWA MODELU (Prosty i Czysty)
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    # Jedna warstwa LSTM (nie Stacked, żeby nie przekombinować)
    LSTM(NEURONS, return_sequences=False, kernel_regularizer=l2(L2_REG)),
    Dropout(DROPOUT),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacki
estop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n--- Rozpoczynam Trening ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[estop],
    verbose=1
)

# 5. DIAGNOSTYKA I WYNIKI
print("\n--- Generowanie Raportu ---")

# Predykcje
p_train = model.predict(X_train).flatten()
p_val   = model.predict(X_val).flatten()

# Funkcja Strategii (Short / Flat / Long)
def run_strategy(probs, returns, t_short, t_long):
    positions = []
    for p in probs:
        if p > t_long: positions.append(1)   # Long
        elif p < t_short: positions.append(-1) # Short
        else: positions.append(0)            # Flat
    
    positions = np.array(positions)
    # Przesuwamy zwroty: Decyzja dziś (positions) wpływa na wynik jutro (returns przesunięte w targetcie, 
    # ale tutaj mamy 'ret_val' który jest z tego samego dnia co target.
    # W create_dataset: y to target[i+TIMESTEPS]. ret to log_return[i+TIMESTEPS].
    # Target[t] to czy cena wzrośnie t->t+1. Log_return[t] to zmiana t-1->t.
    # Czekaj! Musimy użyć 'actual_return' (future return).
    # Poprawka: W tym skrypcie używamy log_return z *następnego* dnia do obliczenia wyniku strategii.
    # Ponieważ ret_train zawiera historyczne zwroty, musimy wziąć 'next day return'.
    
    # Dla uproszczenia: w create_dataset pobierzmy target (kierunek) i zwrot z TEGO SAMEGO momentu (przyszłość).
    # W pandas: data['log_return'].shift(-1) to jest to co chcemy złapać.
    # Musimy to pobrać z DataFrame'a poprawnie.
    return positions

# Naprawa wektora zwrotów do Equity (musimy wziąć future returns)
# W create_dataset pobraliśmy 'log_return' z bieżącego wiersza. 
# Ale my przewidujemy 'target', który jest shift(-1).
# Więc musimy pobrać shiftnięte zwroty.
future_ret_train = train_df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
future_ret_val   = val_df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values

pos_train = run_strategy(p_train, future_ret_train, THRESH_SHORT, THRESH_LONG)
pos_val   = run_strategy(p_val,   future_ret_val,   THRESH_SHORT, THRESH_LONG)

strat_res_train = pos_train * future_ret_train
strat_res_val   = pos_val * future_ret_val

# Obliczanie Win Rate (Tylko aktywne dni)
def calc_winrate(pos, ret):
    active = pos != 0
    if np.sum(active) == 0: return 0
    # Win: (Pos 1 & Ret > 0) lub (Pos -1 & Ret < 0)
    wins = np.sign(pos[active]) == np.sign(ret[active])
    return np.mean(wins) * 100

wr_train = calc_winrate(pos_train, future_ret_train)
wr_val   = calc_winrate(pos_val,   future_ret_val)

# Gini
try: gini_train = 2 * roc_auc_score(y_train, p_train) - 1
except: gini_train = 0
try: gini_val = 2 * roc_auc_score(y_val, p_val) - 1
except: gini_val = 0

# Wyniki zbiorcze
print("\n" + "="*40)
print(f"RAPORT KOŃCOWY (Train: {TRAIN_START}-{TRAIN_END} | Val: {VAL_START}-{VAL_END})")
print("="*40)
print(f"{'Metryka':<20} | {'TRENING':<10} | {'WALIDACJA':<10}")
print("-" * 45)
print(f"{'Win Rate':<20} | {wr_train:.2f}%     | {wr_val:.2f}%")
print(f"{'Gini Ratio':<20} | {gini_train:.3f}      | {gini_val:.3f}")
print("-" * 45)

# DETEKTOR OVERFITTINGU
diff_wr = wr_train - wr_val
print("\n--- DIAGNOZA MODELU ---")
if diff_wr > 10:
    print(f"⚠️  RYZYKO PRZETRENOWANIA! (WinRate spada o {diff_wr:.1f}% na walidacji)")
    print("Sugestia: Zwiększ DROPOUT, Zwiększ L2_REG lub Zmniejsz NEURONS.")
elif wr_val < 50:
    print("⚠️  MODEL NIE DZIAŁA (WinRate < 50%).")
    print("Sugestia: Zmień INPUT_FEATURES lub Zwiększ TIMESTEPS.")
else:
    print("✅  MODEL STABILNY. (Różnica w normie, wynik dodatni).")

# Exporty
df_val_res = pd.DataFrame(index=idx_val)
df_val_res['Actual_Ret'] = future_ret_val
df_val_res['Prob'] = p_val
df_val_res['Position'] = pos_val
df_val_res['Strat_Ret'] = strat_res_val
df_val_res.to_csv('restart_val_results.csv')

# Wykresy
plt.figure(figsize=(12, 8))

# 1. Krzywa Uczenia (Loss)
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Krzywa Uczenia (Loss)')
plt.legend()

# 2. Equity Curve (Walidacja)
plt.subplot(2, 1, 2)
# Konwersja log returns na equity
equity_bh = np.exp(np.cumsum(future_ret_val)) 
equity_strat = np.exp(np.cumsum(strat_res_val))
plt.plot(idx_val, equity_bh, label='Buy & Hold (Val)', color='grey', alpha=0.5)
plt.plot(idx_val, equity_strat, label='LSTM Strategy (Val)', color='green')
plt.title(f'Equity Curve Walidacja ({VAL_START} - {VAL_END})')
plt.legend()

plt.tight_layout()
plt.savefig('restart_report.png')
print("Wygenerowano: restart_report.png, restart_val_results.csv")