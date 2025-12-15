import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- 1. KONFIGURACJA I PARAMETRY ---
# Dokładna nazwa Twojego pliku
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 

TIMESTEPS = 12  # 12 świeczek 4h = 48h historii (okno patrzenia wstecz)

# Progi decyzyjne (Twoja strategia)
THRESH_SHORT = 0.48
THRESH_LONG = 0.52

# --- 2. DEFINICJA OKRESÓW CZASOWYCH ---
# 4 lata treningu (2018-2021)
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE   = '2021-12-31'

# 1 rok walidacji (2022)
VAL_START_DATE   = '2022-01-01'
VAL_END_DATE     = '2022-12-31'

# Test: Od 2023 do końca danych (czyli do 31.12.2024)
TEST_START_DATE  = '2023-01-01'

# --- 3. WCZYTANIE I PRZYGOTOWANIE DANYCH ---
print(f"--- Wczytywanie danych z: {INPUT_FILE} ---")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Nie znaleziono pliku: {INPUT_FILE}. Upewnij się, że nazwa jest poprawna.")

# Wczytujemy z parsowaniem dat
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)

# Sortujemy chronologicznie (na wszelki wypadek)
data = data.sort_index()

# Upewniamy się, że kolumny są z dużej litery (Open, Close, Volume)
data.columns = [c.capitalize() for c in data.columns]

print(f"Zakres danych w pliku: {data.index.min()} do {data.index.max()}")

# --- 4. FEATURE ENGINEERING (Cechy) ---

# Log Returns (Logarytmiczna stopa zwrotu)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# Relative Volume (Wolumen względem średniej z 30 dni)
# 30 dni * 6 świeczek (4h) dziennie = 180 okresów
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=180).mean()

# Wskaźniki techniczne
data['RSI'] = ta.rsi(data['Close'], length=14)
macd = ta.macd(data['Close'])
data['MACD'] = macd.iloc[:, 0] # Bierzemy główną linię MACD

# Kodowanie Czasu (Sinus/Cosinus godziny) - ważne dla danych 4h
# Pozwala modelowi zrozumieć cykliczność dnia
data['Hour_Sin'] = np.sin(2 * np.pi * data.index.hour / 24)
data['Hour_Cos'] = np.cos(2 * np.pi * data.index.hour / 24)

# Lags (Opóźnione zwroty - pamięć dla drzew)
lags = 5
for lag in range(1, lags + 1):
    data[f'log_return_lag_{lag}'] = data['log_return'].shift(lag)

# Lista cech wejściowych
features = ['log_return', 'Rel_Vol', 'RSI', 'MACD', 'Hour_Sin', 'Hour_Cos'] + [f'log_return_lag_{i}' for i in range(1, lags + 1)]

# --- 5. TARGET I CZYSZCZENIE ---

# Target: Czy następna świeca 4h zamknie się wyżej? (1 = Tak, 0 = Nie)
data['target'] = np.where(data['log_return'].shift(-1) > 0, 1, 0)

# Rzeczywisty zwrot z następnej świecy (do backtestu)
data['actual_return'] = data['log_return'].shift(-1)

# Usuwamy NaN powstałe przy wskaźnikach i lagach
data = data.dropna()
print(f"Liczba danych 4h po czyszczeniu: {len(data)}")

# --- 6. PODZIAŁ NA ZBIORY (TRAIN / VAL / TEST) ---
# Używamy .loc z datami, co jest najbardziej precyzyjne
train_df = data.loc[TRAIN_START_DATE:TRAIN_END_DATE].copy()
val_df   = data.loc[VAL_START_DATE:VAL_END_DATE].copy()
test_df  = data.loc[TEST_START_DATE:].copy()

print(f"\nPodział danych:")
print(f"Train: {len(train_df)} świec ({TRAIN_START_DATE} - {TRAIN_END_DATE})")
print(f"Val:   {len(val_df)} świec ({VAL_START_DATE} - {VAL_END_DATE})")
print(f"Test:  {len(test_df)} świec ({TEST_START_DATE} - koniec)")

# Ważenie klas (dla zbalansowania modelu)
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['target']), y=train_df['target'])
class_weights_dict = dict(enumerate(class_weights))

# --- 7. SKALOWANIE DANYCH ---
scaler = StandardScaler()
# Trenujemy skaler TYLKO na zbiorze treningowym, żeby nie "podglądać" przyszłości
scaler.fit(train_df[features])

# Transformujemy wszystkie zbiory tym samym skalerem
train_df[features] = scaler.transform(train_df[features])
val_df[features]   = scaler.transform(val_df[features])
test_df[features]  = scaler.transform(test_df[features])

# --- 8. TWORZENIE SEKWENCJI (Dla LSTM) ---
def create_sequences(data_df, feature_cols, target_col, time_steps=TIMESTEPS):
    X, y = [], []
    data_features = data_df[feature_cols].values
    data_target = data_df[target_col].values
    
    for i in range(len(data_features) - time_steps):
        X.append(data_features[i:(i + time_steps)])
        y.append(data_target[i + time_steps])
        
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_df, features, 'target')
X_val, y_val     = create_sequences(val_df, features, 'target')
X_test, y_test   = create_sequences(test_df, features, 'target')

# Spłaszczanie danych dla Random Forest i XGBoost (one nie obsługują sekwencji 3D)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

# --- 9. MODELOWANIE ---

# A. LSTM (Bidirectional)
print("\n--- Trenowanie LSTM ---")
lstm_model = Sequential()
lstm_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))))
lstm_model.add(Dropout(0.4))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001)
]

history = lstm_model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=64, # Większy batch dla danych 4h
    validation_data=(X_val, y_val), 
    class_weight=class_weights_dict,
    shuffle=False, 
    callbacks=callbacks, 
    verbose=1
)

# B. Random Forest
print("\n--- Trenowanie Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
rf_model.fit(X_train_flat, y_train)

# C. XGBoost
print("\n--- Trenowanie XGBoost ---")
xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=1.5, n_jobs=-1, random_state=42)
xgb_model.fit(X_train_flat, y_train)

# --- 10. GENEROWANIE WYNIKÓW I STRATEGIA ---
print("\n--- Generowanie wyników strategii ---")

# Pobieramy prawdopodobieństwa (pewność modelu)
probs_lstm = lstm_model.predict(X_test).flatten()
probs_rf   = rf_model.predict_proba(X_test_flat)[:, 1]
probs_xgb  = xgb_model.predict_proba(X_test_flat)[:, 1]

# Rzeczywiste zwroty w okresie testowym (musimy uciąć pierwsze TIMESTEPS wierszy, bo one poszły na sekwencje)
returns_test = test_df['actual_return'].values[TIMESTEPS:]
dates_test   = test_df.index[TIMESTEPS:]

# --- KLUCZOWA FUNKCJA STRATEGII (Z Twoimi progami) ---
def run_strategy_with_neutral_zone(probs, returns, thresh_short=0.48, thresh_long=0.52):
    positions = []
    for p in probs:
        if p < thresh_short:
            positions.append(-1) # Short (Pewność spadku)
        elif p > thresh_long:
            positions.append(1)  # Long (Pewność wzrostu)
        else:
            positions.append(0)  # Neutral/Flat (Niepewność 48-52%)
            
    positions = np.array(positions)
    
    # Wynik to pozycja * zwrot (dla Shorta: -1 * ujemny zwrot = zysk)
    strategy_returns = positions * returns
    return strategy_returns

# Obliczanie krzywych kapitału (Equity Curves)
equity_bh   = (1 + returns_test).cumprod() # Buy & Hold
equity_lstm = (1 + run_strategy_with_neutral_zone(probs_lstm, returns_test, THRESH_SHORT, THRESH_LONG)).cumprod()
equity_rf   = (1 + run_strategy_with_neutral_zone(probs_rf, returns_test, THRESH_SHORT, THRESH_LONG)).cumprod()
equity_xgb  = (1 + run_strategy_with_neutral_zone(probs_xgb, returns_test, THRESH_SHORT, THRESH_LONG)).cumprod()

# --- 11. OBLICZANIE METRYK (Sharpe, Drawdown) ---
def calculate_metrics(equity_curve, strategy_name):
    # Całkowity zwrot
    total_return = (equity_curve[-1] - 1) * 100
    
    # Dzienne zwroty (zmieniamy equity na pct_change)
    daily_rets = pd.Series(equity_curve).pct_change().dropna()
    
    # Sharpe Ratio (zakładamy 6 okresów dziennie * 365 dni = 2190 okresów w roku dla danych 4h)
    # Annualized Sharpe
    if daily_rets.std() == 0: sharpe = 0
    else: sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(365 * 6)
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min() * 100
    
    return {
        "Strategy": strategy_name,
        "Return %": round(total_return, 2),
        "Sharpe": round(sharpe, 2),
        "Max DD %": round(max_dd, 2)
    }

metrics = []
metrics.append(calculate_metrics(equity_bh, "Buy & Hold"))
metrics.append(calculate_metrics(equity_lstm, "LSTM 4h"))
metrics.append(calculate_metrics(equity_rf, "Random Forest"))
metrics.append(calculate_metrics(equity_xgb, "XGBoost"))

metrics_df = pd.DataFrame(metrics)
print("\n=== WYNIKI KOŃCOWE (2023-2024) ===")
print(metrics_df.to_string(index=False))

# --- 12. WYKRESY ---
plt.figure(figsize=(14, 8))
plt.plot(dates_test, equity_bh, label='Buy & Hold', color='gray', alpha=0.5)
plt.plot(dates_test, equity_lstm, label='LSTM', linewidth=1.5)
plt.plot(dates_test, equity_rf, label='Random Forest', linewidth=1.5)
plt.plot(dates_test, equity_xgb, label='XGBoost', linewidth=1.5)

plt.title(f'Backtest 4h (Strategia: Short < {THRESH_SHORT*100:.0f}% | Long > {THRESH_LONG*100:.0f}%)')
plt.ylabel('Equity (Kapitał)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('strategy_4h_results.png')
print("\nWykres zapisano jako: strategy_4h_results.png")