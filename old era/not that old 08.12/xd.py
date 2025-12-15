import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ==============================================================================
# A. KONFIGURACJA ZWYCIĘZCY (TUTAJ WKLEJ WYNIKI Z OPTUNY)
# ==============================================================================
# Przykładowe wartości (Zastąp je tymi z konsoli!):
FILTERS       = 32       # Ilość filtrów CNN
KERNEL_SIZE   = 2        # Wielkość okna splotu
LSTM_UNITS    = 50       # Ilość jednostek pamięci [cite: 66, 126]
DROPOUT       = 0.3      # Poziom zapominania
LEARNING_RATE = 0.001    # Szybkość uczenia
REG_STRENGTH  = 0.0003   # Siła regularyzacji L2
THRESH_DIST   = 0.02     # Margines pewności (Snajper)

# Ustawienia stałe (Zgodne z nowym artykułem)
TIMESTEPS     = 10       # Krótkie okno 10-dniowe [cite: 57, 76]
N_MODELS      = 5        # Ilość modeli w zespole (Ensemble)
EPOCHS        = 50       # Epoki treningu
BATCH_SIZE    = 64       # [cite: 80, 92]

# ==============================================================================
# B. PRZYGOTOWANIE DANYCH (FUSION DATASET)
# ==============================================================================
print("--- [FUSION EXECUTOR: CNN-LSTM ENSEMBLE] ---")

# 1. Pobieranie danych (BTC + Macro)
tickers = {'BTC': 'BTC-USD', 'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'OIL': 'CL=F'}
dfs = []
print("Pobieranie danych rynkowych...")

for name, ticker in tickers.items():
    try:
        d = yf.download(ticker, start="2014-01-01", end="2024-12-31", progress=False)
        if isinstance(d.columns, pd.MultiIndex): d = d['Close']
        else: d = d[['Close']]
        d.columns = [name]
        dfs.append(d)
    except Exception as e:
        print(f"Błąd: {e}")

if not dfs: raise ValueError("Brak danych!")

# Łączenie i czyszczenie
raw_data = pd.concat(dfs, axis=1).fillna(method='ffill').dropna()

# 2. Feature Engineering
data = pd.DataFrame(index=raw_data.index)

# Zwroty logarytmiczne (Log Returns) - lepsze dla sieci neuronowych
for col in raw_data.columns:
    data[f'{col}_ret'] = np.log(raw_data[col] / raw_data[col].shift(1))

# RSI (Wskaźnik techniczny dla BTC)
delta = raw_data['BTC'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['BTC_rsi'] = (100 - (100 / (1 + rs))) / 100.0

# Target: 1 jeśli cena jutro wzrośnie, 0 jeśli spadnie
data['target'] = np.where(raw_data['BTC'].shift(-1) > raw_data['BTC'], 1, 0)
data.dropna(inplace=True)

# Wybór cech (Input Features)
features = [c for c in data.columns if c != 'target']
print(f"Użyte wskaźniki: {features}")

# 3. Podział Chronologiczny
TRAIN_END = '2022-12-31'
train_df = data.loc[:TRAIN_END].copy()
val_df   = data.loc['2023-01-01':].copy()

# 4. Skalowanie (Fitujemy tylko na treningu!)
scaler = MinMaxScaler()
scaler.fit(train_df[features])

def create_sequences(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    # Returny do obliczania Sharpe Ratio (przesunięte o 1 dzień)
    rets = df['BTC_ret'].shift(-1).fillna(0).values
    
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
        
    return np.array(X), np.array(y), rets[TIMESTEPS : len(X_sc)]

X_train, y_train, r_train = create_sequences(train_df)
X_val, y_val, r_val       = create_sequences(val_df)

print(f"Dane Treningowe: {X_train.shape}")
print(f"Dane Walidacyjne: {X_val.shape}")

# ==============================================================================
# C. TRENING ENSEMBLE (5x CNN-LSTM)
# ==============================================================================
models = []
print(f"\nRozpoczynam trening zespołu {N_MODELS} modeli...")

for i in range(N_MODELS):
    # Architektura Hybrydowa z Artykułu [cite: 67, 79-88]
    model = Sequential([
        Input(shape=(TIMESTEPS, len(features))),
        
        # 1. CNN: Widzi kształty (Pattern Recognition)
        Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu', padding='same', 
               kernel_regularizer=l2(REG_STRENGTH)),
        MaxPooling1D(pool_size=2),
        
        # 2. LSTM: Widzi czas (Sequence Learning)
        LSTM(LSTM_UNITS, return_sequences=False, kernel_regularizer=l2(REG_STRENGTH)),
        
        BatchNormalization(),
        Dropout(DROPOUT),
        
        # 3. Decyzja
        Dense(16, activation='relu', kernel_regularizer=l2(REG_STRENGTH)),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # Trening z lekkim przetasowaniem (shuffle=True) dla różnorodności modeli
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=True)
    models.append(model)
    print(f" -> Model {i+1} gotowy.")

# ==============================================================================
# D. RAPORT KOŃCOWY
# ==============================================================================
def evaluate_ensemble(X, y_true, returns, name):
    # Uśrednianie predykcji wszystkich modeli
    preds = np.mean([m.predict(X, verbose=0).flatten() for m in models], axis=0)
    
    # Logika Snajpera
    upper, lower = 0.5 + THRESH_DIST, 0.5 - THRESH_DIST
    signals = np.where(preds > upper, 1, np.where(preds < lower, -1, 0))
    active_mask = (signals != 0)
    
    n_trades = np.sum(active_mask)
    action_rate = (n_trades / len(y_true)) * 100
    
    if n_trades == 0:
        return {"Period": name, "Win Rate": "0.0%", "Trades": 0, "Sharpe": 0.0}
    
    # Win Rate
    # (Predykcja > 0.5 to '1', <= 0.5 to '0')
    active_preds_bin = (preds[active_mask] > 0.5).astype(int)
    active_actuals = y_true[active_mask]
    win_rate = accuracy_score(active_actuals, active_preds_bin) * 100
    
    # Sharpe Ratio (Szacunkowy)
    strat_rets = signals * returns
    sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(365) if np.std(strat_rets) != 0 else 0
    
    return {
        "Period": name,
        "Win Rate": f"{win_rate:.2f}%",
        "Trades": n_trades,
        "Action Rate": f"{action_rate:.1f}%",
        "Sharpe": f"{sharpe:.2f}"
    }

print("\n" + "="*65)
print("   RAPORT KOŃCOWY (FUSION BOT: CNN-LSTM)   ")
print("="*65)

stats_train = evaluate_ensemble(X_train, y_train, r_train, "TRAINING (2014-2022)")
stats_val   = evaluate_ensemble(X_val, y_val, r_val, "VALIDATION (2023-2024)")

results = pd.DataFrame([stats_train, stats_val])
print(results.to_string(index=False))
print("-" * 65)

# Interpretacja wyniku
wr_val = float(stats_val['Win Rate'].strip('%'))
print("\n[WERDYKT]:")
if wr_val > 54.0:
    print("✅ Model zyskowny i stabilny. Można testować na koncie demo.")
elif wr_val > 51.0:
    print("⚠️ Model zarabia, ale marża błędu jest mała. Wymaga dobrego zarządzania ryzykiem.")
else:
    print("❌ Model nadal rzuca monetą. Zwiększ REG_STRENGTH lub zmień dane wejściowe.")