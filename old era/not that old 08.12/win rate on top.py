import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam

# ==========================================
# KONFIGURACJA (WYNIKI Z OPTUNY WKLEJ TUTAJ)
# ==========================================
NEURONS       = 32       # <-- Zmień
DROPOUT       = 0.2741033615674956      # <-- Zmień
LEARNING_RATE = 0.00030573272576434903    # <-- Zmień
THRESH_DIST   = 0.03265567383847591     # <-- Zmień (np. jeśli Optuna dała 0.04, to progi są 0.46/0.54)

INPUT_FILE = 'BTC_USD_1d_2014_2024.csv'
TIMESTEPS = 21 
N_MODELS = 5  # Ensemble

# ==========================================
# 1. DANE (Identyczna logika co wyżej)
# ==========================================
print("--- PRZYGOTOWANIE DANYCH ---")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("Brak pliku CSV!")

data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# Wskaźniki
delta = data['Close'].diff()
rs = (delta.where(delta > 0, 0)).rolling(14).mean() / (-delta.where(delta < 0, 0)).rolling(14).mean()
data['rsi_norm'] = (100 - (100 / (1 + rs))) / 100.0
data['sma_50'] = data['Close'].rolling(50).mean()
data['dist_sma50'] = (data['Close'] - data['sma_50']) / data['sma_50']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1
data['volatility_14'] = data['log_return'].rolling(14).std()
data.dropna(inplace=True)

features = ['log_return', 'rsi_norm', 'dist_sma50', 'momentum_3d', 'volatility_14']

# Split Chronologiczny
TRAIN_END = '2022-12-31'
train_df = data.loc[:TRAIN_END].copy()
val_df   = data.loc['2023-01-01':].copy()

# Skalowanie (Fit na Train)
scaler = MinMaxScaler()
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    future_returns = df['log_return'].shift(-1).fillna(0).values
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    return np.array(X), np.array(y), future_returns[TIMESTEPS : len(X_sc)]

X_train, y_train, r_train = create_dataset(train_df)
X_val, y_val, r_val       = create_dataset(val_df)

# ==========================================
# 2. TRENING ENSEMBLE
# ==========================================
models = []
print(f"Trenowanie zespołu {N_MODELS} modeli...")

for i in range(N_MODELS):
    model = Sequential([
        Input(shape=(TIMESTEPS, len(features))),
        Bidirectional(LSTM(NEURONS, return_sequences=False)),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=60, batch_size=32, verbose=0, shuffle=True)
    models.append(model)
    print(f"  -> Model {i+1} gotowy.")

# ==========================================
# 3. RAPORT KOŃCOWY
# ==========================================
def get_ensemble_stats(X, y, rets, name):
    # Średnia predykcja ze wszystkich modeli
    preds = np.mean([m.predict(X, verbose=0).flatten() for m in models], axis=0)
    
    # Progi
    upper = 0.5 + THRESH_DIST
    lower = 0.5 - THRESH_DIST
    
    # Logika
    signals = np.where(preds > upper, 1, np.where(preds < lower, -1, 0))
    active_mask = signals != 0
    n_trades = np.sum(active_mask)
    
    if n_trades == 0:
        win_rate = 0; sharpe = 0
    else:
        # Win Rate
        pred_cls = (preds[active_mask] > 0.5).astype(int)
        true_cls = y[active_mask]
        win_rate = accuracy_score(true_cls, pred_cls) * 100
        
        # Sharpe
        strat_rets = signals * rets
        sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(365) if np.std(strat_rets) != 0 else 0

    try: gini = 2 * roc_auc_score(y, preds) - 1
    except: gini = 0

    return {
        "Period": name,
        "Win Rate %": round(win_rate, 2),
        "Action Rate %": round((n_trades/len(y))*100, 2),
        "Trades": n_trades,
        "Total Ops": len(y),
        "Gini": round(gini, 3),
        "Sharpe": round(sharpe, 2)
    }

stat_train = get_ensemble_stats(X_train, y_train, r_train, "TRAINING (2014-2022)")
stat_val   = get_ensemble_stats(X_val, y_val, r_val, "VALIDATION (2023-2024)")

results = pd.DataFrame([stat_train, stat_val])

print("\n" + "="*80)
print("   FINAL PERFORMANCE REPORT (ENSEMBLE)   ")
print("="*80)
print(results.to_string(index=False))
print("-" * 80)