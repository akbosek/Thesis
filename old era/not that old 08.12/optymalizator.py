import pandas as pd
import numpy as np
import yfinance as yf
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# ==========================================
# 1. KONFIGURACJA DANYCH (MACRO + BTC)
# ==========================================
TIMESTEPS = 1          # Zgodnie z artykułem dla modelu D1 [cite: 343]
TRAIN_END_DATE = '2022-12-31'

print("--- [SMART OPTIMIZER: PAPER FINE-TUNING] ---")

# A. Dane
tickers = {'BTC': 'BTC-USD', 'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'OIL': 'CL=F'}
dfs = []
for k, v in tickers.items():
    d = yf.download(v, start="2014-01-01", end="2024-12-31", progress=False)
    if isinstance(d.columns, pd.MultiIndex): d = d['Close']
    else: d = d[['Close']]
    d.columns = [k]
    dfs.append(d)

data = pd.concat(dfs, axis=1).fillna(method='ffill').dropna()

# B. Feature Engineering
# Target: Kierunek jutro (1=Up, 0=Down)
data['target_dir'] = np.where(data['BTC'].shift(-1) > data['BTC'], 1, 0)
# Inputy: Zmiana procentowa (dla stacjonarności)
for c in data.columns[:-1]:
    data[f'{c}_ret'] = np.log(data[c] / data[c].shift(1))

data.dropna(inplace=True)
# Używamy zwrotów logarytmicznych jako input
features = [c for c in data.columns if '_ret' in c]
print(f"Cechy: {features}")

# C. Split & Scale
train = data.loc[:TRAIN_END_DATE]
val = data.loc['2023-01-01':]

scaler = MinMaxScaler()
scaler.fit(train[features])

def mk_ds(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:i+TIMESTEPS])
        y.append(df['target_dir'].iloc[i+TIMESTEPS])
    return np.array(X), np.array(y)

X_train, y_train = mk_ds(train)
X_val, y_val = mk_ds(val)

# ==========================================
# 2. OPTUNA: SZUKANIE WOKÓŁ WARTOŚCI Z PAPERU
# ==========================================
def objective(trial):
    # --- HIPERPARAMETRY (Wąskie zakresy wokół artykułu) ---
    
    # Artykuł: 50 neuronów. 
    # My sprawdzamy: 32 do 80.
    neurons = trial.suggest_int('neurons', 32, 80, step=8)
    
    # Artykuł: Dropout 0.01. 
    # To ryzykowne. Sprawdzamy od 0.01 do 0.25.
    dropout = trial.suggest_float('dropout', 0.01, 0.25)
    
    # Artykuł: Learning Rate 0.001.
    # Sprawdzamy okolice.
    lr = trial.suggest_float('lr', 0.0005, 0.005, log=True)
    
    # Artykuł: Adamax[cite: 351].
    # Czy Adam może być lepszy na nowych danych? Niech AI zdecyduje.
    optimizer_name = trial.suggest_categorical('optimizer', ['Adamax', 'Adam'])
    
    # Batch size: Artykuł 72. Sprawdźmy 32, 64, 72.
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 72])

    # --- MODEL (Standard LSTM - jak w artykule) ---
    model = Sequential([
        Input(shape=(TIMESTEPS, len(features))),
        LSTM(neurons, return_sequences=False),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    
    if optimizer_name == 'Adamax':
        opt = Adamax(learning_rate=lr)
    else:
        opt = Adam(learning_rate=lr)
        
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Early Stopping
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    
    # Trening
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=50, # Jak w artykule 
              batch_size=batch_size, 
              callbacks=[es], 
              verbose=0,
              shuffle=False)

    # Ewaluacja
    preds = model.predict(X_val, verbose=0).flatten()
    
    # Win Rate z delikatnym filtrem (0.5 +/- 0.01)
    signals = np.where(preds > 0.51, 1, np.where(preds < 0.49, 0, -1))
    active = (signals != -1)
    
    if np.sum(active) == 0: return 0.0
    
    wr = accuracy_score(y_val[active], signals[active]) * 100
    
    # Constraint: Wymagamy minimum aktywności
    if np.mean(active) < 0.10: return 0.0
    
    return wr

# ==========================================
# 3. START
# ==========================================
if __name__ == "__main__":
    print("Rozpoczynam 'Fine-Tuning' parametrów z artykułu...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    best = study.best_trial
    print("\n" + "="*50)
    print(">>> ZWYCIĘSKIE PARAMETRY (VS PAPER) <<<")
    print(f"NEURONS       = {best.params['neurons']} (Paper: 50)")
    print(f"DROPOUT       = {best.params['dropout']:.4f} (Paper: 0.01)")
    print(f"OPTIMIZER     = {best.params['optimizer']} (Paper: Adamax)")
    print(f"LEARNING_RATE = {best.params['lr']:.5f} (Paper: 0.001)")
    print(f"BATCH_SIZE    = {best.params['batch_size']} (Paper: 72)")
    print("="*50)
    print(f"Wynik Win Rate: {best.value:.2f}%")