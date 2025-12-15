import pandas as pd
import numpy as np
import os
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ==========================================
# KONFIGURACJA HYBRYDY
# ==========================================
INPUT_FILE = 'BTC_USD_1d_2014_2024.csv' 
TIMESTEPS = 21 
PREDICTION_HORIZON = 1 # 1 = Jutro (zostawiamy 1, ale model będzie miał CNN do pomocy)
N_INTERNAL_MODELS = 3  # Zmniejszyłem do 3 dla szybkości, bo CNN jest nieco wolniejsze

TRAIN_START = '2014-09-17'
TRAIN_END   = '2022-12-31'
VAL_START   = '2023-01-01'
VAL_END     = '2024-12-31'

# ==========================================
# 1. PRZYGOTOWANIE DANYCH
# ==========================================
def prepare_data():
    if not os.path.exists(INPUT_FILE):
        return None, None, None, None, None, None
        
    data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
    
    # TARGET: Czy cena za X dni będzie wyższa niż dziś?
    data['target'] = np.where(data['Close'].shift(-PREDICTION_HORIZON) > data['Close'], 1, 0)
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Feature Engineering
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['rsi_norm'] = (100 - (100 / (1 + rs))) / 100.0
    
    data['sma_50'] = data['Close'].rolling(50).mean()
    data['dist_sma50'] = (data['Close'] - data['sma_50']) / data['sma_50']
    
    # Volatility
    data['volatility_14'] = data['log_return'].rolling(14).std()
    
    data.dropna(inplace=True)
    features = ['log_return', 'rsi_norm', 'dist_sma50', 'volatility_14']
    
    train_df = data.loc[TRAIN_START:TRAIN_END].copy()
    val_df   = data.loc[VAL_START:VAL_END].copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train_df[features])
    
    def create_sequences(df):
        X_sc = scaler.transform(df[features])
        X, y = [], []
        # Returny przesunięte o horyzont predykcji
        rets = df['log_return'].shift(-PREDICTION_HORIZON).fillna(0).values 
        
        for i in range(len(X_sc) - TIMESTEPS):
            X.append(X_sc[i:(i + TIMESTEPS)])
            y.append(df['target'].iloc[i + TIMESTEPS])
        
        return np.array(X), np.array(y), rets[TIMESTEPS : len(X_sc)]

    return (*create_sequences(train_df), *create_sequences(val_df))

X_train, y_train, r_train, X_val, y_val, r_val = prepare_data()

# Dummy check
if X_train is None:
    X_train = np.random.rand(1000, TIMESTEPS, 4)
    y_train = np.random.randint(0, 2, 1000)
    r_train = np.random.randn(1000)
    X_val   = np.random.rand(200, TIMESTEPS, 4)
    y_val   = np.random.randint(0, 2, 200)
    r_val   = np.random.randn(200)

# ==========================================
# 2. METRYKI
# ==========================================
def calculate_metrics(y_true, raw_probs, returns, threshold_dist):
    upper = 0.5 + threshold_dist
    lower = 0.5 - threshold_dist
    signals = np.where(raw_probs > upper, 1, np.where(raw_probs < lower, -1, 0))
    
    active = (signals != 0)
    if np.sum(active) == 0: return {"WinRate": 0.0, "ActionRate": 0.0, "Sharpe": 0.0}
    
    win_rate = accuracy_score(y_true[active], (raw_probs[active] > 0.5).astype(int)) * 100
    strat_rets = signals * returns
    sharpe = (np.mean(strat_rets)/np.std(strat_rets))*np.sqrt(365) if np.std(strat_rets) != 0 else 0
    
    return {"WinRate": win_rate, "ActionRate": np.mean(active), "Sharpe": sharpe}

# ==========================================
# 3. OBJECTIVE (CNN + GRU)
# ==========================================
def objective(trial):
    # Parametry Hybrydy
    neurons = trial.suggest_int('neurons', 32, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float('lr', 0.0001, 0.002, log=True)
    threshold_dist = trial.suggest_float('threshold_dist', 0.01, 0.06)
    
    # Parametry CNN
    filters = trial.suggest_categorical('filters', [32, 64])
    kernel_size = trial.suggest_int('kernel_size', 2, 4)

    val_preds_sum = np.zeros(len(y_val))
    
    for i in range(N_INTERNAL_MODELS):
        model = Sequential([
            Input(shape=(TIMESTEPS, X_train.shape[2])),
            
            # --- BLOK CNN (Oczyszczanie danych) ---
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            
            # --- BLOK GRU (Pamięć sekwencyjna) ---
            GRU(neurons, return_sequences=False),
            
            BatchNormalization(),
            Dropout(dropout),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0)
        
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=32, callbacks=[es], verbose=0)
        val_preds_sum += model.predict(X_val, verbose=0).flatten()
        tf.keras.backend.clear_session()

    avg_preds = val_preds_sum / N_INTERNAL_MODELS
    stats = calculate_metrics(y_val, avg_preds, r_val, threshold_dist)
    
    trial.set_user_attr("Val_WR", stats['WinRate'])
    trial.set_user_attr("Action_Rate", stats['ActionRate'])
    
    if stats['ActionRate'] < 0.10: return 0.0
    return stats['WinRate']

# ==========================================
# 4. START
# ==========================================
if __name__ == "__main__":
    print("--- OPTYMALIZACJA HYBRYDY (CNN-GRU) ---")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100)
    
    best = study.best_trial
    print("\n" + "="*50)
    print(">>> PARAMETRY DLA HYBRYDY <<<")
    print(f"NEURONS       = {best.params['neurons']}")
    print(f"FILTERS       = {best.params['filters']}")
    print(f"KERNEL_SIZE   = {best.params['kernel_size']}")
    print(f"DROPOUT       = {best.params['dropout']}")
    print(f"LEARNING_RATE = {best.params['lr']}")
    print(f"THRESH_DIST   = {best.params['threshold_dist']}")
    print("="*50)
    print(f"Wynik WR: {best.value:.2f}% (Aktywność: {best.user_attrs['Action_Rate']*100:.1f}%)")