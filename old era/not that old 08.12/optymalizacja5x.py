import pandas as pd
import numpy as np
import os
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# ==========================================
# KONFIGURACJA
# ==========================================
INPUT_FILE = 'BTC_USD_1d_2014_2024.csv' 
TIMESTEPS = 21 
N_INTERNAL_MODELS = 5 # Tyle modeli trenujemy w JEDNEJ próbie Optuny (symulacja finału)

# Daty
TRAIN_START = '2014-09-17'
TRAIN_END   = '2022-12-31'
VAL_START   = '2023-01-01'
VAL_END     = '2024-12-31'

# ==========================================
# 1. PRZYGOTOWANIE DANYCH
# ==========================================
def prepare_data():
    if not os.path.exists(INPUT_FILE):
        print("⚠️ Generuję dane losowe (Brak CSV)")
        return None, None, None, None, None, None
        
    data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
    
    # Feature Engineering
    data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['rsi_norm'] = (100 - (100 / (1 + rs))) / 100.0
    
    data['sma_50'] = data['Close'].rolling(50).mean()
    data['dist_sma50'] = (data['Close'] - data['sma_50']) / data['sma_50']
    data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1
    data['volatility_14'] = data['log_return'].rolling(14).std()

    data.dropna(inplace=True)
    features = ['log_return', 'rsi_norm', 'dist_sma50', 'momentum_3d', 'volatility_14']
    
    # Split
    train_df = data.loc[TRAIN_START:TRAIN_END].copy()
    val_df   = data.loc[VAL_START:VAL_END].copy()
    
    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(train_df[features])
    
    def create_sequences(df):
        X_sc = scaler.transform(df[features])
        X, y = [], []
        future_returns = df['log_return'].shift(-1).fillna(0).values 
        
        for i in range(len(X_sc) - TIMESTEPS):
            X.append(X_sc[i:(i + TIMESTEPS)])
            y.append(df['target'].iloc[i + TIMESTEPS])
        
        rets = future_returns[TIMESTEPS : len(X_sc)]
        return np.array(X), np.array(y), rets

    X_train, y_train, r_train = create_sequences(train_df)
    X_val, y_val, r_val       = create_sequences(val_df)
    
    return X_train, y_train, r_train, X_val, y_val, r_val

X_train, y_train, r_train, X_val, y_val, r_val = prepare_data()

if X_train is None: # Dummy data
    X_train = np.random.rand(1000, TIMESTEPS, 5)
    y_train = np.random.randint(0, 2, 1000)
    r_train = np.random.randn(1000) * 0.01
    X_val   = np.random.rand(200, TIMESTEPS, 5)
    y_val   = np.random.randint(0, 2, 200)
    r_val   = np.random.randn(200) * 0.01

# ==========================================
# 2. SILNIK METRYK
# ==========================================
def calculate_metrics(y_true, raw_probs, returns, threshold_dist):
    upper = 0.5 + threshold_dist
    lower = 0.5 - threshold_dist
    signals = np.where(raw_probs > upper, 1, np.where(raw_probs < lower, -1, 0))
    
    active_mask = (signals != 0)
    n_trades = np.sum(active_mask)
    action_rate = n_trades / len(y_true) if len(y_true) > 0 else 0
    
    if n_trades == 0:
        win_rate = 0.0
    else:
        pred_class = (raw_probs[active_mask] > 0.5).astype(int)
        true_class = y_true[active_mask]
        win_rate = accuracy_score(true_class, pred_class) * 100

    strat_rets = signals * returns
    if np.std(strat_rets) == 0: sharpe = 0.0
    else: sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(365)

    try: gini = 2 * roc_auc_score(y_true, raw_probs) - 1
    except: gini = 0.0

    return {"WinRate": win_rate, "ActionRate": action_rate, "Sharpe": sharpe, "Gini": gini}

# ==========================================
# 3. FUNKCJA CELU (ENSEMBLE TRAINING)
# ==========================================
def objective(trial):
    # Parametry losowane raz dla całego zespołu 5 modeli
    neurons = trial.suggest_int('neurons', 32, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.15, 0.45)
    lr = trial.suggest_float('lr', 0.0001, 0.002, log=True)
    threshold_dist = trial.suggest_float('threshold_dist', 0.01, 0.06)
    
    # Kontenery na predykcje
    val_preds_sum = np.zeros(len(y_val))
    train_preds_sum = np.zeros(len(y_train))
    
    # --- PĘTLA ENSEMBLE (To jest ta nowość) ---
    for i in range(N_INTERNAL_MODELS):
        model = Sequential([
            Input(shape=(TIMESTEPS, X_train.shape[2])),
            Bidirectional(LSTM(neurons, return_sequences=False)),
            BatchNormalization(),
            Dropout(dropout),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Każdy model ma własny EarlyStopping
        es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0)
        
        model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=40, # Nieco mniej epok dla prędkości
            batch_size=32, 
            callbacks=[es], 
            verbose=0
        )
        
        # Sumujemy predykcje
        val_preds_sum += model.predict(X_val, verbose=0).flatten()
        train_preds_sum += model.predict(X_train, verbose=0).flatten()
        
        # Czyścimy pamięć (ważne przy pętli!)
        tf.keras.backend.clear_session()

    # --- UŚREDNIANIE (ENSEMBLE) ---
    avg_preds_val = val_preds_sum / N_INTERNAL_MODELS
    avg_preds_train = train_preds_sum / N_INTERNAL_MODELS
    
    # --- OCENA STATYSTYK ---
    stats_train = calculate_metrics(y_train, avg_preds_train, r_train, threshold_dist)
    stats_val   = calculate_metrics(y_val, avg_preds_val, r_val, threshold_dist)
    
    # Zapisujemy do raportu
    trial.set_user_attr("Train_WR", stats_train['WinRate'])
    trial.set_user_attr("Val_WR", stats_val['WinRate'])
    trial.set_user_attr("Train_Sharpe", stats_train['Sharpe'])
    trial.set_user_attr("Val_Sharpe", stats_val['Sharpe'])
    trial.set_user_attr("Action_Rate", stats_val['ActionRate'])
    
    # Hard Constraint
    if stats_val['ActionRate'] < 0.10:
        return 0.0

    return stats_val['WinRate']

# ==========================================
# 4. START
# ==========================================
if __name__ == "__main__":
    print(f"--- START OPTYMALIZACJI (ENSEMBLE x{N_INTERNAL_MODELS}) ---")
    print("To potrwa dłużej, ale wynik będzie wiarygodny.")
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # Mniej triali, bo każdy trwa 5x dłużej (np. 20 prób = 100 treningów łącznie)
    study.optimize(objective, n_trials=20) 

    best = study.best_trial
    
    print("\n" + "#"*60)
    print(">>> KOPIUJ PONIŻSZY BLOK DO MODELU DOCELOWEGO <<<")
    print("#"*60)
    print(f"NEURONS       = {best.params['neurons']}")
    print(f"DROPOUT       = {best.params['dropout']}")
    print(f"LEARNING_RATE = {best.params['lr']}")
    print(f"THRESH_DIST   = {best.params['threshold_dist']}")
    print("#"*60 + "\n")
    
    print(f"Statystyki zwycięzcy (Ensemble x{N_INTERNAL_MODELS}):")
    print(f"VAL Win Rate:   {best.value:.2f}%")
    print(f"TRAIN Win Rate: {best.user_attrs['Train_WR']:.2f}%")
    print(f"Aktywność:      {best.user_attrs['Action_Rate']*100:.1f}%")
    print(f"VAL Sharpe:     {best.user_attrs['Val_Sharpe']:.2f}")