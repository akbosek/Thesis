import pandas as pd
import numpy as np
import os
import optuna
import sys
import logging

# Wyciszenie logów
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ==========================================
#      0. KONFIGURACJA (SYMETRYCZNA)
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'
N_TRIALS   = 5000   
N_SPLITS   = 4     

PARAM_RANGES = {
    'lookback': [14, 60],       
    'neurons':  [32, 96],      
    'dropout':  [0.1, 0.5],     
    'lr':       [0.0001, 0.005], 
    'batch_size': [16, 64],     
    'epochs':   [20, 60],       
    
    # --- ZMIANA: SYMETRIA ---
    # Zamiast dwóch progów, szukamy "odchylenia od 0.5".
    # 0.03 oznacza: Long > 0.53, Short < 0.47 (Bardzo aktywnie)
    # 0.15 oznacza: Long > 0.65, Short < 0.35 (Bardzo bezpiecznie)
    'threshold_dist': [0.03, 0.15] 
}

# ==========================================
#      1. DANE
# ==========================================
def load_and_engineer_data(filepath):
    if not os.path.exists(filepath): raise FileNotFoundError(f"Brak pliku {filepath}")
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    df.sort_index(inplace=True)
    df = df.astype('float32')
    
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1)).astype('float32')
    df['target']     = np.where(df['Close'].shift(-1) > df['Close'], 1.0, 0.0).astype('float32')
    df['volatility_30'] = df['log_return'].rolling(window=30).std().astype('float32')
    df['dist_high']     = ((df['High'] - df['Close']) / df['Close']).astype('float32')
    df['dist_low']      = ((df['Close'] - df['Low']) / df['Close']).astype('float32')
    df['momentum_3d']   = (df['Close'] / df['Close'].shift(3) - 1).astype('float32')
    df['SMA_200']  = df['Close'].rolling(window=200).mean().astype('float32')
    df['Trend_Up'] = (df['Close'] > df['SMA_200']).astype('float32')
    df.dropna(inplace=True)
    return df

def walk_forward_split(df, n_splits=4, train_years=2, test_years=1):
    unique_years = sorted(df.index.year.unique())
    if len(unique_years) < train_years + test_years: raise ValueError("Za mało danych!")
    splits = []
    start_idx = len(unique_years) - n_splits - test_years - train_years + 1
    if start_idx < 0: start_idx = 0
    for i in range(n_splits):
        t_start = unique_years[start_idx + i]
        t_end   = unique_years[start_idx + i + train_years - 1]
        test_yr = unique_years[start_idx + i + train_years]
        splits.append((df[(df.index.year >= t_start) & (df.index.year <= t_end)], df[df.index.year == test_yr]))
    return splits

def create_sequences(data_x, data_y, lookback):
    X, y = [], []
    for i in range(len(data_x) - lookback):
        X.append(data_x[i:(i + lookback)])
        y.append(data_y[i + lookback])
    return np.array(X), np.array(y)

# ==========================================
#      2. FUNKCJA CELU (SYMETRYCZNA)
# ==========================================
def objective_fast_hybrid(trial):
    lookback = trial.suggest_int('lookback', *PARAM_RANGES['lookback'])
    neurons  = trial.suggest_int('neurons', *PARAM_RANGES['neurons'])
    dropout  = trial.suggest_float('dropout', *PARAM_RANGES['dropout'])
    lr       = trial.suggest_float('lr', *PARAM_RANGES['lr'], log=True)
    batch_sz = trial.suggest_categorical('batch_size', PARAM_RANGES['batch_size'])
    epochs   = trial.suggest_int('epochs', *PARAM_RANGES['epochs'])
    
    # --- SYMETRYCZNE PROGI ---
    dist = trial.suggest_float('threshold_dist', *PARAM_RANGES['threshold_dist'])
    t_long  = 0.5 + dist
    t_short = 0.5 - dist
    
    features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']
    wf_scores, wf_wrs, wf_rets = [], [], []
    
    splits = walk_forward_split(full_df, n_splits=N_SPLITS)
    
    for step, (train_subset, test_subset) in enumerate(splits):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_subset[features])
        train_x_sc = scaler.transform(train_subset[features])
        test_x_sc  = scaler.transform(test_subset[features])
        
        X_train, y_train = create_sequences(train_x_sc, train_subset['target'].values, lookback)
        X_test, _        = create_sequences(test_x_sc, test_subset['target'].values, lookback)
        
        if len(X_train) < 100 or len(X_test) < 20: return -999.0
            
        model = Sequential([
            Input(shape=(lookback, len(features))),
            LSTM(neurons, return_sequences=False), 
            BatchNormalization(),
            Dropout(dropout),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_sz, verbose=0, callbacks=[es], shuffle=False)
        
        preds = model.predict(X_test, verbose=0).flatten()
        test_data_slice = test_subset.iloc[lookback:]
        market_ret = test_data_slice['log_return'].shift(-1).fillna(0).values
        trend      = test_data_slice['Trend_Up'].values
        
        # --- LOGIKA DECYZYJNA ---
        signals = np.zeros(len(preds), dtype=np.int8)
        signals[preds > t_long] = 1
        signals[preds < t_short] = -1
        
        signals[(signals == 1) & (trend == 0)] = 0
        signals[(signals == -1) & (trend == 1)] = 0
        
        strat_ret = signals * market_ret
        n_trades = np.count_nonzero(signals)
        
        if n_trades < 10:
            score, wr, total_ret = -50.0, 0.0, 0.0
        else:
            wins = np.sum(np.sign(signals[signals!=0]) == np.sign(market_ret[signals!=0]))
            wr = (wins / n_trades) * 100
            total_ret = (np.exp(np.sum(strat_ret)) - 1) * 100
            
            if total_ret > 0: score = total_ret * (wr / 100.0)
            else: score = total_ret * 2.0
        
        wf_scores.append(score)
        wf_wrs.append(wr)
        wf_rets.append(total_ret)
        
        del model
        tf.keras.backend.clear_session()
        
        intermediate_val = np.mean(wf_scores)
        trial.report(intermediate_val, step)
        if trial.should_prune(): raise optuna.TrialPruned()

    avg_score = np.mean(wf_scores)
    trial.set_user_attr("win_rate", np.mean(wf_wrs))
    trial.set_user_attr("return", np.mean(wf_rets))
    
    return avg_score

# ==========================================
#      3. URUCHOMIENIE I WYNIKI
# ==========================================
if __name__ == "__main__":
    print("--- Start Optymalizacji (Symmetry + Fast) ---")
    full_df = load_and_engineer_data(INPUT_FILE)
    
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=N_SPLITS, reduction_factor=3)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    
    try:
        # n_jobs=4 dla Ryzena. Jeśli błędy -> daj 1.
        study.optimize(objective_fast_hybrid, n_trials=N_TRIALS, n_jobs=1)
    except KeyboardInterrupt:
        print("\nPrzerwano.")

    print("\n" + "="*90)
    print(f"   TOP 10 KONFIGURACJI (SYMETRYCZNE)   ")
    print("="*90)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    
    results_list = []
    for i, t in enumerate(completed_trials[:10]):
        # Obliczamy faktyczne progi do wyświetlenia w tabeli
        dist = t.params['threshold_dist']
        calc_t_long  = 0.5 + dist
        calc_t_short = 0.5 - dist
        
        row = {
            "Rank": i + 1,
            "Score": round(t.value, 2),
            "WinRate %": round(t.user_attrs.get("win_rate", 0), 1),
            "Return %": round(t.user_attrs.get("return", 0), 1),
            "Lookback": t.params['lookback'],
            "Neurons": t.params['neurons'],
            "LR": f"{t.params['lr']:.5f}",
            "Epochs": t.params['epochs'],
            # Pokazujemy obliczone wartości, żebyś mógł je wkleić do V10
            "Calc T_Long": round(calc_t_long, 3),
            "Calc T_Short": round(calc_t_short, 3)
        }
        results_list.append(row)
        
    df_results = pd.DataFrame(results_list)
    if not df_results.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_results.to_string(index=False))
    else:
        print("Brak danych.")
    print("="*90)
    print("Wskazówka: Wpisz wartości 'Calc T_Long' i 'Calc T_Short' do skryptu V10.")