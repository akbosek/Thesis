import pandas as pd
import numpy as np
import os
import optuna
import sys
import logging

# Wyciszenie logów TensorFlow (przyspiesza konsolę)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ==========================================
#      KONFIGURACJA
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'
N_TRIALS = 100   # Możemy dać więcej prób, bo skrypt jest szybki
N_SPLITS = 4     # Liczba okresów Walk-Forward

# ==========================================
#      1. SZYBKIE ŁADOWANIE DANYCH
# ==========================================
def load_and_engineer_data(filepath):
    if not os.path.exists(filepath): raise FileNotFoundError(f"Brak pliku {filepath}")
    
    # Parsowanie dat spowalnia, ale jest konieczne raz
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    df.sort_index(inplace=True)
    
    # Rzutowanie na float32 (kluczowe dla szybkości na CPU/GPU)
    df = df.astype('float32')
    
    # Feature Engineering
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1)).astype('float32')
    df['target']     = np.where(df['Close'].shift(-1) > df['Close'], 1.0, 0.0).astype('float32')
    
    df['volatility_30'] = df['log_return'].rolling(window=30).std().astype('float32')
    df['dist_high']     = ((df['High'] - df['Close']) / df['Close']).astype('float32')
    df['dist_low']      = ((df['Close'] - df['Low']) / df['Close']).astype('float32')
    df['momentum_3d']   = (df['Close'] / df['Close'].shift(3) - 1).astype('float32')
    
    # Filtr Trendu (Globalny)
    df['SMA_200']  = df['Close'].rolling(window=200).mean().astype('float32')
    df['Trend_Up'] = (df['Close'] > df['SMA_200']).astype('float32')
    
    df.dropna(inplace=True)
    return df

# ==========================================
#      2. MECHANIKA WALK-FORWARD
# ==========================================
def walk_forward_split(df, n_splits=4, train_years=2, test_years=1):
    unique_years = sorted(df.index.year.unique())
    if len(unique_years) < train_years + test_years:
        raise ValueError("Za mało lat danych!")

    splits = []
    start_idx = len(unique_years) - n_splits - test_years - train_years + 1
    if start_idx < 0: start_idx = 0
    
    for i in range(n_splits):
        t_start = unique_years[start_idx + i]
        t_end   = unique_years[start_idx + i + train_years - 1]
        test_yr = unique_years[start_idx + i + train_years]
        
        train_mask = (df.index.year >= t_start) & (df.index.year <= t_end)
        test_mask  = (df.index.year == test_yr)
        
        splits.append((df[train_mask], df[test_mask]))
    return splits

def create_sequences(data_x, data_y, lookback):
    # Wersja zoptymalizowana pod NumPy
    X, y = [], []
    for i in range(len(data_x) - lookback):
        X.append(data_x[i:(i + lookback)])
        y.append(data_y[i + lookback])
    return np.array(X), np.array(y)

# ==========================================
#      3. OBJECTIVE FUNCTION (SZYBKA)
# ==========================================
def objective_fast_hybrid(trial):
    # Parametry do optymalizacji
    params = {
        'lookback': trial.suggest_int('lookback', 14, 60),
        'neurons':  trial.suggest_int('neurons', 32, 128),
        'dropout':  trial.suggest_float('dropout', 0.1, 0.5),
        'lr':       trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
        'epochs':   trial.suggest_int('epochs', 10, 40),
        't_long':   trial.suggest_float('t_long', 0.55, 0.85), # Wyższe progi dla bezpieczeństwa
        't_short':  trial.suggest_float('t_short', 0.15, 0.45)
    }
    
    features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']
    wf_scores = []
    
    splits = walk_forward_split(full_df, n_splits=N_SPLITS)
    
    for step, (train_subset, test_subset) in enumerate(splits):
        # A. Skalowanie
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit tylko na treningu
        scaler.fit(train_subset[features])
        
        # Transformacja (szybka operacja wektorowa)
        train_x_sc = scaler.transform(train_subset[features])
        test_x_sc  = scaler.transform(test_subset[features])
        
        # B. Sekwencje
        X_train, y_train = create_sequences(train_x_sc, train_subset['target'].values, params['lookback'])
        X_test, _        = create_sequences(test_x_sc, test_subset['target'].values, params['lookback'])
        
        if len(X_train) < 100 or len(X_test) < 20: 
            return -999.0 # Fail
            
        # C. Model (Lekki)
        model = Sequential([
            Input(shape=(params['lookback'], len(features))),
            LSTM(params['neurons'], return_sequences=False), 
            BatchNormalization(),
            Dropout(params['dropout']),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=params['lr']), 
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        # Early Stopping (agresywny dla szybkości)
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        
        model.fit(X_train, y_train, 
                  epochs=params['epochs'], 
                  batch_size=params['batch_size'], 
                  verbose=0, callbacks=[es], shuffle=False)
        
        # D. Predykcja i Logika (Wektoryzacja NumPy - 100x szybciej niż pętle)
        preds = model.predict(X_test, verbose=0).flatten()
        
        # Wyrównanie danych
        test_data_slice = test_subset.iloc[params['lookback']:]
        market_ret = test_data_slice['log_return'].shift(-1).fillna(0).values
        trend      = test_data_slice['Trend_Up'].values
        
        # --- VECTORIZED STRATEGY LOGIC ---
        # 1. Inicjalizacja sygnałów zerami
        signals = np.zeros(len(preds), dtype=np.int8)
        
        # 2. Thresholding (Long/Short)
        signals[preds > params['t_long']] = 1
        signals[preds < params['t_short']] = -1
        
        # 3. Filtr Trendu (SMA 200)
        # Zeruj LONG jeśli Trend jest 0 (Bessa)
        signals[(signals == 1) & (trend == 0)] = 0
        # Zeruj SHORT jeśli Trend jest 1 (Hossa)
        signals[(signals == -1) & (trend == 1)] = 0
        
        # E. Wyniki
        strat_ret = signals * market_ret
        n_trades = np.count_nonzero(signals)
        
        if n_trades < 10:
            score = -50.0 # Kara za brak aktywności
        else:
            # Win Rate Calculation
            # Zysk: (pos=1 i ret>0) LUB (pos=-1 i ret<0) -> sign(pos) == sign(ret)
            # Uwaga: dla ret=0 (brak zmiany) sign=0, więc nie liczy jako win, co jest ok.
            wins = np.sum(np.sign(signals[signals!=0]) == np.sign(market_ret[signals!=0]))
            wr = (wins / n_trades) * 100
            
            # Total Return
            total_ret = (np.exp(np.sum(strat_ret)) - 1) * 100
            
            # --- HYBRID SCORE FORMULA ---
            # Balance: Return * (WinRate/100). 
            # Przykład: 50% zysku przy 60% WR -> Score 30.
            # Przykład: 50% zysku przy 40% WR -> Score 20.
            if total_ret > 0:
                score = total_ret * (wr / 100.0)
            else:
                score = total_ret * 2.0 # Mocna kara za stratę
                
        wf_scores.append(score)
        
        # F. Sprzątanie
        del model
        tf.keras.backend.clear_session()
        
        # --- PRUNING (Optymalizacja Czasu) ---
        # Zgłaszamy wynik do Optuny. Jeśli jest fatalny, przerywamy pętlę lat (po co liczyć 2022, jak 2021 był klapą?)
        intermediate_val = np.mean(wf_scores)
        trial.report(intermediate_val, step)
        
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(wf_scores)

# ==========================================
#      URUCHOMIENIE
# ==========================================
if __name__ == "__main__":
    print("--- Ładowanie danych... ---")
    full_df = load_and_engineer_data(INPUT_FILE)
    print(f"Dane gotowe: {len(full_df)} wierszy.")
    
    # Pruner: Hyperband to algorytm, który bardzo szybko odcina słabe gałęzie
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=N_SPLITS, reduction_factor=3)
    
    study = optuna.create_study(direction='maximize', pruner=pruner)
    
    print(f"--- Start Optymalizacji ({N_TRIALS} prób) ---")
    print("Cel: Maksymalizacja (Return * WinRate)")
    
    # n_jobs=1 jest bezpieczne dla TensorFlow. 
    # Jeśli masz mocny CPU i dużo RAM, możesz spróbować n_jobs=2, ale przy GPU często powoduje to błędy.
    try:
        study.optimize(objective_fast_hybrid, n_trials=N_TRIALS, n_jobs=1)
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika.")
    
    print("\n" + "="*60)
    print("   ZWYCIĘSKIE PARAMETRY (HYBRID MAX)   ")
    print("="*60)
    best = study.best_params
    for k, v in best.items():
        print(f"{k.ljust(15)}: {v}")
    
    print("-" * 60)
    print(f"Najlepszy Score: {study.best_value:.2f}")
    print("(Score = Return % * WinRate % / 100)")
    print("="*60)