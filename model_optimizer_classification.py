import os
# --- SILENCE LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Import all optimizers
from tensorflow.keras.optimizers import Adam, AdamW, Adamax, Nadam 

# --- CONFIGURATION ---
CONFIG = {
    'PERIODS': {
        'TRAIN_START': '2018-01-01',
        'TRAIN_END':   '2021-12-31',
        'VAL_START':   '2022-01-01',
        'VAL_END':     '2023-12-31',
        'TEST_START':  '2024-01-01',
        'TEST_END':    '2025-12-31' 
    },
    
    # Hyperparameter Grid
    'GRID': {
        'LOOKBACK': [7, 14, 21, 28],
        'L1_UNITS': [32, 48, 64, 96, 128],    
        'L2_UNITS': [8, 16, 24, 32, 48, 64],     
        'DROPOUT': [0.2, 0.3, 0.35, 0.4, 0.45, 0.5],
        'LAYERS': [1, 2],
        'LEARNING_RATE': [0.001, 0.0005, 0.0001],
        # NEW: The Optimizer Tournament
        'OPTIMIZER': ['adam', 'adamw', 'adamax', 'nadam'] 
    },
    
    'MAX_TRIALS': 200,       # Increased slightly to cover new optimizers
    'TOP_N_RESULTS': 15,     
    'SEED': 2026
}

DATA_PATH = 'processed_data.csv'
RESULTS_DIR = "results_optimizer_class"
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

# --- UTILS ---
def get_data_chunk_class(df, start_str, end_str, lookback, scaler_X, fit_scalers=False):
    try: start_idx = df.index.get_loc(start_str)
    except KeyError: start_idx = df.index.searchsorted(start_str)
    try: end_idx = df.index.get_loc(end_str)
    except KeyError: end_idx = df.index.searchsorted(end_str) - 1
    
    padded_start = max(0, start_idx - lookback)
    chunk = df.iloc[padded_start : end_idx + 1]
    feature_cols = [c for c in df.columns if c != 'TARGET']
    
    if fit_scalers:
        X_scaled = scaler_X.fit_transform(chunk[feature_cols])
    else:
        X_scaled = scaler_X.transform(chunk[feature_cols])
    
    y_raw = chunk['TARGET'].values
    y_binary = (y_raw > 0).astype(int).reshape(-1, 1)
        
    return X_scaled, y_binary, chunk.index[lookback:], feature_cols

def create_sequences(X, y, dates, lookback):
    Xs, ys, ds = [], [], []
    if len(X) <= lookback: return np.array([]), np.array([]), np.array([])
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback])
        ds.append(dates[i])
    return np.array(Xs), np.array(ys), np.array(ds)

def prepare_data(df, lookback):
    scaler_X = MinMaxScaler((-1, 1))
    X_tr, y_tr, d_tr, feats = get_data_chunk_class(df, CONFIG['PERIODS']['TRAIN_START'], CONFIG['PERIODS']['TRAIN_END'], lookback, scaler_X, True)
    X_val, y_val, d_val, _ = get_data_chunk_class(df, CONFIG['PERIODS']['VAL_START'], CONFIG['PERIODS']['VAL_END'], lookback, scaler_X, False)
    return {'train': create_sequences(X_tr, y_tr, d_tr, lookback), 'val': create_sequences(X_val, y_val, d_val, lookback), 'features': feats}

def train_evaluate_model(params, df):
    lookback = params['LOOKBACK']
    data = prepare_data(df, lookback)
    X_t, y_t, _ = data['train']
    X_v, y_v, _ = data['val']
    
    if len(X_t) == 0: return {'val_winrate': 0}

    # Weights
    y_train_flat = y_t.flatten()
    weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
    class_weights = {0: weights[0], 1: weights[1]}

    # Model
    model = Sequential()
    model.add(Input(shape=(lookback, X_t.shape[2])))
    
    if params['LAYERS'] == 2:
        model.add(LSTM(params['L1_UNITS'], return_sequences=True, kernel_regularizer=l2(0.001)))
        model.add(Dropout(params['DROPOUT']))
        model.add(LSTM(params['L2_UNITS'], return_sequences=False, kernel_regularizer=l2(0.001)))
    else:
        model.add(LSTM(params['L1_UNITS'], return_sequences=False, kernel_regularizer=l2(0.001)))
        
    model.add(Dropout(params['DROPOUT']))
    model.add(Dense(1, activation='sigmoid'))
    
    # --- DYNAMIC OPTIMIZER SELECTION ---
    lr = params['LEARNING_RATE']
    opt_name = params['OPTIMIZER']
    
    if opt_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif opt_name == 'adamw':
        # AdamW needs weight decay (standard default is ~0.004)
        optimizer = AdamW(learning_rate=lr, weight_decay=0.004)
    elif opt_name == 'adamax':
        optimizer = Adamax(learning_rate=lr)
    elif opt_name == 'nadam':
        optimizer = Nadam(learning_rate=lr)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=0)
    
    model.fit(
        X_t, y_t, 
        validation_data=(X_v, y_v), 
        epochs=30, 
        batch_size=32, 
        callbacks=[early_stop, reduce_lr], 
        class_weight=class_weights, 
        verbose=0
    )
    
    # Eval
    probs = model.predict(X_v, verbose=0).flatten()
    actuals = y_v.flatten()
    preds = (probs > 0.5).astype(int)
    
    win_rate = accuracy_score(actuals, preds)
    
    return {'val_winrate': win_rate, 'params': params, 'model': model, 'X_val': X_v, 'feature_names': data['features']}

def run_optimizer():
    try: df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    except: print("Error: processed_data.csv not found."); exit()

    keys, values = zip(*CONFIG['GRID'].items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if CONFIG['MAX_TRIALS'] and CONFIG['MAX_TRIALS'] < len(all_combinations):
        test_combinations = random.sample(all_combinations, CONFIG['MAX_TRIALS'])
    else:
        test_combinations = all_combinations

    results_log = []
    print(f"--- Starting Optimization ({len(test_combinations)} runs) ---")
    print(f"--- Testing: Adam vs AdamW vs Adamax vs Nadam ---")
    
    best_score = -1
    best_run = None

    for i, params in enumerate(test_combinations):
        print(f"[{i+1}/{len(test_combinations)}] Testing {params}...", end=" ")
        try:
            res = train_evaluate_model(params, df)
            print(f"-> WinRate: {res['val_winrate']:.2%}")
            results_log.append({**params, 'Val_WinRate': res['val_winrate']})
            
            if res['val_winrate'] > best_score:
                best_score = res['val_winrate']
                best_run = res
        except Exception as e:
            print(f"-> Failed: {e}")

    df_results = pd.DataFrame(results_log).sort_values('Val_WinRate', ascending=False)
    print(f"\n=== TOP {CONFIG['TOP_N_RESULTS']} CONFIGURATIONS ===")
    print(df_results.head(CONFIG['TOP_N_RESULTS']))
    df_results.to_csv(os.path.join(RESULTS_DIR, 'optimizer_results.csv'), index=False)
    
    if best_run:
        print("\nCalculating Feature Importance for Best Model...")
        model = best_run['model']
        X = best_run['X_val']
        feats = best_run['feature_names']
        
        base_pred = model.predict(X, verbose=0).flatten()
        importances = {}
        for i in range(X.shape[2]):
            X_shuff = X.copy()
            np.random.shuffle(X_shuff[:, :, i])
            shuff_pred = model.predict(X_shuff, verbose=0).flatten()
            importances[feats[i]] = np.mean(np.abs(base_pred - shuff_pred))
            
        plt.figure(figsize=(10, 6))
        pd.Series(importances).sort_values().plot(kind='barh', color='teal')
        plt.title('Feature Importance (Impact on Probability)')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))
        plt.show()

if __name__ == "__main__":
    run_optimizer()