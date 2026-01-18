import os
# --- SILENCE LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Import the specific optimizers
from tensorflow.keras.optimizers import Adam, AdamW, Adamax, Nadam

# --- THESIS CONFIGURATION ---
CONFIG = {
    # 1. Date Ranges
    'PERIODS': {
        'TRAIN_START': '2018-01-01',
        'TRAIN_END':   '2021-12-31',
        'VAL_START':   '2022-01-01',
        'VAL_END':     '2023-12-31',
        'TEST_START':  '2024-01-01',
        'TEST_END':    '2025-12-31' 
    },
    
    # 2. Optimal Params (UPDATE FROM OPTIMIZER RESULTS)
    'LOOKBACK': 28,          
    'LAYERS': 1,            
    'L1_UNITS': 96,         
    'L2_UNITS': 48,         
    'DROPOUT': 0.2,         
    'LEARNING_RATE': 0.001, # Update this with your best result (e.g., 0.0001)
    'OPTIMIZER': 'nadam',    # Update this with your best result (adam, adamw, adamax, or nadam)

    # 3. Strategy Settings
    'EPOCHS': 50,
    'BATCH_SIZE': 32,
    'L2_REG': 0.002,
    'FEE': 0,
    'SEED': 2026            
}

# --- SETUP DIRECTORIES ---
RESULTS_DIR = f"results_class_nothresh_{CONFIG['SEED']}"
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])

def load_data(filepath):
    try: return pd.read_csv(filepath, index_col=0, parse_dates=True)
    except: print("Error: 'processed_data.csv' not found."); exit()

def get_data_chunk_classification(df, start_str, end_str, lookback, scaler_X, fit_scalers=False):
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

def build_model_classification(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    if CONFIG['LAYERS'] == 2:
        model.add(LSTM(CONFIG['L1_UNITS'], return_sequences=True, kernel_regularizer=l2(CONFIG['L2_REG'])))
        model.add(Dropout(CONFIG['DROPOUT']))
        model.add(LSTM(CONFIG['L2_UNITS'], return_sequences=False, kernel_regularizer=l2(CONFIG['L2_REG'])))
    else:
        model.add(LSTM(CONFIG['L1_UNITS'], return_sequences=False, kernel_regularizer=l2(CONFIG['L2_REG'])))
        
    model.add(Dropout(CONFIG['DROPOUT']))
    model.add(Dense(1, activation='sigmoid'))
    
    # --- DYNAMIC OPTIMIZER SELECTION ---
    lr = CONFIG['LEARNING_RATE']
    opt_name = CONFIG['OPTIMIZER'].lower()
    
    if opt_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif opt_name == 'adamw':
        optimizer = AdamW(learning_rate=lr, weight_decay=0.004)
    elif opt_name == 'adamax':
        optimizer = Adamax(learning_rate=lr)
    elif opt_name == 'nadam':
        optimizer = Nadam(learning_rate=lr)
    else:
        print(f"Warning: Unknown optimizer '{opt_name}'. Defaulting to Adam.")
        optimizer = Adam(learning_rate=lr)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def calculate_advanced_metrics(results_df, fee=0.001):
    # Win Rate (All days, since we always trade)
    true_dir = (results_df['Actual_LogRet'] > 0).astype(int)
    pred_dir = (results_df['Signal'] > 0).astype(int)
    win_rate = accuracy_score(true_dir, pred_dir)

    # Gini
    try: gini = 2 * roc_auc_score(true_dir, results_df['Pred_Prob']) - 1
    except: gini = 0
        
    # Net Returns
    trades = np.abs(results_df['Signal'] - results_df['Signal'].shift(1).fillna(0)) / 2
    results_df['Net_Ret'] = (results_df['Signal'] * results_df['Actual_LogRet']) - (trades * fee)
    
    days = len(results_df)
    ann_ret = (np.exp(results_df['Net_Ret'].sum()) ** (365/days) - 1) if days > 0 else 0
    daily_std = results_df['Net_Ret'].std()
    sharpe = (results_df['Net_Ret'].mean()/daily_std * np.sqrt(365)) if daily_std > 0 else 0
    
    return {'WinRate': win_rate, 'Gini': gini, 'AnnReturn': ann_ret, 'Sharpe': sharpe, 'Df': results_df}

def save_plot(fig, filename):
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_visualizations(results, metrics, phase_name):
    print(f"  -> Generating Plots for: {phase_name}")
    results['Cum_Benchmark'] = np.exp(results['Actual_LogRet'].cumsum())
    results['Cum_Strategy'] = np.exp(results['Net_Ret'].cumsum())
    
    # 1. Equity
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(results['Cum_Strategy'], label='Strategy (Net)', color='#2ca02c', linewidth=2)
    plt.plot(results['Cum_Benchmark'], label='Benchmark', color='#7f7f7f', alpha=0.6)
    plt.title(f'{phase_name}: Equity (Sharpe: {metrics["Sharpe"]:.2f} | Ann. Ret: {metrics["AnnReturn"]*100:.1f}%)')
    plt.legend(); plt.grid(True, alpha=0.3)
    save_plot(fig1, f"{phase_name}_1_EquityCurve.png")
    
    # 2. Confusion Matrix
    cm = confusion_matrix((results['Actual_LogRet']>0).astype(int), (results['Signal']>0).astype(int))
    fig2, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=['Short', 'Long']).plot(cmap='Blues', ax=ax, colorbar=False)
    plt.title(f'{phase_name}: Conf Matrix (Win Rate: {metrics["WinRate"]*100:.1f}%)')
    save_plot(fig2, f"{phase_name}_2_ConfMatrix.png")
    
    # 3. Yearly
    results['Year'] = results.index.year
    yearly = results.groupby('Year').apply(lambda x: pd.Series({
        'Strategy': (np.exp(x['Net_Ret'].sum()) - 1) * 100,
        'Benchmark': (np.exp(x['Actual_LogRet'].sum()) - 1) * 100
    }), include_groups=False)
    fig3 = plt.figure(figsize=(10, 5))
    yearly.plot(kind='bar', ax=plt.gca(), color=['#2ca02c', '#7f7f7f'])
    plt.title(f'{phase_name}: Yearly Returns'); plt.grid(axis='y', alpha=0.3)
    save_plot(fig3, f"{phase_name}_3_YearlyBar.png")

    # 4. Drawdown
    dd = (results['Cum_Strategy'] / results['Cum_Strategy'].cummax()) - 1
    fig4 = plt.figure(figsize=(10, 4))
    plt.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
    plt.title(f'{phase_name}: Drawdown'); plt.grid(True, alpha=0.3)
    save_plot(fig4, f"{phase_name}_4_Drawdown.png")
    
    # 5. Distribution
    fig5 = plt.figure(figsize=(10, 5))
    sns.histplot(results['Net_Ret'], bins=50, kde=True, color='green', label='Strategy', stat='density', alpha=0.4)
    sns.histplot(results['Actual_LogRet'], bins=50, kde=True, color='gray', label='Benchmark', stat='density', alpha=0.4)
    plt.title(f'{phase_name}: Return Distribution'); plt.legend()
    save_plot(fig5, f"{phase_name}_5_Distribution.png")

    # 6. Rolling Sharpe
    roll = results['Net_Ret'].rolling(180).apply(lambda x: (x.mean()/x.std()*np.sqrt(365)) if x.std()>0 else 0)
    fig6 = plt.figure(figsize=(10, 5))
    plt.plot(roll, label='Rolling Sharpe', color='blue'); plt.axhline(0, color='k', linestyle='--')
    plt.title(f'{phase_name}: 6-Month Rolling Sharpe'); plt.legend(); plt.grid(True, alpha=0.3)
    save_plot(fig6, f"{phase_name}_6_RollingSharpe.png")

    # 7. ROC Curve
    true_cls = (results['Actual_LogRet'] > 0).astype(int)
    if len(np.unique(true_cls)) > 1:
        fpr, tpr, _ = roc_curve(true_cls, results['Pred_Prob'])
        fig7 = plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'AUC = {(metrics["Gini"]+1)/2:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'{phase_name}: ROC Curve'); plt.legend()
        save_plot(fig7, f"{phase_name}_7_ROC_Curve.png")

    # 8. Signals Zoom
    zoom = results.iloc[-100:].copy() if len(results) > 100 else results.copy()
    price = np.exp(zoom['Actual_LogRet'].cumsum())
    fig8 = plt.figure(figsize=(12, 6))
    plt.plot(price.index, price, color='k', alpha=0.5, label='Price')
    plt.scatter(zoom[zoom['Signal']==1].index, price[zoom['Signal']==1], marker='^', color='g', s=70, label='Long')
    plt.scatter(zoom[zoom['Signal']==-1].index, price[zoom['Signal']==-1], marker='v', color='r', s=70, label='Short')
    plt.title(f'{phase_name}: Trade Signals'); plt.legend()
    save_plot(fig8, f"{phase_name}_8_Signals_Zoom.png")

def analyze_feature_importance(model, X, feature_names):
    print("\n--- Calculating Feature Importance (Permutation) ---")
    base_pred = model.predict(X, verbose=0).flatten()
    importances = {}
    for i in range(X.shape[2]):
        X_shuff = X.copy()
        np.random.shuffle(X_shuff[:, :, i])
        shuff_pred = model.predict(X_shuff, verbose=0).flatten()
        importances[feature_names[i]] = np.mean(np.abs(base_pred - shuff_pred))
    
    plt.figure(figsize=(10, 6))
    pd.Series(importances).sort_values().plot(kind='barh', color='teal')
    plt.title('Feature Importance (Impact on Probability)')
    save_plot(plt.gcf(), "Final_Feature_Importance.png")

def run_pipeline():
    df = load_data('processed_data.csv')
    scaler_X = MinMaxScaler((-1,1))
    
    X_t, y_t, d_t, feats = get_data_chunk_classification(df, CONFIG['PERIODS']['TRAIN_START'], CONFIG['PERIODS']['TRAIN_END'], CONFIG['LOOKBACK'], scaler_X, True)
    X_v, y_v, d_v, _ = get_data_chunk_classification(df, CONFIG['PERIODS']['VAL_START'], CONFIG['PERIODS']['VAL_END'], CONFIG['LOOKBACK'], scaler_X, False)
    X_te, y_te, d_te, _ = get_data_chunk_classification(df, CONFIG['PERIODS']['TEST_START'], CONFIG['PERIODS']['TEST_END'], CONFIG['LOOKBACK'], scaler_X, False)
    
    Xt_seq, yt_seq, dt_seq = create_sequences(X_t, y_t, d_t, CONFIG['LOOKBACK'])
    Xv_seq, yv_seq, dv_seq = create_sequences(X_v, y_v, d_v, CONFIG['LOOKBACK'])
    Xte_seq, yte_seq, dte_seq = create_sequences(X_te, y_te, d_te, CONFIG['LOOKBACK'])
    
    # 2. Weights
    y_train_flat = y_t.flatten()
    weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
    class_weights = {0: weights[0], 1: weights[1]}
    print(f"Class Weights applied: {class_weights}")
    
    print("Building & Training Model...")
    model = build_model_classification((CONFIG['LOOKBACK'], Xt_seq.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(Xt_seq, yt_seq, validation_data=(Xv_seq, yv_seq), 
              epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'], 
              callbacks=[early_stop], verbose=1, class_weight=class_weights)
    
    analyze_feature_importance(model, Xv_seq, feats)
    
    summary = []
    
    for name, X, y, dates in [('Train', Xt_seq, y_t, dt_seq), ('Val', Xv_seq, y_v, dv_seq), ('Test', Xte_seq, y_te, dte_seq)]:
        if len(X) == 0: continue
        print(f"\nEvaluating: {name}...")
        
        preds_prob = model.predict(X, verbose=0).flatten()
        actual_returns = df.loc[dates]['TARGET'].values
        
        # FORCED SIGNAL GENERATION (Threshold effectively 0.5)
        # Prob > 0.5 -> Long (1)
        # Prob <= 0.5 -> Short (-1)
        signals = np.where(preds_prob > 0.5, 1, -1)
        
        res_df = pd.DataFrame({'Actual_LogRet': actual_returns, 'Pred_Prob': preds_prob, 'Signal': signals}, index=dates)
        
        m = calculate_advanced_metrics(res_df, CONFIG['FEE'])
        generate_visualizations(m['Df'], m, name)
        
        print(f"  > WinRate: {m['WinRate']:.2%}")
        print(f"  > Sharpe:  {m['Sharpe']:.2f}")
        print(f"  > Ann Ret: {m['AnnReturn']:.2%}")
        
        summary.append({'Phase': name, 'WinRate': m['WinRate'], 'Sharpe': m['Sharpe'], 'AnnReturn': m['AnnReturn'], 'Gini': m['Gini']})

    print("\n=== FINAL REPORT ===")
    print(pd.DataFrame(summary))
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, 'final_metrics_summary.csv'), index=False)
    print(f"\n[DONE] Results saved to: {RESULTS_DIR}/")

if __name__ == "__main__":
    run_pipeline()