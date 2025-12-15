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
# Daty (Twoje ustawienia)
TRAIN_START = '2014-09-17'
TRAIN_END   = '2022-12-31'
VAL_START   = '2023-01-01'
VAL_END     = '2024-12-31'

# ==========================================
# 1. PRZYGOTOWANIE DANYCH (SAFE MODE)
# ==========================================
def prepare_data():
    if not os.path.exists(INPUT_FILE):
        print("⚠️ Generuję dane losowe (Brak pliku CSV)")
        return None, None, None, None, None, None
        
    data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
    
    # Target
    data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Wskaźniki
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
    
    # PODZIAŁ (Bez shuffle, chronologicznie)
    train_df = data.loc[TRAIN_START:TRAIN_END].copy()
    val_df   = data.loc[VAL_START:VAL_END].copy()
    
    # SKALOWANIE (Fit tylko na treningu - zapobiega Data Leakage)
    scaler = MinMaxScaler()
    scaler.fit(train_df[features])
    
    # Funkcja tworząca okna
    def create_sequences(df):
        X_sc = scaler.transform(df[features])
        X, y = [], []
        # Returny do Sharpe Ratio (przesunięte o 1 dzień w przyszłość względem okna)
        future_returns = df['log_return'].shift(-1).fillna(0).values 
        
        for i in range(len(X_sc) - TIMESTEPS):
            X.append(X_sc[i:(i + TIMESTEPS)])
            y.append(df['target'].iloc[i + TIMESTEPS])
        
        # Returns musimy przyciąć do długości y
        rets = future_returns[TIMESTEPS : len(X_sc)]
        return np.array(X), np.array(y), rets

    X_train, y_train, r_train = create_sequences(train_df)
    X_val, y_val, r_val       = create_sequences(val_df)
    
    return X_train, y_train, r_train, X_val, y_val, r_val

X_train, y_train, r_train, X_val, y_val, r_val = prepare_data()

# Jeśli brak danych (test mode)
if X_train is None:
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
    # Progi
    upper = 0.5 + threshold_dist
    lower = 0.5 - threshold_dist
    
    # Decyzje (1: Long, -1: Short, 0: Flat)
    signals = np.where(raw_probs > upper, 1, np.where(raw_probs < lower, -1, 0))
    
    # Podstawowe liczniki
    total_obs = len(y_true)
    active_mask = (signals != 0)
    n_trades = np.sum(active_mask)
    action_rate = n_trades / total_obs
    
    # Win Rate
    if n_trades == 0:
        win_rate = 0.0
    else:
        # Znak zwrotu vs Znak sygnału (dla aktywnych)
        # Uwaga: y_true to [0, 1]. Zamieniamy na [-1, 1] dla porównania kierunku? 
        # Prościej: Jeśli signal=1 i y=1 (Wzrost) -> Win. Jeśli signal=-1 i y=0 (Spadek) -> Win.
        
        # Konwersja y_true (0/1) na kierunek ceny (-1/1) jest trudna bez surowych returnów.
        # Użyjmy logiki accuracy na klasach dla uproszczenia, jak wcześniej:
        # Long (>upper) vs y=1 | Short (<lower) vs y=0
        
        pred_class_active = (raw_probs[active_mask] > 0.5).astype(int)
        true_class_active = y_true[active_mask]
        win_rate = accuracy_score(true_class_active, pred_class_active) * 100

    # Sharpe Ratio
    # Strategy returns = Signal * Market Return
    strat_rets = signals * returns
    if np.std(strat_rets) == 0:
        sharpe = 0.0
    else:
        # Annualized Sharpe (zakładając dane dzienne crypto ~365 dni)
        sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(365)

    # Gini Ratio (2*AUC - 1)
    try:
        auc = roc_auc_score(y_true, raw_probs)
        gini = 2 * auc - 1
    except:
        gini = 0.0

    return {
        "WinRate": win_rate,
        "ActionRate": action_rate,
        "Trades": n_trades,
        "Total": total_obs,
        "Sharpe": sharpe,
        "Gini": gini
    }

# ==========================================
# 3. FUNKCJA CELU
# ==========================================
def objective(trial):
    # Hiperparametry
    neurons = trial.suggest_int('neurons', 32, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.15, 0.45)
    lr = trial.suggest_float('lr', 0.0001, 0.002, log=True)
    threshold_dist = trial.suggest_float('threshold_dist', 0.01, 0.06)
    
    model = Sequential([
        Input(shape=(TIMESTEPS, X_train.shape[2])),
        Bidirectional(LSTM(neurons, return_sequences=False)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, 
        batch_size=32,
        callbacks=[es],
        verbose=0
    )
    
    # --- OCENA (TRAIN vs VAL) ---
    preds_train = model.predict(X_train, verbose=0).flatten()
    preds_val   = model.predict(X_val, verbose=0).flatten()
    
    stats_train = calculate_metrics(y_train, preds_train, r_train, threshold_dist)
    stats_val   = calculate_metrics(y_val, preds_val, r_val, threshold_dist)
    
    # Raportowanie do Optuny (abyś widział to w logach)
    trial.set_user_attr("Train_WR", stats_train['WinRate'])
    trial.set_user_attr("Val_WR", stats_val['WinRate'])
    trial.set_user_attr("Train_Sharpe", stats_train['Sharpe'])
    trial.set_user_attr("Val_Sharpe", stats_val['Sharpe'])
    trial.set_user_attr("Action_Rate", stats_val['ActionRate'])
    
    # Kryterium odcięcia (Action Rate < 10% na walidacji)
    if stats_val['ActionRate'] < 0.10:
        return 0.0

    return stats_val['WinRate'] # Optymalizujemy pod Win Rate na walidacji

# ==========================================
# 4. START
# ==========================================
if __name__ == "__main__":
    print("--- OPTYMALIZATOR ROZPOCZĘTY ---")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=50)

    best = study.best_trial
    print("\n" + "="*60)
    print(f"🏆 NAJLEPSZY MODEL (TRIAL {best.number})")
    print("="*60)
    print(f"VAL Win Rate:   {best.value:.2f}%")
    print(f"TRAIN Win Rate: {best.user_attrs['Train_WR']:.2f}%  <-- Sprawdź Overfitting!")
    print(f"VAL Sharpe:     {best.user_attrs['Val_Sharpe']:.2f}")
    print(f"Aktywność:      {best.user_attrs['Action_Rate']*100:.1f}%")
    print("-" * 60)
    print("PARAMETRY:")
    for k,v in best.params.items():
        print(f"  {k}: {v}")