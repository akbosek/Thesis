import pandas as pd
import numpy as np
import os
import optuna
import logging
import sys

# Wyłączamy logi TensorFlow, żeby widzieć postęp Optuny
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ==========================================
#      KONFIGURACJA DANYCH I BADANIA
# ==========================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'
N_TRIALS = 50  # Ile różnych kombinacji parametrów sprawdzić (im więcej tym lepiej, np. 100)
N_SPLITS = 4   # Ile okresów Walk-Forward (np. 4 lata testów rok po roku)

# Zakresy parametrów do przeszukania
SEARCH_SPACE = {
    'lookback': [10, 60],        # Długość sekwencji (dni)
    'neurons': [32, 128],        # Złożoność modelu
    'dropout': [0.1, 0.5],       # Ochrona przed overfittingiem
    'lr': [0.0001, 0.005],       # Szybkość uczenia
    'batch_size': [16, 64],
    'thresh_long': [0.55, 0.80], # Jak pewny musi być model, by grać LONG
    'thresh_short': [0.20, 0.45] # Jak pewny musi być model, by grać SHORT
}
# ==========================================

# 1. PRZYGOTOWANIE DANYCH (BEZ SKALOWANIA - TO ROBIMY PÓŹNIEJ)
def load_and_engineer_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Brak pliku {filepath}")
    
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    df.sort_index(inplace=True)
    
    # Targety i feature'y
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    # Target: 1 jeśli jutro cena wzrośnie
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # Feature Engineering
    df['volatility_30'] = df['log_return'].rolling(window=30).std()
    df['dist_high'] = (df['High'] - df['Close']) / df['Close']
    df['dist_low']  = (df['Close'] - df['Low']) / df['Close']
    df['momentum_3d'] = df['Close'] / df['Close'].shift(3) - 1
    
    # Filtr Trendu (SMA 200) - obliczamy globalnie, bo to wskaźnik opóźniony (nie ma wycieku przyszłości)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Trend_Up'] = (df['Close'] > df['SMA_200']).astype(int)
    
    df.dropna(inplace=True)
    return df

# 2. GENERATOR WALK-FORWARD (KLUCZOWY ELEMENT)
def walk_forward_split(df, n_splits=4, train_years=2, test_years=1):
    """
    Generuje indeksy dla walidacji kroczącej.
    Niestandardowy split, aby zachować chronologię rynkową.
    """
    unique_years = sorted(df.index.year.unique())
    # Upewniamy się, że mamy dość lat
    if len(unique_years) < train_years + test_years:
        raise ValueError("Za mało danych na Walk-Forward!")

    splits = []
    # Zaczynamy tak, by ostatni test był na ostatnim dostępnym roku
    start_idx = len(unique_years) - n_splits - test_years - train_years + 1
    if start_idx < 0: start_idx = 0
    
    for i in range(n_splits):
        # Definiujemy lata
        current_train_start_yr = unique_years[start_idx + i]
        current_train_end_yr   = unique_years[start_idx + i + train_years - 1]
        current_test_yr        = unique_years[start_idx + i + train_years]
        
        # Maski
        train_mask = (df.index.year >= current_train_start_yr) & (df.index.year <= current_train_end_yr)
        test_mask  = (df.index.year == current_test_yr)
        
        splits.append((df[train_mask], df[test_mask], current_test_yr))
        
    return splits

# 3. BUDOWA MODELU
def build_model(input_shape, params):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(params['neurons'], return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = Adam(learning_rate=params['lr'])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. FUNKCJA TWORZĄCA OKNA CZASOWE (X, y)
def create_sequences(data_df, feature_cols, target_col, lookback):
    # Konwersja do numpy
    data_x = data_df[feature_cols].values
    data_y = data_df[target_col].values
    
    X, y = [], []
    for i in range(len(data_x) - lookback):
        X.append(data_x[i:(i + lookback)])
        y.append(data_y[i + lookback])
        
    return np.array(X), np.array(y), data_df.index[lookback:]

# 5. FUNKCJA CELU OPTUNY
def objective(trial):
    # A. Losowanie hiperparametrów
    params = {
        'lookback': trial.suggest_int('lookback', *SEARCH_SPACE['lookback']),
        'neurons':  trial.suggest_int('neurons', *SEARCH_SPACE['neurons']),
        'dropout':  trial.suggest_float('dropout', *SEARCH_SPACE['dropout']),
        'lr':       trial.suggest_float('lr', *SEARCH_SPACE['lr'], log=True),
        'batch_size': trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size']),
        't_long':   trial.suggest_float('t_long', *SEARCH_SPACE['thresh_long']),
        't_short':  trial.suggest_float('t_short', *SEARCH_SPACE['thresh_short'])
    }
    
    # Ładujemy dane (są w pamięci globalnej 'full_df')
    features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']
    
    # B. Walk-Forward Loop
    wf_scores = []
    wf_trades = []
    
    splits = walk_forward_split(full_df, n_splits=N_SPLITS)
    
    for train_subset, test_subset, test_year in splits:
        # --- ZAPOBIEGANIE WYCIEKOWI DANYCH ---
        # Skaler trenujemy TYLKO na train_subset
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_subset_sc = train_subset.copy()
        test_subset_sc  = test_subset.copy()
        
        scaler.fit(train_subset[features])
        
        train_subset_sc[features] = scaler.transform(train_subset[features])
        test_subset_sc[features]  = scaler.transform(test_subset[features])
        
        # Tworzenie sekwencji
        X_train, y_train, _ = create_sequences(train_subset_sc, features, 'target', params['lookback'])
        X_test, y_test, idx_test = create_sequences(test_subset_sc, features, 'target', params['lookback'])
        
        if len(X_train) < 100 or len(X_test) < 50:
            return -999 # Za mało danych w tym foldzie
            
        # Trenowanie modelu (z Early Stopping, żeby nie marnować czasu na overfitting)
        model = build_model((params['lookback'], len(features)), params)
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        
        model.fit(X_train, y_train, 
                  epochs=15, # W optymalizacji wystarczy mniej epok, by złapać potencjał
                  batch_size=params['batch_size'], 
                  verbose=0, 
                  callbacks=[es],
                  shuffle=False) # Shuffle False dla LSTM często jest lepsze (zachowuje sekwencje w batchach)
        
        # Predykcja
        preds = model.predict(X_test, verbose=0).flatten()
        
        # --- LOGIKA STRATEGII (HYBRYDA) ---
        # Pobieramy oryginalne zwroty i trend dla okresu testowego
        # Musimy dopasować indeksy (ucinamy początek o lookback)
        test_data_aligned = test_subset.iloc[params['lookback']:].copy()
        market_ret = test_data_aligned['log_return'].shift(-1).fillna(0).values
        trend_filter = test_data_aligned['Trend_Up'].values
        
        final_pos = []
        for i in range(len(preds)):
            p = preds[i]
            t = trend_filter[i]
            
            # Logika:
            # Long: Pred > t_long ORAZ Trend wzrostowy
            # Short: Pred < t_short ORAZ Trend spadkowy (opcjonalnie, lub po prostu brak Longa)
            # Tu użyjemy Twojej logiki z filtra:
            
            signal = 0
            if p > params['t_long']: signal = 1
            elif p < params['t_short']: signal = -1
            
            # Filtr SMA
            if signal == 1 and t == 0: signal = 0  # Blokada Longa w Bessie
            if signal == -1 and t == 1: signal = 0 # Blokada Shorta w Hossie
            
            final_pos.append(signal)
            
        final_pos = np.array(final_pos)
        strat_ret = final_pos * market_ret
        
        # Metryki dla tego Foldu
        n_trades = np.sum(final_pos != 0)
        
        if n_trades < 10: 
            wf_scores.append(-1.0) # Kara za brak handlu
            wf_trades.append(n_trades)
            continue

        # Sharpe Ratio (annualized)
        std = np.std(strat_ret)
        if std == 0: sharpe = 0
        else: sharpe = (np.mean(strat_ret) / std) * np.sqrt(365)
        
        wf_scores.append(sharpe)
        wf_trades.append(n_trades)
        
        # Clean up memory
        del model
        tf.keras.backend.clear_session()

    # AGREGACJA WYNIKÓW
    avg_sharpe = np.mean(wf_scores)
    avg_trades = np.mean(wf_trades)
    
    # Penalizacja jeśli średni wynik jest ujemny lub mało tradów
    if avg_trades < 15:
        return -10.0
        
    print(f"Trial done. Sharpe: {avg_sharpe:.2f} | Trades Avg: {avg_trades:.1f}")
    return avg_sharpe

# ==========================================
#      URUCHOMIENIE
# ==========================================
if __name__ == "__main__":
    full_df = load_and_engineer_data(INPUT_FILE)
    print(f"Dane załadowane: {len(full_df)} wierszy.")
    
    # Tworzymy badanie Optuna
    study = optuna.create_study(direction='maximize')
    
    print(f"Rozpoczynam optymalizację Walk-Forward ({N_TRIALS} prób)...")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("Przerwano przez użytkownika.")
    
    print("\n" + "="*50)
    print("NAJLEPSZE PARAMETRY (ODPORNE NA PRZETRENOWANIE):")
    print("="*50)
    best = study.best_params
    for k, v in best.items():
        print(f"{k}: {v}")
    
    print(f"\nNajlepszy średni Sharpe Ratio z testów: {study.best_value:.4f}")
    
    print("\nCo dalej?")
    print("1. Wpisz te parametry do swojego skryptu V10.")
    print("2. Uruchom V10 na 2024 rok jako ostateczny 'OOS Test'.")