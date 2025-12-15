import pandas as pd
import numpy as np
import yfinance as yf
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. KONFIGURACJA HYBRYDOWA (PAPER + NASZE MACRO)
# ==============================================================================
# ZMIANA: Skracamy okno do 10 dni (zgodnie z nowym artykułem)
TIMESTEPS = 10         
PREDICTION_HORIZON = 1 
TRAIN_END_DATE = '2022-12-31'

print("--- [FUSION BOT: CNN-LSTM + MACRO DATA] ---")

# ==============================================================================
# 2. ŁADOWANIE DANYCH (GLOBALNE - Naprawa błędu "not defined")
# ==============================================================================
tickers = {
    'BTC': 'BTC-USD',
    'SP500': '^GSPC',  # Dodajemy kontekst giełdowy
    'NASDAQ': '^IXIC', 
    'OIL': 'CL=F'      # Dodajemy surowce
}

dfs = []
print("Pobieranie danych...")
for name, ticker in tickers.items():
    try:
        # Poberanie bez multiindexu
        d = yf.download(ticker, start="2014-01-01", end="2024-12-31", progress=False)
        if isinstance(d.columns, pd.MultiIndex): 
            d = d['Close']
        else: 
            d = d[['Close']]
        d.columns = [name]
        dfs.append(d)
    except Exception as e:
        print(f"Błąd pobierania {name}: {e}")

# Łączenie
if not dfs:
    raise ValueError("Nie udało się pobrać danych. Sprawdź połączenie z internetem.")
    
raw_data = pd.concat(dfs, axis=1).fillna(method='ffill').dropna()

# Feature Engineering
data = pd.DataFrame(index=raw_data.index)
# 1. Zwroty logarytmiczne (dla wszystkich aktywów)
for col in raw_data.columns:
    data[f'{col}_ret'] = np.log(raw_data[col] / raw_data[col].shift(1))

# 2. Target (Kierunek Ceny BTC Jutro)
data['target'] = np.where(raw_data['BTC'].shift(-PREDICTION_HORIZON) > raw_data['BTC'], 1, 0)
data.dropna(inplace=True)

# Wybór cech
features = [c for c in data.columns if c != 'target']
print(f"Cechy wejściowe ({len(features)}): {features}")

# Podział i Skalowanie
train_df = data.loc[:TRAIN_END_DATE]
val_df   = data.loc['2023-01-01':]

scaler = MinMaxScaler()
scaler.fit(train_df[features])

# Funkcja tworząca okna (zwraca numpy arrays)
def create_dataset_arrays(df, steps):
    if len(df) < steps:
        return np.array([]), np.array([])
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - steps):
        X.append(X_sc[i:(i + steps)])
        y.append(df['target'].iloc[i + steps])
    return np.array(X), np.array(y)

# --- ZMIENNE GLOBALNE (Tutaj naprawiamy błąd widoczności) ---
X_train, y_train = create_dataset_arrays(train_df, TIMESTEPS)
X_val, y_val     = create_dataset_arrays(val_df, TIMESTEPS)

print(f"Gotowe zestawy danych:\n X_train: {X_train.shape}\n X_val:   {X_val.shape}")

# ==============================================================================
# 3. OPTYMALIZATOR (Architektura CNN-LSTM z Artykułu)
# ==============================================================================
def objective(trial):
    # Zakresy hiperparametrów
    # Artykuł sugerował: Conv1D (16-32 filtry) -> LSTM (50 jednostek)
    
    # Warstwa CNN (Wydobywanie cech)
    filters = trial.suggest_categorical('filters', [16, 32, 64])
    kernel_size = trial.suggest_int('kernel_size', 2, 3) # Krótkie wzorce (2-3 dni)
    
    # Warstwa LSTM (Pamięć)
    lstm_units = trial.suggest_int('lstm_units', 32, 64, step=16)
    
    # Regularyzacja (Nasza lekcja "Anti-Overfit")
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float('lr', 0.0001, 0.002, log=True)
    reg_strength = trial.suggest_float('reg_strength', 1e-5, 1e-3, log=True)
    
    # Próg decyzyjny
    threshold_dist = trial.suggest_float('threshold_dist', 0.0, 0.05)

    # Budowa Modelu (Inspirowana Rysunkiem 1 z PDF)
    model = Sequential()
    model.add(Input(shape=(TIMESTEPS, len(features))))
    
    # 1. Conv1D - "Patrzenie na wykres"
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same', 
                     kernel_regularizer=l2(reg_strength)))
    model.add(MaxPooling1D(pool_size=2))
    
    # 2. LSTM - "Pamiętanie sekwencji" (Odbiera sygnał z CNN)
    model.add(LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2(reg_strength)))
    
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    # 3. Decyzja
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(reg_strength)))
    model.add(Dense(1, activation='sigmoid'))

    # Optimizer (Paper używał Adama dla CNN-LSTM)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

    # Trening
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=40, 
              batch_size=64, # Paper używał 64
              callbacks=[es], 
              verbose=0,
              shuffle=False)

    # Ewaluacja
    preds = model.predict(X_val, verbose=0).flatten()
    
    # Logika Snajpera
    upper, lower = 0.5 + threshold_dist, 0.5 - threshold_dist
    signals = np.where(preds > upper, 1, np.where(preds < lower, -1, 0))
    active_mask = (signals != 0)
    
    if np.sum(active_mask) == 0: return 0.0
        
    win_rate = accuracy_score(y_val[active_mask], (preds[active_mask] > 0.5).astype(int)) * 100
    action_rate = np.mean(active_mask)

    # Constraint: Musi grać min 10% czasu
    if action_rate < 0.10: return 0.0
    
    # Dodatkowe zabezpieczenie przed overfittingiem (różnica Train vs Val)
    # Można dodać, jeśli nadal wynik będzie 50%
    
    trial.set_user_attr("Action_Rate", action_rate)
    return win_rate

# ==============================================================================
# 4. URUCHOMIENIE
# ==============================================================================
if __name__ == "__main__":
    if len(X_train) == 0:
        print("BŁĄD: Pusty zbiór treningowy. Sprawdź daty i pobieranie danych.")
    else:
        print("Rozpoczynam poszukiwanie parametrów dla hybrydy CNN-LSTM...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        
        best = study.best_trial
        print("\n" + "="*50)
        print(">>> ZWYCIĘZCA (CNN-LSTM HYBRID) <<<")
        print(f"FILTERS       = {best.params['filters']}")
        print(f"KERNEL_SIZE   = {best.params['kernel_size']}")
        print(f"LSTM_UNITS    = {best.params['lstm_units']}")
        print(f"DROPOUT       = {best.params['dropout']:.4f}")
        print(f"LEARNING_RATE = {best.params['lr']:.5f}")
        print(f"REG_STRENGTH  = {best.params['reg_strength']:.6f}")
        print(f"THRESH_DIST   = {best.params['threshold_dist']:.4f}")
        print("="*50)
        print(f"Wynik Val WR: {best.value:.2f}% (Aktywność: {best.user_attrs['Action_Rate']*100:.1f}%)")