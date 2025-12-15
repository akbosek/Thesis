import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adamax

# --- KONFIGURACJA CISZY ---
optuna.logging.set_verbosity(optuna.logging.INFO)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# 1. KONFIGURACJA STEROWANIA
# ==============================================================================
DATA_FILE      = 'bitcoin_2018_feb_data.csv'
VAL_START_DATE = '2022-01-01'
OUTPUT_FILE    = 'WYNIKI_OPTUNA_SMART.csv' # Plik służy do ZAPISU i ODCZYTU
NUM_TRIALS     = 1000  # Ustawiamy dużo, bo i tak możesz przerwać Ctrl+C
EPOCHS_OPTUNA  = 40    # 30 epok wystarczy do testów

# ==============================================================================
# 2. PRZYGOTOWANIE DANYCH
# ==============================================================================
if not os.path.exists(DATA_FILE): raise FileNotFoundError(f"Brak pliku {DATA_FILE}")

df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
features = ['BTC_Close', 'BTC_Volume', 'Mayer_Ratio', 'RSI']
target = 'Target'

train_df = df[df.index < VAL_START_DATE].copy()
val_df   = df[df.index >= VAL_START_DATE].copy()

scaler = MinMaxScaler()
X_train_raw = scaler.fit_transform(train_df[features])
X_val_raw   = scaler.transform(val_df[features])
y_train_raw = train_df[target].values
y_val_raw   = val_df[target].values

cw = compute_class_weight('balanced', classes=np.unique(y_train_raw), y=y_train_raw)
cw_dict = {0: cw[0], 1: cw[1]}

def create_dataset(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)

# ==============================================================================
# 3. FUNKCJA ZAPISUJĄCA (CALLBACK)
# ==============================================================================
def save_trial_callback(study, trial):
    """Dopisuje wynik każdej próby do pliku CSV w czasie rzeczywistym."""
    try:
        trial_data = trial.params.copy()
        trial_data['number'] = trial.number
        trial_data['val_loss'] = trial.value
        trial_data['val_accuracy'] = trial.user_attrs.get('val_accuracy', 0.0)
        
        df_row = pd.DataFrame([trial_data])
        
        # Kolejność kolumn dla porządku
        cols_order = ['number', 'val_loss', 'val_accuracy', 'lookback', 'lstm_1', 'lstm_2', 'dense', 'dropout', 'lr', 'batch_size']
        cols_to_save = [c for c in cols_order if c in df_row.columns]
        df_row = df_row[cols_to_save]

        file_exists = os.path.isfile(OUTPUT_FILE)
        df_row.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)
    except Exception as e:
        print(f"[ERROR] Błąd zapisu do CSV: {e}")

# ==============================================================================
# 4. FUNKCJA IMPORTUJĄCA (SEEDING) - TO JEST IMPORT!
# ==============================================================================
def load_best_trials_from_csv(study, filename):
    """Wczytuje najlepsze wyniki z pliku i ustawia je jako punkt startowy."""
    if not os.path.exists(filename):
        print(f"[INFO] Plik {filename} nie istnieje. Zaczynam czyste badanie.")
        return

    try:
        print(f"[INFO] Analizuję plik {filename}...")
        df_old = pd.read_csv(filename)
        
        if 'val_loss' not in df_old.columns:
            print("[INFO] Plik istnieje, ale brak kolumny 'val_loss'. Pomijam import.")
            return

        # Sortujemy od najmniejszego Loss (najlepsze modele)
        # Bierzemy TOP 15 unikalnych konfiguracji
        top_trials = df_old.sort_values(by='val_loss', ascending=True).head(15)
        
        count = 0
        for _, row in top_trials.iterrows():
            params = {}
            # Mapowanie kolumn CSV -> Parametry Optuny
            mapping = {
                'lookback': int, 'lstm_1': int, 'lstm_2': int, 'dense': int, 
                'batch_size': int, 'dropout': float, 'lr': float
            }
            
            valid = True
            for key, dtype in mapping.items():
                # Obsługa nazw kolumn (czasem CSV ma 'params_lr', czasem 'lr')
                # Sprawdzamy oba warianty
                val = None
                if key in row: val = row[key]
                elif f"params_{key}" in row: val = row[f"params_{key}"]
                
                if val is not None:
                    params[key] = dtype(val)
                else:
                    valid = False # Brak parametru w CSV
            
            if valid:
                study.enqueue_trial(params)
                count += 1
        
        print(f"[SUKCES] Zaimportowano {count} najlepszych konfiguracji z przeszłości jako 'Warm Start'.")
        print("Optuna sprawdzi je ponownie, a potem zacznie szukać w ich okolicy.\n")

    except Exception as e:
        print(f"[WARNING] Błąd importu pliku (zignorowano): {e}")

# ==============================================================================
# 5. CEL (OBJECTIVE)
# ==============================================================================
def objective(trial):
    # Zakresy parametrów
    p_lookback = trial.suggest_categorical('lookback', [14, 21, 28])
    p_lstm_1   = trial.suggest_int('lstm_1', 32, 192, step=16)
    p_lstm_2   = trial.suggest_int('lstm_2', 16, 96, step=8)
    p_dense    = trial.suggest_int('dense', 8, 48, step=4)
    p_dropout  = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    
    # Zmieniony zakres LR na 1e-5 (pomoc w wyjściu z 0.693)
    p_lr       = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    p_batch    = trial.suggest_categorical('batch_size', [16, 24, 32, 40, 48, 56, 64])

    X_train, y_train = create_dataset(X_train_raw, y_train_raw, p_lookback)
    X_val, y_val     = create_dataset(X_val_raw, y_val_raw, p_lookback)

    model = Sequential([
        Input(shape=(p_lookback, len(features))),
        LSTM(p_lstm_1, return_sequences=True, kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(p_dropout),
        LSTM(p_lstm_2, return_sequences=False, kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(p_dropout),
        Dense(p_dense, activation='relu', kernel_regularizer=l2(0.005)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adamax(learning_rate=p_lr), loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    model.fit(
        X_train, y_train, epochs=EPOCHS_OPTUNA, batch_size=p_batch,
        validation_data=(X_val, y_val), callbacks=[es],
        class_weight=cw_dict, verbose=0
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    trial.set_user_attr("val_accuracy", val_acc)
    return val_loss

# ==============================================================================
# 6. URUCHOMIENIE
# ==============================================================================
print("--- START SYSTEMU OPTYMALIZACJI ---")
print(f"Plik bazy wiedzy: {OUTPUT_FILE}")
print("Aby przerwać i zapisać, naciśnij CTRL+C w konsoli.")

# Tworzymy badanie
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

# IMPORT DANYCH (Jeśli plik istnieje)
load_best_trials_from_csv(study, OUTPUT_FILE)

try:
    # Uruchomienie z zapisywaniem ciągłym
    study.optimize(objective, n_trials=NUM_TRIALS, callbacks=[save_trial_callback])
    
except KeyboardInterrupt:
    print("\n\n[!!!] PRZERWANO PRZEZ UŻYTKOWNIKA (CTRL+C) [!!!]")
    print("Kończę obecny proces i zapisuję stan...")

# ==============================================================================
# 7. PODSUMOWANIE
# ==============================================================================
if len(study.trials) > 0:
    best = study.best_params
    best_loss = study.best_value
    best_acc = study.best_trial.user_attrs.get('val_accuracy', 0.0)

    print("\n==================================================")
    print(f"LIDER RANKINGU (Najniższy Loss):")
    print(f" > Val_Loss:     {best_loss:.4f}")
    print(f" > Val_Accuracy: {best_acc:.2%}")
    print(f" > Parametry:    {best}")
    print("==================================================")
    
    print("\nGOTOWY CONFIG DO WKLEJENIA:")
    print(f"HP_LOOKBACK   = {best['lookback']}")
    print(f"HP_LSTM_1     = {best['lstm_1']}")
    print(f"HP_LSTM_2     = {best['lstm_2']}")
    print(f"HP_DENSE      = {best['dense']}")
    print(f"HP_DROPOUT    = {best['dropout']}")
    print(f"HP_LR         = {best['lr']:.6f}")
    print(f"HP_BATCH_SIZE = {best['batch_size']}")
else:
    print("Brak wyników.")