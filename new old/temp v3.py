import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

# Importy Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adamax

# ==============================================================================
# 1. KONFIGURACJA HIPERPARAMETRÓW
# ==============================================================================

# --- PLIKI ---
DATA_FILE      = 'bitcoin_2018_feb_data.csv'
VAL_START_DATE = '2022-01-01'
OUTPUT_DIR     = 'RAPORT_BET_SIZING'  # Nowy folder na wyniki

# --- TRENING ---
HP_SEED        = 42
HP_EPOCHS      = 100
HP_BATCH_SIZE  = 16
HP_LR          = 0.001

# --- LOGIKA POZYCJI (NOWOŚĆ!) ---
# Model obliczy 'Środek' (Medianę) automatycznie na treningu.
# Tutaj ustawiasz tylko, jak szeroko od środka zaczynamy grać.

HP_THRESHOLD_OFFSET = 0.10  # O ile odsunąć się od mediany? 
                            # Jeśli Mediana=0.50, a Offset=0.10:
                            # LONG gramy od 0.60, SHORT gramy poniżej 0.40.
                            # Pomiędzy 0.40 a 0.60 nie robimy nic.

HP_MIN_POS_SIZE     = 0.20  # Minimalna wielkość pozycji (na granicy progu) - 20%
HP_MAX_POS_SIZE     = 1.00  # Maksymalna wielkość pozycji (przy pewności 100% lub 0%)

# --- ARCHITEKTURA ---
HP_LOOKBACK    = 30
HP_LSTM_1      = 128
HP_LSTM_2      = 64
HP_DENSE       = 16
HP_DROPOUT     = 0.2
HP_L2          = 0.005
HP_OPTIMIZER   = Adamax(learning_rate=HP_LR)

# ==============================================================================
# 2. INICJALIZACJA
# ==============================================================================
os.environ['PYTHONHASHSEED'] = str(HP_SEED)
random.seed(HP_SEED)
np.random.seed(HP_SEED)
tf.random.set_seed(HP_SEED)

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 3. DANE
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

def create_dataset(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_dataset(X_train_raw, y_train_raw, HP_LOOKBACK)
X_val, y_val     = create_dataset(X_val_raw, y_val_raw, HP_LOOKBACK)

cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {0: cw[0], 1: cw[1]}

# ==============================================================================
# 4. MODEL
# ==============================================================================
model = Sequential([
    Input(shape=(HP_LOOKBACK, len(features))),
    LSTM(HP_LSTM_1, return_sequences=True, kernel_regularizer=l2(HP_L2)),
    BatchNormalization(),
    Dropout(HP_DROPOUT),
    LSTM(HP_LSTM_2, return_sequences=False, kernel_regularizer=l2(HP_L2)),
    BatchNormalization(),
    Dropout(HP_DROPOUT),
    Dense(HP_DENSE, activation='relu', kernel_regularizer=l2(HP_L2)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=HP_OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])

# ==============================================================================
# 5. TRENING
# ==============================================================================
checkpoint = ModelCheckpoint(f"{OUTPUT_DIR}/best_model.keras", monitor='val_loss', save_best_only=True, verbose=0)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("\n--- [TRENING] ---")
model.fit(
    X_train, y_train, epochs=HP_EPOCHS, batch_size=HP_BATCH_SIZE,
    validation_data=(X_val, y_val), callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=cw_dict, verbose=1
)

# ==============================================================================
# 6. KALIBRACJA (OBLICZANIE ŚRODKA)
# ==============================================================================
print("\n--- [KALIBRACJA CENTRUM] ---")
train_probs = model.predict(X_train, verbose=0).flatten()
# Wyznaczamy środek układu jako medianę predykcji
CENTER_POINT = np.median(train_probs) 

# Wyliczamy granice
thresh_long  = CENTER_POINT + HP_THRESHOLD_OFFSET
thresh_short = CENTER_POINT - HP_THRESHOLD_OFFSET

print(f" > Mediana (Środek): {CENTER_POINT:.4f}")
print(f" > Strefa SHORT:     < {thresh_short:.4f}")
print(f" > Strefa NEUTRAL:   {thresh_short:.4f} - {thresh_long:.4f}")
print(f" > Strefa LONG:      > {thresh_long:.4f}")

# ==============================================================================
# 7. LOGIKA POZYCYJNA (FUNKCJA SKALUJĄCA)
# ==============================================================================
def calculate_position_size(prob, center, offset, min_size, max_size):
    """
    Oblicza wielkość pozycji (-1.0 do 1.0) w zależności od pewności modelu.
    """
    upper = center + offset
    lower = center - offset
    
    # LONG
    if prob > upper:
        # Jak daleko jesteśmy od progu w stronę 1.0?
        # Skalujemy od 0 (przy progu) do 1 (przy 1.0)
        dist_norm = (prob - upper) / (1.0 - upper + 1e-9)
        # Interpolacja liniowa między min_size a max_size
        size = min_size + (max_size - min_size) * dist_norm
        return min(size, max_size) # Cap na max_size
        
    # SHORT
    elif prob < lower:
        # Jak daleko jesteśmy od progu w stronę 0.0?
        dist_norm = (lower - prob) / (lower - 0.0 + 1e-9)
        size = min_size + (max_size - min_size) * dist_norm
        return -min(size, max_size) # Zwracamy ujemną wartość (Short)
        
    # NEUTRAL
    else:
        return 0.0

# Wektoryzacja funkcji, żeby działała szybko na całych tablicach
v_calc_pos = np.vectorize(calculate_position_size)

# ==============================================================================
# 8. RAPORTOWANIE
# ==============================================================================
def generate_report(X, y_true, prices, name, center):
    print(f"\n>>> RAPORT: {name} <<<")
    probs = model.predict(X, verbose=0).flatten()
    
    # 1. Obliczamy pozycje (zmienne wielkości: np. 0.2, 0.55, 1.0, -0.3...)
    positions = v_calc_pos(probs, center, HP_THRESHOLD_OFFSET, HP_MIN_POS_SIZE, HP_MAX_POS_SIZE)
    
    # 2. Symulacja Equity
    equity = [100.0]
    market = [100.0]
    real_returns = prices.pct_change().shift(-1).dropna()
    min_len = min(len(positions), len(real_returns))
    
    active_pos = positions[:min_len]
    active_rets = real_returns.values[:min_len]
    dates = real_returns.index[:min_len]
    
    for i in range(min_len):
        ret = active_rets[i]
        pos = active_pos[i] # Tutaj pos może być np. 0.5 (połowa kapitału)
        
        # Zysk = Zmiana ceny * Wielkość pozycji
        strategy_ret = pos * ret 
        
        equity.append(equity[-1] * (1 + strategy_ret))
        market.append(market[-1] * (1 + ret))
        
    final_eq = equity[-1] - 100.0
    final_mkt = market[-1] - 100.0
    
    # Statystyki pozycji
    long_count  = np.sum(active_pos > 0)
    short_count = np.sum(active_pos < 0)
    avg_long_size = np.mean(active_pos[active_pos > 0]) if long_count > 0 else 0
    avg_short_size = np.mean(abs(active_pos[active_pos < 0])) if short_count > 0 else 0

    print(f" > Wynik:      Model={final_eq:.2f}% (Rynek={final_mkt:.2f}%)")
    print(f" > Transakcje: L={long_count} (śr. wielkość {avg_long_size:.2f}), S={short_count} (śr. wielkość {avg_short_size:.2f})")
    
    # --- WYKRES 1: Equity ---
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity[1:], label=f'Model Scaled ({final_eq:.0f}%)', color='blue', lw=2)
    plt.plot(dates, market[1:], label=f'Rynek ({final_mkt:.0f}%)', color='gray', ls='--', alpha=0.6)
    plt.title(f'{name} - Krzywa Kapitału (Zmienna Pozycja)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{name}_1_Equity.png")
    plt.close()
    
    # --- WYKRES 2: Wielkość Pozycji w Czasie (Nowy!) ---
    plt.figure(figsize=(12, 4))
    plt.fill_between(dates, active_pos, 0, where=(active_pos>0), color='green', alpha=0.5, label='LONG Size')
    plt.fill_between(dates, active_pos, 0, where=(active_pos<0), color='red', alpha=0.5, label='SHORT Size')
    plt.axhline(0, color='black', lw=1)
    plt.title(f'{name} - Zaangażowanie Kapitału (-100% do +100%)')
    plt.ylabel('Wielkość Pozycji')
    plt.legend(loc='upper right')
    plt.savefig(f"{OUTPUT_DIR}/{name}_2_PositionSize.png")
    plt.close()

    # --- WYKRES 3: Histogram Pewności z Pasami ---
    plt.figure(figsize=(10, 5))
    plt.hist(probs, bins=50, color='orange', alpha=0.7, edgecolor='black')
    plt.axvline(center, color='black', linestyle='-', label='Mediana')
    plt.axvline(center + HP_THRESHOLD_OFFSET, color='green', linestyle='--', label='Start Long')
    plt.axvline(center - HP_THRESHOLD_OFFSET, color='red', linestyle='--', label='Start Short')
    plt.title(f'{name} - Rozkład Pewności z progami')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{name}_3_Confidence.png")
    plt.close()

train_prices = train_df['BTC_Close'].iloc[HP_LOOKBACK:]
val_prices   = val_df['BTC_Close'].iloc[HP_LOOKBACK:]

generate_report(X_train, y_train, train_prices, "TRENING", CENTER_POINT)
generate_report(X_val, y_val, val_prices, "WALIDACJA", CENTER_POINT)

print(f"\n✅ ZAKOŃCZONO. Sprawdź wykres 'PositionSize' w folderze {OUTPUT_DIR}.")