import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, Bidirectional, Conv1D, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# ==============================================================================
# A. KONFIGURACJA ZWYCIĘZCY (TUTAJ WKLEJ WYNIKI Z OPTUNY)
# ==============================================================================
# Parametry Modelu (Zastąp swoimi z Optuny)
NEURONS       = 64
DROPOUT       = 0.3
LEARNING_RATE = 0.001
REG_STRENGTH  = 0.0001
USE_CNN       = True
FILTERS       = 32
OPTIMIZER     = 'Adam'  # 'Adam' lub 'Adamax'

# --- METODA 1: ASYMETRYCZNE PROGI (Kalibracja) ---
# Skoro model jest "optymistą" (średnia > 0.5), podnosimy poprzeczkę dla Longów
# i ułatwiamy Shorty.
THRESH_LONG   = 0.55    # Musi być > 0.55 żeby kupić (pewny wzrost)
THRESH_SHORT  = 0.51    # <--- KLUCZOWE: Jeśli spadnie poniżej 0.51, gramy SHORT

# --- METODA 2: WAGI KLAS (Trening) ---
USE_CLASS_WEIGHTS  = True  # Czy włączyć "karanie" za ignorowanie spadków?
SHORT_WEIGHT_BOOST = 2.0   # Mnożnik: Jak bardzo ważniejsze są spadki? (2.0 = 2x ważniejsze)

# Stałe systemowe
TIMESTEPS     = 30
N_MODELS      = 5
EPOCHS        = 50
BATCH_SIZE    = 64
DATA_FILE     = 'data_master_v1.csv'

# ==============================================================================
# B. ŁADOWANIE DANYCH
# ==============================================================================
print("--- [FINAL EXECUTOR: BALANCED EDITION] ---")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Brak pliku {DATA_FILE}! Uruchom najpierw skrypt 'data_downloader.py'.")

print(f"Wczytuję dane z {DATA_FILE}...")
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

# Definicja cech
features = [c for c in data.columns if c not in ['target', 'BTC_price']]
print(f"Cechy wejściowe: {features}")

# Podział Chronologiczny
TRAIN_END = '2022-12-31'
train_df = data.loc[:TRAIN_END].copy()
val_df   = data.loc['2023-01-01':].copy()

# Skalowanie
scaler = MinMaxScaler()
scaler.fit(train_df[features])

def create_dataset(df):
    if len(df) < TIMESTEPS: return np.array([]), np.array([]), np.array([])
    X_sc = scaler.transform(df[features])
    X, y = [], []
    
    # Szukanie kolumny zwrotów do Sharpe Ratio
    ret_col = [c for c in df.columns if 'BTC' in c and 'ret' in c][0]
    rets = df[ret_col].shift(-1).fillna(0).values
    
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
        
    return np.array(X), np.array(y), rets[TIMESTEPS : len(X_sc)]

X_train, y_train, r_train = create_dataset(train_df)
X_val, y_val, r_val       = create_dataset(val_df)

print(f"Dane Treningowe: {X_train.shape}")

# --- OBLICZANIE WAG KLAS (METODA 2) ---
if USE_CLASS_WEIGHTS:
    print("\n⚖️  Obliczam wagi klas (Balansowanie)...")
    y_flat = y_train.flatten()
    # Obliczamy wagi tak, by klasy były równe
    weights = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
    
    # Tworzymy słownik wag
    # weights[0] to waga dla spadków, weights[1] dla wzrostów
    class_weights_dict = {
        0: weights[0] * SHORT_WEIGHT_BOOST, # Podbijamy wagę spadków!
        1: weights[1]
    }
    print(f"   Wagi: Spadek (0) = {class_weights_dict[0]:.2f}, Wzrost (1) = {class_weights_dict[1]:.2f}")
else:
    class_weights_dict = None

# ==============================================================================
# C. TRENING ENSEMBLE
# ==============================================================================
def build_model():
    input_layer = Input(shape=(TIMESTEPS, len(features)))
    x = input_layer
    
    if USE_CNN:
        x = Conv1D(filters=FILTERS, kernel_size=2, activation='relu', padding='same', 
                   kernel_regularizer=l2(REG_STRENGTH))(x)
    
    lstm_out = Bidirectional(LSTM(NEURONS, return_sequences=True, 
                                  kernel_regularizer=l2(REG_STRENGTH)))(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(DROPOUT)(lstm_out)
    
    attention_out = Attention()([lstm_out, lstm_out])
    context = GlobalAveragePooling1D()(attention_out)
    
    dense = Dense(16, activation='relu', kernel_regularizer=l2(REG_STRENGTH))(context)
    output = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=input_layer, outputs=output)
    
    opt = Adam(learning_rate=LEARNING_RATE) if OPTIMIZER == 'Adam' else Adamax(learning_rate=LEARNING_RATE)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

models = []
print(f"\nRozpoczynam trening zespołu {N_MODELS} modeli...")

for i in range(N_MODELS):
    model = build_model()
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0)
    
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              callbacks=[es],
              verbose=0, 
              shuffle=True,
              class_weight=class_weights_dict) # <--- APLIKACJA WAG (METODA 2)
              
    models.append(model)
    print(f" -> Model {i+1} gotowy.")

# ==============================================================================
# D. ANALIZA Z WYNIKAMI PARYSKIMI (METODA 1: PROGI)
# ==============================================================================
def calculate_metrics_asymmetric(X, y_true, returns, name):
    # Predykcja Ensemble
    preds_probs = np.mean([m.predict(X, verbose=0).flatten() for m in models], axis=0)
    
    # --- LOGIKA ASYMETRYCZNA (METODA 1) ---
    signals = np.where(preds_probs > THRESH_LONG, 1, 
              np.where(preds_probs < THRESH_SHORT, -1, 0))
    # --------------------------------------
    
    active_mask = (signals != 0)
    n_trades = int(np.sum(active_mask))
    action_rate = n_trades / len(y_true) if len(y_true) > 0 else 0
    
    # Win Rate
    if n_trades == 0:
        win_rate = 0.0
    else:
        # Dla aktywnych: Jeśli signal=1 to szukamy 1, jeśli signal=-1 to szukamy 0
        # Konwersja sygnałów na oczekiwany target (1 lub 0)
        # Signal 1 -> Target 1
        # Signal -1 -> Target 0
        predicted_target = np.where(signals[active_mask] == 1, 1, 0)
        actual_target = y_true[active_mask]
        
        win_rate = accuracy_score(actual_target, predicted_target) * 100
            
    # Sharpe
    try:
        strat_rets = signals * returns
        if np.std(strat_rets) == 0: sharpe = 0.0
        else: sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(365)
    except: sharpe = 0.0
            
    # Gini
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, preds_probs)
            gini = 2 * auc - 1
        else: gini = 0.0
    except: gini = 0.0

    return {
        "Period": name, "Win Rate": win_rate, "Action Rate": action_rate * 100,
        "Trades": n_trades, "Sharpe": sharpe, "Gini": gini
    }

print("\n" + "="*80)
print(f"   RAPORT KOŃCOWY (PROGI: Long>{THRESH_LONG} | Short<{THRESH_SHORT})   ")
print("="*80)

stats_train = calculate_metrics_asymmetric(X_train, y_train, r_train, "TRAINING")
stats_val   = calculate_metrics_asymmetric(X_val, y_val, r_val, "VALIDATION")

results_df = pd.DataFrame([stats_train, stats_val])
print_df = results_df.copy()
print_df['Win Rate'] = print_df['Win Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Action Rate'] = print_df['Action Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Sharpe'] = print_df['Sharpe'].apply(lambda x: f"{x:.2f}")
print_df['Gini'] = print_df['Gini'].apply(lambda x: f"{x:.3f}")

print(print_df.to_string(index=False))
print("-" * 80)

# ==============================================================================
# F. WYKRESY DLA PREZENTACJI (NAPRAWIONE)
# ==============================================================================
def generate_presentation_material():
    print("\n--- [GENEROWANIE WYKRESÓW] ---")
    dates = val_df.index[TIMESTEPS:]
    raw_preds = np.mean([m.predict(X_val, verbose=0).flatten() for m in models], axis=0)
    
    # Logika asymetryczna
    signals = np.where(raw_preds > THRESH_LONG, 1, 
              np.where(raw_preds < THRESH_SHORT, -1, 0))
    
    # --- NAPRAWA BŁĘDU BTC ---
    # Szukamy kolumny z ceną (zwykle 'BTC' lub 'BTC_price')
    price_col = 'BTC_price' if 'BTC_price' in val_df.columns else 'BTC'
    close_prices = val_df[price_col].iloc[TIMESTEPS:].values
    
    # 1. Equity Curve
    strat_log_returns = signals * r_val
    cum_market = np.cumsum(r_val)
    cum_strategy = np.cumsum(strat_log_returns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cum_strategy, label='AI Model (Long+Short)', color='green', linewidth=2)
    plt.plot(dates, cum_market, label='Bitcoin (Buy&Hold)', color='gray', linestyle='--', alpha=0.6)
    plt.title(f'Equity Curve (Sharpe: {stats_val["Sharpe"]:.2f})', fontsize=14)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('Wykres_1_Equity_Curve.png')
    print("✅ Wykres 1")

    # 2. Sygnały
    plt.figure(figsize=(14, 7))
    plt.plot(dates, close_prices, label='Cena BTC', color='black', alpha=0.5)
    longs = signals == 1
    shorts = signals == -1
    
    if np.sum(longs) > 0:
        plt.scatter(dates[longs], close_prices[longs], marker='^', color='green', s=80, label='Long', zorder=5)
    if np.sum(shorts) > 0:
        plt.scatter(dates[shorts], close_prices[shorts], marker='v', color='red', s=80, label='Short', zorder=5)
    
    plt.title('Decyzje Modelu (Zielone=Kupno, Czerwone=Sprzedaż)', fontsize=14)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('Wykres_2_Sygnaly.png')
    print("✅ Wykres 2 (Sprawdź czy są czerwone trójkąty!)")

    # 3. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_preds, bins=50, kde=True, color='purple')
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.axvline(x=THRESH_LONG, color='green', linestyle=':', label='Próg Long')
    plt.axvline(x=THRESH_SHORT, color='red', linestyle=':', label='Próg Short')
    plt.title('Histogram Pewności Modelu', fontsize=14)
    plt.legend(); plt.tight_layout()
    plt.savefig('Wykres_3_Histogram.png')
    print("✅ Wykres 3")

    # 4. Macierz (Dla transakcji)
    active_mask = signals != 0
    if np.sum(active_mask) > 0:
        # Konwersja sygnałów na klasy: 1 (Long) -> 1 (Wzrost), -1 (Short) -> 0 (Spadek)
        # Celem jest sprawdzenie czy trafiliśmy kierunek
        pred_classes = np.where(signals[active_mask] == 1, 1, 0)
        true_classes = y_val[active_mask]
        
        cm = confusion_matrix(true_classes, pred_classes, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Spadek (0)', 'Wzrost (1)'], 
                    yticklabels=['Spadek (0)', 'Wzrost (1)'])
        plt.title('Trafność Decyzji', fontsize=14)
        plt.xlabel('Przewidywanie Modelu'); plt.ylabel('Faktyczny Rynek')
        plt.tight_layout()
        plt.savefig('Wykres_4_Macierz.png')
        print("✅ Wykres 4")

generate_presentation_material()