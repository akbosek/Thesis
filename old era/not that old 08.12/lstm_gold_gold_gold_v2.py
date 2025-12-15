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
# A. KONFIGURACJA (CENTER SHIFT + CLASS WEIGHTS)
# ==============================================================================
NEURONS       = 64
DROPOUT       = 0.3
LEARNING_RATE = 0.001
REG_STRENGTH  = 0.0001
USE_CNN       = True
FILTERS       = 32
OPTIMIZER     = 'Adam'

# --- KALIBRACJA SYGNAŁÓW ---
THRESH_DIST   = 0.02   # Margines "strefy ciszy" (+/- 2%)
MODEL_CENTER  = 0.53   # Przesunięcie środka (Bo model jest optymistą)

# Jak to działa?
# BUY  > 0.53 + 0.02 = 0.55
# SELL < 0.53 - 0.02 = 0.51 (Dzięki temu łatwiej zagrać Shorta)

# --- WAGI KLAS (TRENING) ---
USE_CLASS_WEIGHTS  = True
SHORT_WEIGHT_BOOST = 1.5   # Spadki są o 50% ważniejsze dla funkcji błędu

# Reszta bez zmian
TIMESTEPS     = 30
N_MODELS      = 5
EPOCHS        = 50
BATCH_SIZE    = 64
DATA_FILE     = 'data_master_v1.csv'

# ==============================================================================
# B. ŁADOWANIE DANYCH
# ==============================================================================
print("--- [FINAL EXECUTOR: CENTER SHIFT EDITION] ---")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Brak pliku {DATA_FILE}! Uruchom najpierw skrypt 'data_downloader.py'.")

print(f"Wczytuję dane z {DATA_FILE}...")
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

features = [c for c in data.columns if c not in ['target', 'BTC_price']]
print(f"Cechy wejściowe: {features}")

TRAIN_END = '2022-12-31'
train_df = data.loc[:TRAIN_END].copy()
val_df   = data.loc['2023-01-01':].copy()

scaler = MinMaxScaler()
scaler.fit(train_df[features])

def create_dataset(df):
    if len(df) < TIMESTEPS: return np.array([]), np.array([]), np.array([])
    X_sc = scaler.transform(df[features])
    X, y = [], []
    ret_col = [c for c in df.columns if 'BTC' in c and 'ret' in c][0]
    rets = df[ret_col].shift(-1).fillna(0).values
    
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
        
    return np.array(X), np.array(y), rets[TIMESTEPS : len(X_sc)]

X_train, y_train, r_train = create_dataset(train_df)
X_val, y_val, r_val       = create_dataset(val_df)

# --- WAGI KLAS ---
if USE_CLASS_WEIGHTS:
    print("\n⚖️  Obliczam wagi klas...")
    y_flat = y_train.flatten()
    weights = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
    class_weights_dict = {
        0: weights[0] * SHORT_WEIGHT_BOOST, 
        1: weights[1]
    }
    print(f"   Wagi: Spadek (0)={class_weights_dict[0]:.2f}, Wzrost (1)={class_weights_dict[1]:.2f}")
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
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, 
              batch_size=BATCH_SIZE, callbacks=[es], verbose=0, shuffle=True,
              class_weight=class_weights_dict)
    models.append(model)
    print(f" -> Model {i+1} gotowy.")

# ==============================================================================
# D. FUNKCJA EVALUATE (Z MODEL_CENTER)
# ==============================================================================
def calculate_safe_metrics(X, y_true, returns, name):
    print(f"   > Obliczam metryki dla: {name}...")
    try:
        preds_probs = np.mean([m.predict(X, verbose=0).flatten() for m in models], axis=0)
        
        # --- LOGIKA Z PRZESUNIĘCIEM ŚRODKA ---
        upper = MODEL_CENTER + THRESH_DIST
        lower = MODEL_CENTER - THRESH_DIST
        
        signals = np.where(preds_probs > upper, 1, np.where(preds_probs < lower, -1, 0))
        active_mask = (signals != 0)
        
        n_trades = int(np.sum(active_mask))
        action_rate = n_trades / len(y_true) if len(y_true) > 0 else 0
        
        if n_trades == 0:
            win_rate = 0.0
        else:
            # Predykcja dla aktywnych (zgodna z logiką sygnału)
            # Jeśli sygnał=1 (Long), to przewidujemy 1. Jeśli -1 (Short), to 0.
            predicted_class = np.where(signals[active_mask] == 1, 1, 0)
            win_rate = accuracy_score(y_true[active_mask], predicted_class) * 100
            
        try:
            strat_rets = signals * returns
            sharpe = (np.mean(strat_rets)/np.std(strat_rets))*np.sqrt(365) if np.std(strat_rets)!=0 else 0
        except: sharpe = 0.0
            
        try:
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, preds_probs)
                gini = 2 * auc - 1
            else: gini = 0.0
        except: gini = 0.0

        return {"Period": name, "Win Rate": win_rate, "Action Rate": action_rate*100, 
                "Trades": n_trades, "Sharpe": sharpe, "Gini": gini}
        
    except Exception as e:
        print(f"!!! Błąd: {e}")
        return {"Period": name, "Win Rate": 0, "Action Rate": 0, "Trades": 0, "Sharpe": 0, "Gini": 0}

print("\n" + "="*80)
print(f"   RAPORT KOŃCOWY (CENTER: {MODEL_CENTER} | DIST: {THRESH_DIST})   ")
print("="*80)

stats_train = calculate_safe_metrics(X_train, y_train, r_train, "TRAINING")
stats_val   = calculate_safe_metrics(X_val, y_val, r_val, "VALIDATION")

results_df = pd.DataFrame([stats_train, stats_val])
print_df = results_df.copy()
print_df['Win Rate'] = print_df['Win Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Action Rate'] = print_df['Action Rate'].apply(lambda x: f"{x:.2f}%")
print_df['Sharpe'] = print_df['Sharpe'].apply(lambda x: f"{x:.2f}")
print_df['Gini'] = print_df['Gini'].apply(lambda x: f"{x:.3f}")

print(print_df.to_string(index=False))
print("-" * 80)

# ==============================================================================
# F. WYKRESY (ZGODNE Z MODEL_CENTER)
# ==============================================================================
def generate_presentation_material():
    print("\n--- [GENEROWANIE WYKRESÓW] ---")
    dates = val_df.index[TIMESTEPS:]
    raw_preds = np.mean([m.predict(X_val, verbose=0).flatten() for m in models], axis=0)
    
    # --- LOGIKA Z PRZESUNIĘCIEM ---
    upper = MODEL_CENTER + THRESH_DIST
    lower = MODEL_CENTER - THRESH_DIST
    signals = np.where(raw_preds > upper, 1, np.where(raw_preds < lower, -1, 0))
    
    price_col = 'BTC_price' if 'BTC_price' in val_df.columns else 'BTC'
    close_prices = val_df[price_col].iloc[TIMESTEPS:].values
    
    # 1. Equity Curve
    strat_log_returns = signals * r_val
    cum_market = np.cumsum(r_val)
    cum_strategy = np.cumsum(strat_log_returns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cum_strategy, label='Model AI', color='green', linewidth=2)
    plt.plot(dates, cum_market, label='Bitcoin', color='gray', linestyle='--', alpha=0.6)
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
    
    plt.title(f'Decyzje Modelu (Center={MODEL_CENTER})', fontsize=14)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('Wykres_2_Sygnaly.png')
    print("✅ Wykres 2")

    # 3. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_preds, bins=50, kde=True, color='purple')
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.axvline(x=upper, color='green', linestyle=':', label=f'Long ({upper:.2f})')
    plt.axvline(x=lower, color='red', linestyle=':', label=f'Short ({lower:.2f})')
    plt.title('Histogram Pewności', fontsize=14)
    plt.legend(); plt.tight_layout()
    plt.savefig('Wykres_3_Histogram.png')
    print("✅ Wykres 3")

    # 4. Macierz
    active_mask = signals != 0
    if np.sum(active_mask) > 0:
        pred_classes = np.where(signals[active_mask] == 1, 1, 0)
        true_classes = y_val[active_mask]
        cm = confusion_matrix(true_classes, pred_classes, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spadek', 'Wzrost'], yticklabels=['Spadek', 'Wzrost'])
        plt.title('Macierz Trafień', fontsize=14)
        plt.xlabel('Przewidywanie'); plt.ylabel('Rynek'); plt.tight_layout()
        plt.savefig('Wykres_4_Macierz.png')
        print("✅ Wykres 4")

generate_presentation_material()