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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ==========================================
# 1. KONFIGURACJA (REGULARYZACJA)
# ==========================================
DATA_FILE      = 'bitcoin_2018_feb_data.csv'
VAL_START_DATE = '2022-01-01'
OUTPUT_DIR     = 'RAPORT_REGULARIZED'

HP_SEED        = 42
HP_LOOKBACK    = 30
HP_EPOCHS      = 150
HP_BATCH_SIZE  = 32
HP_LR          = 0.0005

# ZMIANA 1: Mocniejsza regularyzacja (trudniej wkuwać na pamięć)
HP_DROPOUT     = 0.4     
HP_L2          = 0.002   

# ZMIANA 2: Martwa Strefa (Dead Zone)
THRESH_LONG    = 0.52    # Kupuj tylko jak jesteś pewien > 52%
THRESH_SHORT   = 0.48    # Sprzedawaj tylko jak jesteś pewien < 48%
# Pomiędzy 0.48 a 0.52 model nie robi nic (Cash)

os.environ['PYTHONHASHSEED'] = str(HP_SEED)
random.seed(HP_SEED)
np.random.seed(HP_SEED)
tf.random.set_seed(HP_SEED)

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. DANE
# ==========================================
if not os.path.exists(DATA_FILE): raise FileNotFoundError("Brak pliku danych!")
df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

features = ['BTC_Close', 'BTC_Volume', 'Mayer_Ratio', 'RSI', 'SPX', 'Oil', 'VIX', 'FNG']
target = 'Target'

train_df = df[df.index < VAL_START_DATE].copy()
val_df   = df[df.index >= VAL_START_DATE].copy()

print(f" > Trening: {len(train_df)} dni")
print(f" > Walidacja: {len(val_df)} dni")

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

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {0: class_weights[0], 1: class_weights[1]}

# ==========================================
# 3. MODEL (LŻEJSZY I BARDZIEJ KRYTYCZNY)
# ==========================================
model = Sequential([
    Input(shape=(HP_LOOKBACK, len(features))),
    
    # Mniej neuronów (32 zamiast 64) + Większy Dropout
    LSTM(32, return_sequences=True, kernel_regularizer=l2(HP_L2)),
    BatchNormalization(),
    Dropout(HP_DROPOUT),
    
    LSTM(16, return_sequences=False, kernel_regularizer=l2(HP_L2)),
    BatchNormalization(),
    Dropout(HP_DROPOUT),
    
    Dense(8, activation='relu', kernel_regularizer=l2(HP_L2)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=HP_LR), loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. TRENING
# ==========================================
print("\n--- [START TRENINGU] ---")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=HP_EPOCHS,
    batch_size=HP_BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    class_weight=cw_dict,
    verbose=1
)

# ==========================================
# 5. RAPORTOWANIE Z "MARTWĄ STREFĄ"
# ==========================================
def generate_report(X, y_true, prices, name):
    print(f"\n>>> RAPORT: {name} <<<")
    probs = model.predict(X, verbose=0).flatten()
    
    # --- LOGIKA DECYZYJNA (Dead Zone) ---
    # 1 = Long, 0 = Short/Cash, -1 = Brak decyzji (Flat)
    actions = []
    for p in probs:
        if p > THRESH_LONG:
            actions.append(1) # Long
        elif p < THRESH_SHORT:
            actions.append(0) # Short/Cash
        else:
            actions.append(-1) # Flat (nie robimy nic)
    actions = np.array(actions)
    
    # Metryki liczymy tylko dla podjętych decyzji (ignorujemy Flat)
    active_mask = actions != -1
    if np.sum(active_mask) > 0:
        acc = accuracy_score(y_true[active_mask], actions[active_mask])
        cm = confusion_matrix(y_true[active_mask], actions[active_mask])
    else:
        acc = 0
        cm = np.zeros((2,2))

    fpr, tpr, _ = roc_curve(y_true, probs)
    gini = 2 * auc(fpr, tpr) - 1
    
    # Equity Curve (%)
    equity = [100.0]
    market = [100.0]
    
    real_returns = prices.pct_change().shift(-1).dropna()
    min_len = min(len(actions), len(real_returns))
    
    # Wyrównanie
    active_acts  = actions[:min_len]
    active_rets  = real_returns.values[:min_len]
    dates        = real_returns.index[:min_len]
    active_prices= prices.values[:min_len]
    
    trades_count = 0
    for i in range(min_len):
        ret = active_rets[i]
        act = active_acts[i]
        
        if act == 1:   # Long
            equity.append(equity[-1] * (1 + ret))
            trades_count += 1
        elif act == 0: # Cash (lub Short)
            equity.append(equity[-1]) 
            trades_count += 1
        else:          # Flat (Dead Zone)
            equity.append(equity[-1]) 
            
        market.append(market[-1] * (1 + ret))
        
    final_eq = equity[-1] - 100.0
    final_mkt = market[-1] - 100.0
    
    print(f" > Accuracy (Aktywne): {acc:.2%}")
    print(f" > Gini:     {gini:.4f}")
    print(f" > Zysk:     {final_eq:.2f}% (Rynek: {final_mkt:.2f}%)")
    print(f" > Akcja:    {trades_count}/{len(actions)} dni ({trades_count/len(actions):.1%} czasu na rynku)")

    # WYKRESY
    # 1. Equity
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity[1:], label=f'Model ({final_eq:.0f}%)', color='green', linewidth=2)
    plt.plot(dates, market[1:], label=f'Rynek ({final_mkt:.0f}%)', color='gray', linestyle='--', alpha=0.7)
    plt.title(f'{name} - Equity Curve (Selective)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{name}_1_Equity.png")
    plt.close()
    
    # 2. Sygnały (Tylko aktywne)
    plt.figure(figsize=(14, 7))
    plt.plot(dates, active_prices, color='black', alpha=0.5, label='Cena')
    buy_idx = [i for i, x in enumerate(active_acts) if x == 1]
    sell_idx = [i for i, x in enumerate(active_acts) if x == 0]
    
    if len(buy_idx)>0: plt.scatter(dates[buy_idx], active_prices[buy_idx], marker='^', color='green', label='Long')
    if len(sell_idx)>0: plt.scatter(dates[sell_idx], active_prices[sell_idx], marker='v', color='red', label='Cash')
    plt.title(f'{name} - Sygnały (Filtrowane)')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{name}_2_Signals.png")
    plt.close()
    
    # 3. Lorenz Curve (ROC) - O TO PROSIŁEŚ
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC (Gini={gini:.2f})', color='purple', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - Krzywa Lorenza / ROC')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{name}_4_Lorenz_ROC.png")
    plt.close()
    
    # 4. Histogram Pewności (z liniami progów)
    plt.figure(figsize=(8, 4))
    plt.hist(probs, bins=50, color='purple', alpha=0.7)
    plt.axvline(THRESH_LONG, color='green', linestyle='--', label='Próg Long')
    plt.axvline(THRESH_SHORT, color='red', linestyle='--', label='Próg Short')
    plt.title(f'{name} - Rozkład Pewności')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{name}_5_Confidence.png")
    plt.close()

train_prices = train_df['BTC_Close'].iloc[HP_LOOKBACK:]
val_prices   = val_df['BTC_Close'].iloc[HP_LOOKBACK:]

generate_report(X_train, y_train, train_prices, "TRENING")
generate_report(X_val, y_val, val_prices, "WALIDACJA")

print(f"\n✅ ZAKOŃCZONO. Wyniki w folderze: {OUTPUT_DIR}")