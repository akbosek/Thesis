import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler # ZMIANA: Lepszy do wykrywania odchyleń
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight # ZMIANA: Klucz do balansu

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==============================================================================
#      PANEL STEROWANIA (BALANCED MODE)
# ==============================================================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# OKRESY (STRICT MODE)
TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' 
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31'

# ENSEMBLE
N_MODELS = 5 

# PARAMETRY MODELU
TIMESTEPS = 14       
NEURONS   = 64       
DROPOUT   = 0.3      
LEARNING_RATE = 0.001
EPOCHS    = 40       
BATCH_SIZE = 32

# PROGI
THRESH_SHORT = 0.49
THRESH_LONG  = 0.51
# ==============================================================================

print(f"--- [LSTM BALANCED: CLASS WEIGHTS & SCALER] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# --- 1. DANE ---
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

# Dodajemy dzień tygodnia (Seasonality)
data['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
data['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)

data.dropna(inplace=True)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d', 'day_sin', 'day_cos']

# --- 2. PODZIAŁ ---
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

# --- 3. SKALOWANIE (StandardScaler - Zmiana kluczowa) ---
# StandardScaler centruje dane wokół 0. To pomaga LSTM widzieć "górki i dołki"
scaler = StandardScaler() 
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    future_returns = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    return np.array(X), np.array(y), future_returns, df.index[TIMESTEPS:]

X_train, y_train, ret_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, idx_val = create_dataset(val_df)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# --- 3B. WAGI KLAS (THE FIX) ---
# Obliczamy, czy klasy są niezbalansowane i generujemy wagi
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Wagi klas (0=Down, 1=Up): {class_weights_dict}")
print("(Jeśli waga dla 0 jest wyższa, model będzie mocniej karany za przegapienie spadku)")

# --- 4. TRENING ENSEMBLE ---
models = []
p_val_list = []

print(f"\n--- Trening {N_MODELS} zbalansowanych modeli... ---")

for i in range(N_MODELS):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(NEURONS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Dodajemy class_weight do treningu!
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=False,
        class_weight=class_weights_dict # <--- TUTAJ JEST MAGIA
    )
    models.append(model)
    p_val_list.append(model.predict(X_val, verbose=0).flatten())

print("--- Ensemble Gotowy ---")

# --- 5. ANALIZA ---
p_train_avg = np.mean([m.predict(X_train, verbose=0).flatten() for m in models], axis=0)
p_val_avg   = np.mean(p_val_list, axis=0)

def analyze(probs, y_true, ret, name):
    pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    strat_ret = pos * ret
    
    active = pos != 0
    trades = np.sum(active)
    if trades > 0:
        wr = (np.sign(pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    # Confusion Matrix
    preds_bin = (probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, preds_bin)
    
    # Sprawdzamy czy model widzi spadki (True Negatives)
    true_downs = cm[0][0]
    total_downs = np.sum(cm[0])
    recall_down = true_downs / total_downs if total_downs > 0 else 0
    
    cum = np.exp(np.cumsum(strat_ret))
    total_ret = (cum[-1] - 1) * 100
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    try: gini = 2 * roc_auc_score(y_true, probs) - 1
    except: gini = 0
    
    return {
        "Name": name, "WinRate %": wr, "Gini": gini, "Sharpe": sharpe, 
        "Return %": total_ret, "Trades": trades, "Recall Down": recall_down,
        "ConfMatrix": cm
    }

m_train = analyze(p_train_avg, y_train, ret_train, "TRAIN")
m_val   = analyze(p_val_avg, y_val, ret_val, "VAL")

# --- 6. RAPORT ---
print("\n" + "="*70)
print("   RAPORT ZBALANSOWANY (Czy model widzi spadki?)   ")
print("="*70)
res_df = pd.DataFrame([
    {k: v for k, v in m_train.items() if k != 'ConfMatrix'},
    {k: v for k, v in m_val.items() if k != 'ConfMatrix'}
])
print(res_df.round(3).to_string(index=False))
print("-" * 70)

print("\n🔍 SZCZEGÓŁY MACIERZY POMYŁEK (VAL):")
cm = m_val['ConfMatrix']
print(f"[[{cm[0][0]:^5} {cm[0][1]:^5}]  <-- Wiersz SPADKI (Trafione / Pomylone)")
print(f" [{cm[1][0]:^5} {cm[1][1]:^5}]] <-- Wiersz WZROSTY (Pomylone / Trafione)")

print("\n🤖 [AI ADVISOR - REALITY CHECK]:")
# Nowa logika Advisora
if m_val['Recall Down'] < 0.10:
    print("❌ MODEL JEST PERMABULLEM. Nadal ignoruje spadki (Recall Down < 10%).")
    print("   -> Sugestia: Model może potrzebować wejścia typu 'RSI' lub 'MACD', żeby widzieć przegrzanie.")
elif m_val['WinRate %'] < 50:
    print("⚠️ MODEL WALCZY. Widzi spadki, ale traci na skuteczności.")
    print("   -> To normalne przy balansowaniu. Teraz spróbuj dostroić TIMESTEPS (np. 21).")
else:
    print("✅ SUKCES! Model przewiduje w obie strony i ma WinRate > 50%.")
    print("   To jest fundament pod prawdziwy trading.")

print("="*70)

# Wykres
plt.figure(figsize=(10, 6))
cum_bh = np.exp(np.cumsum(ret_val))
cum_strat = np.exp(np.cumsum(np.where(p_val_avg > THRESH_LONG, 1, np.where(p_val_avg < THRESH_SHORT, -1, 0)) * ret_val))

plt.plot(idx_val, cum_bh, label='Buy & Hold', color='gray', alpha=0.5)
plt.plot(idx_val, cum_strat, label='Balanced Strategy', color='purple', linewidth=2)
plt.title('Walidacja 2023 (Zbalansowana)')
plt.legend()
plt.savefig('balanced_report.png')
print("Zapisano: balanced_report.png")