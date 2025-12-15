import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==============================================================================
#      PANEL STEROWANIA (TWOJE SUWAKI)
# ==============================================================================
INPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

# 1. OKRESY (STRICT MODE - BEZ TESTU 2024)
TRAIN_START = '2017-09-01'
TRAIN_END   = '2022-12-31' 
VAL_START   = '2023-01-01'
VAL_END     = '2023-12-31'

# 2. ENSEMBLE (NOWOŚĆ - STABILIZACJA)
N_MODELS = 5  # Ilu "ekspertów" trenujemy? (3-5 to optimum)

# 3. HIPERPARAMETRY POJEDYNCZEGO MODELU
TIMESTEPS = 14       
NEURONS   = 64       
DROPOUT   = 0.3      # Lekko zwiększone dla bezpieczeństwa
LEARNING_RATE = 0.001
EPOCHS    = 40       # Mniej epok, bo mamy ensemble
BATCH_SIZE = 32

# 4. STRATEGIA (PROGI)
THRESH_SHORT = 0.48
THRESH_LONG  = 0.52
# ==============================================================================

print(f"--- [LSTM FINAL STABLE: {N_MODELS}x ENSEMBLE] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# --- 1. DANE & FEATURE ENGINEERING ---
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Cechy (Sprawdzone)
data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

data.dropna(inplace=True)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']

# --- 2. PODZIAŁ ---
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

# --- 3. SKALOWANIE ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[features])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(df['target'].iloc[i + TIMESTEPS])
    # Future Returns
    future_returns = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    return np.array(X), np.array(y), future_returns, df.index[TIMESTEPS:]

X_train, y_train, ret_train, idx_train = create_dataset(train_df)
X_val, y_val, ret_val, idx_val = create_dataset(val_df)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# --- 4. TRENING ENSEMBLE (PĘTLA) ---
models = []
print(f"\n--- Uruchamiam Trening {N_MODELS} Modeli... ---")

for i in range(N_MODELS):
    print(f"   -> Trenowanie Eksperta #{i+1}...")
    
    # Budowa (Za każdym razem od nowa - inna inicjalizacja wag)
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(NEURONS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Trening
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0, # Cicho
        shuffle=False 
    )
    models.append(model)

print("--- Ensemble Gotowy ---")

# --- 5. AGREGACJA PREDYKCJI (ŚREDNIA) ---
def get_ensemble_preds(X):
    # Pobieramy predykcje od każdego modelu
    all_preds = [m.predict(X, verbose=0).flatten() for m in models]
    # Wyciągamy średnią (To jest ta synchronizacja)
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds

p_train = get_ensemble_preds(X_train)
p_val   = get_ensemble_preds(X_val)

# --- 6. ZAAWANSOWANA DIAGNOSTYKA ---
def analyze_performance(probs, y_true, ret, name):
    # 1. Strategia
    pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    strat_ret = pos * ret
    
    # 2. Metryki Finansowe
    active = pos != 0
    trades = np.sum(active)
    if trades > 0:
        wr = (np.sign(pos[active]) == np.sign(ret[active])).mean() * 100
    else: wr = 0
    
    cum = np.exp(np.cumsum(strat_ret))
    total_ret = (cum[-1] - 1) * 100
    bh_ret = (np.exp(np.cumsum(ret))[-1] - 1) * 100
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    # 3. Metryki ML (Gini, Confusion Matrix)
    try: gini = 2 * roc_auc_score(y_true, probs) - 1
    except: gini = 0
    
    # Binaryzacja do macierzy (prosta > 0.5)
    preds_binary = (probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, preds_binary)
    cr = classification_report(y_true, preds_binary, output_dict=True)
    
    return {
        "Name": name,
        "WinRate %": wr, "Gini": gini, "Sharpe": sharpe, 
        "Return %": total_ret, "BH Return %": bh_ret, "Trades": trades,
        "ConfMatrix": cm, "ClassReport": cr, "StdDev": np.std(probs)
    }

m_train = analyze_performance(p_train, y_train, ret_train, "TRAIN")
m_val   = analyze_performance(p_val, y_val, ret_val, "VAL")

# --- 7. RAPORT ---
print("\n" + "="*60)
print("   RAPORT STABILNOŚCI (ENSEMBLE)   ")
print("="*60)
res_df = pd.DataFrame([
    {k: v for k, v in m_train.items() if k not in ['ConfMatrix', 'ClassReport']},
    {k: v for k, v in m_val.items() if k not in ['ConfMatrix', 'ClassReport']}
])
print(res_df.round(3).to_string(index=False))
print("-" * 60)

# --- 8. SZCZEGÓŁY KLASYFIKACJI (Dla Walidacji) ---
print("\n🔍 SZCZEGÓŁY WALIDACJI (Czy model widzi spadki?):")
cm = m_val['ConfMatrix']
print(f"Macierz Pomyłek (Confusion Matrix):\n{cm}")
print(f"   [True Down  False Up]\n   [False Down True Up]")
print(f"\nPrecyzja dla Spadków (0): {m_val['ClassReport']['0']['precision']:.2f}")
print(f"Precyzja dla Wzrostów (1): {m_val['ClassReport']['1']['precision']:.2f}")

# --- 9. AI ADVISOR ---
print("\n🤖 [AI ADVISOR - STRATEGICZNY]:")
val_wr = m_val['WinRate %']
val_sharpe = m_val['Sharpe']
val_trades = m_val['Trades']
diff_wr = m_train['WinRate %'] - val_wr

if val_trades < 30:
    print(f"❌ MAŁA AKTYWNOŚĆ ({val_trades}). Ensemble uśrednił wyniki do środka (0.5).")
    print(f"   -> Sugestia: ZWĘŹ PROGI. Ustaw THRESH_SHORT={THRESH_SHORT+0.01:.2f} / THRESH_LONG={THRESH_LONG-0.01:.2f}")
elif diff_wr > 8:
    print(f"⚠️ PRZETRENOWANIE. Różnica WinRate {diff_wr:.1f}%.")
    print("   -> Sugestia: Zmniejsz NEURONS do 32 lub zwiększ DROPOUT do 0.4.")
elif val_sharpe > 1.0:
    print("✅ WYNIK DOSKONAŁY. Stabilny, zyskowny, bezpieczny.")
elif val_sharpe > 0.5:
    print("✅ WYNIK DOBRY. Model zarabia stabilnie.")
else:
    print("ℹ️ Model walczy. Sprawdź, czy nie gra przeciwko trendowi (Confusion Matrix).")

print("="*60)

# WYKRESY
plt.figure(figsize=(14, 6))

# 1. Equity
plt.subplot(1, 2, 1)
cum_bh = np.exp(np.cumsum(ret_val))
cum_strat = np.exp(np.cumsum(np.where(p_val > THRESH_LONG, 1, np.where(p_val < THRESH_SHORT, -1, 0)) * ret_val))
plt.plot(idx_val, cum_bh, label='Buy & Hold', color='gray', alpha=0.5)
plt.plot(idx_val, cum_strat, label=f'Ensemble Strategy', color='purple', linewidth=2)
plt.title('Equity Curve (Validation)')
plt.legend()

# 2. Confusion Matrix Heatmap
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['True Down', 'True Up'])
plt.title('Confusion Matrix (Validation)')

plt.tight_layout()
plt.savefig('final_stable_report.png')
print("Zapisano: final_stable_report.png")