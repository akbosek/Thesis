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
#      PANEL STEROWANIA (WIN RATE MODE)
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

# PROGI (Szukamy balansu)
THRESH_SHORT = 0.495
THRESH_LONG  = 0.505
# ==============================================================================

print(f"--- [LSTM WINRATE BOOSTER: {N_MODELS}x ENSEMBLE] ---")
if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"Brak pliku {INPUT_FILE}!")

# --- 1. DANE & FEATURE ENGINEERING (ROZSZERZONE) ---
data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# A. Standardowe Cechy
data['volatility_30'] = data['log_return'].rolling(window=30).std()
data['dist_high'] = (data['High'] - data['Close']) / data['Close']
data['dist_low']  = (data['Close'] - data['Low']) / data['Close']
data['momentum_3d'] = data['Close'] / data['Close'].shift(3) - 1

# B. NOWOŚĆ: Cechy Czasowe (Seasonality)
# Model nauczy się specyfiki weekendów vs dni roboczych
data['day_of_week'] = data.index.dayofweek
data['is_weekend'] = np.where(data['day_of_week'] >= 5, 1, 0)

# Kodowanie cykliczne dnia tygodnia (sin/cos) - lepsze dla sieci neuronowych
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

data.dropna(inplace=True)

# Lista cech (teraz zawiera czas)
features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d', 'day_sin', 'day_cos']

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
    future_returns = df['log_return'].shift(-1).iloc[TIMESTEPS:].fillna(0).values
    # Zwracamy też index i surowe dane do analizy
    return np.array(X), np.array(y), future_returns, df.index[TIMESTEPS:], df.iloc[TIMESTEPS:]

X_train, y_train, ret_train, idx_train, df_train_cut = create_dataset(train_df)
X_val, y_val, ret_val, idx_val, df_val_cut = create_dataset(val_df)

print(f"Features: {features}")
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# --- 4. TRENING ENSEMBLE ---
models = []
val_predictions_matrix = np.zeros((len(X_val), N_MODELS)) # Do przechowywania głosów każdego modelu

print(f"\n--- Uruchamiam Trening {N_MODELS} Modeli... ---")

for i in range(N_MODELS):
    print(f"   -> Trenowanie Eksperta #{i+1}...")
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(NEURONS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=False)
    models.append(model)
    
    # Zbieramy głosy na walidacji od razu
    val_predictions_matrix[:, i] = model.predict(X_val, verbose=0).flatten()

print("--- Ensemble Gotowy ---")

# --- 5. AGREGACJA ---
p_train_avg = np.mean([m.predict(X_train, verbose=0).flatten() for m in models], axis=0)
p_val_avg   = np.mean(val_predictions_matrix, axis=1)

# --- 6. DIAGNOSTYKA I EXPORTY ---
def analyze_performance(probs, y_true, ret, df_source, name):
    pos = np.where(probs > THRESH_LONG, 1, np.where(probs < THRESH_SHORT, -1, 0))
    strat_ret = pos * ret
    
    active = pos != 0
    trades = np.sum(active)
    
    if trades > 0:
        # Win Rate ogólny
        wr = (np.sign(pos[active]) == np.sign(ret[active])).mean() * 100
        
        # Win Rate w WEEKENDY vs DNI ROBOCZE (Analiza sezonowości)
        weekend_mask = (active) & (df_source['is_weekend'] == 1)
        weekday_mask = (active) & (df_source['is_weekend'] == 0)
        
        wr_weekend = 0
        if np.sum(weekend_mask) > 0:
            wr_weekend = (np.sign(pos[weekend_mask]) == np.sign(ret[weekend_mask])).mean() * 100
            
        wr_weekday = 0
        if np.sum(weekday_mask) > 0:
            wr_weekday = (np.sign(pos[weekday_mask]) == np.sign(ret[weekday_mask])).mean() * 100
            
    else: wr, wr_weekend, wr_weekday = 0, 0, 0
    
    cum = np.exp(np.cumsum(strat_ret))
    total_ret = (cum[-1] - 1) * 100
    
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    
    try: gini = 2 * roc_auc_score(y_true, probs) - 1
    except: gini = 0
    
    preds_binary = (probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, preds_binary)
    
    return {
        "Name": name,
        "WinRate %": wr, "WR Weekend %": wr_weekend, "WR Weekday %": wr_weekday,
        "Gini": gini, "Sharpe": sharpe, "Return %": total_ret, "Trades": trades,
        "ConfMatrix": cm
    }

m_train = analyze_performance(p_train_avg, y_train, ret_train, df_train_cut, "TRAIN")
m_val   = analyze_performance(p_val_avg, y_val, ret_val, df_val_cut, "VAL")

# --- 7. RAPORT ---
print("\n" + "="*80)
print("   RAPORT OPTYMALIZACJI (SEASONALITY CHECK)   ")
print("="*80)
res_df = pd.DataFrame([
    {k: v for k, v in m_train.items() if k != 'ConfMatrix'},
    {k: v for k, v in m_val.items() if k != 'ConfMatrix'}
])
print(res_df.round(2).to_string(index=False))
print("-" * 80)

# --- 8. ZAAWANSOWANE EXPORTY ---
# Export 1: Szczegóły Głosowania (Czy modele są zgodne?)
df_votes = pd.DataFrame(val_predictions_matrix, index=idx_val, columns=[f'Model_{i+1}' for i in range(N_MODELS)])
df_votes['Average_Prob'] = p_val_avg
df_votes['Actual_Target'] = y_val
df_votes['Is_Weekend'] = df_val_cut['is_weekend'].values
# Odchylenie standardowe głosów (im mniejsze, tym bardziej modele są zgodne)
df_votes['Vote_Consensus_Std'] = df_votes.iloc[:, :N_MODELS].std(axis=1)

df_votes.to_csv('ensemble_voting_details.csv')
print("Zapisano: ensemble_voting_details.csv (Analiza zgody modeli)")

# --- 9. AI ADVISOR (OPTIMIZATION MODE) ---
print("\n🤖 [AI ADVISOR - OPTYMALIZACJA WIN RATE]:")

val_wr = m_val['WinRate %']
wr_weekend = m_val['WR Weekend %']
wr_weekday = m_val['WR Weekday %']

# Analiza Sezonowości
if abs(wr_weekend - wr_weekday) > 5.0:
    print(f"💡 ZNALEZIONO WZORZEC: Duża różnica skuteczności Weekend ({wr_weekend:.1f}%) vs Tydzień ({wr_weekday:.1f}%).")
    if wr_weekend < 50:
        print("   -> SUGESTIA: Wyłącz trading w weekendy! Model gubi się przy niskiej płynności.")
    elif wr_weekday < 50:
        print("   -> SUGESTIA: Model lepiej gra w weekendy. Rozważ strategię weekendową.")
else:
    print("ℹ️ Brak wyraźnej różnicy Weekend/Tydzień. Cechy czasowe działają stabilnie.")

# Analiza Poziomu Win Rate
if val_wr < 53.0:
    print("🔧 Win Rate < 53%. Aby go podbić:")
    print("   1. ZWĘŹ PROGI: Ustaw THRESH_SHORT=0.495, THRESH_LONG=0.505 (Graj tylko pewniaki).")
    print("   2. SPRAWDŹ EXPORT: Otwórz 'ensemble_voting_details.csv'.")
    print("      Czy modele często się kłócą? (Wysokie 'Vote_Consensus_Std').")
    print("      Jeśli tak, zwiększ liczbę epok (EPOCHS=60) lub zmniejsz Learning Rate.")
elif val_wr >= 55.0:
    print("🚀 Win Rate > 55%! To doskonały wynik.")
    print("   -> Możesz spróbować dodać LEWAR x2 dla sygnałów z wysoką zgodnością modeli.")

print("="*80)

# Wykres
plt.figure(figsize=(10, 5))
plt.plot(idx_val, p_val_avg, label='Ensemble Avg Prob', color='purple', alpha=0.6)
plt.axhline(0.5, color='black', linestyle='--')
plt.title('Pewność Decyzji (Ensemble Average)')
plt.savefig('winrate_booster_probs.png')
print("Zapisano: winrate_booster_probs.png")