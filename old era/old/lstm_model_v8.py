import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Imports for Stationarity & Metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Imports for Models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- 1. KONFIGURACJA ---
# Nazwa Twojego pliku (musi być w tym samym folderze)
INPUT_FILE = 'BTC_USDT_4h_Binance 01.01.2018-31.12.2024.csv' 

TIMESTEPS = 12 
THRESH_SHORT = 0.48
THRESH_LONG = 0.52
TRANSACTION_FEE = 0.001 

# Daty (Sztywne ramy czasowe)
TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE   = '2022-12-31'
TEST_START_DATE = '2023-01-01'

# --- 2. WCZYTANIE I CZAS ---
print(f"--- [DIAGNOSTIC V8] Wczytywanie danych ---")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Brak pliku: {INPUT_FILE}")

data = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
data.columns = [c.capitalize() for c in data.columns]

# Konwersja na strefę czasową Warszawy
try:
    data.index = data.index.tz_localize('UTC').tz_convert('Europe/Warsaw')
except TypeError:
    data.index = data.index.tz_convert('Europe/Warsaw')

# --- 3. FEATURE ENGINEERING ---
# Log Returns
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# Relative Volume (30 dni * 6 świec 4h = 180)
data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(window=180).mean()

# Indicators
data['RSI'] = ta.rsi(data['Close'], length=14)
data['MACD'] = ta.macd(data['Close']).iloc[:, 0]

# Time features
data['Hour_Sin'] = np.sin(2 * np.pi * data.index.hour / 24)
data['Hour_Cos'] = np.cos(2 * np.pi * data.index.hour / 24)

# Lags
lags = 3
for lag in range(1, lags + 1):
    data[f'log_return_lag_{lag}'] = data['log_return'].shift(lag)

features = ['log_return', 'Rel_Vol', 'RSI', 'MACD', 'Hour_Sin', 'Hour_Cos'] + [f'log_return_lag_{i}' for i in range(1, lags + 1)]

# Target & Actual Return
data['target'] = np.where(data['log_return'].shift(-1) > 0, 1, 0)
data['actual_return'] = data['log_return'].shift(-1)

data = data.dropna()

# --- 4. PODZIAŁ ---
train_df = data.loc[:TRAIN_END_DATE].copy()
val_df   = data.loc[VAL_START_DATE:VAL_END_DATE].copy()
test_df  = data.loc[TEST_START_DATE:].copy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

class_weights = compute_class_weight('balanced', classes=np.unique(train_df['target']), y=train_df['target'])
class_weights_dict = dict(enumerate(class_weights))

# --- 5. SKALOWANIE ---
scaler = StandardScaler()
scaler.fit(train_df[features])

# Helper function to prepare X/y for any set
def prepare_set(df):
    X_scaled = scaler.transform(df[features])
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - TIMESTEPS):
        X_seq.append(X_scaled[i:(i + TIMESTEPS)])
        y_seq.append(df['target'].iloc[i + TIMESTEPS])
    # Zwracamy też indeks i zwroty (do CSV)
    return np.array(X_seq), np.array(y_seq), df.index[TIMESTEPS:], df['actual_return'].iloc[TIMESTEPS:].values

X_train, y_train, idx_train, ret_train = prepare_set(train_df)
X_val, y_val, idx_val, ret_val = prepare_set(val_df)
X_test, y_test, idx_test, ret_test = prepare_set(test_df)

# Flatten for Trees
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

# --- 6. MODELOWANIE ---

# LSTM
print("\n--- Training LSTM ---")
lstm_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)]

history = lstm_model.fit(
    X_train, y_train, epochs=30, batch_size=64, 
    validation_data=(X_val, y_val), class_weight=class_weights_dict, 
    shuffle=False, callbacks=callbacks, verbose=1
)

# RF & XGB
print("\n--- Training Trees ---")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train_flat, y_train)
xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, scale_pos_weight=1.5, n_jobs=-1, random_state=42).fit(X_train_flat, y_train)

# --- 7. DIAGNOSTYKA I EKSPORT ---
print("\n--- Generowanie Pełnego Raportu Diagnostycznego ---")

def get_predictions(model, X, is_deep=False):
    if is_deep: return model.predict(X).flatten()
    return model.predict_proba(X)[:, 1]

# Pobieramy predykcje dla WSZYSTKICH zbiorów
preds = {
    'Train': {
        'LSTM': get_predictions(lstm_model, X_train, True),
        'RF': get_predictions(rf_model, X_train_flat),
        'XGB': get_predictions(xgb_model, X_train_flat),
        'Target': y_train, 'Returns': ret_train, 'Index': idx_train
    },
    'Val': {
        'LSTM': get_predictions(lstm_model, X_val, True),
        'RF': get_predictions(rf_model, X_val_flat),
        'XGB': get_predictions(xgb_model, X_val_flat),
        'Target': y_val, 'Returns': ret_val, 'Index': idx_val
    },
    'Test': {
        'LSTM': get_predictions(lstm_model, X_test, True),
        'RF': get_predictions(rf_model, X_test_flat),
        'XGB': get_predictions(xgb_model, X_test_flat),
        'Target': y_test, 'Returns': ret_test, 'Index': idx_test
    }
}

# --- A. Tabela Metryk (Gini, Sharpe) ---
metrics_summary = []

def calc_advanced_metrics(probs, targets, returns, period_name, model_name):
    try:
        auc = roc_auc_score(targets, probs)
        gini = 2 * auc - 1
    except: gini = 0
    
    # Simple strategy for metric
    pos = np.where(probs > 0.52, 1, 0)
    strat_ret = pos * returns
    if np.std(strat_ret) == 0: sharpe = 0
    else: sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365 * 6)
    
    return {
        'Period': period_name,
        'Model': model_name,
        'Gini': round(gini, 3),
        'Sharpe': round(sharpe, 2),
        'Mean_Prob': round(np.mean(probs), 3)
    }

for period in ['Train', 'Val', 'Test']:
    for model_name in ['LSTM', 'RF', 'XGB']:
        m = calc_advanced_metrics(preds[period][model_name], preds[period]['Target'], preds[period]['Returns'], period, model_name)
        metrics_summary.append(m)

df_metrics = pd.DataFrame(metrics_summary)
print("\n=== RAPORT PORÓWNAWCZY (Train vs Val vs Test) ===")
print(df_metrics.pivot(index='Model', columns='Period', values=['Gini', 'Sharpe']))

# --- B. Eksport Pełnych Danych do CSV ---
df_export = pd.DataFrame()

for period in ['Train', 'Val', 'Test']:
    temp_df = pd.DataFrame(index=preds[period]['Index'])
    temp_df['Period'] = period
    temp_df['Actual_Return'] = preds[period]['Returns']
    temp_df['Target_Class'] = preds[period]['Target']
    
    # Dodajemy predykcje
    temp_df['LSTM_Prob'] = preds[period]['LSTM']
    temp_df['RF_Prob'] = preds[period]['RF']
    temp_df['XGB_Prob'] = preds[period]['XGB']
    
    # Obliczamy wynik strategii
    for m in ['LSTM', 'RF', 'XGB']:
        prob = temp_df[f'{m}_Prob']
        pos = np.where(prob < THRESH_SHORT, -1, np.where(prob > THRESH_LONG, 1, 0))
        # Variable Size: agresywniejsze wejście
        size = np.where(pos==1, (prob-THRESH_LONG)/(1-THRESH_LONG), (THRESH_SHORT-prob)/THRESH_SHORT)
        size = np.clip(size * 2, 0, 1) 
        
        fee_cost = np.abs(pos) * TRANSACTION_FEE * 0.1 # Fee
        temp_df[f'{m}_Strat_Ret'] = (pos * size * temp_df['Actual_Return']) - fee_cost

    df_export = pd.concat([df_export, temp_df])

df_export.to_csv('full_diagnostic_data.csv')
print("\n>>> Zapisano pełne dane: full_diagnostic_data.csv")

# --- C. Wykresy Diagnostyczne ---
plt.figure(figsize=(15, 10))

# 1. Gini Evolution
plt.subplot(2, 2, 1)
sns.barplot(data=df_metrics, x='Model', y='Gini', hue='Period', palette='viridis')
plt.title('Stabilność Modelu (Gini)')
plt.axhline(0, color='black', linewidth=0.5)

# 2. Probability Distribution
plt.subplot(2, 2, 2)
test_set = df_export[df_export['Period']=='Test']
sns.kdeplot(test_set['LSTM_Prob'], label='LSTM', fill=True, alpha=0.3)
sns.kdeplot(test_set['RF_Prob'], label='RF', fill=True, alpha=0.3)
sns.kdeplot(test_set['XGB_Prob'], label='XGB', fill=True, alpha=0.3)
plt.axvline(0.5, color='red', linestyle='--')
plt.title('Rozkład Pewności (Test Set)')
plt.legend()

# 3. Equity Curve
plt.subplot(2, 1, 2)
plt.plot(test_set.index, (1 + test_set['Actual_Return']).cumprod(), label='Buy & Hold', color='grey', alpha=0.5)
plt.plot(test_set.index, (1 + test_set['LSTM_Strat_Ret']).cumprod(), label='LSTM Strategy')
plt.plot(test_set.index, (1 + test_set['RF_Strat_Ret']).cumprod(), label='RF Strategy')
plt.plot(test_set.index, (1 + test_set['XGB_Strat_Ret']).cumprod(), label='XGB Strategy')
plt.title('Equity Curve (2023-2024)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostic_dashboard.png')
print(">>> Zapisano wykresy: diagnostic_dashboard.png")