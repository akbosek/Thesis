import pandas as pd
import numpy as np
import yfinance as yf
import optuna
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, Bidirectional, Attention, GlobalAveragePooling1D, Concatenate, Conv1D
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. KONFIGURACJA (RESEARCH GRADE)
# ==============================================================================
TIMESTEPS = 30          # Długie okno, bo Attention potrafi z niego wyłowić perełki
PREDICTION_HORIZON = 1  # Przewidujemy kierunek na jutro
TRAIN_END_DATE = '2022-12-31'

print("--- [RESEARCH BOT: Bi-LSTM + ATTENTION + VIX] ---")

# ==============================================================================
# 2. DANE: BTC + MACRO + SENTYMENT (VIX)
# ==============================================================================
# Zgodnie z artykułem, dodajemy VIX jako miarę ryzyka/strachu.
tickers = {
    'BTC': 'BTC-USD',
    'SP500': '^GSPC', 
    'NASDAQ': '^IXIC', 
    'OIL': 'CL=F',
    'VIX': '^VIX'      # <--- INDEKS STRACHU (Fear Gauge)
}

print("Pobieranie danych (w tym VIX)...")
dfs = []
for name, ticker in tickers.items():
    try:
        d = yf.download(ticker, start="2014-01-01", end="2024-12-31", progress=False)
        if isinstance(d.columns, pd.MultiIndex): d = d['Close']
        else: d = d[['Close']]
        d.columns = [name]
        dfs.append(d)
    except Exception as e:
        print(f"Błąd pobierania {name}: {e}")

if not dfs: raise ValueError("Brak danych!")

raw_data = pd.concat(dfs, axis=1).fillna(method='ffill').dropna()

# Feature Engineering
data = pd.DataFrame(index=raw_data.index)

# 1. Zwroty (Log Returns) dla cen aktywów
for col in ['BTC', 'SP500', 'NASDAQ', 'OIL']:
    data[f'{col}_ret'] = np.log(raw_data[col] / raw_data[col].shift(1))

# 2. VIX zostawiamy jako poziom (to już jest wskaźnik znormalizowany w punktach)
# Ale znormalizujemy go później MinMaxem.
data['VIX_level'] = raw_data['VIX']

# 3. RSI (Momentum techniczne)
delta = raw_data['BTC'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['BTC_rsi'] = (100 - (100 / (1 + rs))) / 100.0

# Target
data['target'] = np.where(raw_data['BTC'].shift(-1) > raw_data['BTC'], 1, 0)
data.dropna(inplace=True)

features = [c for c in data.columns if c != 'target']
print(f"Cechy wejściowe: {features}")

# Podział i Skalowanie
train_df = data.loc[:TRAIN_END_DATE]
val_df   = data.loc['2023-01-01':]

scaler = MinMaxScaler()
scaler.fit(train_df[features])

# Funkcja tworząca sekwencje
def create_dataset_arrays(df, steps):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - steps):
        X.append(X_sc[i:(i + steps)])
        y.append(df['target'].iloc[i + steps])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset_arrays(train_df, TIMESTEPS)
X_val, y_val     = create_dataset_arrays(val_df, TIMESTEPS)

print(f"Dane gotowe. X_train: {X_train.shape}")

# ==============================================================================
# 3. MODEL (FUNCTIONAL API: ATTENTION + BI-LSTM)
# ==============================================================================
def objective(trial):
    # --- HIPERPARAMETRY ---
    neurons = trial.suggest_int('neurons', 32, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float('lr', 0.0001, 0.002, log=True)
    reg_strength = trial.suggest_float('reg_strength', 1e-5, 1e-3, log=True)
    threshold_dist = trial.suggest_float('threshold_dist', 0.0, 0.04)
    
    # Czy użyć CNN przed LSTM? (Opcja hybrydowa z artykułu)
    use_cnn = trial.suggest_categorical('use_cnn', [True, False])
    filters = trial.suggest_categorical('filters', [32, 64])

    # --- BUDOWA MODELU (Functional API) ---
    input_layer = Input(shape=(TIMESTEPS, len(features)))
    
    x = input_layer
    
    # 1. Opcjonalny CNN (Wydobycie cech lokalnych)
    if use_cnn:
        x = Conv1D(filters=filters, kernel_size=2, activation='relu', padding='same', 
                   kernel_regularizer=l2(reg_strength))(x)
        # Nie dajemy Poolingu tutaj, żeby zachować sekwencję czasową dla Attention
    
    # 2. Bi-LSTM (Musi zwracać sekwencje dla Attention!)
    # return_sequences=True jest kluczowe, bo Attention musi widzieć każdy krok czasowy
    lstm_out = Bidirectional(LSTM(neurons, return_sequences=True, 
                                  kernel_regularizer=l2(reg_strength)))(x)
    
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(dropout)(lstm_out)
    
    # 3. MECHANIZM UWAGI (ATTENTION)
    # Self-Attention: Sieć pyta "które momenty z przeszłości pasują do obecnego kontekstu?"
    # query=lstm_out, value=lstm_out
    attention_out = Attention()([lstm_out, lstm_out])
    
    # 4. Agregacja
    # Łączymy to, co zrozumiało LSTM (lstm_out) z tym, co podkreśliła Uwaga (attention_out)
    # Możemy użyć GlobalAveragePooling, żeby spłaszczyć wynik do 1 wektora
    context_vector = GlobalAveragePooling1D()(attention_out)
    
    # 5. Głowica Decyzyjna
    dense_out = Dense(32, activation='relu', kernel_regularizer=l2(reg_strength))(context_vector)
    dense_out = Dropout(dropout)(dense_out)
    output_layer = Dense(1, activation='sigmoid')(dense_out)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Optimizer - Artykuł sugerował Adamax, sprawdźmy też Adama
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Adamax'])
    if optimizer_name == 'Adam': opt = Adam(learning_rate=lr)
    else: opt = Adamax(learning_rate=lr)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # --- TRENING ---
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=40, 
              batch_size=64, 
              callbacks=[es], 
              verbose=0,
              shuffle=True) # Shuffle pomaga przy Attention

    # --- EWALUACJA SNAJPERA ---
    preds = model.predict(X_val, verbose=0).flatten()
    
    upper, lower = 0.5 + threshold_dist, 0.5 - threshold_dist
    signals = np.where(preds > upper, 1, np.where(preds < lower, -1, 0))
    active_mask = (signals != 0)
    
    if np.sum(active_mask) == 0: return 0.0
    
    win_rate = accuracy_score(y_val[active_mask], (preds[active_mask] > 0.5).astype(int)) * 100
    action_rate = np.mean(active_mask)
    
    trial.set_user_attr("Action_Rate", action_rate)
    
    if action_rate < 0.10: return 0.0
    
    # Raportowanie na bieżąco
    print(f"[Trial {trial.number}] WR: {win_rate:.2f}% | Act: {action_rate*100:.1f}% | CNN: {use_cnn} | Opt: {optimizer_name}")
    
    return win_rate

# ==============================================================================
# 4. START
# ==============================================================================
if __name__ == "__main__":
    print("Szukam najlepszej architektury (Attention + VIX)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    best = study.best_trial
    print("\n" + "="*50)
    print(">>> ZWYCIĘZCA (RESEARCH GRADE) <<<")
    print("="*50)
    for k, v in best.params.items():
        print(f"{k} = {v}")
    print("="*50)
    print(f"Win Rate: {best.value:.2f}% (Aktywność: {best.user_attrs['Action_Rate']*100:.1f}%)")