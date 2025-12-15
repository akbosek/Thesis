import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adamax
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# ==========================================
# 1. KONFIGURACJA (WG ARTYKUŁU)
# ==========================================
# Parametry "Model 1" z artykułu:
BATCH_SIZE    = 72       # 
HIDDEN_UNITS  = 50       # 
DROPOUT       = 0.01     #  - Bardzo mały dropout!
LEARNING_RATE = 0.001    # 
EPOCHS        = 50       # 
TIMESTEPS     = 1        # [cite: 218] - Model 1-dniowy patrzył 1 krok wstecz

TRAIN_END_DATE = '2022-12-31' # Artykuł miał dane do 2022

print("--- [IMPLEMENTACJA MODELU Z ARTYKUŁU: STANDARD LSTM] ---")

# ==========================================
# 2. DANE (MACRO + BTC)
# ==========================================
# Artykuł wskazał, że dla D1 (1 dzień) kluczowe są: BTC, S&P500, NASDAQ, Ropa.
tickers = {
    'BTC': 'BTC-USD',
    'SP500': '^GSPC',  # S&P 500
    'NASDAQ': '^IXIC', # NASDAQ
    'OIL': 'CL=F'      # Crude Oil WTI
}

dfs = []
for name, ticker in tickers.items():
    print(f"Pobieranie {name}...")
    d = yf.download(ticker, start="2014-01-01", end="2024-12-31", progress=False)
    if isinstance(d.columns, pd.MultiIndex): d = d['Close']
    else: d = d[['Close']]
    d.columns = [name]
    dfs.append(d)

# Łączenie i ffill (giełdy tradycyjne mają luki w weekendy)
data = pd.concat(dfs, axis=1).fillna(method='ffill').dropna()

# Target: Przewidujemy cenę JUTRO (w artykule to była regresja, my badamy kierunek)
data['target_price'] = data['BTC'].shift(-1)
data['target_dir'] = np.where(data['target_price'] > data['BTC'], 1, 0)
data.dropna(inplace=True)

# Wybór cech (Zgodnie z selekcją GA z artykułu dla Modelu 1)
features = ['BTC', 'SP500', 'NASDAQ', 'OIL'] 
print(f"Cechy wejściowe (zgodnie z artykułem): {features}")

# Podział
train = data.loc[:TRAIN_END_DATE]
val = data.loc['2023-01-01':]

# Skalowanie
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train[features])

# Funkcja dataset (TIMESTEPS=1 jak w Modelu 1)
def mk_ds(df):
    X_sc = scaler.transform(df[features])
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:i+TIMESTEPS])
        y.append(df['target_dir'].iloc[i+TIMESTEPS])
    return np.array(X), np.array(y)

X_train, y_train = mk_ds(train)
X_val, y_val = mk_ds(val)

# ==========================================
# 3. MODEL (CZYSTY STANDARD LSTM)
# ==========================================
# Architektura dokładnie jak na Rysunku 2 i w opisie sekcji 5 [cite: 350, 397]
model = Sequential([
    Input(shape=(TIMESTEPS, len(features))),
    
    # Pojedyncza warstwa LSTM (nie Bi-LSTM)
    LSTM(HIDDEN_UNITS, return_sequences=False),
    
    # Minimalny Dropout
    Dropout(DROPOUT),
    
    # Wyjście
    Dense(1, activation='sigmoid') # Sigmoid bo klasyfikacja (kierunek)
])

# Użycie ADAMAX (zgodnie z )
opt = Adamax(learning_rate=LEARNING_RATE)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

print(f"Trenowanie modelu (Optimizer: Adamax, Neurons: {HIDDEN_UNITS})...")
model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE, 
          verbose=1,
          shuffle=False) # Dane szeregów czasowych w artykule

# ==========================================
# 4. WYNIK
# ==========================================
preds = model.predict(X_val, verbose=0).flatten()
# Progi standardowe (lub zoptymalizowane przez Ciebie wcześniej)
signals = np.where(preds > 0.51, 1, np.where(preds < 0.49, 0, -1))
active = (signals != -1)

if np.sum(active) > 0:
    wr = accuracy_score(y_val[active], signals[active]) * 100
    print(f"\nWIN RATE: {wr:.2f}%")
else:
    print("Brak sygnałów.")