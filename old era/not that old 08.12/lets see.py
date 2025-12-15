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
# 1. KONFIGURACJA ZGODNA Z ARTYKUŁEM
# ==========================================
# Parametry z tabeli i tekstu artykułu [cite: 350, 351]
BATCH_SIZE    = 72       
HIDDEN_UNITS  = 50       
DROPOUT       = 0.01     
LEARNING_RATE = 0.001    
EPOCHS        = 50       
TIMESTEPS     = 1        # Artykuł używał lag=1 dla modelu D1 [cite: 218]

# Daty (Twoje ustawienia)
TRAIN_START = '2014-09-17'
TRAIN_END   = '2022-12-31'
VAL_START   = '2023-01-01'
VAL_END     = '2024-12-31'

# Plik bazowy (Twoje OHLCV)
INPUT_FILE = 'BTC_USD_1d_2014_2024.csv' 

# ==========================================
# 2. POBIERANIE I ŁĄCZENIE DANYCH (MACRO)
# ==========================================
print("--- [1. TWORZENIE DATASETU 'PAPER-LIKE'] ---")

# 1. Wczytaj Twoje dane BTC
if try:
    df_btc = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
except FileNotFoundError:
    print(f"Brak pliku {INPUT_FILE}. Pobieram BTC z Yahoo...")
    df_btc = yf.download("BTC-USD", start="2014-01-01", end="2024-12-31")
    df_btc.columns = [c[0] for c in df_btc.columns] # Spłaszczanie multiindexu jeśli jest

# Uporządkowanie kolumn BTC
df_btc = df_btc[['Close']].rename(columns={'Close': 'btc'})

# 2. Pobierz dane MAKRO (Kluczowe wg artykułu )
print("Pobieranie danych makro (S&P500, NASDAQ, Oil)...")
tickers = {
    'sandp500': '^GSPC',  # S&P 500
    'nasdaq': '^IXIC',    # NASDAQ
    'crude_oil': 'CL=F'   # Crude Oil
}

df_macro = yf.download(list(tickers.values()), start="2014-01-01", end="2024-12-31")['Close']
df_macro.columns = list(tickers.keys())

# 3. Łączenie (Merge)
# Forward Fill jest konieczny, bo giełdy tradycyjne (Macro) nie działają w weekendy, a krypto tak.
data = df_btc.join(df_macro).fillna(method='ffill').dropna()

# 4. Feature Engineering zgodnie z artykułem
# Artykuł używał danych opóźnionych (lagged) o 1 dzień 
# My zrobimy to w sekwencjonowaniu, ale tutaj przygotujmy zmienne.
features = ['btc', 'sandp500', 'nasdaq', 'crude_oil']
print(f"Użyte zmienne (zgodnie z Feature Selection z artykułu): {features}")

# Target: Artykuł przewidywał CENĘ (Regression)[cite: 12], my chcemy KIERUNEK (Classification).
# Zrobimy hybrydę: Przewidzimy cenę jak w artykule, a potem ocenimy WinRate.
data['target_price'] = data['btc'].shift(-1) # Cena jutro
data.dropna(inplace=True)

# Podział
train_df = data.loc[TRAIN_START:TRAIN_END].copy()
val_df   = data.loc[VAL_START:VAL_END].copy()

# Skalowanie (Artykuł skalował do [0-1] [cite: 311])
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1)) # Osobny skaler dla ceny docelowej

# Fit tylko na treningu
scaler.fit(train_df[features])
scaler_y.fit(train_df[['target_price']])

def create_dataset(df):
    X_sc = scaler.transform(df[features])
    y_sc = scaler_y.transform(df[['target_price']])
    
    X, y = [], []
    for i in range(len(X_sc) - TIMESTEPS):
        X.append(X_sc[i:(i + TIMESTEPS)])
        y.append(y_sc[i + TIMESTEPS])
        
    return np.array(X), np.array(y), df['btc'].iloc[TIMESTEPS:].values

X_train, y_train, prices_train = create_dataset(train_df)
X_val, y_val, prices_val       = create_dataset(val_df)

print(f"Kształt danych: {X_train.shape}")

# ==========================================
# 3. MODEL (ARCHITEKTURA Z ARTYKUŁU)
# ==========================================
print("\n--- [2. BUDOWA MODELU WG ARTYKUŁU] ---")
#[cite: 350, 351]: Batch 72, Hidden 50, Layers 1, Dropout 0.01, Adamax

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # Pojedyncza warstwa LSTM (jak w artykule)
    LSTM(HIDDEN_UNITS, return_sequences=False),
    
    # Bardzo mały Dropout (jak w artykule)
    Dropout(DROPOUT),
    
    # Wyjście (Cena)
    Dense(1) 
])

# Użycie Adamax (jak w artykule)
model.compile(optimizer=Adamax(learning_rate=LEARNING_RATE), loss='mse')

# ==========================================
# 4. TRENING
# ==========================================
print("Rozpoczynam trening...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    shuffle=False # Szeregi czasowe w artykule często nie są tasowane
)

# ==========================================
# 5. EWALUACJA (WIN RATE)
# ==========================================
print("\n--- [3. WYNIKI] ---")

# Predykcja ceny
preds_sc = model.predict(X_val)
preds_real = scaler_y.inverse_transform(preds_sc).flatten()

# Obliczenie Win Rate na podstawie przewidzianej ceny vs dzisiejszej ceny
# Jeśli model przewiduje Price_Tomorrow > Price_Today, to sygnał BUY (1)
# Artykuł skupiał się na R2 i RMSE[cite: 314], ale my przeliczymy to na Twój Win Rate.

# Ceny "dzisiaj" (ostatnia znana cena w oknie)
# Ponieważ TIMESTEPS=1, X_val[:, 0, 0] to znormalizowana cena 'btc' dzisiaj.
# Musimy odwrócić skalowanie dla kolumny 'btc' (index 0 w features)
dummy_matrix = np.zeros((len(X_val), len(features)))
dummy_matrix[:, 0] = X_val[:, -1, 0] 
current_prices = scaler.inverse_transform(dummy_matrix)[:, 0]

# Logika kierunku
predicted_direction = np.where(preds_real > current_prices, 1, 0)
true_future_prices = scaler_y.inverse_transform(y_val).flatten()
true_direction = np.where(true_future_prices > current_prices, 1, 0)

# Wynik
win_rate = accuracy_score(true_direction, predicted_direction) * 100

print(f"WIN RATE (Direction): {win_rate:.2f}%")

# Sprawdzenie metryk z artykułu (R2 / RMSE) dla porównania
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(true_future_prices, preds_real)
rmse = np.sqrt(mean_squared_error(true_future_prices, preds_real))

print(f"R^2 (Metryka z artykułu): {r2:.4f} (W artykule osiągnęli 0.98 na D1 )")
print(f"RMSE: {rmse:.2f}")

if r2 > 0.90:
    print("✅ Model odwzorował sukces z artykułu! Bardzo wysokie dopasowanie ceny.")
else:
    print("⚠️ R^2 niższe niż w artykule. Możliwe przyczyny: brak danych blockchain (Difficulty, Hashrate).")