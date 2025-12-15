import yfinance as yf
# ==============================================================================
# 2. DANE: SYSTEM "JEDEN PLIK" (BTC + MACRO + VIX)
# ==============================================================================
DATA_FILE = 'data_master_v1.csv'  # <-- To będzie Twój stały plik z danymi

def get_market_data():
    """
    Sprawdza czy dane są na dysku. Jak nie - pobiera i zapisuje.
    Zwraca gotowy DataFrame.
    """
    import os
    
    # 1. Sprawdzenie czy plik istnieje
    if os.path.exists(DATA_FILE):
        print(f"✅ Znaleziono plik lokalny: {DATA_FILE}. Wczytuję...")
        df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        return df
    
    # 2. Jeśli brak pliku - pobieramy z sieci
    print(f"⚠️ Brak pliku {DATA_FILE}. Pobieram świeże dane z Yahoo Finance...")
    
    tickers = {
        'BTC': 'BTC-USD',
        'SP500': '^GSPC', 
        'NASDAQ': '^IXIC', 
        'OIL': 'CL=F',
        'VIX': '^VIX'      # Indeks Strachu
    }
    
    dfs = []
    for name, ticker in tickers.items():
        try:
            d = yf.download(ticker, start="2014-01-01", end="2024-12-31", progress=False)
            
            # Obsługa MultiIndex (nowe wersje yfinance)
            if isinstance(d.columns, pd.MultiIndex): 
                d = d['Close']
            else: 
                d = d[['Close']]
                
            d.columns = [name]
            dfs.append(d)
        except Exception as e:
            print(f"❌ Błąd pobierania {name}: {e}")

    if not dfs: 
        raise ValueError("Krytyczny błąd: Nie udało się pobrać danych!")

    # Łączenie i czyszczenie
    raw_data = pd.concat(dfs, axis=1).fillna(method='ffill').dropna()
    
    # Zapis do pliku na przyszłość
    raw_data.to_csv(DATA_FILE)
    print(f"💾 Zapisano nowe dane do: {DATA_FILE} (Użyjemy ich następnym razem)")
    
    return raw_data

# --- GŁÓWNA LOGIKA PRZETWARZANIA (Uruchamiana zawsze) ---
raw_data = get_market_data()

# Feature Engineering
data = pd.DataFrame(index=raw_data.index)

# A. Zwroty logarytmiczne
for col in raw_data.columns:
    data[f'{col}_ret'] = np.log(raw_data[col] / raw_data[col].shift(1))

# B. Zachowujemy poziomy VIX (znormalizujemy je później)
data['VIX_level'] = raw_data['VIX']

# C. Wskaźniki Techniczne (RSI dla BTC)
delta = raw_data['BTC'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['BTC_rsi'] = (100 - (100 / (1 + rs))) / 100.0

# D. Target (Kierunek Ceny)
data['target'] = np.where(raw_data['BTC'].shift(-1) > raw_data['BTC'], 1, 0)

data.dropna(inplace=True)

# Definicja cech
features = [c for c in data.columns if c != 'target']
print(f"Cechy wejściowe ({len(features)}): {features}")

# Podział i Skalowanie
TRAIN_END = '2022-12-31'
train_df = data.loc[:TRAIN_END].copy()
val_df   = data.loc['2023-01-01':].copy()

scaler = MinMaxScaler()
scaler.fit(train_df[features]) # Fit tylko na treningu!

# Funkcja tworząca sekwencje
def create_dataset_arrays(df, steps):
    if len(df) < steps: return np.array([]), np.array([])
    X_sc = scaler.transform(df[features])
    X, y = [], []
    # Returny do Sharpe (przesunięte o 1 dzień w przyszłość)
    # Jeśli kolumna BTC_ret istnieje, używamy jej.
    rets = df['BTC_ret'].shift(-1).fillna(0).values
    
    for i in range(len(X_sc) - steps):
        X.append(X_sc[i:(i + steps)])
        y.append(df['target'].iloc[i + steps])
        
    return np.array(X), np.array(y), rets[steps:]

# Generowanie zmiennych globalnych
X_train, y_train, r_train = create_dataset_arrays(train_df, TIMESTEPS)
X_val, y_val, r_val       = create_dataset_arrays(val_df, TIMESTEPS)

print(f"Gotowe. Train: {X_train.shape}, Val: {X_val.shape}")