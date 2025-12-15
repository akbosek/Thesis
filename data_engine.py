import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pandas_ta as ta
import os

# ==========================================
# KONFIGURACJA (START FEAR & GREED)
# ==========================================
# Pobieramy wcześniej, żeby Mayer Multiple (200 dni) był gotowy na start
DOWNLOAD_START = "2017-06-01" 

# OFICJALNY START: 1 Lutego 2018 (Pierwszy dzień FNG)
OFFICIAL_START = "2018-02-01"
END_DATE       = "2024-12-31"

OUTPUT_FILE    = "bitcoin_2018_feb_data.csv"

print("--- [1. DATA ENGINE: REAL DATA (FEB 2018)] ---")

# 1. POBIERANIE DANYCH RYNKOWYCH
print(" > Pobieranie BTC i Makro...")
tickers = {
    'BTC-USD': 'BTC_Close',
    '^GSPC': 'SPX',
    '^VIX': 'VIX',
    'CL=F': 'Oil'
}

dfs = []
fetch_end = "2025-01-05" 

for t, name in tickers.items():
    try:
        df = yf.download(t, start=DOWNLOAD_START, end=fetch_end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if name == 'BTC_Close':
            df = df[['Close', 'Volume']].rename(columns={'Close': name, 'Volume': 'BTC_Volume'})
        else:
            df = df[['Close']].rename(columns={'Close': name})
        dfs.append(df)
    except:
        print(f" ! Błąd dla {t}")

master_df = dfs[0]
for d in dfs[1:]:
    master_df = master_df.join(d, how='left')

# 2. POBIERANIE FEAR & GREED (API)
print(" > Pobieranie Fear & Greed...")
try:
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    r = requests.get(url, timeout=10)
    data = r.json()['data']
    fng = pd.DataFrame(data)
    fng['timestamp'] = pd.to_datetime(fng['timestamp'], unit='s')
    fng = fng.set_index('timestamp').sort_index()
    fng = fng[['value']].rename(columns={'value': 'FNG'}).astype(float) / 100.0
    
    master_df = master_df.join(fng, how='left')
except:
    print(" ! Błąd API FNG")

# 3. CZYSZCZENIE
master_df.ffill(inplace=True)

# 4. WSKAŹNIKI
print(" > Obliczanie Mayer Multiple i RSI...")
master_df['SMA_200'] = master_df['BTC_Close'].rolling(window=200).mean()
master_df['Mayer_Ratio'] = master_df['BTC_Close'] / master_df['SMA_200']
master_df['RSI'] = ta.rsi(master_df['BTC_Close'], length=14) / 100.0

# 5. TARGET
master_df['Target'] = (master_df['BTC_Close'].shift(-1) > master_df['BTC_Close']).astype(int)

# 6. CIĘCIE (OFFICIAL START)
# Wycinamy od 1 Lutego 2018
final_df = master_df.loc[OFFICIAL_START : END_DATE].copy()
final_df.dropna(inplace=True)

final_df.to_csv(OUTPUT_FILE)
print(f"✅ GOTOWE. Plik: {OUTPUT_FILE}")
print(f"   Start danych: {final_df.index.min().date()}")
print(f"   Koniec danych: {final_df.index.max().date()}")
print(f"   Liczba dni: {len(final_df)}")