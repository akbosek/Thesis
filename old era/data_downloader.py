import pandas as pd
import numpy as np
import yfinance as yf
import os

# ==============================================================================
# KONFIGURACJA
# ==============================================================================
OUTPUT_FILE = 'data_master_v1.csv'
START_DATE  = "2014-01-01"
END_DATE    = "2024-12-31"

# Lista aktywów do pobrania (Zgodnie z artykułami naukowymi)
# BTC = Target
# SP500, NASDAQ, OIL = Makroekonomia
# VIX = Sentyment (Indeks Strachu)
TICKERS = {
    'BTC': 'BTC-USD',
    'SP500': '^GSPC', 
    'NASDAQ': '^IXIC', 
    'OIL': 'CL=F',
    'VIX': '^VIX'
}

def generate_master_data():
    print(f"--- ROZPOCZYNAM POBIERANIE DANYCH ({START_DATE} - {END_DATE}) ---")
    
    collected_dfs = []
    
    # 1. POBIERANIE Z YAHOO FINANCE
    for name, symbol in TICKERS.items():
        print(f"⏳ Pobieranie: {name} ({symbol})...")
        try:
            # progress=False wyłącza pasek ładowania, żeby nie śmiecić w konsoli
            df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
            
            # Obsługa MultiIndex (nowa wersja yfinance zwraca czasem dziwne kolumny)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            else:
                df = df[['Close']]
            
            # Zmieniamy nazwę kolumny na nazwę aktywa (np. 'Close' -> 'BTC')
            df.columns = [name]
            collected_dfs.append(df)
            print(f"   ✅ Sukces: {name} ({len(df)} wierszy)")
            
        except Exception as e:
            print(f"   ❌ BŁĄD przy {name}: {e}")
            return

    # 2. ŁĄCZENIE I CZYSZCZENIE (SYNCHRONIZACJA RYNKÓW)
    print("\n--- PRZETWARZANIE I FEATURE ENGINEERING ---")
    print("Łączenie rynków (Krypto 24/7 vs Giełda Pn-Pt)...")
    
    # ffill() jest krytyczny: Uzupełnia ceny z piątku dla giełd tradycyjnych (SP500) 
    # na sobotę i niedzielę, żeby pasowały do Bitcoina.
    data = pd.concat(collected_dfs, axis=1).fillna(method='ffill')
    
    # Usuwamy wiersze, gdzie nadal są braki (np. na samym początku historii)
    data.dropna(inplace=True)
    
    # 3. WYLICZANIE WSKAŹNIKÓW (FEATURE ENGINEERING)
    # Tworzymy kopię, żeby nie modyfikować oryginału w pętli
    processed_data = pd.DataFrame(index=data.index)
    
    # A. Zachowujemy surową cenę BTC (do wizualizacji lub późniejszych obliczeń)
    processed_data['BTC_price'] = data['BTC']
    
    # B. Zwroty Logarytmiczne (Log Returns) dla wszystkich aktywów
    # To jest lepsze dla sieci neuronowych niż surowa cena
    for col in data.columns:
        processed_data[f'{col}_ret'] = np.log(data[col] / data[col].shift(1))
    
    # C. VIX Level (Indeks Strachu)
    # Dla VIX ważny jest sam poziom (np. 20 vs 80), a nie tylko zmiana procentowa
    processed_data['VIX_level'] = data['VIX']
    
    # D. RSI dla Bitcoina (Siła Trendu)
    delta = data['BTC'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    processed_data['BTC_rsi'] = (100 - (100 / (1 + rs))) / 100.0 # Skalujemy do 0-1
    
    # E. TARGET (Co przewidujemy?)
    # 1 = Cena JUTRO wyższa niż DZIŚ, 0 = Cena JUTRO niższa
    processed_data['target'] = np.where(data['BTC'].shift(-1) > data['BTC'], 1, 0)
    
    # Usuwamy NaN powstałe przy liczeniu RSI i zwrotów
    processed_data.dropna(inplace=True)
    
    # 4. ZAPIS DO PLIKU
    processed_data.to_csv(OUTPUT_FILE)
    
    print("\n" + "="*50)
    print(f"🎉 SUKCES! Plik '{OUTPUT_FILE}' został utworzony.")
    print(f"📊 Liczba wierszy (dni): {len(processed_data)}")
    print(f"📝 Zapisane kolumny: {list(processed_data.columns)}")
    print("="*50)
    print("\nPodgląd ostatnich 5 dni:")
    print(processed_data.tail())

if __name__ == "__main__":
    # Sprawdzenie czy biblioteki są zainstalowane
    try:
        import yfinance
    except ImportError:
        print("BŁĄD: Brakuje biblioteki yfinance.")
        print("Wpisz w terminalu: pip install yfinance pandas numpy")
    else:
        generate_master_data()