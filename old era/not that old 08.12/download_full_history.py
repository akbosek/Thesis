import yfinance as yf
import pandas as pd
import os

def download_btc_history():
    print("--- Pobieranie pełnej historii BTC-USD (Yahoo Finance) ---")
    
    # Pobieramy maksymalny dostępny zakres
    # Yahoo Finance zazwyczaj ma dane BTC-USD od 2014-09-17
    data = yf.download("BTC-USD", start="2014-01-01", end="2024-12-31", progress=True)
    
    if len(data) == 0:
        print("Błąd: Nie udało się pobrać danych. Sprawdź połączenie internetowe.")
        return

    # Czyszczenie danych (Yahoo zwraca MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Formatowanie kolumn
    data = data.rename(columns={
        "Open": "Open", "High": "High", "Low": "Low", 
        "Close": "Close", "Volume": "Volume"
    })
    
    # Zapis do CSV
    filename = "BTC_USD_1d_2014_2024.csv"
    data.to_csv(filename)
    print(f"\nSukces! Zapisano {len(data)} dni do pliku: {filename}")
    print(f"Zakres danych: {data.index.min().date()} do {data.index.max().date()}")

if __name__ == "__main__":
    try:
        import yfinance
        download_btc_history()
    except ImportError:
        print("Brakuje biblioteki yfinance. Zainstaluj ją komendą:")
        print("pip install yfinance")