import ccxt
import pandas as pd
import time
from datetime import datetime

# --- KONFIGURACJA ---
EXCHANGE = ccxt.binance()
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'        # Interwał Dzienny (zgodnie z życzeniem)
START_DATE = '2017-08-17 00:00:00' # Start Binance BTC/USDT
OUTPUT_FILE = 'BTC_USDT_1d_2017_2024.csv'

def fetch_data():
    print(f"--- Pobieranie danych {SYMBOL} ({TIMEFRAME}) od {START_DATE} ---")
    since = EXCHANGE.parse8601(START_DATE)
    all_candles = []
    
    while True:
        try:
            candles = EXCHANGE.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=1000)
            if not candles: break
            
            all_candles.extend(candles)
            last_timestamp = candles[-1][0]
            since = last_timestamp + 1
            
            print(f"Pobrano do: {datetime.fromtimestamp(last_timestamp/1000)} | Ilość dni: {len(all_candles)}")
            
            # Pobieramy do "teraz"
            if last_timestamp >= EXCHANGE.milliseconds() - 1000*60*60*24:
                break
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Błąd: {e}, czekam...")
            time.sleep(5)

    print("--- Zapisywanie ---")
    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df.drop(columns=['Timestamp'], inplace=True) # Wolumen usuwamy w modelu, tu niech zostanie w pliku "na wszelki wypadek"
    
    df.to_csv(OUTPUT_FILE)
    print(f"Gotowe! Plik: {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_data()