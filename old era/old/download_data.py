import ccxt
import pandas as pd
import time
from datetime import datetime

# --- KONFIGURACJA ---
EXCHANGE = ccxt.binance()
SYMBOL = 'BTC/USDT'
TIMEFRAME = '4h'
START_DATE = '2018-01-01 00:00:00'
OUTPUT_FILE = 'BTC_USDT_4h_Binance.csv'

def fetch_data():
    print(f"--- Rozpoczynam pobieranie danych {SYMBOL} ({TIMEFRAME}) od {START_DATE} ---")
    since = EXCHANGE.parse8601(START_DATE)
    all_candles = []
    
    while True:
        try:
            candles = EXCHANGE.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=1000)
            if not candles: break
            
            all_candles.extend(candles)
            last_timestamp = candles[-1][0]
            since = last_timestamp + 1
            
            current_date = datetime.fromtimestamp(last_timestamp / 1000)
            print(f"Pobrano do: {current_date} | Wierszy: {len(all_candles)}")
            
            if last_timestamp >= EXCHANGE.milliseconds() - 1000 * 60 * 60 * 4:
                break
            time.sleep(0.1) # Rate limit
            
        except Exception as e:
            print(f"Błąd: {e}. Ponawiam...")
            time.sleep(5)

    print("--- Zapisywanie pliku ---")
    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df.drop(columns=['Timestamp'], inplace=True)
    
    df.to_csv(OUTPUT_FILE)
    print(f"GOTOWE! Dane zapisano w: {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_data()