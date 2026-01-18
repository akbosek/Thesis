import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
START_DATE = "2016-01-01"
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')

# 1. EXPANDED ASSET LIST (Make sure this matches exactly)
TICKERS = {
    'BTC': 'BTC-USD',
    'SP500': '^GSPC',   # S&P 500
    'NASDAQ': '^NDX',   # Nasdaq 100 (Tech)
    'ETH': 'ETH-USD',   # Ethereum
    'DXY': 'DX-Y.NYB',  # Dollar Index
    'VIX': '^VIX',      # Volatility Index
    'US10Y': '^TNX',    # 10Y Treasury Yield
    'GOLD': 'GC=F',     # Gold Futures
    'OIL': 'CL=F',      # Crude Oil
    'MSTR': 'MSTR'      # MicroStrategy
}

# 2. LOWER THRESHOLD TO SAVE PRICE/VOLUME
# Previous 0.03 was too strict. 0.012 (1.2%) keeps Price (2.5%) and Volume (2.8%)
CORR_THRESHOLD = 0.012 
OUTPUT_FILE = 'processed_data.csv'

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def fetch_and_process_data():
    print("--- Step 1: Fetching Data ---")
    
    # 1. Fetch BTC (Anchor)
    print(f"Fetching {TICKERS['BTC']} (Anchor)...")
    btc = yf.download(TICKERS['BTC'], start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
    
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    if 'Adj Close' in btc.columns:
        btc['Close'] = btc['Adj Close']
    
    # Indicators
    btc['BTC_Vol_30d'] = btc['Close'].pct_change().rolling(30).std()
    btc['BTC_RSI'] = calculate_rsi(btc['Close'])
    btc['BTC_MACD'] = calculate_macd(btc['Close'])
    btc['BTC_ATR'] = calculate_atr(btc['High'], btc['Low'], btc['Close'])
    btc['BTC_Volume_Log'] = np.log(btc['Volume'] + 1)
    btc['BTC_LogRet'] = np.log(btc['Close'] / btc['Close'].shift(1))
    
    # Base DF
    df = btc[['BTC_LogRet', 'BTC_Vol_30d', 'BTC_RSI', 'BTC_MACD', 'BTC_ATR', 'BTC_Volume_Log']].copy()
    
    # 2. Fetch Macros
    for name, ticker in TICKERS.items():
        if name == 'BTC': continue
        
        print(f"Fetching {name} ({ticker})...")
        try:
            macro = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
            
            if isinstance(macro.columns, pd.MultiIndex):
                macro.columns = macro.columns.get_level_values(0)
            
            price_col = 'Adj Close' if 'Adj Close' in macro.columns else 'Close'
            
            # Weekend Fix
            macro = macro.reindex(df.index)
            macro[price_col] = macro[price_col].ffill()
            
            # Log Returns
            macro_ret = np.log(macro[price_col] / macro[price_col].shift(1))
            macro_ret.name = f'{name}_LogRet'
            macro_ret = macro_ret.fillna(0.0)
            
            df = df.join(macro_ret)
            
        except Exception as e:
            print(f"Warning: Failed to fetch {name}: {e}")

    # Target
    df['TARGET'] = df['BTC_LogRet'].shift(-1)
    df.dropna(inplace=True)

    print(f"\nData aligned. Shape: {df.shape}")
    
    # --- Step 2: Feature Selection ---
    print("\n--- Step 2: Scientific Feature Selection ---")
    correlations = df.corr()['TARGET'].drop('TARGET')
    
    print("Feature Correlations with Target:")
    print(correlations.sort_values(ascending=False))
    
    # Gatekeeper
    selected_features = correlations[abs(correlations) > CORR_THRESHOLD].index.tolist()
    
    # SAFETY NET: If BTC Price/Volume are dropped, force them back in
    must_have = ['BTC_LogRet', 'BTC_Volume_Log']
    for f in must_have:
        if f not in selected_features and f in df.columns:
            print(f"  -> Rescuing {f} (Critical for LSTM)")
            selected_features.append(f)
    
    if not selected_features:
        print("WARNING: Threshold too strict. Reverting to all.")
        selected_features = correlations.index.tolist()
    else:
        print(f"\n[PASS] {len(selected_features)} features selected (Threshold > {CORR_THRESHOLD} or Whitelisted)")
        print(f"Final Features: {selected_features}")

    final_df = df[selected_features + ['TARGET']]
    final_df.to_csv(OUTPUT_FILE)
    print(f"\nSuccess! Processed data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_and_process_data()