# 📈 Bitcoin Direction Predictor (LSTM Neural Network)
**Bachelor's Thesis Project in Quantitative Finance & Machine Learning**

## 🎯 Overview
This repository contains a production-ready, two-step Python pipeline designed to predict the daily directional movement of Bitcoin (BTC-USD) using a Long Short-Term Memory (LSTM) Deep Learning architecture.

The project goes beyond standard accuracy metrics by framing the prediction as a quantitative trading problem, introducing a **3-State Confidence Logic (Long / Short / Do Nothing)** and generating simulated equity curves.

## 🚀 Pipeline Structure
1. `01_data_generator.py`: Fetches 24/7 crypto data and 5/2 traditional market data (S&P500, Gold, NVDA, DXY) via `yfinance`. Synchronizes time-series using forward-fill, engineers log-returns, and builds the target variable.
2. `02_lstm_model.py`: Implements a strictly chronological split (Train/Val/Test) to prevent look-ahead bias. Builds a regularized 2-layer LSTM (L2 + Dropout) and evaluates it using advanced financial metrics (ROC AUC, Gini, Conditional Win Rate).

## 📊 Current Baseline Results
The baseline model achieves the following on the **Out-of-Time Test Set (2025)**:
* **Variant A (Always in Market):** ~54% Accuracy | AUC: 0.508
* **Variant B (3-State Logic):** Proves that restricting trades to high-confidence signals (Coverage ~4-10%) drastically increases the Conditional Win Rate (up to ~76% on Validation data).

*(See the `/plots` folder for full ROC curves, Confusion Matrices, and Simulated Equity curves).*

## 🛠️ Next Steps (Roadmap)
- [x] Baseline architecture and data synchronization.
- [x] Anti-overfitting measures (Chronological splitting, L2, Dropout).
- [ ] Hyperparameter tuning (GridSearch / Optuna) to improve Test Set AUC.
- [ ] Walk-forward validation (Rolling Window).
