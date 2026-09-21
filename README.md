# MarketSignal

**Walk-forward ML research platform for 5-day stock-direction classification and strategy backtesting.**

## What it does
MarketSignal estimates whether a stock will close higher or lower five trading days after a given trading day. It combines technical and market-regime features with chronological walk-forward validation and a long/flat backtest.

This is a research and education project, not a trading recommendation system.

## Methodology

### Target
`Target = 1` when `Close[t+5] > Close[t]`, otherwise `0`. The final five rows are excluded because their outcomes are not known.

### Features
The model uses moving averages, momentum, return volatility, volume behavior, RSI, MACD, Bollinger position, VIX, relative strength versus SPY, candlestick structure, and overnight gaps.

Current Yahoo Finance fundamentals are intentionally excluded. Current P/E, margin, growth, and debt/equity values are not point-in-time historical observations, so applying them to old rows can introduce look-ahead bias.

### Walk-forward validation
The first 70% of observations form the initial training set. The model predicts the next unseen block, expands the training window, and retrains on a configurable schedule. No random shuffling is used.

Reported classification metrics: accuracy, precision, recall, F1, ROC-AUC, and log loss.

### Backtest
Signals are generated from the model probability. High-confidence UP predictions create a long position; high-confidence DOWN predictions move the strategy to flat; low-confidence predictions retain the previous position.

Signals formed from a day's close are applied to the following day's return. The backtest includes configurable transaction costs and slippage and compares the strategy with buy-and-hold SPY.

Reported strategy metrics: total return, annualized return, volatility, Sharpe ratio, maximum drawdown, and win rate.

### Model comparison
The app compares Logistic Regression, Random Forest, and XGBoost using the same chronological walk-forward process.

## Project structure
```text
MarketSignal/
├── app.py
├── requirements.txt
├── runtime.txt
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── signals.py
│   └── backtest.py
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_backtest.py
└── .github/workflows/tests.yml
```

## Important limitations
- Yahoo Finance is not an institutional point-in-time market-data source.
- Transaction costs and slippage are simplified assumptions.
- The strategy is long/flat rather than a full execution simulator.
- The 60% threshold is a configurable research parameter, not a guaranteed optimal cutoff.
- Historical backtests do not establish future performance.
- This project is for educational and portfolio purposes and is not financial advice.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author
**Othman Dabarani — Finance & Computer Science, University of Ottawa**

[GitHub](https://github.com/odabarani/MarketSignal)