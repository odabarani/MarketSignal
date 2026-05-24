[README.md](https://github.com/user-attachments/files/28197204/README.md)
# MarketSignal

**ML-powered stock direction prediction with backtesting and risk metrics**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit)](https://stockpredictor-od.streamlit.app/)
&nbsp;
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
&nbsp;
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)

---

## Live Demo

**[stockpredictor-od.streamlit.app](https://stockpredictor-od.streamlit.app/)**

<img width="1206" height="1132" alt="Screenshot 2026-05-24 at 2 22 06 PM" src="https://github.com/user-attachments/assets/ebf27da8-1217-4b53-9c9f-0606727f68a1" />

---

## What It Does

MarketSignal is a Streamlit web app that predicts the 5-day price direction (up or down) of any US-listed stock using an XGBoost classifier trained on 20 technical and fundamental signals.

Enter any ticker — the app trains a model on historical data, generates a BUY / SELL / HOLD signal with a confidence score, visualizes price history with a 30-day projected range, and runs a backtest reporting Sharpe ratio and max drawdown on the out-of-sample test period.

---

## Features

**Technical Signals**
- Moving averages (5-day, 20-day), momentum, volatility
- RSI, MACD, MACD signal line, Bollinger Band position
- Volume change, volume ratio
- Candlestick structure: body size, upper wick, lower wick, gap

**Macro & Relative Signals**
- VIX (market fear index)
- RS_SPY (price strength relative to S&P 500)

**Fundamental Signals**
- P/E ratio, profit margin, revenue growth, debt-to-equity

---

## Tech Stack

| Layer | Tools |
|---|---|
| Model | XGBoost, scikit-learn |
| Data | yfinance, pandas, NumPy |
| Visualization | Plotly |
| App | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git / GitHub |

---

## Project Structure

```
stock_predictor/
│
├── app.py                  # Main Streamlit app
├── requirements.txt
├── runtime.txt             # Python 3.11
│
└── src/
    ├── data_loader.py      # Pulls historical price data via yfinance
    ├── features.py         # Computes all 20 technical/fundamental features
    ├── model.py            # Trains XGBoost classifier, returns CV accuracy
    ├── signals.py          # Converts prediction + confidence into BUY/SELL/HOLD
    └── backtest.py         # Simulates $10,000 portfolio; computes Sharpe + max drawdown
```

---

## How to Run Locally

```bash
git clone https://github.com/odabarani/stock_predictor.git
cd stock_predictor
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.11. No API keys needed.

---

## Backtesting Methodology

- Data split: **70% train / 30% test** (chronological — no shuffling)
- Starting capital: **$10,000**
- Signals with confidence above 65% trigger a position
- Reports: total return, Sharpe ratio, max drawdown, peak portfolio value

---

## Known Limitations

- **Fundamental data is not point-in-time.** P/E ratio, profit margin, revenue growth, and debt-to-equity are sourced as current values via yfinance, not historical snapshots. This introduces mild look-ahead bias in the fundamental features.
- **No transaction costs.** The backtest assumes zero friction. Real-world slippage and commissions would reduce returns.
- **No benchmark comparison.** Performance is not compared against a buy-and-hold SPY baseline. A positive return does not imply alpha.
- **This is not financial advice.** MarketSignal is a portfolio project built for educational purposes.

---

## Author

**Othman Dabarani** — Finance & Computer Science, University of Ottawa

[LinkedIn](https://linkedin.com/in/othmandabarani) · [GitHub](https://github.com/odabarani)
