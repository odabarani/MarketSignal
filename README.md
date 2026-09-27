# MarketSignal

MarketSignal is a **research dashboard** that estimates whether a stock will close higher in five trading sessions.

It includes:

- **Walk-forward validation** — trains on the past and tests on later data.
- **Label embargo** — blocks five-day outcomes that were not known at training time.
- **Current inference** — uses the newest feature-ready market row.
- **Next-open backtest** — decisions made after the close enter at the next open.
- **Trading friction** — includes transaction cost and slippage.
- **Baselines** — compares the strategy with the stock, SPY, and a majority-class prediction.
- **Calibration** — reports **Brier score**, log loss, and a probability chart.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project map

- `app.py` — Streamlit dashboard
- `src/features.py` — features and 5-day target
- `src/model.py` — models and walk-forward validation
- `src/backtest.py` — execution and portfolio math
- `tests/` — regression tests

Yahoo Finance is used for market data. This project is for education and research, not financial advice.
