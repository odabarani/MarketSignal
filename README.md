# MarketSignal

MarketSignal is a **Dash research app** for 5-day stock-direction experiments.

It uses **walk-forward validation**, a **label embargo**, **next-open execution**, trading costs, baselines, and probability calibration.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

## Build the Mac app

Double-click `build_macos.command`, or run:

```bash
./build_macos.command
```

The regular Mac application is created at `dist/MarketSignal.app`.

## Publish it

`render.yaml` and `Procfile` are included for public hosting. Connect this GitHub repository to Render and deploy the detected web service.

Yahoo Finance provides the market data. This project is for education and research, not financial advice.

## Research pipeline

Build a multi-stock dataset before testing new models:

```bash
python research_pipeline.py --tickers AAPL MSFT NVDA AMZN GOOGL
```

This creates **DOWN**, **NO TRADE**, and **UP** targets using next-open returns,
costs, an SPY comparison, and locked **train**, **validation**, and **test** dates.
Generated datasets stay local. Use a dated ticker list to reduce survivorship bias.

Model selection uses validation data only. Open the final test once with
`--evaluate-test`; the result and configuration are added to the experiment ledger.
