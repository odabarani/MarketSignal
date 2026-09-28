# MarketSignal

MarketSignal is a **Dash research app** for 5-day stock-direction experiments.

It uses **walk-forward validation**, a **label embargo**, **next-open execution**, trading costs, baselines, and a probability calibration chart.

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

Research targets mean **underperform SPY**, **within the cost band**, and
**outperform SPY** over five sessions from the next open. These differ from the
app's stock-direction labels. A supplied ticker list is not a historical universe.

Compare raw indicators, **price-relative features**, and **market context** on
one saved dataset with fixed train/validation dates:

```bash
python research_pipeline.py --input data/research_panel.csv --output data/comparison.csv --compare-features
```

Every comparison is logged locally with features, settings, and code/data hashes.
`--evaluate-test` opens the final test; the same ledger blocks overlapping reuse.
Keep that ledger. An already viewed test period is no longer untouched.
See [research results](docs/research-results.md) for the latest validation findings.
