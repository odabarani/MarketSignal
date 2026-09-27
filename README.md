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
