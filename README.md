# MarketSignal

MarketSignal is a **Dash research app** for 5-day stock-direction experiments.

It uses **walk-forward validation**, a **label embargo**, **next-open execution**, trading costs, baselines, and probability calibration.

## Run locally

Use Python 3.11.16 for the validated workflow. The lockfile includes platform and
Python-version markers; other interpreter versions are not covered by the current
readiness evidence.

```bash
pip install --require-hashes -r requirements.lock
python app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

Research begins when you select **Run research**. Aligned daily stock, SPY and
VIX data is retained in `data/snapshots.sqlite`. Results show the dataset ID;
model fits reuse that exact snapshot. Provider checks expire after one hour and
refresh on the next research request. Failed refreshes produce an error rather
than silently using expired data. The latest market date is shown separately
from the download date: refreshing does not guarantee recent provider data.

Set `MARKETSIGNAL_DATA_DIR` to choose a different local storage directory. Hosted
instances need persistent storage to retain snapshots across deployments. The
store currently has no automatic retention cleanup; monitor disk usage. Snapshot
prices use 15-digit JSON precision, and calculations use the retained values.
Downloads request Yahoo corporate actions with `auto_adjust=False`. Yahoo's OHLC
history is already split-normalized; the app does not apply split factors again.
Simulated returns credit cash dividends when held into their ex-date, charge
entry/exit costs and final liquidation, and approximate fractional reinvestment.
Buying at the ex-date open does not earn that dividend. The forecast remains
five-session **price direction**, distinct from dividend-inclusive simulated returns.
Benchmarks include dividends but exclude trading costs. Revised provider histories
are not point-in-time archives; split-normalization and simplified dividend timing
remain research limitations. Missing benchmark sessions or interior feature gaps
stop a run rather than silently changing the execution timeline.

The first beta uses the XNYS regular US equity session calendar. It checks for
missing/unexpected dates and measures data lag against completed sessions,
including holidays, early closes and daylight-saving changes. More than two
completed sessions of lag blocks new research; stale retained data triggers an
immediate provider refresh first. Differently scheduled or suspended assets are
unsupported in this workflow. This calendar check does not identify an asset's
listing venue or establish point-in-time price availability.

Legacy snapshots retain their original price policy and identity. New requests
refresh those snapshots to include actions; old saved reports remain unchanged.

After a successful run, select **Download research report** to save a JSON report.
It contains the completed run's parameters, dataset identity, evaluation sample,
classification and strategy metrics, stock/SPY benchmark metrics, simulated equity
curves, Python/package versions, source-file hashes and limitations. Editing the
sidebar does not change the report. Reports are retained alongside snapshots in
the same SQLite store; exporting does not download data or fit a model again.
Undefined numerical metrics are represented as JSON `null`.
New reports use schema version 2 and include training-derived baseline accuracy,
entry/exit counts, exposure and the final-liquidation policy. Earlier retained
reports keep their original schema and calculation assumptions.

The export references its retained market dataset; it does not include raw provider
prices or model artifacts. Keep the local store if you need the original data.
Offline replay refits from the retained dataset and verifies the metrics and curves.
Use the same source/runtime versions; different-hardware reproducibility is unverified.
Reports are currently anonymous research artifacts, not private account records.

### Offline report replay

Keep the downloaded report and its original SQLite snapshot store. From the same
source revision and Python/package versions, run:

```bash
python -m src.replay --report report.json --store data/snapshots.sqlite
```

Replay verifies the report checksum, recorded source/lockfile hashes and runtime,
then refits on the exact retained dataset without provider access or database
writes. It checks sample dates, model estimates, metrics and every equity-curve
value with absolute/relative tolerance `1e-8`. It exits nonzero on a mismatch.
It deliberately skips current-data freshness rules when replaying historical data.
Report timestamps and the freshness observation at creation are not recomputed.

Reports from changed code or package versions must be replayed with their recorded
environment; they are not silently upgraded. The checksum detects accidental
changes, not authenticity. No model artifact or downloaded provider prices are
embedded in the report. Separate-process and clean-venv replay passed on this host,
including a live AAPL report with 880 evaluated predictions; different-machine/
hardware reproduction requires its own check.

`requirements.txt` declares supported ranges; `requirements.lock` pins resolved
dependencies and artifact hashes for installation, CI and hosting. Refresh it
intentionally with `uv pip compile --python-version 3.11 --universal
--generate-hashes --output-file requirements.lock requirements.txt`, then rerun
the suite and deployment/replay checks before using the new lock.

### Background research

The beta runs research and comparisons in one persistent child process. The web
process returns a job ID immediately and the page polls once per second. Four
jobs may wait behind the active job; identical queued/running requests share a
job ID. New jobs are limited to 12 per minute across the web process. A 180-second
timeout terminates the child and starts a replacement for remaining work. Native
compute threads are limited to two during tasks. Provider requests have a
15-second timeout. Previous results remain visible if a new job fails.

Completed job responses expire after 15 minutes and are capped at 16; retained
reports and snapshots remain on disk. The worker's in-memory model caches survive
between successful jobs but not a timeout, crash or restart. Jobs are not durable:
a web-process restart drops queued work, and the page asks the user to rerun it.

Use **one Gunicorn web worker**, as in the supplied deployment configuration.
The queue and admission limit are process-local: multiple web workers/replicas
would each have their own limits and would not reliably route polling to the
correct job. Move to a shared durable queue before scaling out. Per-user quotas,
queue fairness, production memory limits and restart deployment tests remain
launch requirements.

See [the public-beta audit and backlog](docs/public-beta.md) for launch criteria.
See [the operations guide](docs/operations.md) for readiness probes, request IDs,
deployment smoke checks and remaining hosting requirements.
See [locked historical studies](docs/locked-studies.md) for the dashboard and CLI workflow
that freezes validation-selected settings before a separate final evaluation.

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

## Install on your phone or computer

MarketSignal now supports installation as a browser app on iPhone, Windows and
Mac, with a home-screen/dock icon and standalone window. Use the same hosted HTTPS
address on each device. See [installation and hosting](docs/install-app.md) for
steps, persistence limits and release checks. Internet is required for research.
