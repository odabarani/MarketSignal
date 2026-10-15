# Paper tracking

Run `python paper_signal.py` after the US market close. It waits for a complete
daily bar and commits one forecast every five sessions. Each signal uses only
outcomes available by that close and assumes execution at the next session's
open. The five-session return target exits at the sixth subsequent open.

The log compares the historical-mean, Ridge, and XGBoost forecasts with cash,
SPY, equal-weight volatility sizing, and momentum. It preserves prior holdings,
charges 15, 30, or 50 basis points per side in portfolio marks, and records
realized forecast errors once the exit open is available. Reconciliation uses
the horizon saved with each forecast, even if a later policy changes. Paper
equity marks remain invested and do not assume a terminal sale.

Each signal is append-only in `data/paper/signals.jsonl`; its compressed adjusted
price snapshot is saved under `data/paper/snapshots/`. When a forecast matures,
the exact updated data used to calculate its outcome is saved under
`data/paper/reconciliation-snapshots/`, with its hash and the reconciliation
code fingerprint in the event. Keep the ledger and both snapshot folders
together as the research record. Run the command again after future closes to
reconcile matured forecasts and create a new signal when due. It never sends
orders to a broker.

Paper results are prospective evidence for this fixed research policy. They do
not prove future profitability. Yahoo may revise adjusted history, costs remain
assumptions, and the ETF universe does not replace point-in-time individual
stocks or delisted coverage.

## Separate prospective stock test

The fixed 37-stock roster in
[`configs/paper-stock-universe-v1.json`](../configs/paper-stock-universe-v1.json)
starts a separate prospective test. Its roster is spread across 11 sectors and
has a content hash recorded with each new signal. Keep the file unchanged; make
a new version and a new ledger if the roster changes.

Run it after the market close with its own ledger and snapshots:

```bash
python paper_signal.py \
  --universe-file configs/paper-stock-universe-v1.json \
  --ledger data/paper/stocks-v1/signals.jsonl \
  --snapshot-dir data/paper/stocks-v1/snapshots \
  --reconciliation-snapshot-dir data/paper/stocks-v1/reconciliation-snapshots
```

This begins a test of future predictions on a frozen stock basket; it does not
turn that basket's historical prices into a point-in-time backtest. The current
roster was chosen in 2026, so historical training and comparisons still have
survivor-selection and ticker-coverage limits. Public price feeds may also omit
a stock after it delists. Do not replace a missing ticker or score a terminal
outcome until its exit value is verified. Keep this stream separate from the
ETF ledger and do not combine their returns into one claim.
