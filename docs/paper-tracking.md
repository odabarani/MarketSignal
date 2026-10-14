# Paper tracking

Run `python paper_signal.py` after the US market close. It waits for a complete
daily bar and commits one forecast every five sessions. Each signal uses only
outcomes available by that close and assumes execution at the next session's
open. The five-session return target exits at the sixth subsequent open.

The log compares the historical-mean, Ridge, and XGBoost forecasts with cash,
SPY, equal-weight volatility sizing, and momentum. It preserves prior holdings,
charges 15, 30, or 50 basis points per side in portfolio marks, and records
realized forecast errors once the exit open is available. Paper equity marks
remain invested and do not assume a terminal sale.

Each signal is append-only in `data/paper/signals.jsonl`; its compressed adjusted
price snapshot is saved under `data/paper/snapshots/`. The log includes hashes,
model settings, predictions, weights, execution timing, and later outcomes. Keep
both paths together as the research record. Run the command again after future
closes to reconcile matured forecasts and create a new signal when due. It never
sends orders to a broker.

Paper results are prospective evidence for this fixed research policy. They do
not prove future profitability. Yahoo may revise adjusted history, costs remain
assumptions, and the ETF universe does not replace point-in-time individual
stocks or delisted coverage.
