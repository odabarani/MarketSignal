# Locked historical evaluation

The public dashboard remains exploratory: users can change settings after seeing
results. Its exported reports explicitly identify that protocol. The locked-study panel and operator CLI provide validation-only selection and a frozen final check.

In the app, choose a ticker and use **Prepare / resume ticker study**, then
**Run validation & freeze settings**, then **Evaluate final period once**.
Preparation downloads data without displaying exploratory scores. Each stage
uses the bounded research queue. The browser tab retains study/job IDs across
reloads; the server retains the authoritative protocol, selection and result.
Download the study record to preserve those records. Existing studies for the
same snapshot are shared and resumed, not duplicated; there is no account privacy.
After a server restart, a retained study can still be viewed and downloaded.
Interrupted stages require operator review; final stages cannot be reset in the UI.

## Procedure

Start from a retained, action-aware dataset snapshot. Its ID appears in a research
report. Run from the repository using its pinned Python environment:

```bash
python -m src.studies prepare --store data/snapshots.sqlite --snapshot-id DATASET_ID
python -m src.studies validate --store data/snapshots.sqlite --study-id STUDY_ID
python -m src.studies finalize --store data/snapshots.sqlite --study-id STUDY_ID
python -m src.studies status --store data/snapshots.sqlite --study-id STUDY_ID
```

`prepare` returns STUDY_ID and records the exact dataset, source/runtime, 60/20/20
chronological date boundaries, XGBoost configuration, costs and candidate grid.
The grid is thresholds 55%, 60%, 65% crossed with retraining every 21, 42, 63
sessions. Costs are 10 bps plus 5 bps slippage; initial capital is USD 10,000.

`validate` slices raw history at the validation boundary *before* creating features
and targets. Targets ending after that boundary are unavailable and excluded.
It records all nine trials, selecting the highest net strategy Sharpe; candidate
order breaks ties. Predictions are reused across threshold choices with identical
retrain intervals. Settings are frozen automatically; the CLI has no manual
threshold override after selection.

`finalize` evaluates the frozen settings on later prediction dates. The refit rule
is predeclared expanding walk-forward: earlier final-period labels may train later
fits once known, without changing strategy parameters. Final-period scores never
select parameters. Costs, dividends, final liquidation and gross benchmark
comparisons match the dashboard simulation.

The final JSON is labeled `locked_historical_final_evaluation` and includes the
protocol, validation trials, frozen settings, final sample and metrics. Redirect
stdout to save an artifact if needed. No source/lockfile changes should occur
between preparation and evaluation; differing recorded versions are rejected.

## Exposure and recovery

- One study is allowed per snapshot in the local store. An atomic state transition
  prevents concurrent scoring of a stage. Protocols, selections and results carry
  integrity checksums; these detect accidental changes, not malicious authorship.
- Completed final requests return the saved result without recomputing or selecting
  again. Starting another study on that snapshot is rejected.
- A failed final stage is consumed, since data may already have been inspected.
  It is not silently retried. Interrupted stages remain marked running/consumed;
  inspect the incident and preserve records rather than reset the ledger casually.
- A failed validation stage can retry the same recorded grid and protocol, since
  no final scores have been exposed. It cannot change the grid, dates or costs.
- CLI failures return nonzero. Maintain consistent backups of the SQLite store.
  These studies currently use local storage and have no account-level privacy or
  durable distributed-worker orchestration.

## Interpretation

Historical holdout discipline cannot make previously viewed prices/results
unseen. A researcher can bypass the local ledger by copying datasets or creating
another store; this is a workflow safeguard, not a security boundary. A genuinely
prospective study needs an independently recorded protocol before future data
arrives. The current fixed universe may also have survivorship/selection bias.

Nine-candidate selection still overfits validation to some degree. Final returns
do not establish profitability, calibrated probabilities or economic significance.
Overlapping outcomes, revised prices, simplified reinvestment/execution and gross
benchmarks remain limitations. Multi-asset/regime robustness and independent
review are still open.
