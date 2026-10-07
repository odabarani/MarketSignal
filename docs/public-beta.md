# Public research beta

## Product scope

First release: help an individual investigate a US stock using historical daily
data, a five-session direction estimate, and a cost-aware simulated strategy.
The user chooses a ticker, runs research, inspects evidence and benchmarks, then
exports a reproducible result. Public users are the audience; this is not yet a
production launch. Frontier-economy exchange coverage is outside this initial
scope and has not been validated.

The first beta does not execute orders, connect brokerage accounts, or promise
investment returns. Paper trading is a later milestone.

## Repository audit

Findings are based on source inspection and local tests, not a live-data or
production load audit.

| Area | Existing capability | Remaining work |
| --- | --- | --- |
| Research | Chronological walk-forward evaluation; label availability filtering; fixed model seeds | Independent review of fit-time boundaries; regime and universe testing; locked final evaluation |
| Simulation | Next-open long/cash, cash dividends, entry/exit costs, final liquidation, stock/SPY comparisons | Execution sensitivity, asset-venue identification and point-in-time data review |
| Probabilities | Brier score, log loss, empirical calibration chart | The model is not explicitly probability-calibrated; chart is diagnostic. Add calibration fitted only on historical training data if evidence supports it |
| Baselines | Dividend-inclusive stock/SPY curves, train-derived majority predictions and retrospective majority accuracy | Benchmarks are gross of costs; disclose selection effects and test different regimes |
| Targets | Dashboard predicts close-to-close UP/DOWN; research pipeline builds cost-aware excess-return labels | Clearly distinguish the separate targets; do not claim dashboard uses the pipeline's three-class labels |
| Data | Yahoo corporate actions with strict SPY/VIX session alignment; live access verified locally | Licensing, alternate exchange schedules and point-in-time archives |
| Caching | Retained SQLite snapshots, one-hour provider-check expiry, fits keyed by dataset identity | Persistent hosting volume, retention policy, cross-worker refresh deduplication and load testing |
| Experiments | CLI dataset fingerprint and JSON-lines experiment helper | Connect provenance to dashboard; retain dataset, settings, code/dependency versions, model seed and results together |
| Hosting | Render blueprint, Gunicorn, CI pytest | Dependency lock, background jobs, request limits, monitoring, health checks, deployment smoke and load tests |
| Users | Single shared research dashboard | Decide anonymous versus authenticated beta; add persistence and access controls before private saved research |

## First delivered slice

- Welcome screen explains the workflow without downloading data or fitting on page load.
- Run button is disabled while its research callback runs.
- Results show market date, history size, prediction period, evaluated sample size,
  and the fact that overlapping targets are not independent observations.
- Results show completed-session lag using the XNYS regular US equity calendar.
  New research rejects more than two completed sessions of lag; stale retained
  data triggers immediate provider refresh before fitting.
- Market requests now retain content-addressed snapshots and validate required
  prices, OHLC consistency, finite values and date ordering. Expired provider
  checks refresh on demand; errors do not silently fall back to expired data.
- Results identify their snapshot and its acquisition date. Fits and comparisons
  are tied to that dataset; comparisons no longer follow edited sidebar controls.
- Server-side settings reject invalid tickers, unsupported ranges and nonfinite values.
- Completed runs now retain downloadable JSON reports with parameters, dataset
  identity, source hashes, runtime versions, metrics, benchmark metrics and equity
  curves. Export loads the retained report without downloading or refitting, and
  ignores mutable sidebar settings. Undefined metrics become JSON null.
- Research and comparisons now use one persistent subprocess with four waiting
  slots, pending-request deduplication, a 12-new-jobs/minute process admission
  limit, 180-second hard task timeouts and two native compute threads. The page
  shows status and preserves previous results on failure. Completed responses
  have a 15-minute/16-result retention bound. Worker death and timeout recovery
  are covered by actual subprocess tests.

Queue limits: this is a single-web-process design, matching the current Gunicorn
configuration. Queue state is not durable or shared across replicas. Pending
requests share work, but admission controls do not establish per-user fairness.
Production load, memory limits, gateway controls and web-server restart behavior
still need deployment validation.

Deployment foundation now includes liveness/readiness probes, correlated HTTP/job
logs, generic unexpected-error responses and a reusable deployment smoke script.
Render's configured health check uses `/ready`. External monitoring/alerts,
persistent hosted storage, backup recovery and a real staging release remain open.

Remaining snapshot limits: no alternate exchange-calendar support, no point-in-time
corporate-action archive, no retention cleanup or multi-worker download deduplication. Local
storage is not a production persistence guarantee. The provider may return old
data after a successful refresh; the date warning remains necessary.

Report limits: the download references a locally retained snapshot but does not
include raw prices or model artifacts. Offline replay is implemented with source,
lockfile and runtime checks; different-machine reproduction is unverified.
Reports have no user ownership or privacy boundary; add access control
before treating saved research as private. Both snapshots and reports need a
production retention and backup policy.

## Prioritized backlog and acceptance criteria

### P0 — Trustworthy, reproducible data

1. Confirm an appropriate provider and public-display/export rights. Document
   entitlement, coverage and permitted uses; verify AAPL, SPY and VIX access from
   the deployed environment. Yahoo access was blocked during onboarding and
   must be rechecked after network settings apply.
2. Store versioned data snapshots with acquisition time, source, date range,
   corporate-action policy and fingerprint. Reject malformed, duplicated or
   misaligned sessions. Test revisions, missing benchmarks and stale responses.
3. Replace indefinite caches with snapshot-aware expiry. A refresh must produce
   a new result identity; repeated requests for the same snapshot should reuse work.
4. Export a research report containing snapshot identity, full parameters,
   code/dependency versions, sample dates, metrics and limitations. A replay using
   the retained snapshot must reproduce the result within documented tolerance.

### P0 — Independent research review

1. Audit time boundaries with adversarial tests showing no unavailable label can
   enter a fit. Separate research tuning from a locked final test period.
2. Verify split/dividend handling and benchmark methodology. Test a known
   corporate-action example and missing-session example before publishing returns.
3. Report trade count, time in market, turnover and benchmark metrics alongside
   strategy metrics. Document zero risk-free rate and 252-session annualization.
4. Evaluate several assets and market regimes; disclose selection bias and
   overlapping outcomes. Do not use repeated experiments to advertise the best
   historical run as expected performance.

### P0 — Reliable public requests

1. Validate ticker, numerical ranges and supported history on the server before
   downloads or fitting. Invalid and nonfinite values must produce useful errors.
2. Move fits to bounded background jobs with status, timeout and deduplication.
   Limit request rates and concurrency. Test concurrent requests, provider outage
   and worker restart; no unbounded fits or resource exhaustion.
3. Log internal failures with request IDs; show actionable public errors without
   exposing exception internals. Add health checks and error monitoring.
4. Pin tested dependencies and verify clean installation in CI. The existing
   broad requirements ranges do not guarantee reproducible future installations.

### P1 — Polished beta workflow

1. Let users export completed research before adding accounts. Results must keep
   the parameters of the completed run even when controls are subsequently edited.
2. Bind model comparison to the displayed run's ticker/snapshot, not mutable
   sidebar controls; label which run each comparison belongs to.
3. Validate keyboard navigation, readable mobile charts, loading feedback and
   recovery from failed runs in a real browser. Local HTTP tests alone do not
   demonstrate accessibility or visual quality.
4. If accounts and private saved runs are included, use managed authentication,
   server-side authorization, retention/deletion controls and backup/restore tests.

### P1 — Release and operation

1. Deploy a staging instance with HTTPS, environment-specific configuration and
   a repeatable release process. Run an actual research request, export and failure
   recovery smoke test there.
2. Establish monitoring, resource budgets, support contact, rollback procedure and
   an incident response owner. Test rollback before inviting users.
3. Complete data-rights and jurisdiction-appropriate product/legal review, privacy
   notice and terms. A research disclaimer alone does not establish compliance.
4. Invite a small beta cohort; measure run completion, time to result, failure
   rate and whether users understand the results. Broaden access after the P0
   criteria and staging checks pass.

## Later milestone: paper trading

Use a separate simulated portfolio ledger with timestamped orders and fills,
position limits, stale-data protection, reconciliation and an audit trail. Keep
historical backtests distinct from forward paper performance. Live trading needs
its own broker integration, security, risk controls and release review.

## Research audit corrections

The dashboard and CLI locked-study workflow records 60/20/20 chronological date
boundaries and a fixed nine-candidate grid. Validation slices raw history before
feature/label generation and freezes settings without final scores. Final
evaluation consumes its period once; later calls return the retained result.
The editable research dashboard/reports remain explicitly exploratory. The separate
locked-study panel supports queued stages, reload recovery, and retained-record
downloads. See locked-studies.md for historical-data and shared-ledger limitations.

Cash dividends now enter strategy/benchmark returns only for held ex-dates; final
liquidation pays friction and entry friction reduces invested capital. Yahoo
split-normalized OHLC is not adjusted twice. The forecast remains price direction,
while the separate pipeline's excess-return target now credits held dividends.
Training baseline predictions use only labels known at fit time. Fit occurs after
the session close, so labels ending at that close are available. Invalid label
contracts and interior feature gaps fail visibly. Legacy dataset/report identities
are preserved; refreshed datasets request actions. Revised histories are still
not point-in-time archives, and ex-date dividend reinvestment is simplified.

Exchange-calendar checks cover holidays, early closes, daylight saving, gaps and
unfinished sessions. Differently scheduled/suspended assets are rejected. The
calendar does not prove the listing venue or point-in-time provider availability.
