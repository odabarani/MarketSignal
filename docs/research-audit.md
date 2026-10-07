# Research calculation audit

This is a source review with regression checks, not an independent scientific
review or evidence of a profitable strategy.

## Corrections delivered

| Issue | Current behavior | Evidence |
| --- | --- | --- |
| Cash dividends omitted | Stock/SPY returns include dividends for held ex-dates; entry on ex-date does not earn that dividend | Known-value dividend/drop, entry-date and cash-position tests |
| Split handling unclear | Request action metadata; retain provider split-normalized OHLC without multiplying split factors again | Constant normalized split fixture; live AAPL 4-for-1 event on 2020-08-31 |
| Entry cost treatment | Costs reduce invested capital before the interval return | Known one-interval arithmetic test |
| Terminal exposure uncharged | Remaining long exposure is liquidated at final realized open with friction | Entry/final-exit arithmetic and turnover test |
| Evaluation-majority baseline | Show a separate training-majority baseline refitted using known labels | Opposing training/evaluation majority test |
| Invalid label availability | Reject nonbinary targets and labels available at/before feature date | Label-contract tests; poisoning future labels leaves first fit unchanged |
| Missing benchmark/feature rows | Stop instead of silently bridging missing benchmark or interior feature rows | Missing-session regression tests |
| Old provenance after policy change | Preserve old snapshot IDs/policies and reports; refresh legacy latest data with action metadata | SQLite migration and refresh test |
| Ambiguous win rate | Report positive interval rate explicitly; do not call it profitable-trade rate | Metric definition in backtest module |

The dashboard target remains whether the close five sessions later exceeds the
decision-date close. Dividend-inclusive simulated portfolio returns answer a
different question. The separate multi-stock pipeline uses cost-aware excess
returns from next-open entry, including dividends earned after entry.

Model fitting is defined as after the session close. A historical label ending at
that same close is available; future-ending labels are excluded. This convention
does not imply an order could fill at that close: execution remains next open.

## Important remaining limits

- Yahoo histories may be revised and retrospectively split-normalized. They are
  not archived point-in-time feature inputs; a stronger provider/archive is needed
  for a strict historical information-set claim.
- Dividends are credited and fractionally reinvested at the ex-date open as a
  total-return approximation. Actual payment dates, tax, lot sizing and cash drag
  are not modeled.
- Gross benchmarks exclude transaction costs. Strategy costs are configurable,
  uniform per turnover, and do not model liquidity, impact or partial fills.
- Missing benchmark/feature rows and XNYS regular-session gaps now fail. Calendar
  checks cover holidays, early closes and daylight saving, with a two-completed-
  session freshness cap. Asset venue identification, alternate schedules,
  suspensions and delisted-universe coverage remain unsupported.
- Five-session forecast outcomes overlap. The simple evaluation metrics do not
  provide independent-trial confidence intervals or guarantee calibrated estimates.
- The dashboard does not have a locked final test after strategy selection.
  Repeatedly changing thresholds/model choices can overfit the visible period.
- Multi-asset/regime robustness, survivorship bias, economic significance and
  independent methodology review remain prerequisites for public efficacy claims.
- Local tests and live-provider checks do not establish public data rights or
  production reliability.

Next research milestones: integrate locked evaluation into the public workflow;
cost-sensitive multi-asset/regime checks; independent review and prospective
protocol registration.

Offline replay is now implemented and tested against exported reports. A fresh
Python 3.11 virtual environment installed from the hashed lockfile reproduced a
live AAPL report's metrics, estimates and equity curves across 880 predictions,
without changing the retained SQLite database. This is same-host evidence, not
a different-machine/hardware reproducibility claim. Source/runtime mismatches and
tampered reports fail visibly. Locked strategy selection/final evaluation is now implemented in the operator
workflow described in locked-studies.md. A live AAPL CLI study recorded all nine
validation candidates and 587 final predictions; repeated final requests returned
the identical retained result. This is workflow validation, not an efficacy claim.
Next milestones are public integration, multi-asset/regime checks and independent
review. Historical data already viewed remains retrospective evidence.
