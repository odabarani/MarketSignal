# Expected-return experiment

This phase predicts **five-session absolute returns**, from the next open to
the sixth open. It keeps the overnight returns that the first liquidation
diagnostic missed. Existing positions stay invested until an allocation changes;
costs apply only to the traded difference and final liquidation.

The settings below are fixed before inspecting this experiment's results:

- **Models:** historical mean, Ridge regression (alpha 100), and shallow XGBoost
  regression (100 trees, depth 2). No parameter sweep or random time splits.
- **Inputs:** price-relative indicators and four market-context indicators.
  Future return labels and current analyst opinions are excluded.
- **Entry:** forecast is at least 30 bps round-trip costs plus 20 bps minimum edge.
  An existing stock may stay while its forecast remains positive. Choose up to
  three stocks; cap each at 35% of equity at rebalance.
- **Sizing:** scale allocations using trailing 20-session volatility toward a
  conservative 15% annual risk budget. This estimate cannot cap future losses.
- **Timing:** rebalance every five sessions at the next open. Cash earns zero;
  the final position is liquidated at the common ending open.
- **Selection:** 504 initial training dates, 126 selection dates, then 126
  evaluation dates. Purge outcomes unavailable at either fitting boundary.
  Select the highest net compounded return among candidates with at most 20%
  selection drawdown; cash and simple baselines are eligible. Refit the model
  only on outcomes available before evaluation, then freeze it for that block.
- **Stress:** report 15, 30, and 50 bps per side using the same allocations.

The chronological selector is a complete research strategy, not a promise that
the highest-return historical model will win in the future. Selection windows
are short, estimates are noisy, and this five-stock survivor-selected data has
already been inspected. Broader historical coverage and prospective paper
trading remain required before promotion. Price returns omit dividends, taxes,
cash interest, FX, and execution impact.

[Ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
shrinks coefficients through regularization. Comparing return forecasts follows
the general research direction of [Gu, Kelly, and Xiu](https://www.nber.org/papers/w25398);
this small daily experiment does not replicate their broader asset-pricing study.

```bash
python return_pipeline.py --input data/feature_comparison.csv
```

The matching manifest supplies the development cutoff. Outputs include every
candidate, selection decisions, daily equity, trades, cost stress, annual returns,
and an offline comparison chart. Each run is recorded in the experiment ledger.

## First results — October 13, 2026

Thirteen evaluation windows cover August 2, 2017 to February 2, 2024, with 8,160
stock-day forecasts per model. At the assumed 15 bps per side:

| Portfolio | Annualized net return | Max drawdown |
| --- | ---: | ---: |
| Equal-weight stocks with volatility sizing | 18.18% | -19.93% |
| Ridge return model with volatility sizing | 17.17% | -17.84% |
| Chronological selector | 13.17% | -16.54% |
| SPY buy-and-hold | 11.04% | -34.10% |
| SPY with volatility sizing | 8.49% | -21.01% |

The chronological selector remains at 11.55% annualized at 50 bps per side,
with a -16.77% drawdown. This is better than the first liquidation diagnostic
on this historical sample. The target, sizing, holding rule, and final endpoint
all changed, so this is not an isolated test of any one change. **The learned models did not beat the simple
equal-weight baseline**. Their return forecast errors also exceeded the
historical-mean model's errors: RMSE 4.78% for Ridge and 4.75% for XGBoost,
versus 4.73% for the historical mean.

Ridge trades approximately 14 times portfolio NAV per year; the equal-weight
rule trades about 3 times. More orders do not necessarily mean more turnover:
small rebalances cost less than replacing an entire position. The selector
chose cash for three of thirteen windows, and otherwise mixed models and
baselines. Its choices used prior selection outcomes, not each window's future.

No candidate is promoted. These stocks were selected with knowledge of today's
survivors, and results have no statistical superiority or untouched-test claim.
The subsequent [ETF test](etf-locked-evaluation.md) broadened the universe and
again found no learned model that beat the simple baselines. The app's live-facing
model is not replaced by this development experiment; further strategy changes
need new forward evidence before promotion.
