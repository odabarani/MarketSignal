# Feature comparison — September 28, 2026

No learned model beat the prior-probability baseline on validation **log loss**
(lower is better). This run does not establish predictive or trading value.

| Features | Best learned model | Log loss | Brier score |
| --- | --- | ---: | ---: |
| Prior baseline (all groups) | — | 0.954756 | 0.587029 |
| Original technical indicators | XGBoost | 0.954794 | 0.586012 |
| Price-relative indicators | XGBoost | 0.956086 | 0.589438 |
| Relative + market context | XGBoost | 0.957458 | 0.590708 |

**Ablation** means comparing feature groups on identical observations. Each group
ran the prior baseline, logistic regression, Random Forest, and XGBoost with fixed
settings. Selection used log loss; the small technical-model Brier improvement
does not override that rule. Differences have not been tested for significance.

The saved AAPL/MSFT/NVDA/AMZN/GOOGL snapshot used 8,465 training rows through
January 24, 2022, and 2,905 validation rows through May 23, 2024. All groups lost
the same 60-session warm-up rows. Boundary labels were purged. The previously
viewed test period was not evaluated again.

**Market context** includes SPY trend/volatility, VIX changes, same-day momentum,
volume and volatility ranks, and breadth within these five stocks. Rankings use
only that day's observations. This is a small, technology-heavy, survivor-selected
sample; breadth is not the breadth of the whole market. Yahoo histories can be
revised, and prices are not a point-in-time corporate-action database.

Snapshot SHA-256:
`8eababf7d8584498791ac9b0dca00a86cf75bcd63b8c1d87dc06c43404708c73`.
Full local manifests and the experiment ledger contain all candidate scores and
parameters. Next: purged walk-forward validation and chronological probability
calibration, then broader point-in-time data and prospective paper evaluation.
