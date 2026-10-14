# Sector ETF locked evaluation

This is the first wider test of the return strategy. A fixed set of 19 US sector,
broad-market, international, and defensive ETFs supplies 2,091 aligned sessions
from 2018-06-19 through 2026-10-13. Prices use Yahoo Finance's dividend- and
split-adjusted OHLC. The frozen local CSV hash is
`be06329c882a81d7401f5b4cc9b00051e39306039b635fe236dd5f146215c67f`.

## Selection and holdout

Data through 2024-05-23 was used for the fixed candidate comparison. The rule
selected the highest annualized net return among candidates whose drawdown was
no worse than 20%, at 15 basis points per side. That chose **equal-weight ETF
volatility sizing**. SPY buy-and-hold returned more during development but its
24.5% drawdown failed that screen. Settings and the selected strategy were then
frozen before evaluating 2024-05-24 through 2026-10-13.

| Candidate | Annualized return | Maximum drawdown | Sharpe |
| --- | ---: | ---: | ---: |
| SPY buy-and-hold | 19.02% | -18.76% | 1.16 |
| SPY volatility target | 14.17% | -15.34% | 1.06 |
| Selected equal-weight ETF strategy | 11.12% | -10.75% | 1.11 |
| Ridge return model | 4.77% | -11.91% | 0.56 |
| Shallow XGBoost return model | -1.40% | -6.85% | -0.40 |

At 50 basis points per side, equal-weight ETF returns were 9.99% annualized and
SPY buy-and-hold returned 18.67%. The holdout therefore gives no evidence that
the learned models improve on simple market exposure. The selected strategy
held losses lower in this period, while SPY earned more and stayed inside the
20% historical drawdown screen. This is one historical holdout, not a claim
that either approach will remain profitable.

The holdout's development-defined VIX median was 18.71. Below or at that level,
SPY returned 107.6% and the selected strategy 58.7% across 422 sessions. Above
it, their returns were -27.2% and -19.1% across 176 sessions. These regime
groups describe this sample; they are not forecasts.

The evaluation and its chart, daily equity, and trades are saved locally under
`data/sector_etf_locked.*`; the run is recorded in `experiments/research.jsonl`.
Yahoo may revise adjusted history, so the local CSV and its hash are needed to
recreate the exact result. The ETF set does not test historical individual-stock
membership. Fixed basis-point costs omit liquidity, market impact, taxes, cash
interest, and FX. More unseen periods and prospective paper tracking are still
needed before making profitability claims.
