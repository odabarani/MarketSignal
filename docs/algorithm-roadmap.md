# MarketSignal algorithm competition

The goal is durable **net returns**: grow capital after execution costs while
controlling losses. Accuracy is a diagnostic, not the competition score.
No candidate has earned a live-trading recommendation yet.

## Rules before the next experiment

- Start with daily US stock data, long positions and cash, without leverage.
- Rank candidates by forward **compounded return**, alongside **Sharpe ratio**,
  **max drawdown**, turnover, and results by year and sector.
- Use a provisional 20% drawdown budget for research promotion. A historical
  budget is a screening rule; it cannot cap future losses. Broader portfolio
  diversification and volatility sizing must be tested before promotion.
- Compare with SPY buy-and-hold, equal-weight stocks, cash, and simple momentum.
  A candidate must justify itself against passive returns and passive risk.
- Record every trial. Choose features and settings on development data, freeze
  one candidate, then evaluate later data once. Repeatedly tuning on inspected
  data creates **backtest overfitting** ([Bailey et al.](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)).

## Evidence so far

- **Execution and target:** the return pipeline buys at the next open, holds
  through the fifth session, exits at the sixth open, charges costs on turnover,
  and keeps daily portfolio values. Absolute-return forecasts were compared
  with simple rules; the learned models did not beat equal-weight stocks in the
  five-stock development sample.
- **Broader comparison:** the [ETF holdout](etf-locked-evaluation.md) favored
  SPY for return. Its uncertainty check suggests lower drawdown for the
  volatility-sized ETF strategy, while its relative-return range still includes
  underperformance and outperformance. The [industry test](industry-portfolio-evaluation.md)
  also underperformed the market and overlaps previously inspected dates.
- **Public-data limit:** available free stock-price panels have material gaps in
  delisted histories. Do not describe those results as survivorship-bias-free.
  [CRSP documents active and inactive security coverage](https://www.crsp.org/research/crsp-us-stock-databases/),
  while a [public S&P 500 price panel](https://github.com/Johnbrick123/sp500-data)
  reports substantial gaps for delisted members. The current historical
  comparisons therefore use public ETF and reconstructed industry portfolios.
- **Forward evidence:** the paper tracker records decisions before outcomes and
  has one signal awaiting its exit data. This is the only current source of
  genuinely unseen performance evidence.

## Next work

1. **Keep paper tracking intact.** Reconcile each forecast only after its exit
   open is observable. Preserve the saved inputs, source hashes, frozen model
   settings, fills, and cost scenarios; never replace an earlier signal.
2. **Use public historical data with clear labels.** Continue testing portfolio
   rules on public ETF and industry series, including market, cash, equal-weight,
   momentum, and volatility-sized baselines. Report results by year and market
   regime. Treat all dates already examined as development data.
3. **Do not promote a learned model yet.** The current models lost to simple
   baselines. Any new features or policy changes must be selected only on
   development data, then frozen before the next future paper observations.
4. **Reassess only after enough prospective evidence accumulates.** Compare net
   return, drawdown, turnover, and forecast calibration against fixed baselines
   across multiple market conditions. A favorable short run is not a live-trading
   recommendation.

## First diagnostic

The separate portfolio pipeline uses fixed five-session cohorts, a 55% UP
probability threshold, the two highest UP-minus-DOWN scores, and a 50% allocation
cap per stock at entry. Unused allocations remain in cash. These settings were
chosen before this diagnostic's returns, and are not optimized or approved.
Momentum uses the two highest trailing 20-session returns under the same cap.
All active rules close and reopen at each cohort boundary; SPY buy-and-hold pays
only its initial entry and final exit costs. The scoreboard also compares each
rule with SPY at the same entry allocation and dates, to separate stock selection
from sitting in cash. This matches cash allocation, not volatility or all risk.

Per-side **cost stress** is 15, 30, and 50 basis points. These are assumptions,
not estimates for a particular account. The report keeps every candidate,
including losing ones, and verifies its source snapshot and prediction report.
It measures reused development data only. Dividend reinvestment, taxes, cash
interest, FX, and market impact remain unmodelled.

Run the diagnostic on matching saved files:

```bash
python strategy_pipeline.py --input data/feature_comparison.csv --report data/walk_forward.json
```

It saves a scoreboard, daily equity, trade records, and an offline interactive
chart in `data/strategy_comparison.*`. Keep the experiment ledger with the data.
See the [first diagnostic results](strategy-results.md) for the current evidence.
The [expected-return experiment](return-experiment.md) documents the later
absolute-return comparison, retained positions, volatility sizing, and
chronological selection.
