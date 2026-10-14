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

## Work order

1. **Execution alignment:** turn saved probabilities into a portfolio that enters
   at the next open and exits at the fifth close. Include both trading fees,
   slippage scenarios, idle cash, final liquidation, and daily valuations.
2. **Better target:** compare predictions of absolute return and downside with
   the existing direction/relative classes. Correctly predicting “beats SPY”
   does not imply making money when the market falls.
3. **Broader evidence:** test across sectors, time periods, and separate assets.
   Obtain dividend/corporate-action data and historical universe membership,
   including delisted stocks. A list of today's winners is inadequate evidence.
4. **Small model competition:** start with regularized linear models, shallow
   boosted trees, and simple rules. Use nested chronological selection; reserve
   later windows for scoring the complete frozen model and portfolio policy.
5. **Portfolio controls:** test diversification, volatility sizing, concentration,
   liquidity limits, and a cost-aware option to stay in cash. Position size and
   prediction quality both affect results.
6. **Prospective paper trading:** save decisions before outcomes arrive, then
   reconcile realistic fills and costs. Current historical dates have been
   inspected; future observations supply the next prospective evidence.

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
The next [expected-return experiment](return-experiment.md) tests retaining
positions, volatility sizing, and selection before each evaluation window.
