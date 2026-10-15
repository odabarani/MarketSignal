# Stock momentum comparison

After exporting the QuantRocket sample, run the fixed comparison:

```bash
python stock_momentum_research.py
```

The report is saved to `data/stock_momentum_development.json`. It compares cash,
SPY buy-and-hold, monthly equal-weight stocks, 12-1 cross-sectional momentum,
momentum held only above SPY's 200-day trend, and momentum sized to a fixed 10%
annualized volatility target with exposure capped at 100%. All rules use the same
evaluation dates. The momentum basket holds the top 10% of eligible stocks with
positive 12-month returns after skipping the latest month.

Eligibility uses a current close, at least 200 observed prices over the previous
252 market sessions, and at least $5 million in adjusted `Close × Volume` proxy
over the previous 20 sessions. This screen uses that amount as a
point-in-time liquidity proxy, not literal nominal trading dollars; each query
ends on its signal date. Eligible securities are Common Stock or Depositary
Receipts according to the latest QuantRocket master snapshot. This static type
label can be revised and is not a point-in-time classification.

Signals are formed at month-end close and traded at the next session's close.
Performance includes 15, 30, and 50 basis points per side, net compounded
returns, Sharpe ratio, maximum drawdown, turnover, calendar-year returns, and
returns above and below SPY's 200-day average. The report also runs two
delisting cases: zero recovery after the last observed price as a conservative
stress, and carrying the last quote as an optimistic assumption. Neither is an
observed payout, and the last quote is not an upper bound on merger proceeds;
the spread between scenarios shows how much missing recovery data matters.
It attributes net portfolio P&L, including trading costs, by sector and year so
concentration and sector drag are visible. Sector labels come from the current
securities master and are descriptive only, not point-in-time historical
classifications.
For each strategy, the report also gives a paired 20-session block-bootstrap
range for compounded excess return versus SPY. It keeps strategy and benchmark
days together and resamples contiguous blocks, but it only describes possible
paths through this same historical sample. It is not a forecast or an
unseen-test result.

Each result also shows excess net return and excess drawdown versus SPY on the
same dates, plus whether it stayed within the project's provisional 20%
research drawdown budget. Passing that historical screen is not a guarantee or
a promotion decision.

This is a historical development comparison, not a hidden test. The five-year
sample has already been examined at the broader market level, so its results
cannot satisfy the goal of genuinely unseen validation. It also uses a current
security-type snapshot, close-price execution, and fixed cost estimates; market
impact, taxes, cash yield, and trade capacity are not fully modeled. Do not
promote or describe any rule as profitable from this report. Preserve each run
under a new filename rather than overwriting an earlier result.
