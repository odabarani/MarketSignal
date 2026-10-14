# First portfolio diagnostic — October 13, 2026

No model is promoted. We now measure actual **portfolio returns**, not just
classification accuracy. This is a new execution rule applied to saved Research
Lab probabilities; it does not replace the app's existing simulation.

The diagnostic runs from August 2, 2017 to February 1, 2024, with 327 fixed
five-session cohorts. Decisions use the prior close, enter at the next open,
and exit at the fifth close. Both entries and exits pay assumed costs. Position
budgets include entry fees; daily closes capture losses during each holding.

At **15 basis points per side**:

| Strategy | Annualized net return | Max drawdown |
| --- | ---: | ---: |
| SPY buy-and-hold | 11.03% | -34.10% |
| Equal-weight five-stock universe, five-session rule | 9.94% | -54.18% |
| XGBoost, raw probabilities, fixed rule | 2.89% | -34.78% |
| Logistic Regression, raw probabilities, fixed rule | 1.17% | -4.56% |
| SPY, five-session liquidation rule | -4.85% | -40.55% |

The logistic rule traded only six stock positions, averaging less than 1% of
capital allocated at cohort entry. Its small drawdown is largely associated
with cash exposure; six trades cannot establish a consistent trading edge.
The report includes a SPY reference with each rule's same entry allocation and
dates. This controls cash allocation and timing, but does not equalize volatility.

**Cost stress:** XGBoost's annualized return falls to -3.67% at 30 bps per side
and -11.76% at 50 bps. Five-session liquidation repays costs even when the next
cohort buys the same stock, and misses the overnight between exit and next
entry. SPY buy-and-hold avoids those repeated costs and stays invested overnight.
This comparison motivates testing lower turnover; it does not identify costs as
the only cause of weak returns.

The five stocks are AAPL, AMZN, GOOGL, MSFT, and NVDA, selected using present-day
knowledge. Prices exclude dividend reinvestment, and the data has already been
inspected. Historical stock membership, delisted coverage, liquidity, taxes,
cash interest, and market impact remain unresolved. These results are
**development diagnostics**, not evidence of future profits.

Next: predict **expected absolute returns** and downside, test a lower-turnover
portfolio rule, and expand the historical universe before selecting a candidate.
All rules and candidates remain in the local report and experiment ledger.
