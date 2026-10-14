# Historical industry strategy comparison

This comparison uses the [Fama/French 49 Industry Portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/Data_Library/det_49_ind_port.html) and daily market/T-bill factors. The downloaded files were created from the 202608 CRSP database. The complete 49-industry panel starts on 1969-07-01; one later session with a missing industry return is omitted. Industry returns are reconstructed value-weighted portfolios, with annual June industry assignments.

## Result

Strategies were compared through 2014, then one choice was frozen using the 30-basis-point cost scenario and the existing 20% drawdown screen. **12–1 industry momentum with a 5% volatility target** was selected. On the reserved 2015–2019 Fama/French interval, it returned **4.33% annualized** with a **12.26% maximum drawdown** at 30 bps per side.

| Strategy | Annualized return | Maximum drawdown | Sharpe vs. T-bills |
| --- | ---: | ---: | ---: |
| Cash / T-bills | 1.01% | 0.00% | 0.00 |
| Frozen 12–1 momentum, 5% volatility target | 4.33% | -12.26% | 0.47 |
| 12–1 momentum, no volatility target | 10.52% | -24.68% | 0.68 |
| Market buy-and-hold | 11.44% | -20.45% | 0.78 |

The selected strategy controlled drawdowns but earned much less than the market. At 50 bps per side it returned 3.51% annualized, versus 11.35% for market buy-and-hold. This is not evidence of a market-beating strategy.

**This is an auxiliary historical test, not a fully unseen project-wide holdout.** Its 2018–2019 dates overlap the earlier ETF development sample, so prior work had already examined that market period. The new industry data and rules were isolated before this run, but calendar overlap weakens independence. Future paper-tracking results are the cleanest evidence still available; the paper ledger has not yet accumulated a mature outcome.

## How it was tested

The development window is 1970-07-01 through 2014-12-31. Fixed candidates include cash, market buy-and-hold, a 200-day market trend, equal-weight industries, and 12–1 momentum across the ten strongest industries. Volatility-target versions use fixed 5%, 7.5%, and 10% targets, with no leverage. Monthly signals take effect after a two-session lag. Costs of 15, 30, and 50 bps per side apply to entry, turnover, and final liquidation. The chosen candidate maximizes development annualized return at 30 bps among candidates within the 20% development drawdown limit.

Download the two release files from the official [industry data page](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html) and [daily factor page](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/Data_Library/f-f_factors.html):

```text
data/research_inputs/fama_french_49_industry_daily_202608.zip
data/research_inputs/fama_french_3_factors_daily_202608.zip
```

The matching download commands are:

```bash
mkdir -p data/research_inputs
curl -L https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/ftp/49_Industry_Portfolios_daily_CSV.zip -o data/research_inputs/fama_french_49_industry_daily_202608.zip
curl -L https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/ftp/F-F_Research_Data_Factors_daily_CSV.zip -o data/research_inputs/fama_french_3_factors_daily_202608.zip
```

Run development first, then run the locked phase once:

```bash
python industry_portfolio_research.py --phase development
python industry_portfolio_research.py --phase locked --development-report data/industry_development.json
```

The reports, growth chart, and daily equity files are saved under `data/` and are not committed. The locked command checks the saved source hashes and code fingerprint, and refuses to overwrite results. The source file hashes for this run are:

```text
49 industries: 8f394fe34bea54d41b9aafed410425ee8f8e252ede3c71a7c1cd20bab83040de
Daily factors: 2f29e22546069914890a712680a6f81f3680ebc3543b52864484d209ca13a7db
```

## Limits

Industry portfolios are not individual-stock predictions or a directly investable 49-fund portfolio. Portfolio-level basis-point costs do not measure trading inside the source portfolios, market impact, taxes, or a particular account's cash rate. The daily T-bill series is rounded to two decimal places in percentage units; this rounds some low daily rates to zero and understates cash returns in parts of 2015–2017. Historical industry assignments and returns may be revised when CRSP is updated. The [Fama/French library](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html) describes its reconstructed history and annual industry assignment.

The result does not replace individual-stock research or forward paper tracking, and it is not a live-trading recommendation.
