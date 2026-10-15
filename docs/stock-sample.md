# Free daily stock sample

MarketSignal can import QuantRocket's free US stock learning sample: daily
prices for all US stocks from 2007 through 2011, including stocks that later
delisted, according to [QuantRocket's free-tier description](https://www.quantrocket.com/pricing/).
The data is public-account access, but it is not installed in this project
automatically.

## Prepare the data

Install QuantRocket and its local services using the [official setup guide](https://www.quantrocket.com/docs/).
In a QuantRocket terminal, create and ingest the daily learning bundle. The
official [bundle guide](https://www.quantrocket.com/docs/#us-stock-eod-zipline-bundle)
documents this sample and its 2007–2011 date range.

```bash
quantrocket zipline create-usstock-bundle usstock-learn-1d --learn
quantrocket zipline ingest usstock-learn-1d
```

Wait for ingestion to finish. Then place `export_quantrocket_stock_sample.py`
and the MarketSignal `src/stock_panel.py` module where the QuantRocket Python
environment can import them, and run:

```bash
python export_quantrocket_stock_sample.py
```

The exporter writes `data/research_inputs/quantrocket_usstock_learn_2007_2011.csv.gz`
plus a JSON manifest with date coverage, security count, source details, and a
SHA-256 file hash. It refuses to overwrite an existing export unless `--force`
is supplied. Generated data is ignored by Git.

## What the export means

Each row is one `Sid` and date, with adjusted `Close` and `Volume`. `Sid` is a
stable security identifier, so ticker changes do not join two companies by
mistake. The bundle adjusts prices as of the end of the query; use price ratios
and returns rather than historical nominal price levels as model inputs.

The source includes delisted securities, but the export does not include cash or
stock consideration paid at delisting. A last observed quote is not necessarily
the investor's final value. Any stock backtest must expose and justify its
terminal-price policy, and must not quietly translate missing prices into
ordinary returns or a total loss. The five-year sample is useful for developing
and falsifying ideas, but it is not a future holdout and cannot establish that a
strategy will keep making money.
