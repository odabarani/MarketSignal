"""Freeze a broad, adjusted-price research panel and reserve later sessions."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import pandas as pd

from src.data_loader import get_research_market_data

DEFAULT_TICKERS = [
    # US equity sectors
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    # Broad US, international and defensive assets
    "SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="Fixed ETF universe; defaults to US sectors and broad/defensive funds.")
    parser.add_argument("--development-end", required=True,
                        help="Last date allowed for model and strategy selection (YYYY-MM-DD).")
    parser.add_argument("--output", default="data/sector_etf_panel.csv")
    args = parser.parse_args(argv)

    output = Path(args.output)
    manifest = output.with_suffix(".manifest.json")
    if output.suffix.lower() != ".csv" or output.resolve() == manifest.resolve():
        parser.error("Output must be a CSV path with a separate JSON manifest.")
    cutoff = pd.Timestamp(args.development_end).normalize()
    tickers = list(dict.fromkeys(symbol.strip().upper() for symbol in args.tickers))
    if not tickers or any(not symbol for symbol in tickers):
        parser.error("Supply at least one ticker symbol.")

    try:
        market = get_research_market_data(tickers)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    frames = []
    for ticker, frame in market.items():
        item = frame.copy()
        item["Ticker"] = ticker
        frames.append(item.reset_index(names="Date"))
    panel = pd.concat(frames, ignore_index=True).sort_values(["Date", "Ticker"])
    dates = pd.DatetimeIndex(panel["Date"].unique()).sort_values()
    if cutoff not in dates:
        parser.error("Development end must be a trading date in the frozen panel.")
    holdout_dates = dates[dates > cutoff]
    if len(holdout_dates) < 126:
        parser.error("The snapshot needs at least 126 later sessions for the locked evaluation.")

    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_type": "sector and multi-asset ETFs",
        "data_vendor": "Yahoo Finance via yfinance",
        "price_basis": "auto_adjust=True OHLC; adjusted for dividends and splits",
        "benchmark": "SPY",
        "tickers": tickers,
        "session_count": len(dates),
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
        "development_end": str(cutoff.date()),
        "validation_end": str(cutoff.date()),
        "holdout_start": str(holdout_dates.min().date()),
        "holdout_sessions": len(holdout_dates),
        "source_hash": sha256(output.read_bytes()).hexdigest(),
        "yfinance_version": version("yfinance"),
        "limitations": [
            "The ETF universe broadens sector and asset coverage but does not represent historical individual-stock membership.",
            "Yahoo adjusted history can be revised; preserve this CSV and its hash with every result.",
            "Adjusted prices model reinvested distributions; taxes, cash interest, FX, liquidity, and market impact remain excluded.",
            "Dates after development_end are reserved for one locked evaluation and must not guide model or policy changes.",
        ],
    }
    manifest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(tickers)} assets across {len(dates)} complete sessions: {output}")
    print(f"Development through {cutoff.date()}; locked evaluation starts {holdout_dates.min().date()}.")
    print(f"Frozen source hash: {config['source_hash']}")


if __name__ == "__main__":
    main()
