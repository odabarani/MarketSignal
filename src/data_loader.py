import yfinance as yf
import pandas as pd
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

START = "2015-01-01"

def _clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def _download_close(ticker, auto_adjust=False, end=None):
    data = yf.download(ticker, start=START, end=end or date.today(), progress=False,
                       auto_adjust=auto_adjust)
    data = _clean_columns(data)
    if data.empty or "Close" not in data:
        raise ValueError(f"No price data was returned for {ticker}.")
    return data.sort_index()

def get_stock_data(ticker):
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Enter a ticker symbol.")

    df = _download_close(ticker)
    vix = _download_close("^VIX")
    spy = _download_close("SPY")

    df["VIX"] = vix["Close"]
    df["SPY_Open"] = spy["Open"]
    df["SPY_Close"] = spy["Close"]
    df = df.dropna(subset=["VIX", "SPY_Open", "SPY_Close"]).copy()

    # Current Yahoo Finance fundamentals are not point-in-time historical
    # observations, so they are intentionally excluded to avoid look-ahead bias.
    return df


def get_market_data(tickers):
    """Download several assets while reusing the same market benchmarks."""
    symbols = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers))
    if not symbols or any(not symbol for symbol in symbols):
        raise ValueError("Provide at least one valid ticker symbol.")

    vix = _download_close("^VIX")["Close"].rename("VIX")
    spy = _download_close("SPY")[["Open", "Close"]].rename(
        columns={"Open": "SPY_Open", "Close": "SPY_Close"}
    )
    result = {}
    for symbol in symbols:
        frame = _download_close(symbol).join([vix, spy], how="left")
        frame = frame.dropna(subset=["VIX", "SPY_Open", "SPY_Close"]).copy()
        if frame.empty:
            raise ValueError(f"No aligned market data was returned for {symbol}.")
        result[symbol] = frame
    return result


def get_research_market_data(tickers, end=None):
    """Download a complete, dividend-adjusted panel for offline research.

    Prices are aligned to the SPY trading calendar. The returned interval starts
    only once every requested asset has begun trading; missing bars inside
    that common interval raise an error instead of silently changing the sample.
    The public app keeps using the unadjusted loader above.
    """
    symbols = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers))
    if not symbols or any(not symbol for symbol in symbols):
        raise ValueError("Provide at least one valid research ticker.")
    if "__SPY__" in symbols:
        raise ValueError("Ticker __SPY__ is reserved for the benchmark.")

    if end is None:
        # Yahoo's end date is exclusive. Include today's daily candle only
        # after a conservative buffer past the US cash-market close.
        eastern = datetime.now(ZoneInfo("America/New_York"))
        end = eastern.date() + timedelta(days=1) if eastern.time() >= time(16, 15) else eastern.date()
    benchmark = _download_close("SPY", auto_adjust=True, end=end)
    volatility = _download_close("^VIX", end=end)["Close"].rename("VIX")
    assets = {symbol: _download_close(symbol, auto_adjust=True, end=end) for symbol in symbols}
    valid_ranges = {}
    for symbol, frame in assets.items():
        valid = frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if valid.empty:
            raise ValueError(f"No complete adjusted price history was returned for {symbol}.")
        valid_ranges[symbol] = (valid.index.min(), valid.index.max())
    common_start = max(start for start, _ in valid_ranges.values())
    common_end = min(end for _, end in valid_ranges.values())
    calendar = benchmark.index[(benchmark.index >= common_start) & (benchmark.index <= common_end)]
    if calendar.empty:
        raise ValueError("The requested assets have no common history with SPY.")
    base = benchmark.reindex(calendar)
    base["VIX"] = volatility.reindex(calendar)
    base["SPY_Open"] = base["Open"]
    base["SPY_Close"] = base["Close"]
    if not base[["SPY_Open", "SPY_Close", "VIX"]].notna().all().all():
        raise ValueError("SPY or VIX has missing observations in the common research interval.")

    result = {}
    for symbol, frame in assets.items():
        aligned = frame.reindex(calendar)
        if not aligned[["Open", "High", "Low", "Close", "Volume"]].notna().all().all():
            raise ValueError(f"{symbol} has missing sessions inside the common research interval.")
        aligned = aligned.join(base[["SPY_Open", "SPY_Close", "VIX"]])
        result[symbol] = aligned
    return result
