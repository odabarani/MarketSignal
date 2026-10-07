import yfinance as yf
import pandas as pd
from datetime import date
from src.market_calendar import validate_us_sessions
from src.errors import DataQualityError

START = "2015-01-01"

def _clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def _download_close(ticker):
    data = yf.download(ticker, start=START, end=date.today(), progress=False,
                       auto_adjust=False, actions=True, timeout=15)
    data = _clean_columns(data)
    if data.empty or "Close" not in data:
        raise ValueError(f"No price data was returned for {ticker}.")
    if ticker != "^VIX" and not {"Dividends", "Stock Splits"} <= set(data.columns):
        raise ValueError(f"Corporate-action data is unavailable for {ticker}.")
    return data.sort_index()


def _align_stock(stock, vix, spy, as_of=None):
    frame = stock.copy()
    frame["VIX"] = vix["Close"]
    frame["SPY_Open"] = spy["Open"]
    frame["SPY_Close"] = spy["Close"]
    frame["SPY_Dividends"] = spy["Dividends"]
    frame["SPY_Stock_Splits"] = spy["Stock Splits"]
    frame = frame.rename(columns={"Stock Splits": "Stock_Splits"})
    if frame[["VIX", "SPY_Open", "SPY_Close", "SPY_Dividends", "SPY_Stock_Splits"]].isna().any().any():
        raise DataQualityError("Benchmark history has missing stock sessions. Research stopped rather than bridging gaps.")
    validate_us_sessions(frame.index, as_of)
    return frame

def get_stock_data(ticker, as_of=None):
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Enter a ticker symbol.")

    df = _download_close(ticker)
    vix = _download_close("^VIX")
    spy = _download_close("SPY")

    df = _align_stock(df, vix, spy, as_of)

    # Current Yahoo Finance fundamentals are not point-in-time historical
    # observations, so they are intentionally excluded to avoid look-ahead bias.
    return df


def get_market_data(tickers, as_of=None):
    """Download several assets while reusing the same market benchmarks."""
    symbols = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers))
    if not symbols or any(not symbol for symbol in symbols):
        raise ValueError("Provide at least one valid ticker symbol.")

    vix = _download_close("^VIX")
    spy = _download_close("SPY")
    result = {}
    for symbol in symbols:
        frame = _align_stock(_download_close(symbol), vix, spy, as_of)
        if frame.empty:
            raise ValueError(f"No aligned market data was returned for {symbol}.")
        result[symbol] = frame
    return result
