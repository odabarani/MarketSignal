import yfinance as yf
import pandas as pd
from datetime import date

START = "2015-01-01"

def _clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def _download_close(ticker):
    data = yf.download(ticker, start=START, end=date.today(), progress=False, auto_adjust=False)
    data = _clean_columns(data)
    if data.empty or "Close" not in data:
        raise ValueError(f"No price data was returned for {ticker}.")
    return data

def get_stock_data(ticker):
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Enter a ticker symbol.")

    df = _download_close(ticker)
    vix = _download_close("^VIX")
    spy = _download_close("SPY")

    df["VIX"] = vix["Close"]
    df["SPY_Close"] = spy["Close"]
    df = df.dropna(subset=["VIX", "SPY_Close"]).copy()

    # Current Yahoo Finance fundamentals are not point-in-time historical
    # observations, so they are intentionally excluded to avoid look-ahead bias.
    return df
