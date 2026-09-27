import numpy as np
import pandas as pd

FORECAST_HORIZON = 5

TECHNICAL_FEATURES = [
    "MA_5", "MA_20", "Momentum", "Volatility",
    "Volume_Change", "Volume_Ratio", "RSI", "MACD",
    "MACD_Signal", "BB_Position", "VIX", "RS_SPY",
    "Body_Size", "Upper_Wick", "Lower_Wick", "Gap",
]

REQUIRED_COLUMNS = {
    "Open", "High", "Low", "Close", "Volume", "VIX", "SPY_Close",
}


def add_features(df):
    """Create features while preserving recent rows that do not have labels yet."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing market-data columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Market data must use a DatetimeIndex.")
    if df.index.has_duplicates:
        raise ValueError("Market data contains duplicate dates.")

    df = df.sort_index().copy()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Momentum"] = df["Close"].pct_change(5, fill_method=None)
    df["Volatility"] = df["Close"].pct_change(fill_method=None).rolling(10).std()

    df["Volume_Change"] = df["Volume"].pct_change(fill_method=None)
    volume_ma20 = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / volume_ma20

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    df["RSI"] = rsi.mask((gain == 0) & (loss == 0), 50.0)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()
    bb_position = (df["Close"] - rolling_mean) / (2 * rolling_std.replace(0, np.nan))
    df["BB_Position"] = bb_position.mask(rolling_std == 0, 0.0)

    df["RS_SPY"] = (
        df["Close"].pct_change(20, fill_method=None)
        - df["SPY_Close"].pct_change(20, fill_method=None)
    )
    df["Body_Size"] = (df["Close"] - df["Open"]).abs()
    df["Upper_Wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["Lower_Wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
    df["Gap"] = df["Open"] / df["Close"].shift(1) - 1

    future_close = df["Close"].shift(-FORECAST_HORIZON)
    df["Target"] = (future_close > df["Close"]).astype(float)
    df.loc[future_close.isna(), "Target"] = np.nan
    df["Target_Available_At"] = pd.Series(
        df.index, index=df.index
    ).shift(-FORECAST_HORIZON)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=TECHNICAL_FEATURES).copy()
