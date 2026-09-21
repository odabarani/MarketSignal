import numpy as np

TECHNICAL_FEATURES = [
    "MA_5", "MA_20", "Momentum", "Volatility",
    "Volume_Change", "Volume_Ratio", "RSI", "MACD",
    "MACD_Signal", "BB_Position", "VIX", "RS_SPY",
    "Body_Size", "Upper_Wick", "Lower_Wick", "Gap",
]

def add_features(df):
    df = df.copy()

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Momentum"] = df["Close"].pct_change(5)
    df["Volatility"] = df["Close"].pct_change().rolling(10).std()

    df["Volume_Change"] = df["Volume"].pct_change()
    volume_ma20 = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / volume_ma20

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()
    df["BB_Position"] = (df["Close"] - rolling_mean) / (2 * rolling_std.replace(0, np.nan))

    df["RS_SPY"] = df["Close"].pct_change(20) - df["SPY_Close"].pct_change(20)

    df["Body_Size"] = (df["Close"] - df["Open"]).abs()
    df["Upper_Wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["Lower_Wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
    df["Gap"] = df["Open"] / df["Close"].shift(1) - 1

    # The final five rows have no known 5-day-ahead outcome.
    future_close = df["Close"].shift(-5)
    df["Target"] = np.where(
        future_close.notna(),
        (future_close > df["Close"]).astype(int),
        np.nan,
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=TECHNICAL_FEATURES + ["Target"]).copy()
    return df
