"""Features observable after the close, computed within assets and within dates."""

import numpy as np
import pandas as pd

from src.features import TECHNICAL_FEATURES

RELATIVE_FEATURES = [
    "Return_1", "Momentum", "Return_20", "Return_60", "Volatility",
    "Volume_Change", "Volume_Ratio", "RSI", "BB_Position", "RS_SPY", "Gap",
    "Distance_MA_5", "Distance_MA_20", "MACD_Relative",
    "MACD_Signal_Relative", "Body_Relative", "Upper_Wick_Relative",
    "Lower_Wick_Relative",
]
CONTEXT_FEATURES = RELATIVE_FEATURES + [
    "VIX", "VIX_Change_5", "SPY_Return_20", "SPY_Trend_60",
    "SPY_Volatility_20", "Momentum_Rank", "Volatility_Rank",
    "Volume_Rank", "Universe_Breadth", "Universe_Return_Dispersion",
]
FEATURE_SETS = {
    "technical": TECHNICAL_FEATURES,
    "relative": RELATIVE_FEATURES,
    "context": CONTEXT_FEATURES,
}


def add_panel_features(panel):
    """Keep every input row; warm-up values remain NaN for a shared sample mask.

    Rankings and breadth describe the supplied universe at that day's close.
    They do not reconstruct historical index membership or future constituents.
    """
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["Date", "Ticker"]:
        raise ValueError("Features require a Date, Ticker MultiIndex.")
    if panel.index.has_duplicates:
        raise ValueError("Feature dates and tickers must be unique.")
    result = panel.sort_index().copy()
    if (result["Close"] <= 0).any():
        raise ValueError("Feature prices must be positive.")
    for days in (1, 20, 60):
        result[f"Return_{days}"] = result.groupby(level="Ticker")["Close"].transform(
            lambda series: series.pct_change(days, fill_method=None)
        )
    for days in (5, 20):
        result[f"Distance_MA_{days}"] = result["Close"] / result[f"MA_{days}"] - 1
    for source, destination in [
        ("MACD", "MACD_Relative"), ("MACD_Signal", "MACD_Signal_Relative"),
        ("Body_Size", "Body_Relative"), ("Upper_Wick", "Upper_Wick_Relative"),
        ("Lower_Wick", "Lower_Wick_Relative"),
    ]:
        result[destination] = result[source] / result["Close"]

    # Compute benchmarks once per date, not across adjacent ticker rows.
    by_date = result.groupby(level="Date")
    if (by_date[["SPY_Close", "VIX"]].nunique(dropna=False) > 1).any().any():
        raise ValueError("Stocks must share the same SPY and VIX observations per date.")
    market = by_date[["SPY_Close", "VIX"]].first()
    market["SPY_Return_20"] = market["SPY_Close"].pct_change(20, fill_method=None)
    market["SPY_Trend_60"] = market["SPY_Close"] / market["SPY_Close"].rolling(60).mean() - 1
    market["SPY_Volatility_20"] = market["SPY_Close"].pct_change(
        fill_method=None
    ).rolling(20).std()
    market["VIX_Change_5"] = market["VIX"].pct_change(5, fill_method=None)
    for column in ["SPY_Return_20", "SPY_Trend_60", "SPY_Volatility_20", "VIX_Change_5"]:
        result[column] = result.index.get_level_values("Date").map(market[column])

    for source, destination in [
        ("Return_20", "Momentum_Rank"), ("Volatility", "Volatility_Rank"),
        ("Volume_Ratio", "Volume_Rank"),
    ]:
        result[destination] = result.groupby(level="Date")[source].rank(pct=True)
    above_average = (result["Close"] > result["MA_20"]).astype(float)
    result["Universe_Breadth"] = above_average.groupby(level="Date").transform("mean")
    result["Universe_Return_Dispersion"] = result.groupby(level="Date")[
        "Return_1"
    ].transform(lambda series: series.std(ddof=0))
    return result.replace([np.inf, -np.inf], np.nan)


def common_feature_rows(frame, feature_sets):
    """Use exactly the same observations for every feature-set comparison."""
    columns = sorted({column for features in feature_sets.values() for column in features})
    return frame.loc[np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)].copy()
