import numpy as np
import pandas as pd
import pytest

from src.features import TECHNICAL_FEATURES, add_features


def make_data(n=100, direction=1):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(100 + direction * np.arange(n, dtype=float), index=idx)
    return pd.DataFrame({
        "Open": close,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1_000_000 + np.arange(n),
        "VIX": 20.0,
        "SPY_Close": 200 + np.arange(n) * 0.5,
        "SPY_Open": 200 + np.arange(n) * 0.5,
    }, index=idx)


def test_features_keep_current_rows_and_mark_unknown_targets():
    raw = make_data()
    result = add_features(raw)

    assert len(result) > 0
    assert result.index[-1] == raw.index[-1]
    assert result["Target"].tail(5).isna().all()
    assert result["Target_Available_At"].tail(5).isna().all()
    assert result.iloc[:-5]["Target"].notna().all()
    assert result[TECHNICAL_FEATURES].notna().all().all()


def test_rsi_handles_one_direction_and_flat_prices():
    rising = add_features(make_data(direction=1))
    falling = add_features(make_data(direction=-1))
    flat_raw = make_data()
    flat_raw[["Open", "High", "Low", "Close"]] = 100.0
    flat = add_features(flat_raw)

    assert rising["RSI"].iloc[-1] == pytest.approx(100)
    assert falling["RSI"].iloc[-1] == pytest.approx(0)
    assert flat["RSI"].iloc[-1] == pytest.approx(50)
    assert flat["BB_Position"].iloc[-1] == pytest.approx(0)


def test_features_reject_duplicate_dates():
    raw = make_data()
    duplicated = pd.concat([raw, raw.iloc[[-1]]])
    with pytest.raises(ValueError, match="duplicate"):
        add_features(duplicated)
