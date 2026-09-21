import numpy as np
import pandas as pd
from src.features import add_features, TECHNICAL_FEATURES

def make_data(n=100):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    return pd.DataFrame({"Open": close * 0.99, "High": close * 1.01, "Low": close * 0.98, "Close": close, "Volume": 1000000, "VIX": 20.0, "SPY_Close": close * 0.5}, index=idx)

def test_features_are_finite_and_target_is_binary():
    result = add_features(make_data())
    assert result[TECHNICAL_FEATURES].notna().all().all()
    assert set(result["Target"].unique()).issubset({0, 1})

def test_last_five_rows_are_excluded_from_labels():
    raw = make_data()
    result = add_features(raw)
    assert len(result) < len(raw) - 4