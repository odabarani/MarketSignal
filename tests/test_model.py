import numpy as np
import pandas as pd
from src.features import add_features, TECHNICAL_FEATURES
from src.model import walk_forward_evaluate

def test_walk_forward_predictions_are_probabilities():
    n = 180
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(42).normal(0, 1, n)), index=idx)
    raw = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1000000, "VIX": 20.0, "SPY_Close": close * 0.9}, index=idx)
    df = add_features(raw)
    _, predictions, metrics = walk_forward_evaluate(df, TECHNICAL_FEATURES, retrain_every=21)
    assert len(predictions) > 0
    assert predictions["Probability"].between(0, 1).all()
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"} <= metrics.keys()