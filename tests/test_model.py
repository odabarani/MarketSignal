import numpy as np
import pandas as pd
import pytest

import src.model as model_module


def make_model_data(n=140):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    feature = np.sin(np.arange(n) / 4)
    data = pd.DataFrame({
        "Feature": feature,
        "Target": (feature > 0).astype(float),
        "Target_Available_At": pd.Series(idx, index=idx).shift(-5),
        "Open": 100 + np.arange(n),
        "SPY_Open": 200 + np.arange(n),
    }, index=idx)
    data.loc[data.index[-5:], "Target"] = np.nan
    return data


def test_walk_forward_embargoes_labels_not_known_at_fit_time(monkeypatch):
    fits = []

    class Recorder:
        def fit(self, x, y):
            fits.append(x.index)
            return self

        def predict_proba(self, x):
            probability = np.where(x["Feature"] >= 0, 0.8, 0.2)
            return np.column_stack([1 - probability, probability])

        def predict(self, x):
            return (x["Feature"] >= 0).astype(int).to_numpy()

    monkeypatch.setattr(model_module, "build_models", lambda: {"XGBoost": Recorder()})
    _, predictions, metrics = model_module.walk_forward_evaluate(
        make_model_data(), ["Feature"], retrain_every=21
    )

    assert len(predictions) > 0
    for fitted_dates, fit_time in zip(fits, predictions["Fit_Time"].unique()):
        available = make_model_data().loc[fitted_dates, "Target_Available_At"]
        assert (available <= fit_time).all()
    assert metrics["accuracy"] == pytest.approx(1)
    assert {"majority_accuracy", "brier", "log_loss"} <= metrics.keys()


def test_walk_forward_rejects_unsorted_dates():
    data = make_model_data().sort_index(ascending=False)
    with pytest.raises(ValueError, match="increasing"):
        model_module.walk_forward_evaluate(data, ["Feature"])
