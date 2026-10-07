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
    # Fit occurs after the session close: labels ending at that close are known.
    assert make_model_data().loc[fits[0], "Target_Available_At"].max() == predictions["Fit_Time"].iloc[0]


def test_walk_forward_rejects_unsorted_dates():
    data = make_model_data().sort_index(ascending=False)
    with pytest.raises(ValueError, match="increasing"):
        model_module.walk_forward_evaluate(data, ["Feature"])


def test_validated_features_use_stable_float64_math():
    data = make_model_data()
    data[["Feature"]] = data[["Feature"]].astype("float32")

    validated = model_module._validated_data(data, ["Feature"])

    assert validated["Feature"].dtype == np.dtype("float64")


@pytest.mark.parametrize("failure", ["nonbinary", "not_future"])
def test_rejects_invalid_label_contract(failure):
    data = make_model_data()
    if failure == "nonbinary":
        data.iloc[0, data.columns.get_loc("Target")] = .5
    else:
        data.iloc[0, data.columns.get_loc("Target_Available_At")] = data.index[0]
    with pytest.raises(ValueError):
        model_module._validated_data(data, ["Feature"])


def test_training_baseline_does_not_choose_evaluation_majority(monkeypatch):
    class ConstantModel:
        def fit(self, x, y):
            return self
        def predict_proba(self, x):
            return np.tile([.5, .5], (len(x), 1))
        def predict(self, x):
            return np.zeros(len(x), dtype=int)
    monkeypatch.setattr(model_module, "build_models", lambda: {"XGBoost": ConstantModel()})
    data = make_model_data()
    data.loc[data.index[:60], "Target"] = 0
    data.loc[data.index[60:-5], "Target"] = 1
    _, evaluated, metrics = model_module.walk_forward_evaluate(data, ["Feature"])
    assert evaluated["BaselinePrediction"].eq(0).all()
    assert metrics["training_baseline_accuracy"] == 0
    assert metrics["majority_accuracy"] == 1


def test_poisoning_unavailable_labels_cannot_change_first_fit(monkeypatch):
    fits = []
    class Recorder:
        def fit(self, x, y):
            fits.append((x.copy(), y.copy()))
            return self
        def predict_proba(self, x):
            return np.tile([.5, .5], (len(x), 1))
        def predict(self, x):
            return np.zeros(len(x), dtype=int)
    monkeypatch.setattr(model_module, "build_models", lambda: {"XGBoost": Recorder()})
    original = make_model_data()
    _, evaluated, _ = model_module.walk_forward_evaluate(original, ["Feature"])
    first_fit = fits[0]
    fit_time = evaluated["Fit_Time"].iloc[0]
    poisoned = original.copy()
    unavailable = (poisoned["Target_Available_At"] > fit_time) & poisoned["Target"].notna()
    poisoned.loc[unavailable, "Target"] = 1 - poisoned.loc[unavailable, "Target"]
    fits.clear()
    model_module.walk_forward_evaluate(poisoned, ["Feature"])
    pd.testing.assert_frame_equal(first_fit[0], fits[0][0])
    pd.testing.assert_series_equal(first_fit[1], fits[0][1])


def test_explicit_evaluation_boundary_is_respected():
    data = make_model_data()
    boundary = data.index[95]
    _, evaluated, _ = model_module.walk_forward_evaluate(data, ["Feature"], evaluation_start=boundary)
    assert evaluated.index.min() == boundary
    assert (evaluated["Fit_Time"] >= boundary).all()
