import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research import build_research_panel, chronological_split
from src.research_model import (
    evaluate_locked_test,
    multiclass_metrics,
    select_model,
)


def make_market_data(n=180, phase=0):
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    wave = np.sin((np.arange(n) + phase) / 3) * 3
    close = pd.Series(100 + np.arange(n) * .04 + wave, index=index)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1_000_000 + (np.arange(n) % 15) * 5_000,
        "VIX": 20 + np.sin(np.arange(n) / 7),
        "SPY_Open": 200 + np.arange(n) * .08,
        "SPY_Close": 200 + np.arange(n) * .08,
    }, index=index)


def make_split():
    panel = build_research_panel({
        "AAA": make_market_data(phase=0),
        "BBB": make_market_data(phase=2),
        "CCC": make_market_data(phase=4),
    })
    return chronological_split(panel)


def test_multiclass_metrics_reward_correct_calibrated_predictions():
    y = np.array([-1, 0, 1])
    probabilities = np.array([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9]])
    metrics = multiclass_metrics(y, y, probabilities)

    assert metrics["accuracy"] == 1
    assert metrics["balanced_accuracy"] == 1
    assert metrics["log_loss"] < .2
    assert metrics["brier"] < .02


def test_selection_uses_validation_and_locked_test_is_separate():
    split = make_split()
    models = {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, solver="liblinear")),
        ]),
    }
    selection = select_model(split, models=models)

    assert selection.selected_name in models
    assert set(selection.validation_results["Model"]) == set(models)
    assert {"log_loss", "brier", "macro_f1"} <= set(selection.validation_results)
    assert isinstance(selection.promoted_over_baseline, bool)

    _, predictions, metrics = evaluate_locked_test(selection, split)
    assert len(predictions) == len(split.test)
    assert set(predictions["Prediction"].unique()) <= {-1, 0, 1}
    assert metrics["log_loss"] >= 0


def test_selection_rejects_non_finite_features():
    split = make_split()
    split.train.iloc[0, split.train.columns.get_loc("MA_5")] = np.inf

    with pytest.raises(ValueError, match="finite"):
        select_model(split, models={"Prior": DummyClassifier(strategy="prior")})
