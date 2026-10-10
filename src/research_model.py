"""Three-class model selection for the multi-stock research panel."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src.features import TECHNICAL_FEATURES
from src.research import DOWN, NO_TRADE, RESEARCH_TARGET, UP

CLASSES = np.array([DOWN, NO_TRADE, UP])
ENCODED_CLASSES = np.array([0, 1, 2])


class StableSGDClassifier(SGDClassifier):
    """Use a stable float64 score path across macOS BLAS implementations."""

    def decision_function(self, x):
        scores = np.einsum(
            "ij,kj->ik", np.asarray(x, dtype="float64"), self.coef_, optimize=False
        ) + self.intercept_
        return scores[:, 0] if scores.shape[1] == 1 else scores


def build_multiclass_models(random_state=42):
    return {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline([
            ("scaler", RobustScaler(quantile_range=(10, 90))),
            ("model", StableSGDClassifier(
                loss="log_loss", penalty="l2", alpha=.001, max_iter=2000,
                tol=1e-4, class_weight="balanced", random_state=random_state,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=180,
            max_depth=4, learning_rate=.03, subsample=.8,
            colsample_bytree=.8, min_child_weight=8, reg_lambda=1.5,
            random_state=random_state, eval_metric="mlogloss", n_jobs=2,
        ),
    }


def _model_data(frame, features):
    required = set(features) | {RESEARCH_TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing research columns: {sorted(missing)}")
    clean = frame.dropna(subset=list(features) + [RESEARCH_TARGET]).copy()
    x = clean[list(features)].astype("float64")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Research features must contain only finite values.")
    if clean.empty:
        raise ValueError("No complete research rows remain.")
    if not clean[RESEARCH_TARGET].isin(CLASSES).all():
        raise ValueError("Research target must contain only DOWN, NO_TRADE, and UP.")
    y = clean[RESEARCH_TARGET].astype(int)
    return clean, x, y


def _encode_target(y):
    return y.astype(int) + 1


def _aligned_probabilities(model, x):
    raw = np.asarray(model.predict_proba(x), dtype="float64")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Model returned invalid class probabilities.")
    probabilities = np.zeros((len(x), 3), dtype="float64")
    for source, label in enumerate(model.classes_.astype(int)):
        probabilities[:, label] = raw[:, source]
    row_total = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_total <= 0):
        raise ValueError("Model returned invalid class probabilities.")
    return probabilities / row_total


def multiclass_metrics(y_true, predictions, probabilities):
    encoded_true = _encode_target(pd.Series(y_true)).to_numpy()
    one_hot = np.eye(3)[encoded_true]
    recalls = recall_score(
        y_true, predictions, labels=CLASSES, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(
            y_true, predictions, labels=CLASSES, average="macro", zero_division=0
        )),
        "log_loss": float(log_loss(encoded_true, probabilities, labels=ENCODED_CLASSES)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "prediction_no_trade_rate": float(np.mean(np.asarray(predictions) == NO_TRADE)),
        "actual_no_trade_rate": float(np.mean(np.asarray(y_true) == NO_TRADE)),
        "recall_down": float(recalls[0]),
        "recall_no_trade": float(recalls[1]),
        "recall_up": float(recalls[2]),
    }


def evaluate_fitted_model(model, frame, features=TECHNICAL_FEATURES):
    clean, x, y = _model_data(frame, features)
    probabilities = _aligned_probabilities(model, x)
    predictions = model.predict(x).astype(int) - 1
    metrics = multiclass_metrics(y.to_numpy(), predictions, probabilities)
    output = clean[[RESEARCH_TARGET, "Forward_Excess_Return"]].copy()
    output["Prediction"] = predictions
    output["P_DOWN"] = probabilities[:, 0]
    output["P_NO_TRADE"] = probabilities[:, 1]
    output["P_UP"] = probabilities[:, 2]
    return output, metrics


@dataclass(frozen=True)
class ModelSelection:
    selected_name: str
    validation_results: pd.DataFrame
    candidates: dict
    promoted_over_baseline: bool
    log_loss_improvement: float


def select_model(split, features=TECHNICAL_FEATURES, models=None):
    """Choose a model using training and validation data only."""
    candidates = models or build_multiclass_models()
    _, train_x, train_y = _model_data(split.train, features)
    if set(train_y.unique()) != set(CLASSES):
        raise ValueError("Training requires all three target classes; widen the training period.")
    rows = []
    fitted = {}
    for name, candidate in candidates.items():
        model = clone(candidate)
        model.fit(train_x, _encode_target(train_y))
        _, metrics = evaluate_fitted_model(model, split.validation, features)
        rows.append({"Model": name, **metrics})
        fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values(
        ["log_loss", "brier", "macro_f1"], ascending=[True, True, False]
    ).reset_index(drop=True)
    selected_name = comparison.iloc[0]["Model"]
    baseline_loss = float(
        comparison.loc[comparison["Model"] == "Prior baseline", "log_loss"].iloc[0]
    ) if "Prior baseline" in set(comparison["Model"]) else np.nan
    selected_loss = float(comparison.iloc[0]["log_loss"])
    improvement = baseline_loss - selected_loss
    promoted = bool(selected_name != "Prior baseline" and improvement > 0)
    return ModelSelection(
        selected_name, comparison, fitted, promoted, float(improvement)
    )


def evaluate_locked_test(selection, split, features=TECHNICAL_FEATURES):
    """Refit the selected model on train+validation, then open the test once."""
    combined = pd.concat([split.train, split.validation]).sort_index()
    _, x, y = _model_data(combined, features)
    model = clone(selection.candidates[selection.selected_name])
    model.fit(x, _encode_target(y))
    predictions, metrics = evaluate_fitted_model(model, split.test, features)
    return model, predictions, metrics
