import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42


def build_models(random_state=RANDOM_STATE):
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, solver="liblinear", random_state=random_state
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, random_state=random_state,
            eval_metric="logloss", n_jobs=2,
        ),
    }


def classification_metrics(y_true, predictions, probabilities):
    up_probability = probabilities[:, 1]
    majority_accuracy = float(pd.Series(y_true).value_counts(normalize=True).max())
    result = {
        "accuracy": accuracy_score(y_true, predictions),
        "majority_accuracy": majority_accuracy,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "brier": brier_score_loss(y_true, up_probability),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }
    result["roc_auc"] = (
        roc_auc_score(y_true, up_probability)
        if len(np.unique(y_true)) == 2 else np.nan
    )
    return result


def _validated_data(df, features):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Feature data must use a DatetimeIndex.")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be unique and increasing.")
    required = set(features) | {"Target", "Target_Available_At"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")
    data = df.dropna(subset=features + ["Target", "Target_Available_At"]).copy()
    data[features] = data[features].astype("float64")
    if not np.isfinite(data[features].to_numpy()).all():
        raise ValueError("Model features must contain only finite values.")
    return data


def walk_forward_evaluate(df, features, initial_train_fraction=0.70,
                          retrain_every=21, model_name="XGBoost"):
    if not 0 < initial_train_fraction < 1:
        raise ValueError("Initial train fraction must be between 0 and 1.")
    if retrain_every < 1:
        raise ValueError("Retrain frequency must be at least one day.")
    if model_name not in build_models():
        raise ValueError(f"Unknown model: {model_name}")

    data = _validated_data(df, features)
    split = int(len(data) * initial_train_fraction)
    if split < 30 or len(data) - split < 30:
        raise ValueError("Not enough historical data for a meaningful out-of-sample test.")

    rows = []
    last_model = None
    for start in range(split, len(data), retrain_every):
        end = min(start + retrain_every, len(data))
        test = data.iloc[start:end]
        fit_time = test.index[0]
        train = data.iloc[:start]
        train = train.loc[train["Target_Available_At"] <= fit_time]
        if train["Target"].nunique() < 2:
            raise ValueError(f"Training data has one target class at {fit_time.date()}.")

        model = build_models()[model_name]
        model.fit(train[features], train["Target"].astype(int))
        probabilities = model.predict_proba(test[features])
        predictions = model.predict(test[features])
        for idx, prediction, probability in zip(
            test.index, predictions, probabilities[:, 1]
        ):
            rows.append((idx, int(prediction), float(probability), fit_time))
        last_model = model

    predictions_df = pd.DataFrame(
        rows, columns=["Date", "Prediction", "Probability", "Fit_Time"]
    ).set_index("Date")
    evaluated = data.join(predictions_df, how="inner")
    probabilities = np.column_stack([
        1 - evaluated["Probability"].to_numpy(),
        evaluated["Probability"].to_numpy(),
    ])
    metrics = classification_metrics(
        evaluated["Target"].astype(int), evaluated["Prediction"], probabilities
    )
    return last_model, evaluated, metrics


def compare_models(df, features, retrain_every=21):
    rows = []
    for model_name in build_models():
        _, _, metrics = walk_forward_evaluate(
            df, features, initial_train_fraction=0.70,
            retrain_every=retrain_every, model_name=model_name,
        )
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows)


def train_final_model(df, features, as_of=None, model_name="XGBoost"):
    if model_name not in build_models():
        raise ValueError(f"Unknown model: {model_name}")
    as_of = pd.Timestamp(as_of if as_of is not None else df.index.max())
    data = _validated_data(df, features)
    data = data.loc[data["Target_Available_At"] <= as_of]
    if data["Target"].nunique() < 2:
        raise ValueError("Current training data does not contain both target classes.")
    model = build_models()[model_name]
    model.fit(data[features], data["Target"].astype(int))
    return model


def predict_latest(df, features, model_name="XGBoost"):
    latest = df.dropna(subset=features).iloc[-1]
    model = train_final_model(df, features, as_of=latest.name, model_name=model_name)
    current_features = pd.DataFrame(
        [latest[features].astype(float).to_dict()], columns=features, index=[latest.name]
    )
    probability = float(model.predict_proba(current_features)[0, 1])
    return model, latest, probability
