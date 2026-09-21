import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42

def build_models(random_state=RANDOM_STATE):
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
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
    result = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }
    result["roc_auc"] = (
        roc_auc_score(y_true, probabilities[:, 1])
        if len(np.unique(y_true)) == 2 else np.nan
    )
    return result

def walk_forward_evaluate(df, features, initial_train_fraction=0.70,
                          retrain_every=21, model_name="XGBoost"):
    data = df.dropna(subset=features + ["Target"]).copy()
    split = int(len(data) * initial_train_fraction)
    if len(data) - split < 30:
        raise ValueError("Not enough historical data for a meaningful out-of-sample test.")

    rows = []
    last_model = None

    for start in range(split, len(data), retrain_every):
        end = min(start + retrain_every, len(data))
        train = data.iloc[:start]
        test = data.iloc[start:end]

        model = build_models()[model_name]
        model.fit(train[features], train["Target"])

        probabilities = model.predict_proba(test[features])
        predictions = model.predict(test[features])

        for idx, prediction, probability in zip(
            test.index, predictions, probabilities[:, 1]
        ):
            rows.append((idx, int(prediction), float(probability)))

        last_model = model

    predictions_df = pd.DataFrame(
        rows, columns=["Date", "Prediction", "Probability"]
    ).set_index("Date")

    evaluated = data.join(predictions_df, how="inner")
    probabilities = np.column_stack([
        1 - evaluated["Probability"].to_numpy(),
        evaluated["Probability"].to_numpy(),
    ])

    metrics = classification_metrics(
        evaluated["Target"], evaluated["Prediction"], probabilities
    )
    return last_model, evaluated, metrics

def compare_models(df, features):
    rows = []
    for model_name in build_models():
        _, _, metrics = walk_forward_evaluate(
            df, features, initial_train_fraction=0.70,
            retrain_every=63, model_name=model_name
        )
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows)

def train_final_model(df, features):
    data = df.dropna(subset=features + ["Target"]).copy()
    model = build_models()["XGBoost"]
    model.fit(data[features], data["Target"])
    return model
