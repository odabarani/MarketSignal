"""Expanding-window validation with a separate chronological calibration block."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.research import RESEARCH_TARGET
from src.research_model import (
    _aligned_probabilities, _encode_target, _model_data,
    build_multiclass_models, multiclass_metrics,
)

PROBABILITY_COLUMNS = ["P_DOWN", "P_NO_TRADE", "P_UP"]
TEMPERATURES = np.array([.5, .75, 1., 1.25, 1.5, 2., 3.])


def temperature_scale(probabilities, temperature):
    """Rescale confidence while preserving the ordering of the three classes."""
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(axis=1), 1)):
        raise ValueError("Expected finite, normalized three-class probabilities.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite.")
    logits = np.log(np.clip(p, 1e-15, 1)) / temperature
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, y):
    encoded = _encode_target(pd.Series(y)).to_numpy()
    # Prefer no adjustment for exact ties. The grid is fixed before evaluation.
    candidates = sorted(TEMPERATURES, key=lambda t: abs(t - 1))
    return float(min(candidates, key=lambda t: -np.log(
        temperature_scale(probabilities, t)[np.arange(len(encoded)), encoded]
    ).mean()))


def walk_forward(panel, features, development_end, min_train_dates=504,
                 calibration_dates=126, evaluation_dates=126, models=None):
    """Evaluate only labels available by development_end; never use later rows.

    Each model fits on the expanding past, temperature fits on the next block,
    and both frozen versions score the following block. Labels crossing either
    fitting boundary or an evaluation block's end are purged.
    """
    if min(min_train_dates, calibration_dates, evaluation_dates) < 10:
        raise ValueError("Each window needs at least ten dates.")
    if panel.index.names != ["Date", "Ticker"] or panel.index.has_duplicates:
        raise ValueError("Expected unique Date, Ticker rows.")
    cutoff = pd.Timestamp(development_end)
    history = panel.loc[panel.index.get_level_values("Date") <= cutoff].sort_index().copy()
    dates = pd.DatetimeIndex(history.index.get_level_values("Date").unique()).sort_values()
    if len(dates) < min_train_dates + calibration_dates + evaluation_dates:
        raise ValueError("Not enough development dates for a complete evaluation window.")
    available = pd.to_datetime(history["Target_Available_At"])
    feature_dates = history.index.get_level_values("Date")
    if (available.notna() & (available <= feature_dates)).any():
        raise ValueError("Forward targets must become available after their feature date.")
    candidates = models if models is not None else build_multiclass_models()
    if "Prior baseline" not in candidates:
        raise ValueError("Include a Prior baseline for a fair comparison.")
    folds, predictions = [], []
    start = min_train_dates + calibration_dates
    for fold, position in enumerate(range(start, len(dates) - evaluation_dates + 1,
                                          evaluation_dates), 1):
        cal_start = dates[position - calibration_dates]
        score_start, score_end = dates[position], dates[position + evaluation_dates - 1]
        train = history.loc[(feature_dates < cal_start) & (available < cal_start)]
        calibration = history.loc[(feature_dates >= cal_start) & (feature_dates < score_start)
                                  & (available < score_start)]
        evaluation = history.loc[(feature_dates >= score_start) & (feature_dates <= score_end)
                                 & (available <= score_end)]
        train, x, y = _model_data(train, features)
        calibration, cal_x, cal_y = _model_data(calibration, features)
        evaluation, eval_x, eval_y = _model_data(evaluation, features)
        if set(y.unique()) != {-1, 0, 1}:
            raise ValueError("Each training window requires all three target classes.")
        for name, candidate in candidates.items():
            fitted = clone(candidate).fit(x, _encode_target(y))
            raw = _aligned_probabilities(fitted, eval_x)
            versions = {"Raw": (1., raw)}
            if name != "Prior baseline":
                temperature = fit_temperature(_aligned_probabilities(fitted, cal_x), cal_y)
                versions["Calibrated"] = (temperature, temperature_scale(raw, temperature))
            for variant, (temperature, probability) in versions.items():
                label = f"{name} · {variant}" if name != "Prior baseline" else name
                metrics = multiclass_metrics(eval_y.to_numpy(), probability.argmax(axis=1) - 1,
                                             probability)
                folds.append({
                    "fold": fold, "candidate": label, "model": name, "variant": variant,
                    "temperature": temperature, "train_start": str(train.index[0][0].date()),
                    "train_end": str(train.index[-1][0].date()),
                    "calibration_start": str(cal_start.date()),
                    "evaluation_start": str(score_start.date()), "evaluation_end": str(score_end.date()),
                    "train_label_end": str(train.Target_Available_At.max().date()),
                    "calibration_label_end": str(calibration.Target_Available_At.max().date()),
                    "train_rows": len(train), "calibration_rows": len(calibration),
                    "evaluation_rows": len(evaluation), **metrics,
                })
                output = evaluation[[RESEARCH_TARGET]].copy()
                output[PROBABILITY_COLUMNS] = probability
                output["candidate"], output["fold"] = label, fold
                predictions.append(output.reset_index())
    results = pd.concat(predictions, ignore_index=True)
    summary, reliability = [], []
    for name, group in results.groupby("candidate", sort=True):
        probability = group[PROBABILITY_COLUMNS].to_numpy()
        summary.append({"candidate": name, "rows": len(group), **multiclass_metrics(
            group[RESEARCH_TARGET].to_numpy(), probability.argmax(axis=1) - 1, probability
        )})
        for label, column in zip([-1, 0, 1], PROBABILITY_COLUMNS):
            bands = np.minimum((group[column] * 10).astype(int), 9)
            for band, points in group.groupby(bands):
                reliability.append({"candidate": name, "class": column, "bin": int(band),
                                    "count": len(points), "predicted": float(points[column].mean()),
                                    "observed": float((points[RESEARCH_TARGET] == label).mean())})
    report = {"folds": folds, "summary": summary, "reliability": reliability,
              "development_end": str(cutoff.date()), "features": list(features),
              "windows": {"minimum_training_dates": min_train_dates,
                          "calibration_dates": calibration_dates, "evaluation_dates": evaluation_dates},
              "temperature_grid": TEMPERATURES.tolist(),
              "unused_tail_dates": len(dates) - (start + len(set(results.fold)) * evaluation_dates)}
    return report, results
