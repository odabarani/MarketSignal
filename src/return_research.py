"""Expected-return experiments with selection strictly before evaluation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from src.features import TECHNICAL_FEATURES, add_features
from src.portfolio import BENCHMARK, _integer, _prices
from src.rebalancing import rebalance_backtest
from src.research_features import RELATIVE_FEATURES, add_panel_features

RETURN_TARGET = "Forward_Open_Return"
AVAILABLE = "Return_Available_At"
RETURN_FEATURES = RELATIVE_FEATURES + ["SPY_Return_20", "SPY_Trend_60", "SPY_Volatility_20", "VIX_Change_5"]
BASELINES = ["Cash", "SPY buy & hold", "SPY volatility target",
             "Equal-weight volatility target", "Momentum volatility target"]
POLICY = {"horizon": 5, "top_n": 3, "max_weight": .35, "volatility_target": .15,
          "minimum_edge": .002, "base_cost_bps": 15, "drawdown_budget": .20}


class StableRidgeRegressor(RegressorMixin, BaseEstimator):
    """Standard intercept + L2 ridge via SVD, without macOS BLAS matmul flags.

    The objective is sum of squared residuals + alpha * squared coefficients.
    Unoptimized einsum keeps the two matrix products off the problematic BLAS
    path; no numerical warnings are suppressed and no target is clipped.
    """

    def __init__(self, alpha=100.):
        self.alpha = alpha

    def fit(self, x, y):
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("Ridge alpha must be positive and finite.")
        x, y = check_X_y(x, y, dtype="float64", y_numeric=True)
        self.n_features_in_ = x.shape[1]
        center, mean = x.mean(axis=0), float(y.mean())
        u, singular, vt = np.linalg.svd(x - center, full_matrices=False)
        projection = np.einsum("ij,i->j", u, y - mean, optimize=False)
        self.coef_ = np.einsum("ij,i->j", vt,
            singular / (singular ** 2 + self.alpha) * projection, optimize=False)
        self.intercept_ = mean - float(np.einsum("i,i->", center, self.coef_, optimize=False))
        if not np.isfinite(self.coef_).all() or not np.isfinite(self.intercept_):
            raise ValueError("Ridge produced invalid parameters.")
        return self

    def predict(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype="float64")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from training.")
        return np.einsum("ij,j->i", x, self.coef_, optimize=False) + self.intercept_


def _policy(settings):
    settings = {} if settings is None else settings
    if set(settings).difference(POLICY):
        raise ValueError("Unknown portfolio policy setting.")
    result = {**POLICY, **settings}
    for key in ["horizon", "top_n"]:
        result[key] = _integer(result[key], key)
    if (not all(np.isfinite(result[key]) for key in ["max_weight", "volatility_target", "minimum_edge", "base_cost_bps", "drawdown_budget"])
            or not 0 < result["max_weight"] <= 1 or result["volatility_target"] <= 0
            or result["minimum_edge"] < 0 or not 0 <= result["base_cost_bps"] < 10000
            or not 0 < result["drawdown_budget"] <= 1):
        raise ValueError("Invalid concentration, risk, edge or cost policy.")
    return result


def build_return_models():
    return {"Historical mean": DummyRegressor(strategy="mean"),
            "Ridge returns": Pipeline([("scaler", RobustScaler()),
                                        ("model", StableRidgeRegressor(alpha=100.))]),
            "Shallow XGBoost returns": XGBRegressor(
                objective="reg:squarederror", n_estimators=100, max_depth=2,
                learning_rate=.03, min_child_weight=20, reg_lambda=10., reg_alpha=.1,
                subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=2)}


def prepare_return_panel(panel, development_end, horizon=5):
    """Build five open-to-open interval returns, after slicing away later prices."""
    horizon = _integer(horizon, "Horizon")
    panel = panel.loc[panel.index.get_level_values("Date") <= pd.Timestamp(development_end)].sort_index().copy()
    _prices(panel)  # Later prices cannot affect even validation of development inputs.
    prices = panel[["Open", "Close", "SPY_Open", "SPY_Close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Development execution prices must be finite and positive.")
    dates = panel.index.get_level_values("Date").unique().sort_values()
    symbols = panel.index.get_level_values("Ticker").unique()
    if len(panel) != len(dates) * len(symbols):
        raise ValueError("This experiment requires a complete fixed-universe session panel.")
    if not np.isfinite(panel.VIX).all() or (panel.VIX < 0).any():
        raise ValueError("Observable VIX histories must be finite and nonnegative.")
    # Ignore cached features: rebuild them from the sliced observable bars.
    for ticker, frame in panel.groupby(level="Ticker", sort=True):
        raw = frame.droplevel("Ticker")
        rebuilt = add_features(raw)
        index = frame.index
        for column in set(TECHNICAL_FEATURES).difference({"VIX"}):
            panel.loc[index, column] = rebuilt[column].reindex(raw.index).to_numpy()
    panel = add_panel_features(panel)
    group = panel.groupby(level="Ticker")
    panel[RETURN_TARGET] = group.Open.shift(-(horizon + 1)) / group.Open.shift(-1) - 1
    expiry = pd.Series(dates, index=dates).shift(-(horizon + 1))
    panel[AVAILABLE] = panel.index.get_level_values("Date").map(expiry)
    panel["Risk_Volatility_20"] = group.Close.transform(
        lambda x: x.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252))
    return panel


def allocation_rules(panel, forecasts, decisions, selected=None, policy=None):
    """Create observable allocations; thresholds are fixed before scoring returns.

    A positive retained forecast can keep a position below the new-entry hurdle.
    Volatility sizing uses a conservative weighted sum of trailing volatilities,
    not a prediction of future portfolio volatility or a loss guarantee.
    """
    policy = _policy(policy)
    symbols = sorted(panel.index.get_level_values("Ticker").unique())
    columns = symbols + [BENCHMARK]
    candidates = sorted(forecasts.candidate.unique())
    names = candidates + BASELINES + (["Chronological selector"] if selected is not None else [])
    if len(set(names)) != len(names):
        raise ValueError("Model names must not collide with baseline names.")
    output = {name: pd.DataFrame(0., index=decisions, columns=columns) for name in names}
    lookups = {name: rows.set_index(["Date", "Ticker"])["expected_return"]
               for name, rows in forecasts.groupby("candidate")}
    if forecasts.duplicated(["candidate", "Date", "Ticker"]).any():
        raise ValueError("Duplicate return forecasts.")
    if not np.isfinite(forecasts.expected_return).all():
        raise ValueError("Return forecasts must be finite.")
    previous = {name: set() for name in names}
    hurdle = 2 * policy["base_cost_bps"] / 10000 + policy["minimum_edge"]
    for date in decisions:
        observed = panel.loc[date].reindex(symbols)
        volatility = observed.Risk_Volatility_20
        market_volatility = float(observed.SPY_Volatility_20.iloc[0] * np.sqrt(252))
        if (not np.isfinite(volatility).all() or (volatility < 0).any()
                or not np.isfinite(market_volatility) or market_volatility < 0):
            raise ValueError("Trailing risk estimates must be finite and nonnegative.")
        for name in names:
            rule = selected[date] if name == "Chronological selector" else name
            weights = pd.Series(0., index=columns)
            if rule == "Cash":
                pass
            elif rule in ["SPY buy & hold", "SPY volatility target"]:
                weights[BENCHMARK] = 1. if rule == "SPY buy & hold" else min(
                    1., policy["volatility_target"] / max(market_volatility, 1e-12))
            else:
                if rule == "Equal-weight volatility target":
                    chosen = symbols
                elif rule == "Momentum volatility target":
                    score = observed.Return_20
                    chosen = score.loc[score > 0].sort_values(ascending=False, kind="stable").head(policy["top_n"]).index
                else:
                    if rule not in lookups or date not in lookups[rule].index.get_level_values("Date"):
                        raise ValueError("Every decision needs forecasts from each required model.")
                    prediction = lookups[rule].loc[date].reindex(symbols)
                    if not np.isfinite(prediction).all():
                        raise ValueError("Every stock needs a finite decision-time forecast.")
                    retained = prediction.index.isin(previous[name])
                    eligible = prediction.loc[(prediction >= hurdle) | (retained & (prediction > 0))]
                    score = eligible + eligible.index.isin(previous[name]) * policy["base_cost_bps"] / 10000
                    chosen = score.sort_values(ascending=False, kind="stable").head(policy["top_n"]).index
                if len(chosen):
                    weights.loc[chosen] = min(policy["max_weight"], 1 / len(chosen))
                    risk = float((weights.reindex(symbols) * volatility).sum())
                    weights *= min(1., policy["volatility_target"] / max(risk, 1e-12))
            output[name].loc[date] = weights
            previous[name] = set(weights.loc[symbols][weights.loc[symbols] > 0].index)
    # A genuine passive benchmark buys once and is held through the common end.
    output["SPY buy & hold"] = output["SPY buy & hold"].iloc[:1]
    return output


def _forecast(model, frame, features, name, fold):
    values = np.asarray(model.predict(frame[features]), dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("Model returned invalid expected returns.")
    result = frame.reset_index()[["Date", "Ticker"]]
    result["expected_return"], result["candidate"], result["fold"] = values, name, fold
    return result


def return_walk_forward(panel, development_end, min_train_dates=504,
                        selection_dates=126, evaluation_dates=126, models=None,
                        features=None, policy=None):
    """Fit, select on earlier realized portfolios, refit, then freeze and evaluate.

    The selector may choose a simple baseline or cash. Model and policy settings
    are fixed across all windows; evaluation outcomes never choose their own
    model. Later evaluation periods are reused development data, not holdouts.
    """
    policy = _policy(policy)
    min_train_dates, selection_dates, evaluation_dates = [
        _integer(count, "Window", 10) for count in [min_train_dates, selection_dates, evaluation_dates]]
    prepared = prepare_return_panel(panel, development_end, policy["horizon"])
    features = RETURN_FEATURES if features is None else list(features)
    if set(features).intersection({RETURN_TARGET, AVAILABLE, "Research_Target", "Forward_Excess_Return", "Target", "Target_Available_At"}):
        raise ValueError("Future outcomes cannot be model features.")
    if (not features or len(set(features)) != len(features)
            or set(features).difference(set(RETURN_FEATURES) | set(TECHNICAL_FEATURES))):
        raise ValueError("Use unique, rebuilt observable research features.")
    dates = prepared.index.get_level_values("Date").unique().sort_values()
    if len(dates) < min_train_dates + selection_dates + evaluation_dates:
        raise ValueError("Not enough dates for complete return evaluation windows.")
    complete = np.isfinite(prepared[features].to_numpy(dtype=float)).all(axis=1)
    eligible = complete & np.isfinite(prepared[RETURN_TARGET]) & prepared[AVAILABLE].notna()
    feature_dates = prepared.index.get_level_values("Date")
    candidates = build_return_models() if models is None else models
    if not candidates or "Historical mean" not in candidates:
        raise ValueError("Include the Historical mean return baseline.")
    forecasts, choices, folds = [], {}, []
    start = min_train_dates + selection_dates
    score_positions = list(range(start, len(dates) - evaluation_dates + 1, evaluation_dates))
    evaluation_limit = dates[score_positions[-1] + evaluation_dates - 1]
    with threadpool_limits(limits=2):
        for fold, position in enumerate(score_positions, 1):
            selection_start, score_start = dates[position - selection_dates], dates[position]
            score_end = dates[position + evaluation_dates - 1]
            train = prepared.loc[eligible & (feature_dates < selection_start) & (prepared[AVAILABLE] < selection_start)]
            selection = prepared.loc[eligible & (feature_dates >= selection_start)
                                     & (feature_dates < score_start) & (prepared[AVAILABLE] < score_start)]
            refit = prepared.loc[eligible & (feature_dates < score_start) & (prepared[AVAILABLE] < score_start)]
            evaluation = prepared.loc[eligible & (feature_dates >= score_start) & (feature_dates <= score_end)
                                      & (prepared[AVAILABLE] <= evaluation_limit)]
            if min(len(train), len(selection), len(refit), len(evaluation)) == 0:
                raise ValueError("A return window has no complete training or evaluation rows.")
            cal_predictions = []
            for name, candidate in candidates.items():
                fitted = clone(candidate).fit(train[features], train[RETURN_TARGET])
                cal_predictions.append(_forecast(fitted, selection, features, name, fold))
                # A fresh expanding fit uses only labels known before this score block.
                refitted = clone(candidate).fit(refit[features], refit[RETURN_TARGET])
                forecasts.append(_forecast(refitted, evaluation, features, name, fold))
            cal_predictions = pd.concat(cal_predictions, ignore_index=True)
            first, last = dates.get_indexer([selection.index[0][0], selection.index[-1][0]])
            decisions = dates[np.arange(first, last + 1, policy["horizon"])]
            liquidation = dates[dates.get_loc(decisions[-1]) + policy["horizon"] + 1]
            if liquidation >= score_start:
                raise ValueError("Selection outcomes cross the evaluation boundary.")
            allocations = allocation_rules(prepared, cal_predictions, decisions, policy=policy)
            scores = []
            for name, weights in allocations.items():
                result = rebalance_backtest(prepared, weights, liquidation, policy["base_cost_bps"])
                scores.append({"candidate": name, **result.metrics})
            qualified = [row for row in scores if row["max_drawdown"] >= -policy["drawdown_budget"]]
            # Stable name tie-break; an equally good cash baseline remains valid.
            selected = sorted(qualified, key=lambda x: (-x["annualized_return"], x["candidate"]))[0]["candidate"]
            for date in dates[position:position + evaluation_dates]:
                choices[date] = selected
            folds.append({"fold": fold, "selected": selected,
                          "selection_start": str(selection_start.date()), "selection_liquidation": str(liquidation.date()),
                          "train_label_end": str(train[AVAILABLE].max().date()),
                          "refit_label_end": str(refit[AVAILABLE].max().date()),
                          "evaluation_start": str(score_start.date()), "evaluation_end": str(score_end.date()),
                          "selection_scores": scores})
    predictions = pd.concat(forecasts, ignore_index=True)
    errors = []
    for name, group in predictions.groupby("candidate"):
        actual = prepared.reindex(pd.MultiIndex.from_frame(group[["Date", "Ticker"]]))[RETURN_TARGET].to_numpy()
        difference = group.expected_return.to_numpy() - actual
        errors.append({"candidate": name, "rows": len(group), "mae": float(np.abs(difference).mean()),
                       "rmse": float(np.sqrt(np.mean(difference ** 2)))})
    first, last = dates.get_indexer([predictions.Date.min(), predictions.Date.max()])
    decisions = dates[np.arange(first, last + 1, policy["horizon"])]
    liquidation = dates[dates.get_loc(decisions[-1]) + policy["horizon"] + 1]
    allocations = allocation_rules(prepared, predictions, decisions, choices, policy)
    results, rows = {}, []
    for cost in [15, 30, 50]:
        for name, weights in allocations.items():
            result = rebalance_backtest(prepared, weights, liquidation, cost)
            results[(name, cost)] = result
            rows.append({"candidate": name, "cost_bps_per_side": cost, **result.metrics})
    report = {"development_end": str(pd.Timestamp(development_end).date()), "features": features,
              "policy": policy, "windows": {"minimum_training_dates": min_train_dates,
              "selection_dates": selection_dates, "evaluation_dates": evaluation_dates},
              "folds": folds, "summary": rows, "forecast_errors": errors,
              "start": str(decisions[0].date()), "end": str(liquidation.date()),
              "status": "Reused development data. No untouched-test or live profitability claim."}
    return report, predictions, results
