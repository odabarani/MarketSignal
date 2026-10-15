import numpy as np
import pandas as pd
import pytest

from industry_momentum_rank_experiment import (
    build_ranked_momentum_weights,
    rank_volatility_adjusted_momentum,
)


def synthetic_industries(periods=420):
    dates = pd.bdate_range("2020-01-02", periods=periods)
    values = np.arange(periods, dtype=float)
    data = {}
    for i in range(12):
        drift = 0.00015 + i * 0.000015
        noise = np.where(values.astype(int) % 2, -1.0, 1.0) * (i + 1) * 0.00004
        data[f"I{i:02d}"] = drift + noise
    return pd.DataFrame(data, index=dates)


def test_volatility_adjusted_ranks_match_trailing_only_formula():
    returns = synthetic_industries()
    decision = returns.index[-1]
    wealth = (1 + returns).cumprod()
    momentum = wealth.shift(21) / wealth.shift(252) - 1
    volatility = returns.rolling(252, min_periods=252).std(ddof=1) * np.sqrt(252)
    scores = (momentum.loc[decision] / volatility.loc[decision]).replace([np.inf, -np.inf], np.nan)
    expected = list(scores.loc[momentum.loc[decision] > 0].dropna()
                    .sort_values(ascending=False, kind="stable").head(10).index)

    assert rank_volatility_adjusted_momentum(returns, decision) == expected


def test_ranked_weights_ignore_future_returns_and_sum_to_one():
    returns = synthetic_industries()
    before = build_ranked_momentum_weights(returns)
    decision = returns.index[returns.index.to_period("M") == pd.Period("2021-01")][-1]
    altered = returns.copy()
    altered.loc[altered.index > decision] = 0.1

    after = build_ranked_momentum_weights(altered)

    pd.testing.assert_frame_equal(before.loc[:decision], after.loc[:decision])
    assert before.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(returns)))


def test_ranked_weights_apply_two_session_execution_lag():
    returns = synthetic_industries()
    weights = build_ranked_momentum_weights(returns)
    decision = returns.index[returns.index.to_period("M") == pd.Period("2021-01")][-1]
    position = returns.index.get_loc(decision)
    selected = rank_volatility_adjusted_momentum(returns, decision)
    broad_volatility = returns.mean(axis=1).rolling(20, min_periods=20).std(ddof=1).iloc[position] * np.sqrt(252)
    expected = pd.Series(0.0, index=[*returns.columns, "Market", "Cash"])
    expected.loc[selected] = min(1.0, 0.05 / broad_volatility) / len(selected)
    expected["Cash"] = 1.0 - expected.drop("Cash").sum()

    pd.testing.assert_series_equal(weights.iloc[position + 2], expected, check_names=False)


def test_ranked_weights_stay_in_cash_until_full_momentum_window_exists():
    dates = pd.bdate_range("2024-01-02", periods=250)
    returns = pd.DataFrame(0.001, index=dates, columns=[f"I{i}" for i in range(10)])

    weights = build_ranked_momentum_weights(returns)

    assert weights.drop(columns="Cash").to_numpy() == pytest.approx(
        np.zeros((len(dates), len(returns.columns) + 1))
    )
    assert weights["Cash"].to_numpy() == pytest.approx(np.ones(len(dates)))


def test_ranked_weights_reject_invalid_target():
    returns = synthetic_industries()

    with pytest.raises(ValueError, match="Target volatility"):
        build_ranked_momentum_weights(returns, target_volatility=0)
