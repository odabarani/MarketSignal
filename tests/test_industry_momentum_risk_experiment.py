import numpy as np
import pandas as pd
import pytest

from industry_momentum_risk_experiment import (
    build_selected_momentum_weights,
    selected_basket_volatility,
    schedule_targets,
)


def test_selected_basket_volatility_uses_selected_assets_through_decision_close():
    dates = pd.bdate_range("2024-01-02", periods=25)
    returns = pd.DataFrame({"A": np.tile([0.02, -0.02], 13)[:25],
                            "B": np.zeros(25), "C": np.tile([0.1, -0.1], 13)[:25]}, index=dates)

    selected_volatility = selected_basket_volatility(returns, ["A", "B"], dates[-1], lookback=20)
    all_asset_volatility = selected_basket_volatility(returns, ["A", "B", "C"], dates[-1], lookback=20)

    assert selected_volatility == pytest.approx(returns.loc[:dates[-1], ["A", "B"]].mean(axis=1).tail(20).std(ddof=1) * np.sqrt(252))
    assert selected_volatility < all_asset_volatility


def test_monthly_targets_wait_two_sessions_before_becoming_effective():
    dates = pd.bdate_range("2024-01-02", periods=8)
    columns = ["A", "Market", "Cash"]
    target = pd.Series([0.4, 0.0, 0.6], index=columns)

    weights = schedule_targets(dates, {dates[2]: target}, columns)

    assert weights.loc[dates[2], "Cash"] == pytest.approx(1.0)
    assert weights.loc[dates[3], "Cash"] == pytest.approx(1.0)
    assert weights.loc[dates[4]].to_list() == pytest.approx([0.4, 0.0, 0.6])


def test_selected_momentum_weights_ignore_future_returns():
    dates = pd.bdate_range("2020-01-02", periods=420)
    values = np.arange(len(dates), dtype=float)
    industries = pd.DataFrame({f"I{i}": 0.0002 + values * (i + 1) * 1e-8
                               for i in range(12)}, index=dates)
    before = build_selected_momentum_weights(industries)
    january_decision = dates[dates.to_period("M") == pd.Period("2021-01")][-1]
    altered = industries.copy()
    altered.loc[altered.index > january_decision] = 0.2
    after = build_selected_momentum_weights(altered)
    pd.testing.assert_frame_equal(before.loc[:january_decision], after.loc[:january_decision])
    assert before.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(dates)))


def test_selected_momentum_weights_reject_invalid_target():
    dates = pd.bdate_range("2024-01-02", periods=5)
    industries = pd.DataFrame({"A": np.zeros(5)}, index=dates)

    with pytest.raises(ValueError, match="Target volatility"):
        build_selected_momentum_weights(industries, target_volatility=0)
