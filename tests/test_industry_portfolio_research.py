import numpy as np
import pandas as pd
import pytest

from industry_portfolio_research import backtest_weights, build_strategy_weights


def test_monthly_weights_wait_for_two_sessions_after_decision_close():
    dates = pd.bdate_range("2022-01-03", periods=80)
    industries = pd.DataFrame({"A": 0.001, "B": 0.002}, index=dates)
    market = pd.Series(0.001, index=dates, name="Market")
    cash = pd.Series(0.0001, index=dates, name="Cash")

    weights = build_strategy_weights(industries, market, cash)["Equal-weight 49 industries"]
    january_decision = dates[dates.to_period("M") == pd.Period("2022-01")][-1]
    position = dates.get_loc(january_decision)

    assert weights.iloc[position][["A", "B"]].sum() == pytest.approx(0.0)
    assert weights.iloc[position + 1][["A", "B"]].sum() == pytest.approx(0.0)
    assert weights.iloc[position + 2][["A", "B"]].to_list() == pytest.approx([0.5, 0.5])


def test_strategy_features_do_not_change_when_future_returns_change():
    dates = pd.bdate_range("2020-01-01", periods=420)
    values = np.arange(len(dates), dtype=float)
    industries = pd.DataFrame({"A": 0.0002 + values * 1e-7,
                               "B": 0.0003 - values * 1e-8}, index=dates)
    market = pd.Series(0.0002, index=dates, name="Market")
    cash = pd.Series(0.00005, index=dates, name="Cash")
    before = build_strategy_weights(industries, market, cash)
    cutoff = dates[300]
    industries.loc[industries.index > cutoff] = 0.9
    market.loc[market.index > cutoff] = -0.9
    after = build_strategy_weights(industries, market, cash)

    for strategy in before:
        pd.testing.assert_frame_equal(before[strategy].loc[:cutoff], after[strategy].loc[:cutoff])


def test_costs_include_initial_entry_and_terminal_liquidation():
    dates = pd.bdate_range("2024-01-02", periods=4)
    industries = pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)
    market = pd.Series(0.0, index=dates, name="Market")
    cash = pd.Series(0.0, index=dates, name="Cash")
    weights = pd.DataFrame(0.0, index=dates, columns=["A", "B", "Market", "Cash"])
    weights["Market"] = 1.0

    equity, _, turnover = backtest_weights(weights, industries, market, cash,
                                            cost_bps=100, start=dates[0], end=dates[-1])

    assert turnover == pytest.approx(2.0)
    assert equity.iloc[-1] == pytest.approx(0.99**2)
