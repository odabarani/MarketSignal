import numpy as np
import pandas as pd
import pytest

from industry_portfolio_research import build_strategy_weights
from industry_trend_filter_experiment import apply_market_trend_filter


def test_trend_filter_moves_to_cash_below_market_trend_without_changing_risk_on_weights():
    dates = pd.bdate_range("2020-01-02", periods=420)
    rising = np.linspace(0.0001, 0.0008, len(dates))
    industries = pd.DataFrame({"A": rising, "B": rising / 2}, index=dates)
    falling_market = pd.Series(-0.001, index=dates, name="Market")
    cash = pd.Series(0.00005, index=dates, name="Cash")
    weights = build_strategy_weights(industries, falling_market, cash)

    base = weights["12-1 momentum, top 10, 5.0% volatility target"]
    trend = weights["Market 200-day trend"]
    filtered = apply_market_trend_filter(base, trend)

    assert filtered.loc[trend.Market == 0, "Cash"].eq(1).all()
    assert filtered.loc[trend.Market == 0].drop(columns="Cash").eq(0).all().all()
    assert filtered.loc[trend.Market > 0.5].equals(base.loc[trend.Market > 0.5])
    assert filtered.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(dates)))


def test_market_filter_rejects_misaligned_weight_panels():
    dates = pd.bdate_range("2024-01-02", periods=3)
    base = pd.DataFrame({"Market": [0.0] * 3, "Cash": [1.0] * 3}, index=dates)
    trend = base.iloc[1:].copy()

    with pytest.raises(ValueError, match="matching dates and assets"):
        apply_market_trend_filter(base, trend)
