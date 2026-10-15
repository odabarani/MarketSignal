import numpy as np
import pandas as pd
import pytest

from src.stock_momentum import (
    build_targets,
    daily_returns,
    make_price_matrices,
    simulate,
    summarize,
    run_experiment,
    volatility_exposure,
    block_bootstrap_excess_return,
    simulate_with_sector_attribution,
)


def metadata_rows():
    return pd.DataFrame({
        "sid": ["SPY-SID", "STOCK-A", "STOCK-B"],
        "symbol": ["SPY", "AAA", "BBB"],
        "date_delisted": [pd.NaT, pd.NaT, pd.NaT],
        "first_price_date": [pd.Timestamp("2007-01-02")] * 3,
        "last_price_date": [pd.Timestamp("2012-01-01")] * 3,
        "security_type": ["ETF", "Common Stock", "Depositary Receipt"],
        "sector": ["Unknown", "Technology", "Financials"],
    })


def monthly_liquidity(dates, sids, value=10_000_000.0):
    dates = pd.DatetimeIndex(dates)
    decisions = pd.Series(dates, index=dates).groupby(dates.to_period("M")).last().tolist()
    return pd.DataFrame([
        {"sid": sid, "date": date, "average_adjusted_dollar_volume": value}
        for date in decisions for sid in sids
    ])


def test_matrices_resolve_spy_and_exclude_etf_from_stock_universe():
    panel = pd.DataFrame({
        "sid": ["SPY-SID", "STOCK-A", "STOCK-B"],
        "date": [pd.Timestamp("2007-01-02")] * 3,
        "close": [100.0, 10.0, 20.0],
        "volume": [1000, 1000, 1000],
    })
    close, spy, stocks = make_price_matrices(panel, metadata_rows())
    assert spy == "SPY-SID"
    assert stocks == ["STOCK-A", "STOCK-B"]
    assert close.loc[pd.Timestamp("2007-01-02"), spy] == 100


def test_monthly_targets_use_only_point_in_time_close_and_liquid_history():
    dates = pd.bdate_range("2007-01-02", periods=300)
    close = pd.DataFrame({
        "SPY-SID": np.linspace(100, 120, len(dates)),
        "STOCK-A": np.linspace(10, 30, len(dates)),
        "STOCK-B": np.linspace(20, 19, len(dates)),
    }, index=dates)
    liquidity = monthly_liquidity(dates, ["STOCK-A", "STOCK-B"])
    targets, regime, start = build_targets(
        close, ["STOCK-A", "STOCK-B"], "SPY-SID", liquidity
    )
    assert targets["12-1 momentum"]
    assert all(start <= date for date in targets["12-1 momentum"])
    assert all("SPY-SID" not in basket for basket in targets["Equal-weight stocks"].values())
    assert regime.index.equals(dates)
    assert all(set(basket).issubset({"STOCK-A"}) for basket in targets["12-1 momentum"].values())


def test_zero_recovery_delisting_is_explicit_and_last_quote_is_separate():
    dates = pd.to_datetime(["2007-01-02", "2007-01-03", "2007-01-04"])
    close = pd.DataFrame({"STOCK-A": [10.0, 9.0, np.nan]}, index=dates)
    metadata = metadata_rows().iloc[[1]].copy()
    metadata.loc[:, "date_delisted"] = pd.Timestamp("2007-01-03")
    metadata.loc[:, "last_price_date"] = pd.Timestamp("2007-01-03")
    conservative = daily_returns(close, metadata, "zero_recovery")
    optimistic = daily_returns(close, metadata, "last_quote")
    assert conservative.loc[dates[2], "STOCK-A"] == -1.0
    assert optimistic.loc[dates[2], "STOCK-A"] == 0.0


def test_simulator_charges_turnover_and_delays_returns_until_after_close_trade():
    dates = pd.to_datetime(["2007-01-02", "2007-01-03", "2007-01-04"])
    returns = pd.DataFrame({"STOCK-A": [0.0, 0.5, 0.1]}, index=dates)
    equity, net, turnover = simulate(
        returns, {dates[1]: {"STOCK-A": 1.0}}, 50, dates[1]
    )
    assert net.loc[dates[1]] == pytest.approx(-0.005)
    assert net.loc[dates[2]] == pytest.approx(0.1)
    assert turnover == pytest.approx(1.0 * 252 / 2)
    assert equity.loc[dates[2]] == pytest.approx(0.995 * 1.1)


def test_sector_attribution_includes_net_trading_costs_and_sums_to_portfolio_return():
    dates = pd.to_datetime(["2007-01-02", "2007-01-03", "2007-01-04"])
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.0], "B": [0.0, -0.10, -0.02]}, index=dates)
    equity, net, turnover, attribution = simulate_with_sector_attribution(
        returns,
        {dates[0]: {"A": 0.5, "B": 0.5}, dates[2]: {"A": 0.0, "B": 1.0}},
        30,
        dates[1],
        {"A": "Technology", "B": "Financials"},
    )
    assert turnover > 0
    assert attribution.loc[dates[1], "Technology"] > 0
    assert attribution.loc[dates[1], "Financials"] < 0
    assert attribution.loc[dates[2], "Financials"] < 0  # includes the rebalance cost
    assert attribution.sum().sum() == pytest.approx(equity.loc[dates[-1]] - 1.0)
    assert (1.0 + net).prod() - 1.0 == pytest.approx(equity.loc[dates[-1]] - 1.0)


def test_summary_reports_years_and_market_regimes():
    dates = pd.to_datetime(["2007-01-02", "2007-01-03", "2008-01-02"])
    equity = pd.Series([1.0, 1.1, 1.21], index=dates)
    returns = pd.Series([0.0, 0.1, 0.1], index=dates)
    regimes = pd.Series(["Below", "Below", "Above"], index=dates)
    result = summarize(equity, returns, regimes, dates[1])
    assert result["yearly_net_returns"]["2007"] == pytest.approx(0.1)
    assert result["market_regime_net_returns"]["Above"]["sessions"] == 1


def test_block_bootstrap_pairs_dates_and_is_reproducible():
    dates = pd.bdate_range("2020-01-01", periods=80)
    benchmark = pd.Series(np.tile([-0.01, 0.01], 40), index=dates)
    strategy = benchmark + 0.001
    first = block_bootstrap_excess_return(
        strategy, benchmark, block_sessions=10, resamples=100, seed=7
    )
    second = block_bootstrap_excess_return(
        strategy, benchmark, block_sessions=10, resamples=100, seed=7
    )
    assert first == second
    assert first["sessions"] == 80
    assert first["percentile_range_95"][0] < first["percentile_range_95"][1]
    assert "not a predictive interval" in first["interpretation"]


def test_identical_strategy_and_benchmark_have_zero_bootstrap_excess():
    returns = pd.Series(np.tile([-0.02, 0.015], 20))
    result = block_bootstrap_excess_return(returns, returns, resamples=30)
    assert result["median_excess_net_return"] == pytest.approx(0.0)
    assert result["percentile_range_95"] == pytest.approx([0.0, 0.0])


def test_block_bootstrap_rejects_misaligned_or_incomplete_daily_returns():
    dates = pd.bdate_range("2020-01-01", periods=3)
    benchmark = pd.Series([0.01, -0.01, 0.0], index=dates)
    with pytest.raises(ValueError, match="identical dates"):
        block_bootstrap_excess_return(
            pd.Series([0.01, -0.01], index=dates[:2]), benchmark
        )
    incomplete = benchmark.copy()
    incomplete.iloc[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        block_bootstrap_excess_return(benchmark, incomplete)


def test_volatility_sizing_is_capped_and_requires_a_nonzero_history():
    low_vol = pd.Series(np.tile([-0.002, 0.002], 10))
    assert volatility_exposure(low_vol) == 1.0
    larger_moves = pd.Series(np.tile([-0.05, 0.05], 10))
    assert volatility_exposure(larger_moves) < 1.0
    assert volatility_exposure(pd.Series([0.01] * 19)) == 0.0
    assert volatility_exposure(pd.Series([0.0] * 20)) == 0.0


def test_delisted_sid_without_last_price_date_is_rejected():
    dates = pd.to_datetime(["2007-01-02", "2007-01-03"])
    close = pd.DataFrame({"STOCK-A": [10.0, np.nan]}, index=dates)
    metadata = metadata_rows().iloc[[1]].copy()
    metadata.loc[:, "date_delisted"] = pd.Timestamp("2007-01-03")
    metadata.loc[:, "last_price_date"] = pd.NaT
    with pytest.raises(ValueError, match="incomplete"):
        daily_returns(close, metadata, "zero_recovery")


def test_full_experiment_is_serializable_and_compares_all_fixed_baselines():
    dates = pd.bdate_range("2007-01-02", periods=300)
    panel = pd.concat([
        pd.DataFrame({"sid": sid, "date": dates, "close": prices,
                      "volume": np.full(len(dates), 1_000_000.0)})
        for sid, prices in [
            ("SPY-SID", np.linspace(100, 120, len(dates))),
            ("STOCK-A", np.linspace(10, 30, len(dates))),
            ("STOCK-B", np.linspace(20, 19, len(dates))),
        ]
    ], ignore_index=True)
    metadata = metadata_rows()
    liquidity = monthly_liquidity(dates, ["STOCK-A", "STOCK-B"])
    report = run_experiment(panel, metadata, liquidity, cost_bps=(30,))
    assert report["phase"] == "historical_development_only"
    assert len(report["results"]) == 12
    assert {row["strategy"] for row in report["results"]} == {
        "Cash", "SPY buy-and-hold", "Equal-weight stocks", "12-1 momentum",
        "12-1 momentum with SPY trend", "12-1 momentum with 10% volatility target",
    }
    spy = next(row for row in report["results"] if row["strategy"] == "SPY buy-and-hold")
    assert spy["excess_net_return_vs_spy"] == pytest.approx(0.0)
    assert all("within_research_drawdown_budget" in row for row in report["results"])
    assert all("block_bootstrap_excess_net_return_vs_spy" in row for row in report["results"])
    assert all("sector_net_pnl_contribution_on_initial_equity" in row for row in report["results"])
    for row in report["results"]:
        sector_contribution = sum(row["sector_net_pnl_contribution_on_initial_equity"].values())
        assert sector_contribution == pytest.approx(row["net_return"])
    import json
    json.dumps(report, allow_nan=False)
