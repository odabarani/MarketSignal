import numpy as np
import pandas as pd
import pytest

from holdout_uncertainty import block_length_sensitivity, compare_equity_curves


def test_paired_bootstrap_is_deterministic_and_preserves_equal_curves():
    dates = pd.bdate_range("2024-01-01", periods=82)
    returns = np.where(np.arange(len(dates)) % 2, 0.01, -0.005)
    wealth = 100 * np.cumprod(1 + returns)
    equity = pd.DataFrame({"Strategy": wealth, "SPY": wealth}, index=dates)

    first = compare_equity_curves(equity, "Strategy", "SPY", block_sessions=10, resamples=500, seed=7)
    second = compare_equity_curves(equity, "Strategy", "SPY", block_sessions=10, resamples=500, seed=7)

    assert first == second
    assert first["observed_annualized_relative_growth"] == pytest.approx(0)
    assert first["bootstrap_annualized_relative_growth_2_5_50_97_5_pct"] == pytest.approx([0, 0, 0])
    assert first["observed_max_drawdown_gap"] == pytest.approx(0)


@pytest.mark.parametrize("block_sessions,resamples", [(0, 500), (10, 99)])
def test_bootstrap_rejects_invalid_settings(block_sessions, resamples):
    equity = pd.DataFrame({"Strategy": [1.0, 1.01, 1.02], "SPY": [1.0, 1.0, 1.0]})

    with pytest.raises(ValueError):
        compare_equity_curves(
            equity, "Strategy", "SPY", block_sessions=block_sessions, resamples=resamples
        )


def test_bootstrap_requires_chronological_unique_date_index():
    equity = pd.DataFrame(
        {"Strategy": [1.0, 1.1, 1.2], "SPY": [1.0, 1.0, 1.0]},
        index=pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"]),
    )

    with pytest.raises(ValueError, match="unique and chronological"):
        compare_equity_curves(equity, "Strategy", "SPY", resamples=500)


def test_block_length_sensitivity_reproduces_reference_length():
    dates = pd.bdate_range("2024-01-01", periods=42)
    strategy = np.cumprod(1 + np.where(np.arange(len(dates)) % 2, 0.01, -0.005))
    benchmark = np.cumprod(1 + np.full(len(dates), 0.001))
    equity = pd.DataFrame({"Strategy": strategy, "SPY": benchmark}, index=dates)

    sensitivity = block_length_sensitivity(
        equity, "Strategy", "SPY", block_sessions=(5, 20), resamples=500, seed=9
    )
    reference = compare_equity_curves(
        equity, "Strategy", "SPY", block_sessions=20, resamples=500, seed=9
    )

    assert [row["block_sessions"] for row in sensitivity] == [5, 20]
    assert sensitivity[1]["relative_growth_interval"] == reference[
        "bootstrap_annualized_relative_growth_2_5_50_97_5_pct"
    ]
    assert sensitivity[1]["share_strategy_lower_drawdown"] == reference[
        "bootstrap_share_strategy_lower_drawdown"
    ]
