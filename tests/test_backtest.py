import pandas as pd
import pytest

from src.backtest import backtest, calculate_metrics, equity_from_returns, trading_metrics


def make_predictions(open_prices, probabilities):
    idx = pd.date_range("2024-01-01", periods=len(open_prices), freq="B")
    return pd.DataFrame({
        "Open": open_prices,
        "SPY_Open": open_prices,
        "Prediction": [int(value >= 0.5) for value in probabilities],
        "Probability": probabilities,
    }, index=idx)


def test_up_then_confident_down_enters_then_exits_at_next_open():
    data = make_predictions([100, 100, 110, 110, 110], [0.9, 0.1, 0.5, 0.5, 0.5])
    equity, detail = backtest(data, transaction_cost_bps=0, slippage_bps=0)

    assert detail["position"].tolist() == [1.0, 0.0, 0.0]
    assert detail["ExecutionDate"].iloc[0] == data.index[1]
    assert detail["RealizationDate"].iloc[0] == data.index[2]
    assert equity.tolist() == pytest.approx([10000, 11000, 11000, 11000])
    assert calculate_metrics(equity)["total_return"] == pytest.approx(0.10)


def test_entry_and_exit_each_pay_friction_once():
    data = make_predictions([100] * 5, [0.9, 0.1, 0.5, 0.5, 0.5])
    equity, detail = backtest(data, transaction_cost_bps=10, slippage_bps=0)

    assert detail["turnover"].tolist() == [1.0, 1.0, 0.0]
    assert equity.iloc[-1] == pytest.approx(10000 * 0.999 * 0.999)


def test_metrics_include_first_loss_and_initial_capital():
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    equity = pd.Series([10000, 9000], index=idx)
    metrics = calculate_metrics(equity)

    assert metrics["total_return"] == pytest.approx(-0.10)
    assert metrics["max_drawdown"] == pytest.approx(-0.10)


def test_strategy_and_benchmark_share_dates_when_returns_match():
    data = make_predictions([100, 100, 110, 121, 133.1], [0.9] * 5)
    equity, detail = backtest(data, transaction_cost_bps=0, slippage_bps=0)
    benchmark_returns = pd.Series(
        detail["benchmark_return"].to_numpy(),
        index=pd.DatetimeIndex(detail["RealizationDate"]),
    )
    benchmark = equity_from_returns(benchmark_returns, 10000, equity.index[0])

    pd.testing.assert_series_equal(equity, benchmark)


def test_held_dividend_offsets_ex_date_price_drop_for_stock_and_spy():
    data = make_predictions([100, 100, 99, 99], [.9] * 4)
    data["Dividends"] = [0, 0, 1, 0]
    data["SPY_Dividends"] = [0, 0, 1, 0]
    equity, detail = backtest(data, transaction_cost_bps=0, slippage_bps=0)
    assert detail["market_return"].tolist() == pytest.approx([0, 0])
    assert detail["benchmark_return"].tolist() == pytest.approx([0, 0])
    assert equity.iloc[-1] == pytest.approx(10000)


def test_buying_on_ex_date_does_not_receive_that_days_dividend():
    data = make_predictions([100, 99, 99, 99], [.9] * 4)
    data["Dividends"] = [0, 1, 0, 0]
    equity, detail = backtest(data, transaction_cost_bps=0, slippage_bps=0)
    assert equity.iloc[-1] == pytest.approx(10000)


def test_cash_position_does_not_earn_dividend():
    data = make_predictions([100, 100, 99, 99], [.1] * 4)
    data["Dividends"] = [0, 0, 1, 0]
    equity, detail = backtest(data, transaction_cost_bps=0, slippage_bps=0)
    assert detail["position"].eq(0).all()
    assert equity.iloc[-1] == pytest.approx(10000)


def test_split_normalized_provider_prices_are_not_adjusted_twice():
    data = make_predictions([100, 100, 100, 100], [.9] * 4)
    data["Stock_Splits"] = [0, 0, 4, 0]
    equity, detail = backtest(data, transaction_cost_bps=0, slippage_bps=0)
    assert equity.iloc[-1] == pytest.approx(10000)


def test_entry_cost_reduces_invested_capital_and_final_exit_is_charged():
    data = make_predictions([100, 100, 110], [.9] * 3)
    equity, detail = backtest(data, transaction_cost_bps=100, slippage_bps=0)
    assert equity.iloc[-1] == pytest.approx(10000 * .99 * 1.10 * .99)
    assert trading_metrics(detail) == {"entries": 1, "exits": 1,
                                       "time_in_market": 1., "one_way_turnover": 2.}
