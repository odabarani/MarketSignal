import pandas as pd
from src.backtest import backtest, calculate_metrics

def test_backtest_uses_next_day_return():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame({"Close": [100, 110, 100, 100, 100], "Prediction": [1, 1, 0, 0, 0], "Probability": [0.8] * 5}, index=idx)
    equity, data = backtest(df, transaction_cost_bps=0, slippage_bps=0)
    assert len(equity) == 4
    assert data["next_return"].iloc[0] == 0.10

def test_metrics_return_expected_keys():
    metrics = calculate_metrics(pd.Series([10000, 10100, 10050, 10200]))
    assert {"total_return", "annualized_return", "volatility", "sharpe", "max_drawdown", "win_rate"} <= metrics.keys()