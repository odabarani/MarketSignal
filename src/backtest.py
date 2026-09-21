import numpy as np
import pandas as pd

def backtest(predictions, starting_capital=10000, transaction_cost_bps=10,
             slippage_bps=5, confidence_threshold=0.60):
    data = predictions.copy()
    required = {"Prediction", "Probability", "Close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {sorted(missing)}")

    data["signal"] = np.where(
        (data["Probability"] >= confidence_threshold) & (data["Prediction"] == 1),
        1.0,
        np.where(
            (data["Probability"] >= confidence_threshold) & (data["Prediction"] == 0),
            0.0, np.nan
        ),
    )
    data["position"] = data["signal"].ffill().fillna(0.0)
    data["next_return"] = data["Close"].pct_change().shift(-1)
    data["turnover"] = data["position"].diff().abs().fillna(data["position"].abs())

    friction = (transaction_cost_bps + slippage_bps) / 10000.0
    data["strategy_return"] = (
        data["position"] * data["next_return"] - data["turnover"] * friction
    )
    data = data.dropna(subset=["strategy_return"]).copy()
    equity = starting_capital * (1 + data["strategy_return"]).cumprod()
    return equity, data

def benchmark_equity(close, starting_capital=10000):
    returns = close.pct_change().fillna(0)
    return starting_capital * (1 + returns).cumprod()

def calculate_metrics(equity):
    values = np.asarray(equity, dtype=float)
    if len(values) < 2:
        return {}
    returns = pd.Series(values).pct_change().dropna()
    annualized_return = (values[-1] / values[0]) ** (252 / len(returns)) - 1
    volatility = returns.std(ddof=1) * np.sqrt(252)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        if returns.std(ddof=1) > 0 else 0.0
    )
    peak = np.maximum.accumulate(values)
    max_drawdown = np.min((values - peak) / peak)
    return {
        "total_return": values[-1] / values[0] - 1,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": float((returns > 0).mean()),
    }
