import numpy as np
import pandas as pd

from src.signals import generate_signal


def equity_from_returns(returns, starting_capital=10000, initial_date=None):
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        raise ValueError("No returns are available for the equity curve.")
    initial_date = pd.Timestamp(initial_date if initial_date is not None else returns.index[0])
    values = [float(starting_capital)]
    values.extend((starting_capital * (1 + returns).cumprod()).tolist())
    index = pd.DatetimeIndex([initial_date, *returns.index])
    return pd.Series(values, index=index, name="Equity")


def backtest(predictions, starting_capital=10000, transaction_cost_bps=10,
             slippage_bps=5, confidence_threshold=0.60):
    data = predictions.sort_index().copy()
    required = {"Prediction", "Probability", "Open"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {sorted(missing)}")
    if len(data) < 3:
        raise ValueError("At least three prediction rows are required for a backtest.")

    data["signal_name"] = [
        generate_signal(prediction, probability, confidence_threshold)
        for prediction, probability in zip(data["Prediction"], data["Probability"])
    ]
    data["signal"] = data["signal_name"].map({"UP": 1.0, "DOWN": 0.0})
    data["position"] = data["signal"].ffill().fillna(0.0)
    data["ExecutionDate"] = pd.Series(data.index, index=data.index).shift(-1)
    data["RealizationDate"] = pd.Series(data.index, index=data.index).shift(-2)
    data["market_return"] = data["Open"].shift(-2) / data["Open"].shift(-1) - 1
    data["turnover"] = data["position"].diff().abs().fillna(data["position"].abs())

    friction = (transaction_cost_bps + slippage_bps) / 10000.0
    data["strategy_return"] = (
        data["position"] * data["market_return"] - data["turnover"] * friction
    )
    if "SPY_Open" in data:
        data["benchmark_return"] = (
            data["SPY_Open"].shift(-2) / data["SPY_Open"].shift(-1) - 1
        )
    data = data.dropna(subset=["strategy_return", "RealizationDate"]).copy()
    data.index.name = "DecisionDate"
    realized = pd.Series(
        data["strategy_return"].to_numpy(),
        index=pd.DatetimeIndex(data["RealizationDate"]),
    )
    equity = equity_from_returns(
        realized, starting_capital, initial_date=data["ExecutionDate"].iloc[0]
    )
    return equity, data


def benchmark_equity(prices, starting_capital=10000):
    prices = pd.Series(prices, dtype=float).dropna().sort_index()
    if len(prices) < 2:
        raise ValueError("At least two benchmark prices are required.")
    return equity_from_returns(
        prices.pct_change(fill_method=None).dropna(),
        starting_capital,
        initial_date=prices.index[0],
    )


def calculate_metrics(equity):
    equity = pd.Series(equity, dtype=float).dropna()
    if len(equity) < 2:
        raise ValueError("At least two equity observations are required.")
    returns = equity.pct_change(fill_method=None).dropna()
    periods = len(returns)
    annualized_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / periods) - 1
    volatility = returns.std(ddof=1) * np.sqrt(252) if periods > 1 else 0.0
    return_std = returns.std(ddof=1) if periods > 1 else 0.0
    sharpe = returns.mean() / return_std * np.sqrt(252) if return_std > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
    }
