"""Cash-funded next-open rebalancing that preserves overnight positions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import calculate_metrics
from src.portfolio import _prices


@dataclass
class RebalanceResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


def rebalance_backtest(panel, weights, liquidation_date=None, cost_bps=15, starting_capital=10000):
    """Trade only weight differences; optionally mark the final holdings to close.

    A close-time target executes at the following session's open. Target weights
    apply to NAV after fees. Bisection solves the self-financing fee equation;
    sales fund purchases and no borrowing is permitted. Daily marks use closes.
    Set ``liquidation_date`` to sell at that open; leave it as ``None`` to mark
    an open paper portfolio without an assumed terminal sale. Splits/dividends
    must be handled by the price source; this engine does not reconstruct them.
    """
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("Per-side costs must be finite in [0, 10000) bps.")
    if not np.isfinite(starting_capital) or starting_capital <= 0:
        raise ValueError("Starting capital must be positive and finite.")
    if (not isinstance(weights.index, pd.DatetimeIndex) or weights.empty
            or weights.index.hasnans or weights.index.has_duplicates
            or not weights.index.is_monotonic_increasing or weights.columns.has_duplicates):
        raise ValueError("Weights need unique increasing decision dates and symbols.")
    weights = weights.astype(float)
    if (not np.isfinite(weights.to_numpy()).all() or (weights < 0).any().any()
            or (weights.sum(axis=1) > 1 + 1e-12).any()):
        raise ValueError("Weights must be finite, long-only and sum to at most one.")
    opens, closes = _prices(panel)
    symbols = weights.columns
    if not set(symbols).issubset(opens.columns):
        raise ValueError("Unknown allocation symbol.")
    positions = opens.index.get_indexer(weights.index)
    if liquidation_date is None:
        final = len(opens.index) - 1
        if (positions < 0).any() or positions[-1] + 1 > final:
            raise ValueError("All decisions need an execution open in the marked period.")
    else:
        final = opens.index.get_indexer([pd.Timestamp(liquidation_date)])[0]
        if (positions < 0).any() or final <= positions[-1] + 1:
            raise ValueError("All decisions need an execution open before final liquidation.")
    opens, closes = opens.reindex(columns=symbols), closes.reindex(columns=symbols)
    first = positions[0]
    execution = dict(zip(positions + 1, weights.to_numpy()))
    shares = np.zeros(len(symbols), dtype=float)
    cash = float(starting_capital)
    friction = float(cost_bps) / 10000
    values, records, turnover = [cash], [], 0.
    for position in range(first + 1, final + 1):
        date = opens.index[position]
        old_shares = shares.copy()
        active = old_shares > 0
        target = execution.get(position)
        if liquidation_date is not None and position == final:
            target = np.zeros(len(symbols))
        if target is not None:
            active |= target > 0
            price = opens.iloc[position].to_numpy()
            if not np.isfinite(price[active]).all() or (price[active] <= 0).any():
                raise ValueError("Missing or invalid execution price for a held or requested asset.")
            # Unused missing-price symbols must not contaminate cash accounting.
            price = np.where(active, price, 1.)
            current = shares * price
            nav = cash + current.sum()
            low, high = 0., nav
            for _ in range(64):
                middle = (low + high) / 2
                required = middle + friction * np.abs(target * middle - current).sum()
                if required > nav:
                    high = middle
                else:
                    low = middle
            after_fees = (low + high) / 2
            desired = target * after_fees / price
            changes = desired - shares
            notional = changes * price
            fees = np.abs(notional) * friction
            cash = float(cash - notional.sum() - fees.sum())
            shares = desired
            if cash < -1e-8 * nav or not np.isclose(cash + (shares * price).sum(), after_fees):
                raise ValueError("Rebalance violated self-financing accounting.")
            cash = max(cash, 0.)
            turnover += float(np.abs(notional).sum() / nav)
            for i in np.flatnonzero(np.abs(notional) > 1e-9 * nav):
                records.append({"date": date, "symbol": symbols[i],
                                "side": "BUY" if changes[i] > 0 else "SELL",
                                "quantity": float(abs(changes[i])), "reference_open": float(price[i]),
                                "notional": float(abs(notional[i])), "fee": float(fees[i]),
                                "cash_after_rebalance": cash})
        if liquidation_date is not None and position == final:
            nav = cash
        else:
            active = shares > 0
            marks = closes.iloc[position].to_numpy()
            if not np.isfinite(marks[active]).all() or (marks[active] <= 0).any():
                raise ValueError("Missing or invalid close for a held asset.")
            nav = cash + (shares[active] * marks[active]).sum()
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("Portfolio NAV must stay positive and finite.")
        values.append(float(nav))
    equity = pd.Series(values, index=opens.index[first:final + 1], name="Equity")
    trades = pd.DataFrame(records, columns=["date", "symbol", "side", "quantity", "reference_open",
                                           "notional", "fee", "cash_after_rebalance"])
    metrics = calculate_metrics(equity)
    metrics["positive_daily_fraction"] = metrics.pop("win_rate")
    metrics.update({"orders": len(trades), "fees_paid": float(trades.fee.sum()),
                    "annualized_turnover": turnover * 252 / (len(equity) - 1),
                    "mean_target_exposure": float(weights.sum(axis=1).mean())})
    return RebalanceResult(equity, trades, metrics)
