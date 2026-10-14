"""Long-only, self-financing portfolio diagnostics for saved research signals.

Decisions use the close, orders enter at the next open, and positions are sold
at the horizon's close. Each cohort uses fresh cash; cohorts never overlap.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import calculate_metrics

BENCHMARK = "__SPY__"
PROBABILITIES = ["P_DOWN", "P_NO_TRADE", "P_UP"]


@dataclass
class PortfolioResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


def _integer(value, name, minimum=1):
    try:
        valid = not isinstance(value, bool) and np.isfinite(value) and int(value) == value and value >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def _prices(panel):
    if (not isinstance(panel.index, pd.MultiIndex)
            or panel.index.names != ["Date", "Ticker"] or panel.index.has_duplicates):
        raise ValueError("Expected unique Date, Ticker price rows.")
    if BENCHMARK in panel.index.get_level_values("Ticker"):
        raise ValueError("The benchmark symbol is reserved.")
    required = {"Open", "Close", "SPY_Open", "SPY_Close"}
    if not required.issubset(panel):
        raise ValueError(f"Missing portfolio prices: {sorted(required.difference(panel))}")
    dates = panel.index.get_level_values("Date")
    if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans:
        raise ValueError("Price dates must be valid timestamps.")
    if (panel.groupby(level="Date")[["SPY_Open", "SPY_Close"]]
            .nunique(dropna=False) > 1).any().any():
        raise ValueError("Benchmark prices must agree across stocks on each date.")
    matrices = []
    for field in ["Open", "Close"]:
        matrix = panel[field].unstack("Ticker").sort_index().astype(float)
        matrix[BENCHMARK] = panel.groupby(level="Date")[f"SPY_{field}"].first()
        matrices.append(matrix)
    return matrices


def portfolio_backtest(panel, weights, horizon=5, cost_bps=15, starting_capital=10000):
    """Simulate specified allocations, with both entry and exit execution costs.

    Weights include entry fees in their budget, so total spending never exceeds
    cash. Daily close valuations retain intra-trade drawdowns. Missing held-asset
    prices are an error, rather than silently removing a losing position.
    """
    horizon = _integer(horizon, "Horizon")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("Per-side costs must be finite and between 0 and 10,000 bps.")
    if not np.isfinite(starting_capital) or starting_capital <= 0:
        raise ValueError("Starting capital must be positive and finite.")
    if (not isinstance(weights.index, pd.DatetimeIndex) or weights.empty
            or weights.index.hasnans or weights.index.has_duplicates
            or not weights.index.is_monotonic_increasing or weights.columns.has_duplicates):
        raise ValueError("Allocations need unique, increasing decision dates and symbols.")
    weights = weights.astype(float)
    if (not np.isfinite(weights.to_numpy()).all() or (weights < 0).any().any()
            or (weights.sum(axis=1) > 1 + 1e-12).any()):
        raise ValueError("Allocations must be finite, long-only, and total at most 100%.")
    opens, closes = _prices(panel)
    if not set(weights.columns).issubset(opens.columns):
        raise ValueError("An allocation references an unknown symbol.")
    positions = opens.index.get_indexer(weights.index)
    if (positions < 0).any() or (positions + horizon >= len(opens)).any():
        raise ValueError("Each decision needs a next open and a complete holding horizon.")
    if (np.diff(positions) < horizon).any():
        raise ValueError("Holding periods cannot overlap.")
    start, end = positions[0], positions[-1] + horizon
    equity = pd.Series(np.nan, index=opens.index[start:end + 1], name="Equity")
    equity.iloc[0] = float(starting_capital)
    nav, previous_exit, records = float(starting_capital), start, []
    friction = float(cost_bps) / 10000
    for decision, position in zip(weights.index, positions):
        entry, exit_position = position + 1, position + horizon
        equity.loc[opens.index[previous_exit + 1:entry]] = nav
        allocation = weights.loc[decision]
        allocation = allocation.loc[allocation > 0]
        symbols = allocation.index
        entry_prices = opens.iloc[entry].reindex(symbols)
        daily_prices = closes.iloc[entry:exit_position + 1].reindex(columns=symbols)
        all_prices = np.r_[entry_prices.to_numpy(), daily_prices.to_numpy().ravel()]
        if not np.isfinite(all_prices).all() or (all_prices <= 0).any():
            raise ValueError("A held asset has missing, non-finite, or non-positive prices.")
        initial_nav = nav
        shares = nav * allocation / ((1 + friction) * entry_prices)
        cash = nav * (1 - allocation.sum())
        values = daily_prices.mul(shares, axis=1).sum(axis=1) + cash
        proceeds = shares * daily_prices.iloc[-1]
        exit_fee = proceeds.sum() * friction
        values.iloc[-1] -= exit_fee
        equity.loc[values.index] = values.to_numpy()
        nav = float(values.iloc[-1])
        for symbol in symbols:
            bought = float(shares[symbol] * entry_prices[symbol])
            sold = float(proceeds[symbol])
            records.append({"decision_date": decision, "entry_date": opens.index[entry],
                            "exit_date": opens.index[exit_position], "symbol": symbol,
                            "weight": float(allocation[symbol]), "entry_nav": initial_nav,
                            "entry_notional": bought, "exit_notional": sold,
                            "fees": (bought + sold) * friction,
                            "net_pnl": sold * (1 - friction) - bought * (1 + friction)})
        previous_exit = exit_position
    if equity.isna().any() or not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("Portfolio accounting produced invalid daily equity.")
    columns = ["decision_date", "entry_date", "exit_date", "symbol", "weight", "entry_nav",
               "entry_notional", "exit_notional", "fees", "net_pnl"]
    trades = pd.DataFrame(records, columns=columns)
    metrics = {**calculate_metrics(equity), "trades": len(trades),
               "cohorts": len(weights), "invested_cohorts": int((weights.sum(axis=1) > 0).sum()),
               "mean_allocated_weight": float(weights.sum(axis=1).mean()),
               "fees_paid": float(trades.fees.sum())}
    metrics["positive_daily_fraction"] = metrics.pop("win_rate")
    return PortfolioResult(equity, trades, metrics)


def strategy_allocations(panel, predictions, development_end, horizon=5,
                         probability_threshold=.55, top_n=2, max_weight=.5):
    """Fixed rules and common calendar for all candidates, without outcome inputs.

    This is a development diagnostic, not a parameter search or a promotion rule.
    Prediction gaps remain in cash, and all baselines use the same daily period.
    """
    horizon, top_n = _integer(horizon, "Horizon"), _integer(top_n, "Top N")
    if not np.isfinite(probability_threshold) or not .5 <= probability_threshold <= 1:
        raise ValueError("Probability threshold must be between 50% and 100%.")
    if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("Maximum stock allocation must be in (0, 1].")
    cutoff = pd.Timestamp(development_end)
    panel = panel.loc[panel.index.get_level_values("Date") <= cutoff].sort_index()
    opens, _ = _prices(panel)
    required = {"Date", "Ticker", "candidate", *PROBABILITIES}
    if predictions.empty or not required.issubset(predictions):
        raise ValueError("Saved candidate probabilities are required.")
    predictions = predictions.copy()
    predictions["Date"] = pd.to_datetime(predictions.Date)
    if predictions.Date.isna().any() or (predictions.Date > cutoff).any():
        raise ValueError("Predictions must be within the saved development period.")
    if predictions[["Ticker", "candidate"]].isna().any().any():
        raise ValueError("Each prediction needs a ticker and candidate.")
    probability = predictions[PROBABILITIES].to_numpy(dtype=float)
    if (not np.isfinite(probability).all() or (probability < 0).any()
            or (probability > 1).any() or not np.allclose(probability.sum(axis=1), 1)):
        raise ValueError("Candidate probabilities must be finite and normalized.")
    if predictions.duplicated(["candidate", "Date", "Ticker"]).any():
        raise ValueError("Duplicate candidate predictions.")
    if set(predictions.Ticker).difference(opens.columns):
        raise ValueError("Prediction tickers are absent from the price snapshot.")
    groups = {name: group.set_index(["Date", "Ticker"]).sort_index()
              for name, group in predictions.groupby("candidate", sort=True)}
    reference = next(iter(groups.values())).index
    if any(not group.index.equals(reference) for group in groups.values()):
        raise ValueError("Candidates must share exactly the same prediction observations.")
    # Prove that execution prices match the saved label horizon, even when the
    # original manifest lacks horizon metadata. No labels select allocations.
    if "Target_Available_At" not in panel:
        raise ValueError("The price snapshot needs saved target availability dates.")
    source = panel.reindex(reference)
    location = opens.index.get_indexer(reference.get_level_values("Date"))
    if (location < 0).any() or (location + horizon >= len(opens)).any():
        raise ValueError("Prediction dates need a complete development holding horizon.")
    expected = opens.index[location + horizon]
    actual = pd.DatetimeIndex(pd.to_datetime(source.Target_Available_At))
    if not actual.equals(expected):
        raise ValueError("Holding horizon does not match the saved prediction target.")
    universe = sorted(set(predictions.Ticker))
    first, last = opens.index.get_indexer([predictions.Date.min(), predictions.Date.max()])
    decisions = opens.index[np.arange(first, last + 1, horizon)]
    allocations = {}
    for name, group in groups.items():
        weights = pd.DataFrame(0., index=decisions, columns=universe)
        for date in decisions:
            if date not in group.index.get_level_values("Date"):
                continue
            signals = group.loc[date].copy()
            signals["score"] = signals.P_UP - signals.P_DOWN
            eligible = signals.loc[(signals.P_UP >= probability_threshold) & (signals.score > 0)]
            # A stable ticker tie-break never consults future price changes.
            chosen = eligible.sort_values("score", ascending=False, kind="stable").head(top_n).index
            if len(chosen):
                weights.loc[date, chosen] = min(max_weight, 1 / len(chosen))
        allocations[f"{name} / fixed rule"] = weights
    allocations["Cash"] = pd.DataFrame(0., index=decisions, columns=universe)
    allocations[f"Equal-weight universe / {horizon} sessions"] = pd.DataFrame(
        1 / len(universe), index=decisions, columns=universe)
    allocations[f"SPY / {horizon} sessions"] = pd.DataFrame(1., index=decisions, columns=[BENCHMARK])
    if "Return_20" not in panel:
        raise ValueError("The momentum baseline needs a trailing Return_20 feature.")
    momentum = pd.DataFrame(0., index=decisions, columns=universe)
    for date in decisions:
        scores = panel.loc[date, "Return_20"].reindex(universe)
        if not np.isfinite(scores).all():
            raise ValueError("The momentum baseline has unavailable trailing features.")
        chosen = scores.sort_values(ascending=False, kind="stable").head(top_n).index
        momentum.loc[date, chosen] = min(max_weight, 1 / len(chosen))
    allocations[f"20-session momentum / {horizon} sessions"] = momentum
    return panel, allocations


def compare_portfolios(panel, predictions, development_end, horizon=5,
                       probability_threshold=.55, top_n=2, max_weight=.5,
                       cost_scenarios=(15, 30, 50)):
    panel, allocations = strategy_allocations(panel, predictions, development_end, horizon,
                                              probability_threshold, top_n, max_weight)
    scenarios = tuple(cost_scenarios)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Provide distinct cost scenarios.")
    dates = panel.index.get_level_values("Date").unique().sort_values()
    decisions = next(iter(allocations.values())).index
    start, last_decision = dates.get_indexer([decisions[0], decisions[-1]])
    full_horizon = last_decision + horizon - start
    holdings = pd.DataFrame(1., index=decisions[:1], columns=[BENCHMARK])
    rows, results = [], {}
    for cost in scenarios:
        for name, weights in allocations.items():
            results[(name, cost)] = portfolio_backtest(panel, weights, horizon, cost)
        results[("SPY buy & hold", cost)] = portfolio_backtest(panel, holdings, full_horizon, cost)
        benchmark = results[("SPY buy & hold", cost)]
        for (name, scenario), result in list(results.items()):
            if scenario != cost:
                continue
            if name == "SPY buy & hold":
                matched = benchmark
            else:
                # Match cash allocation and entry/exit dates, not price volatility.
                matched_weights = allocations[name].sum(axis=1).to_frame(BENCHMARK)
                matched = portfolio_backtest(panel, matched_weights, horizon, cost)
            rows.append({"strategy": name, "cost_bps_per_side": cost, **result.metrics,
                         "return_vs_spy": result.metrics["total_return"] - benchmark.metrics["total_return"],
                         "allocation_matched_spy_return": matched.metrics["total_return"],
                         "return_vs_allocation_matched_spy": result.metrics["total_return"] - matched.metrics["total_return"],
                         "start": str(result.equity.index[0].date()),
                         "end": str(result.equity.index[-1].date())})
    return pd.DataFrame(rows), results
