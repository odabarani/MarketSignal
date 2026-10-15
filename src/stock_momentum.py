"""Pre-registered long-only stock momentum research on daily adjusted prices."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.stock_panel import validate_liquidity_panel

LOOKBACK_SESSIONS = 252
SKIP_SESSIONS = 21
MIN_HISTORY = 200
LIQUIDITY_SESSIONS = 20
MIN_ADJUSTED_DOLLAR_VOLUME = 5_000_000.0
TOP_FRACTION = 0.10
TARGET_VOLATILITY = 0.10
RESEARCH_DRAWDOWN_BUDGET = 0.20
COSTS_BPS = (15, 30, 50)
STRATEGIES = ("Cash", "SPY buy-and-hold", "Equal-weight stocks",
              "12-1 momentum", "12-1 momentum with SPY trend",
              "12-1 momentum with 10% volatility target")
SECURITY_TYPES = {"Common Stock", "Depositary Receipt"}


def volatility_exposure(
    basket_returns: pd.Series, target_volatility: float = TARGET_VOLATILITY
) -> float:
    """Scale a long-only basket to a fixed volatility target, capped at 100%."""
    if not np.isfinite(target_volatility) or not 0 < target_volatility <= 1:
        raise ValueError("Target volatility must be finite and in (0, 1].")
    values = basket_returns.tail(20).to_numpy(dtype=float)
    if len(values) < 20 or not np.isfinite(values).all():
        return 0.0
    volatility = float(np.std(values, ddof=1) * np.sqrt(252))
    if not np.isfinite(volatility) or volatility <= 0:
        return 0.0
    return min(1.0, target_volatility / volatility)


def make_price_matrices(
    panel: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, str, list[str]]:
    """Build price and volume matrices, resolving SPY by its current master symbol."""
    if panel.duplicated(["sid", "date"]).any():
        raise ValueError("Price panel must contain one row per Sid/date.")
    spy = metadata.loc[metadata["symbol"].eq("SPY"), "sid"].astype(str).tolist()
    if len(spy) != 1:
        raise ValueError(f"Expected one SPY Sid in security metadata, found {len(spy)}.")
    spy_sid = spy[0]
    type_by_sid = metadata.set_index(metadata["sid"].astype(str))["security_type"]
    stock_sids = [
        sid for sid in panel["sid"].astype(str).unique()
        if sid != spy_sid and type_by_sid.get(sid) in SECURITY_TYPES
    ]
    if not stock_sids:
        raise ValueError("No common stocks or depositary receipts matched the sample.")
    close = panel.pivot(index="date", columns="sid", values="close").sort_index()
    close.columns = close.columns.astype(str)
    if spy_sid not in close.columns:
        raise ValueError("SPY appears in the securities master but not the price panel.")
    return close, spy_sid, sorted(stock_sids)


def build_targets(
    close: pd.DataFrame,
    stock_sids: list[str],
    spy_sid: str,
    liquidity_panel: pd.DataFrame,
) -> tuple[dict[str, dict[pd.Timestamp, dict[str, float]]], pd.Series, pd.Timestamp]:
    """Create monthly targets using only information available at each close."""
    dates = close.index
    filled = close.ffill()
    momentum = filled.shift(SKIP_SESSIONS) / filled.shift(LOOKBACK_SESSIONS) - 1.0
    observations = close[stock_sids].notna().rolling(
        LOOKBACK_SESSIONS, min_periods=1
    ).sum()
    liquidity_panel = validate_liquidity_panel(liquidity_panel)
    adv = liquidity_panel.pivot(
        index="date", columns="sid", values="average_adjusted_dollar_volume"
    ).reindex(index=dates, columns=stock_sids)
    spy_sma = filled[spy_sid].rolling(200, min_periods=200).mean()
    spy_above_trend = filled[spy_sid] > spy_sma
    stock_daily_returns = filled[stock_sids].pct_change(fill_method=None).fillna(0.0)

    monthly_last = pd.Series(dates, index=dates).groupby(dates.to_period("M")).last()
    raw_targets: dict[str, dict[pd.Timestamp, dict[str, float]]] = {
        name: {} for name in STRATEGIES
    }
    first_valid_effective = None

    for decision in monthly_last.to_list():
        decision_index = dates.get_loc(decision)
        if decision_index + 1 >= len(dates):
            continue
        execution_date = dates[decision_index + 1]
        eligible = close.loc[decision, stock_sids].notna()
        eligible &= observations.loc[decision].ge(MIN_HISTORY)
        if decision not in adv.index:
            if decision_index >= LOOKBACK_SESSIONS:
                raise ValueError(f"Monthly liquidity data is missing decision date {decision.date()}.")
            continue
        eligible &= adv.loc[decision].ge(MIN_ADJUSTED_DOLLAR_VOLUME)
        scores = momentum.loc[decision, stock_sids]
        ranked = scores.loc[eligible].dropna().sort_values(ascending=False, kind="stable")
        if ranked.empty:
            continue
        if first_valid_effective is None:
            first_valid_effective = execution_date

        equal_names = list(ranked.index)
        raw_targets["Equal-weight stocks"][execution_date] = {
            sid: 1.0 / len(equal_names) for sid in equal_names
        }

        winners = ranked[ranked > 0]
        if not winners.empty:
            count = max(1, math.ceil(len(ranked) * TOP_FRACTION))
            selected = list(winners.head(count).index)
            momentum_target = {sid: 1.0 / len(selected) for sid in selected}
        else:
            momentum_target = {}
        raw_targets["12-1 momentum"][execution_date] = momentum_target
        trend_target = momentum_target if bool(spy_above_trend.loc[decision]) else {}
        raw_targets["12-1 momentum with SPY trend"][execution_date] = trend_target
        if momentum_target:
            basket_returns = stock_daily_returns.loc[:decision, list(momentum_target)].mean(axis=1)
            exposure = volatility_exposure(basket_returns)
            scaled_target = {sid: weight * exposure for sid, weight in momentum_target.items()}
        else:
            scaled_target = {}
        raw_targets["12-1 momentum with 10% volatility target"][execution_date] = scaled_target

    if first_valid_effective is None:
        raise ValueError(
            "No eligible monthly stock basket. Check date coverage, liquidity, "
            "and history requirements."
        )
    targets = {
        name: {date: target for date, target in schedule.items() if date >= first_valid_effective}
        for name, schedule in raw_targets.items()
    }
    targets["Cash"] = {}
    targets["SPY buy-and-hold"] = {first_valid_effective: {spy_sid: 1.0}}
    trend_known = spy_above_trend.astype("boolean").shift(1).fillna(False).to_numpy(dtype=bool)
    regime = pd.Series(
        np.where(trend_known, "Above 200-day trend", "At or below 200-day trend"),
        index=dates,
    )
    return targets, regime, first_valid_effective


def daily_returns(
    close: pd.DataFrame,
    metadata: pd.DataFrame,
    delisting_scenario: str,
) -> pd.DataFrame:
    """Build close returns with explicit zero-recovery or last-quote delisting treatment."""
    if delisting_scenario not in {"zero_recovery", "last_quote"}:
        raise ValueError("Delisting scenario must be 'zero_recovery' or 'last_quote'.")
    marks = close.ffill()
    returns = marks.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if delisting_scenario == "zero_recovery":
        end = close.index.max()
        for row in metadata.itertuples(index=False):
            if pd.isna(row.date_delisted) or pd.Timestamp(row.date_delisted) > end:
                continue
            sid = str(row.sid)
            if sid not in returns.columns or pd.isna(row.last_price_date):
                raise ValueError(f"Delisting metadata is incomplete for Sid {sid}.")
            last_price_date = pd.Timestamp(row.last_price_date)
            future_sessions = close.index[close.index > last_price_date]
            if len(future_sessions):
                returns.loc[future_sessions[0], sid] = -1.0
    return returns


def simulate(
    returns: pd.DataFrame,
    targets: dict[pd.Timestamp, dict[str, float]],
    cost_bps: int,
    start: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, float]:
    """Apply close trades after today's return; charge costs on risky traded notional."""
    if cost_bps < 0:
        raise ValueError("Cost must be nonnegative.")
    fee_rate = cost_bps / 10_000.0
    risky_weights: dict[str, float] = {}
    equity = 1.0
    equity_values, net_returns = [], []
    gross_traded = 0.0

    for date, row in returns.iterrows():
        prior_equity = equity
        asset_returns = row
        portfolio_return = sum(
            weight * float(asset_returns.get(sid, 0.0))
            for sid, weight in risky_weights.items()
        )
        if portfolio_return <= -1.0:
            equity = 0.0
            risky_weights = {}
        else:
            equity *= 1.0 + portfolio_return
            cash_weight = max(0.0, 1.0 - sum(risky_weights.values()))
            denominator = 1.0 + portfolio_return
            risky_weights = {
                sid: weight * (1.0 + float(asset_returns.get(sid, 0.0))) / denominator
                for sid, weight in risky_weights.items()
                if weight > 0 and (1.0 + float(asset_returns.get(sid, 0.0))) > 0
            }
            cash_weight = cash_weight / denominator
            total_weight = sum(risky_weights.values()) + cash_weight
            if total_weight > 0 and not np.isclose(total_weight, 1.0):
                risky_weights = {sid: weight / total_weight for sid, weight in risky_weights.items()}

        if date in targets and equity > 0:
            desired = {str(sid): float(weight) for sid, weight in targets[date].items() if weight > 0}
            if any(not np.isfinite(weight) for weight in desired.values()) or sum(desired.values()) > 1 + 1e-10:
                raise ValueError("Target weights must be finite, nonnegative, and at most fully invested.")
            if set(desired).difference(returns.columns):
                raise ValueError("Target contains a security missing from the return panel.")
            traded = set(risky_weights).union(desired)
            traded_notional = sum(abs(desired.get(sid, 0.0) - risky_weights.get(sid, 0.0)) for sid in traded)
            gross_traded += traded_notional
            equity *= max(0.0, 1.0 - fee_rate * traded_notional)
            risky_weights = desired

        equity_values.append(equity)
        net_returns.append(equity / prior_equity - 1.0 if prior_equity > 0 else 0.0)

    equity_series = pd.Series(equity_values, index=returns.index, name="equity")
    return_series = pd.Series(net_returns, index=returns.index, name="net_return")
    years = max(len(returns.loc[start:]) / 252.0, 1 / 252.0)
    return equity_series, return_series, gross_traded / years


def summarize(
    equity: pd.Series,
    returns: pd.Series,
    regime: pd.Series,
    start: pd.Timestamp,
) -> dict[str, Any]:
    """Return comparable net performance, drawdown, turnover, yearly and regime slices."""
    returns = returns.loc[start:]
    equity = equity.loc[start:]
    if returns.empty:
        raise ValueError("Evaluation period is empty.")
    previous_equity = equity.shift(1).loc[returns.index[0]]
    starting_value = float(previous_equity) if pd.notna(previous_equity) else 1.0
    ending_value = float(equity.iloc[-1])
    years = max(len(returns) / 252.0, 1 / 252.0)
    peak = equity.cummax().clip(lower=starting_value)
    drawdown = equity / peak - 1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if returns.std(ddof=1) > 0 else 0.0
    yearly = {
        str(int(year)): float((1.0 + values).prod() - 1.0)
        for year, values in returns.groupby(returns.index.year)
    }
    labels = regime.reindex(returns.index)
    regimes = {
        str(label): {
            "sessions": int(len(values)),
            "net_return": float((1.0 + values).prod() - 1.0),
        }
        for label, indexes in labels.groupby(labels).groups.items()
        if len(values := returns.reindex(indexes))
    }
    return {
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "sessions": int(len(returns)),
        "net_return": ending_value / starting_value - 1.0,
        "annualized_return": (ending_value / starting_value) ** (1.0 / years) - 1.0 if ending_value > 0 else -1.0,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": volatility,
        "sharpe_zero_rf": sharpe,
        "yearly_net_returns": yearly,
        "market_regime_net_returns": regimes,
    }


def run_experiment(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    liquidity_panel: pd.DataFrame,
    cost_bps: tuple[int, ...] = COSTS_BPS,
) -> dict[str, Any]:
    """Compare simple stock momentum with cash, SPY, and equal-weight stocks."""
    close, spy_sid, stock_sids = make_price_matrices(panel, metadata)
    targets, regime, start = build_targets(close, stock_sids, spy_sid, liquidity_panel)
    results = []
    for scenario in ("zero_recovery", "last_quote"):
        returns = daily_returns(close, metadata, scenario)
        for strategy in STRATEGIES:
            for cost in cost_bps:
                equity, net_returns, annual_turnover = simulate(
                    returns, targets[strategy], cost, start
                )
                results.append({
                    "strategy": strategy,
                    "cost_bps_per_side": cost,
                    "delisting_scenario": scenario,
                    "annual_gross_traded_notional": annual_turnover,
                    **summarize(equity, net_returns, regime, start),
                })
    spy_results = {
        (row["cost_bps_per_side"], row["delisting_scenario"]): row
        for row in results if row["strategy"] == "SPY buy-and-hold"
    }
    for row in results:
        spy = spy_results[(row["cost_bps_per_side"], row["delisting_scenario"])]
        row["excess_net_return_vs_spy"] = row["net_return"] - spy["net_return"]
        row["excess_max_drawdown_vs_spy"] = row["max_drawdown"] - spy["max_drawdown"]
        row["within_research_drawdown_budget"] = (
            row["max_drawdown"] >= -RESEARCH_DRAWDOWN_BUDGET
        )
    return {
        "phase": "historical_development_only",
        "status": "Comparison only; no strategy selected or profitability claim.",
        "settings": {
            "momentum_lookback_sessions": LOOKBACK_SESSIONS,
            "momentum_skip_sessions": SKIP_SESSIONS,
            "min_history_observations": MIN_HISTORY,
            "liquidity_lookback_sessions": LIQUIDITY_SESSIONS,
            "liquidity_values_as_of_each_month_end": True,
            "liquidity_measure": "20-session average of adjusted Close times Volume, queried as of each month-end",
            "liquidity_is_nominal_dollar_volume": False,
            "minimum_adjusted_dollar_volume_proxy": MIN_ADJUSTED_DOLLAR_VOLUME,
            "momentum_top_fraction": TOP_FRACTION,
            "volatility_target": TARGET_VOLATILITY,
            "volatility_lookback_sessions": 20,
            "research_drawdown_budget": RESEARCH_DRAWDOWN_BUDGET,
            "rebalancing": "Monthly signal at close, trade at next session close.",
            "universe": "Common Stock and Depositary Receipt per latest master snapshot; no active-only filter.",
            "execution_costs_bps_per_side": list(cost_bps),
        },
        "universe": {
            "spy_sid": spy_sid,
            "stock_security_ids": len(stock_sids),
            "start": str(start.date()),
        },
        "limitations": [
            "Every historical year is development data; none is an untouched holdout.",
            "The current securities-master snapshot classifies security types and delisting events; it is not point-in-time classification data.",
            "Zero recovery after the last price is a conservative stress, not an observed payout; last-quote carry is an optimistic assumption, not an upper bound on merger consideration.",
            "Fixed costs omit market impact, taxes, cash yield, and capacity constraints beyond an adjusted-dollar-volume proxy screen.",
            "Adjusted close-to-close prices approximate total returns and assume trades can execute at the close before costs.",
        ],
        "results": results,
    }
