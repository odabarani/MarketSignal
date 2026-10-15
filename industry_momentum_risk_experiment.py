"""Development-only volatility sizing from the selected momentum basket itself."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from industry_portfolio_research import (
    BASE_COST_BPS,
    COSTS_BPS,
    DEVELOPMENT_END,
    EVALUATION_START,
    EXECUTION_LAG,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP,
    VOLATILITY_LOOKBACK,
    backtest_weights,
    build_strategy_weights,
    code_fingerprint as base_code_fingerprint,
    load_industry_panel,
    summarize,
)

TARGET_VOLATILITY = 0.05
LEGACY_STRATEGY = "12-1 momentum, top 10, 5.0% volatility target"
RISK_STRATEGY = "12-1 momentum, top 10, selected-basket volatility target"
BENCHMARKS = ["Cash / T-bills", "Market buy & hold", "Market 200-day trend"]


def selected_basket_volatility(
    returns: pd.DataFrame,
    selected: list[str],
    decision: pd.Timestamp,
    lookback: int = VOLATILITY_LOOKBACK,
) -> float:
    """Estimate annualized risk from the fixed selected basket through decision close."""
    if not selected:
        return float("nan")
    if len(set(selected)) != len(selected) or set(selected).difference(returns.columns):
        raise ValueError("Selected basket must contain unique assets in the return panel.")
    if decision not in returns.index:
        raise ValueError("Decision date is absent from the return panel.")
    basket_returns = returns.loc[:decision, selected].mean(axis=1).tail(lookback)
    if len(basket_returns) < lookback or not np.isfinite(basket_returns.to_numpy()).all():
        return float("nan")
    return float(basket_returns.std(ddof=1) * np.sqrt(252))


def schedule_targets(
    dates: pd.DatetimeIndex,
    targets: dict[pd.Timestamp, pd.Series],
    columns: list[str],
    execution_lag: int = EXECUTION_LAG,
) -> pd.DataFrame:
    """Apply close-time targets after the fixed execution lag and carry them forward."""
    if (not isinstance(dates, pd.DatetimeIndex) or dates.empty or dates.has_duplicates
            or not dates.is_monotonic_increasing):
        raise ValueError("Scheduling requires unique, increasing decision sessions.")
    if isinstance(execution_lag, bool) or not isinstance(execution_lag, int) or execution_lag < 1:
        raise ValueError("Execution lag must be a positive integer.")
    weights = pd.DataFrame(np.nan, index=dates, columns=columns)
    positions = {date: i for i, date in enumerate(dates)}
    for decision, target in targets.items():
        if decision not in positions or list(target.index) != columns:
            raise ValueError("Targets must use known decision dates and the complete asset list.")
        effective_position = positions[decision] + execution_lag
        if effective_position < len(dates):
            weights.loc[dates[effective_position]] = target
    weights = weights.ffill().fillna({**{name: 0.0 for name in columns if name != "Cash"}, "Cash": 1.0})
    if not np.isfinite(weights.to_numpy()).all() or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Scheduled weights must be finite and include cash.")
    return weights


def build_selected_momentum_weights(
    industries: pd.DataFrame,
    target_volatility: float = TARGET_VOLATILITY,
) -> pd.DataFrame:
    """Build monthly 12-1 momentum weights sized from that selected basket's risk."""
    if (not isinstance(industries.index, pd.DatetimeIndex) or industries.empty
            or industries.index.has_duplicates or not industries.index.is_monotonic_increasing
            or industries.columns.has_duplicates):
        raise ValueError("Industry returns need unique increasing dates and asset columns.")
    if not np.isfinite(target_volatility) or not 0 < target_volatility <= 1:
        raise ValueError("Target volatility must be finite and in (0, 1].")
    if not np.isfinite(industries.to_numpy(dtype=float)).all():
        raise ValueError("Industry returns must be finite.")

    dates = industries.index
    assets = list(industries.columns)
    columns = [*assets, "Market", "Cash"]
    wealth = (1.0 + industries).cumprod()
    momentum = wealth.shift(MOMENTUM_SKIP) / wealth.shift(MOMENTUM_LOOKBACK) - 1.0
    decisions = pd.DatetimeIndex(pd.Series(dates, index=dates).groupby(dates.to_period("M")).last())
    targets: dict[pd.Timestamp, pd.Series] = {}
    for decision in decisions:
        scores = momentum.loc[decision].dropna().sort_values(ascending=False, kind="stable")
        selected = list(scores.head(10).index) if len(scores) >= 10 and scores.iloc[0] > 0 else []
        target = pd.Series(0.0, index=columns)
        if selected:
            volatility = selected_basket_volatility(industries, selected, decision)
            exposure = min(1.0, target_volatility / volatility) if np.isfinite(volatility) and volatility > 0 else 0.0
            target.loc[selected] = exposure / len(selected)
        target["Cash"] = 1.0 - target.drop("Cash").sum()
        targets[decision] = target

    return schedule_targets(dates, targets, columns)


def run_development(industry_zip: Path, factor_zip: Path) -> dict:
    industries, cash, metadata = load_industry_panel(industry_zip, factor_zip)
    market = metadata.pop("market")
    legacy_weights = build_strategy_weights(industries, market, cash)
    weights = {name: legacy_weights[name] for name in BENCHMARKS}
    weights[LEGACY_STRATEGY] = legacy_weights[LEGACY_STRATEGY]
    weights[RISK_STRATEGY] = build_selected_momentum_weights(industries)

    summary, yearly, regimes = [], [], []
    market_wealth = (1.0 + market).cumprod()
    market_ma = market_wealth.rolling(200, min_periods=200).mean()
    regime_signal = pd.Series(
        (market_wealth > market_ma).shift(1).astype("boolean").fillna(False).to_numpy(dtype=bool),
        index=market.index,
    ).map({True: "Above 200-day trend", False: "At or below 200-day trend"})
    for name, allocation in weights.items():
        for cost in COSTS_BPS:
            equity, returns, turnover = backtest_weights(
                allocation, industries, market, cash, cost, EVALUATION_START, DEVELOPMENT_END
            )
            summary.append({"strategy": name, **summarize(equity, returns, cash, turnover, cost)})
            if cost == BASE_COST_BPS:
                annual = returns.groupby(returns.index.year).apply(lambda values: float((1 + values).prod() - 1))
                yearly.extend({"strategy": name, "year": int(year), "net_return": value}
                              for year, value in annual.items())
                active_regimes = regime_signal.reindex(returns.index)
                for label, indexes in active_regimes.groupby(active_regimes).groups.items():
                    values = returns.reindex(indexes).dropna()
                    if not values.empty:
                        regimes.append({"strategy": name, "market_regime": label,
                                         "sessions": len(values),
                                         "net_return": float((1 + values).prod() - 1)})
    fingerprint = sha256(Path(__file__).read_bytes() + base_code_fingerprint().encode("ascii")).hexdigest()
    source_hash = sha256(
        (metadata["industry_zip_sha256"] + metadata["factor_zip_sha256"]).encode("ascii")
    ).hexdigest()
    return {
        "phase": "development_only",
        "status": "Risk-estimator comparison; no strategy promotion or live-use claim.",
        "source_hash": source_hash,
        "sources": metadata,
        "code_fingerprint": fingerprint,
        "development_start": str(EVALUATION_START.date()),
        "development_end": str(DEVELOPMENT_END.date()),
        "candidate_definition": (
            "Keep the existing 12-1 top-10 selection and 5% target, but size exposure from the "
            "trailing 20-session volatility of that selected equal-weight basket rather than "
            "the 49-industry equal-weight basket."
        ),
        "settings": {"costs_bps_per_side": list(COSTS_BPS),
                     "base_cost_bps_per_side": BASE_COST_BPS,
                     "target_volatility": TARGET_VOLATILITY,
                     "volatility_lookback_sessions": VOLATILITY_LOOKBACK,
                     "momentum_lookback_sessions": MOMENTUM_LOOKBACK,
                     "momentum_skip_sessions": MOMENTUM_SKIP,
                     "execution_lag_sessions": EXECUTION_LAG},
        "summary": summary,
        "yearly_returns_at_base_cost": yearly,
        "market_regime_returns_at_base_cost": regimes,
        "limitations": [
            "Development data only; all dates were already used in MarketSignal research.",
            "Industry portfolios are reconstructed aggregates, not individual stocks.",
            "Historical volatility is an imperfect estimate; a 5% target does not cap losses.",
            "Costs are fixed portfolio-level assumptions and omit market impact, taxes, and cash yield differences.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--industries", type=Path,
                        default=Path("data/research_inputs/fama_french_49_industry_daily_202608.zip"))
    parser.add_argument("--factors", type=Path,
                        default=Path("data/research_inputs/fama_french_3_factors_daily_202608.zip"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/industry_momentum_risk_development.json"))
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose another --report path to preserve this run.")
    report = run_development(args.industries, args.factors)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    scores = pd.DataFrame(report["summary"])
    print(scores.loc[scores.cost_bps_per_side == BASE_COST_BPS,
                     ["strategy", "annualized_return", "max_drawdown", "annualized_volatility",
                      "sharpe_vs_cash"]].to_string(index=False))
    print(f"Saved development-only report: {args.report}")


if __name__ == "__main__":
    main()
