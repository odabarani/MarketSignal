"""Development-only test of volatility-adjusted industry momentum ranking."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from industry_momentum_risk_experiment import (
    build_selected_momentum_weights,
    schedule_targets,
    selected_basket_volatility,
)
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

RANKING_VOLATILITY_LOOKBACK = MOMENTUM_LOOKBACK
TARGET_VOLATILITY = 0.05
TOP_N = 10
RAW_STRATEGY = "12-1 momentum, top 10, 5.0% volatility target"
SELECTED_RISK_STRATEGY = "12-1 momentum, top 10, selected-basket volatility target"
RANKED_LEGACY_RISK = "Volatility-adjusted 12-1 momentum, top 10, broad-risk target"
RANKED_SELECTED_RISK = "Volatility-adjusted 12-1 momentum, top 10, selected-basket risk"
BENCHMARKS = ["Cash / T-bills", "Market buy & hold", "Market 200-day trend"]


def rank_volatility_adjusted_momentum(
    returns: pd.DataFrame,
    decision: pd.Timestamp,
    top_n: int = TOP_N,
) -> list[str]:
    """Rank positive 12-1 momentum by its ratio to trailing annualized volatility."""
    if decision not in returns.index:
        raise ValueError("Decision date is absent from the return panel.")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("Top-N must be a positive integer.")
    wealth = (1.0 + returns).cumprod()
    momentum = wealth.shift(MOMENTUM_SKIP) / wealth.shift(MOMENTUM_LOOKBACK) - 1.0
    volatility = returns.rolling(RANKING_VOLATILITY_LOOKBACK,
                                 min_periods=RANKING_VOLATILITY_LOOKBACK).std(ddof=1) * np.sqrt(252)
    scores = (momentum.loc[decision] / volatility.loc[decision]).replace([np.inf, -np.inf], np.nan)
    scores = scores.loc[momentum.loc[decision] > 0].dropna()
    return list(scores.sort_values(ascending=False, kind="stable").head(top_n).index)


def build_ranked_momentum_weights(
    industries: pd.DataFrame,
    target_volatility: float = TARGET_VOLATILITY,
    selected_basket_risk: bool = False,
) -> pd.DataFrame:
    """Build monthly volatility-adjusted momentum weights with fixed execution lag."""
    if (not isinstance(industries.index, pd.DatetimeIndex) or industries.empty
            or industries.index.has_duplicates or not industries.index.is_monotonic_increasing
            or industries.columns.has_duplicates or len(industries.columns) < TOP_N):
        raise ValueError("Industry returns need unique increasing dates and at least ten assets.")
    if not np.isfinite(target_volatility) or not 0 < target_volatility <= 1:
        raise ValueError("Target volatility must be finite and in (0, 1].")
    if not np.isfinite(industries.to_numpy(dtype=float)).all():
        raise ValueError("Industry returns must be finite.")

    dates = industries.index
    assets = list(industries.columns)
    columns = [*assets, "Market", "Cash"]
    broad_volatility = industries.mean(axis=1).rolling(
        VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK
    ).std(ddof=1) * np.sqrt(252)
    decisions = pd.DatetimeIndex(pd.Series(dates, index=dates).groupby(dates.to_period("M")).last())
    targets: dict[pd.Timestamp, pd.Series] = {}
    for decision in decisions:
        selected = rank_volatility_adjusted_momentum(industries, decision)
        target = pd.Series(0.0, index=columns)
        if selected:
            if selected_basket_risk:
                volatility = selected_basket_volatility(industries, selected, decision)
            else:
                volatility = float(broad_volatility.loc[decision])
            exposure = (min(1.0, target_volatility / volatility)
                        if np.isfinite(volatility) and volatility > 0 else 0.0)
            target.loc[selected] = exposure / len(selected)
        target["Cash"] = 1.0 - target.drop("Cash").sum()
        targets[decision] = target
    return schedule_targets(dates, targets, columns, EXECUTION_LAG)


def run_development(industry_zip: Path, factor_zip: Path) -> dict:
    industries, cash, metadata = load_industry_panel(industry_zip, factor_zip)
    market = metadata.pop("market")
    existing = build_strategy_weights(industries, market, cash)
    weights = {name: existing[name] for name in BENCHMARKS}
    weights[RAW_STRATEGY] = existing[RAW_STRATEGY]
    weights[SELECTED_RISK_STRATEGY] = build_selected_momentum_weights(industries)
    weights[RANKED_LEGACY_RISK] = build_ranked_momentum_weights(industries)
    weights[RANKED_SELECTED_RISK] = build_ranked_momentum_weights(
        industries, selected_basket_risk=True
    )

    summary, yearly, regimes = [], [], []
    market_wealth = (1.0 + market).cumprod()
    market_ma = market_wealth.rolling(200, min_periods=200).mean()
    regime = pd.Series(np.where(market_wealth > market_ma, "Above 200-day trend",
                                "At or below 200-day trend"), index=market.index)
    regime.loc[market_ma.isna()] = "Insufficient trend history"
    for name, allocation in weights.items():
        for cost in COSTS_BPS:
            equity, returns, turnover = backtest_weights(
                allocation, industries, market, cash, cost, EVALUATION_START, DEVELOPMENT_END
            )
            summary.append({"strategy": name, **summarize(equity, returns, cash, turnover, cost)})
            if cost == BASE_COST_BPS:
                annual = returns.groupby(returns.index.year).apply(
                    lambda values: float((1.0 + values).prod() - 1.0)
                )
                yearly.extend({"strategy": name, "year": int(year), "net_return": value}
                              for year, value in annual.items())
                active_regime = regime.reindex(returns.index)
                for label, indexes in active_regime.groupby(active_regime).groups.items():
                    values = returns.reindex(indexes).dropna()
                    if not values.empty:
                        yearly_return = float((1.0 + values).prod() - 1.0)
                        regimes.append({"strategy": name, "market_regime": label,
                                        "sessions": len(values), "net_return": yearly_return})

    fingerprint = sha256(Path(__file__).read_bytes()
                         + base_code_fingerprint().encode("ascii")).hexdigest()
    source_hash = sha256((metadata["industry_zip_sha256"]
                          + metadata["factor_zip_sha256"]).encode("ascii")).hexdigest()
    return {
        "phase": "development_only",
        "status": "Fixed volatility-adjusted ranking comparison; no strategy promotion.",
        "source_hash": source_hash,
        "sources": metadata,
        "code_fingerprint": fingerprint,
        "development_start": str(EVALUATION_START.date()),
        "development_end": str(DEVELOPMENT_END.date()),
        "candidate_definition": (
            "Rank industries by positive 12-1 compounded return divided by trailing "
            "252-session annualized volatility; hold up to ten; compare broad-universe "
            "versus selected-basket volatility sizing at a 5% target."
        ),
        "settings": {"costs_bps_per_side": list(COSTS_BPS),
                     "base_cost_bps_per_side": BASE_COST_BPS,
                     "target_volatility": TARGET_VOLATILITY,
                     "ranking_volatility_lookback_sessions": RANKING_VOLATILITY_LOOKBACK,
                     "momentum_lookback_sessions": MOMENTUM_LOOKBACK,
                     "momentum_skip_sessions": MOMENTUM_SKIP,
                     "top_industries": TOP_N,
                     "risk_sizing_lookback_sessions": VOLATILITY_LOOKBACK,
                     "execution_lag_sessions": EXECUTION_LAG},
        "summary": summary,
        "yearly_returns_at_base_cost": yearly,
        "market_regime_returns_at_base_cost": regimes,
        "limitations": [
            "All evaluation dates were previously used for development; this is not a holdout.",
            "Fama/French industry portfolios are reconstructed aggregates, not individual stocks.",
            "Volatility ratios are unstable when estimated risk is low and do not guarantee lower losses.",
            "Fixed costs omit internal portfolio reconstitution, market impact, taxes, and account-specific cash yields.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--industries", type=Path,
                        default=Path("data/research_inputs/fama_french_49_industry_daily_202608.zip"))
    parser.add_argument("--factors", type=Path,
                        default=Path("data/research_inputs/fama_french_3_factors_daily_202608.zip"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/industry_momentum_rank_development.json"))
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose another --report path to preserve this run.")
    report = run_development(args.industries, args.factors)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    scores = pd.DataFrame(report["summary"])
    print(scores.loc[scores.cost_bps_per_side == BASE_COST_BPS,
                     ["strategy", "annualized_return", "max_drawdown",
                      "annualized_volatility", "sharpe_vs_cash"]].to_string(index=False))
    print(f"Saved development-only report: {args.report}")


if __name__ == "__main__":
    main()
