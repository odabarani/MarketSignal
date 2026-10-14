"""Development-only test of a 200-day market filter on industry momentum."""

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
    build_strategy_weights,
    backtest_weights,
    code_fingerprint as base_code_fingerprint,
    load_industry_panel,
    summarize,
)

BASE_STRATEGY = "12-1 momentum, top 10, 5.0% volatility target"
FILTERED_STRATEGY = "12-1 momentum + market 200-day filter, 5.0% volatility target"
BENCHMARKS = ["Cash / T-bills", "Market buy & hold", "Market 200-day trend", BASE_STRATEGY]


def apply_market_trend_filter(base: pd.DataFrame, market_trend: pd.DataFrame) -> pd.DataFrame:
    """Keep the fixed momentum weights only while the lagged trend rule is invested."""
    if not base.index.equals(market_trend.index) or list(base.columns) != list(market_trend.columns):
        raise ValueError("Momentum and trend weights must have matching dates and assets.")
    if "Market" not in base or "Cash" not in base:
        raise ValueError("Weights need Market and Cash columns.")
    filtered = base.copy()
    risk_on = market_trend["Market"] > 0.5
    filtered.loc[~risk_on] = 0.0
    filtered.loc[~risk_on, "Cash"] = 1.0
    if not np.allclose(filtered.sum(axis=1).to_numpy(), 1.0, atol=1e-10):
        raise ValueError("Filtered allocations must remain fully invested, including cash.")
    return filtered


def run_development(industry_zip: Path, factor_zip: Path) -> dict:
    industries, cash, metadata = load_industry_panel(industry_zip, factor_zip)
    market = metadata.pop("market")
    base_weights = build_strategy_weights(industries, market, cash)
    weights = {
        "Cash / T-bills": base_weights["Cash / T-bills"],
        "Market buy & hold": base_weights["Market buy & hold"],
        "Market 200-day trend": base_weights["Market 200-day trend"],
        BASE_STRATEGY: base_weights[BASE_STRATEGY],
        FILTERED_STRATEGY: apply_market_trend_filter(
            base_weights[BASE_STRATEGY], base_weights["Market 200-day trend"]
        ),
    }
    summaries, yearly, regimes = [], [], []
    benchmark_wealth = (1.0 + market).cumprod()
    market_ma = benchmark_wealth.rolling(200, min_periods=200).mean()
    # Classify each close-to-close return using the prior close's known regime.
    regime_signal = pd.Series(
        (benchmark_wealth > market_ma).shift(1).astype("boolean").fillna(False).to_numpy(dtype=bool),
        index=market.index,
    )
    regime = regime_signal.map(
        {True: "Above 200-day trend", False: "At or below 200-day trend"}
    )
    for name, allocation in weights.items():
        for cost in COSTS_BPS:
            equity, returns, turnover = backtest_weights(
                allocation, industries, market, cash, cost, EVALUATION_START, DEVELOPMENT_END
            )
            summaries.append({"strategy": name, **summarize(equity, returns, cash, turnover, cost)})
            if cost == BASE_COST_BPS:
                by_year = returns.groupby(returns.index.year).apply(lambda values: float((1 + values).prod() - 1))
                yearly.extend({"strategy": name, "year": int(year), "net_return": value}
                              for year, value in by_year.items())
                for label, indexes in regime.loc[returns.index].groupby(regime.loc[returns.index]).groups.items():
                    observed = returns.reindex(indexes).dropna()
                    if not observed.empty:
                        regimes.append({"strategy": name, "market_regime": label,
                                         "sessions": len(observed),
                                         "net_return": float((1 + observed).prod() - 1)})
    fingerprint = sha256(
        (Path(__file__).read_bytes() + base_code_fingerprint().encode("ascii"))
    ).hexdigest()
    source_hash = sha256(
        (metadata["industry_zip_sha256"] + metadata["factor_zip_sha256"]).encode("ascii")
    ).hexdigest()
    return {
        "phase": "development_only",
        "status": "Exploratory candidate; not selected for prospective tracking or live use.",
        "source_hash": source_hash,
        "sources": metadata,
        "code_fingerprint": fingerprint,
        "development_start": str(EVALUATION_START.date()),
        "development_end": str(DEVELOPMENT_END.date()),
        "candidate_definition": (
            "Take the existing 12-1 top-10 industry momentum portfolio with a 5% volatility target; "
            "hold cash whenever the already-lagged Market 200-day trend baseline is not invested."
        ),
        "comparison_strategies": [*BENCHMARKS, FILTERED_STRATEGY],
        "settings": {"costs_bps_per_side": list(COSTS_BPS),
                     "base_cost_bps_per_side": BASE_COST_BPS,
                     "trend_lookback_sessions": 200,
                     "execution_lag_sessions": 2,
                     "momentum_lookback_sessions": 252,
                     "momentum_skip_sessions": 21,
                     "volatility_target": 0.05},
        "summary": summaries,
        "yearly_returns_at_base_cost": yearly,
        "market_regime_returns_at_base_cost": regimes,
        "limitations": [
            "Development data only; all dates in this period were previously used by MarketSignal research.",
            "Industry portfolios are reconstructed aggregates, not individual stocks or an investable ETF basket.",
            "Costs are fixed portfolio-level assumptions and do not include taxes or market impact.",
            "This single fixed filter comparison does not establish statistical superiority or future profitability.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--industries", type=Path,
                        default=Path("data/research_inputs/fama_french_49_industry_daily_202608.zip"))
    parser.add_argument("--factors", type=Path,
                        default=Path("data/research_inputs/fama_french_3_factors_daily_202608.zip"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/industry_trend_filter_development.json"))
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose a new --report path to preserve this run.")
    report = run_development(args.industries, args.factors)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    scores = pd.DataFrame(report["summary"])
    print(scores.loc[scores.cost_bps_per_side == BASE_COST_BPS,
                     ["strategy", "annualized_return", "max_drawdown", "sharpe_vs_cash"]].to_string(index=False))
    print(f"Saved development-only report: {args.report}")


if __name__ == "__main__":
    main()
