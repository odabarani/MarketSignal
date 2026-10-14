"""Post-hoc paired block-bootstrap uncertainty for a saved equity curve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _max_drawdown(log_returns: np.ndarray) -> np.ndarray:
    """Return each path's maximum drawdown, with initial wealth set to one."""
    wealth = np.exp(np.cumsum(log_returns, axis=1))
    peaks = np.maximum.accumulate(np.column_stack([np.ones(len(wealth)), wealth]), axis=1)[:, 1:]
    return np.min(wealth / peaks - 1.0, axis=1)


def compare_equity_curves(
    equity: pd.DataFrame,
    strategy: str,
    benchmark: str,
    *,
    block_sessions: int = 20,
    resamples: int = 20_000,
    seed: int = 20_261_014,
) -> dict:
    """Bootstrap paired daily log returns in circular contiguous blocks.

    The return statistic is the strategy's annualized relative wealth growth
    versus the benchmark. The drawdown statistic is strategy MDD minus
    benchmark MDD, so positive values mean a smaller drawdown for the strategy.
    These are descriptive post-hoc uncertainty ranges, not a significance test.
    """
    if not isinstance(equity, pd.DataFrame) or equity.empty:
        raise ValueError("Equity input must be a non-empty DataFrame.")
    if strategy == benchmark or strategy not in equity or benchmark not in equity:
        raise ValueError("Choose two distinct columns present in the equity input.")
    if (isinstance(block_sessions, bool) or not isinstance(block_sessions, int)
            or block_sessions < 1):
        raise ValueError("Block length must be a positive integer.")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")

    pair = equity[[strategy, benchmark]].astype(float)
    if len(pair) < 3 or not np.isfinite(pair.to_numpy()).all() or (pair <= 0).any().any():
        raise ValueError("Equity curves need at least three finite, positive observations.")
    if isinstance(pair.index, pd.DatetimeIndex):
        if pair.index.has_duplicates or not pair.index.is_monotonic_increasing:
            raise ValueError("Date-indexed equity curves must be unique and chronological.")
    log_returns = np.log(pair / pair.shift(1)).dropna().to_numpy()
    sessions = len(log_returns)
    if block_sessions > sessions:
        raise ValueError("Block length cannot exceed the number of return sessions.")

    rng = np.random.default_rng(seed)
    blocks_per_path = int(np.ceil(sessions / block_sessions))
    starts = rng.integers(0, sessions, size=(resamples, blocks_per_path))
    offsets = np.arange(block_sessions)
    positions = (starts[:, :, None] + offsets) % sessions
    positions = positions.reshape(resamples, -1)[:, :sessions]
    sampled = log_returns[positions]
    relative_growth = np.expm1(252.0 * np.mean(sampled[:, :, 0] - sampled[:, :, 1], axis=1))
    drawdown_gap = _max_drawdown(sampled[:, :, 0]) - _max_drawdown(sampled[:, :, 1])

    observed = log_returns
    observed_relative_growth = float(np.expm1(252.0 * np.mean(observed[:, 0] - observed[:, 1])))
    observed_strategy_return = float(np.expm1(252.0 * np.mean(observed[:, 0])))
    observed_benchmark_return = float(np.expm1(252.0 * np.mean(observed[:, 1])))
    observed_drawdown_gap = float(
        _max_drawdown(observed[None, :, 0])[0] - _max_drawdown(observed[None, :, 1])[0]
    )

    def interval(values: np.ndarray) -> list[float]:
        return [float(x) for x in np.quantile(values, [0.025, 0.5, 0.975])]

    return {
        "strategy": strategy,
        "benchmark": benchmark,
        "sessions": sessions,
        "first_date": str(pair.index[1].date()) if isinstance(pair.index, pd.DatetimeIndex) else None,
        "last_date": str(pair.index[-1].date()) if isinstance(pair.index, pd.DatetimeIndex) else None,
        "observed_annualized_return": {
            strategy: observed_strategy_return,
            benchmark: observed_benchmark_return,
        },
        "observed_annualized_relative_growth": observed_relative_growth,
        "bootstrap_annualized_relative_growth_2_5_50_97_5_pct": interval(relative_growth),
        "bootstrap_share_relative_growth_positive": float(np.mean(relative_growth > 0)),
        "observed_max_drawdown_gap": observed_drawdown_gap,
        "bootstrap_max_drawdown_gap_2_5_50_97_5_pct": interval(drawdown_gap),
        "bootstrap_share_strategy_lower_drawdown": float(np.mean(drawdown_gap > 0)),
        "method": "paired circular moving-block bootstrap of daily log returns",
        "block_sessions": block_sessions,
        "resamples": resamples,
        "seed": seed,
        "annualization_sessions": 252,
        "interpretation": (
            "Post-hoc descriptive uncertainty conditional on this observed period; "
            "not a significance test, new holdout, or correction for strategy selection."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sector_etf_locked.equity.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/sector_etf_holdout_uncertainty.json"))
    parser.add_argument("--strategy", default="Equal-weight volatility target")
    parser.add_argument("--benchmark", default="SPY buy & hold")
    parser.add_argument("--block-sessions", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_261_014)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    equity = pd.read_csv(args.input, parse_dates=["Date"]).set_index("Date")
    report = compare_equity_curves(
        equity, args.strategy, args.benchmark, block_sessions=args.block_sessions,
        resamples=args.resamples, seed=args.seed,
    )
    report["input"] = str(args.input)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Observed relative annualized growth: {report['observed_annualized_relative_growth']:.2%}")
    print("95% bootstrap interval:", report["bootstrap_annualized_relative_growth_2_5_50_97_5_pct"])
    print(f"Share of resampled paths with lower strategy drawdown: {report['bootstrap_share_strategy_lower_drawdown']:.1%}")
    print(f"Saved post-hoc uncertainty report: {args.report}")


if __name__ == "__main__":
    main()
