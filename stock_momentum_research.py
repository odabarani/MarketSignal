"""Run the fixed stock momentum comparison on an exported QuantRocket sample."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.fama_french import load_daily_risk_free
from src.stock_momentum import run_experiment
from src.stock_panel import (
    load_liquidity_panel,
    load_security_metadata,
    load_stock_panel,
    validate_learning_sample_coverage,
)


def code_fingerprint() -> str:
    digest = sha256()
    for path in (
        Path(__file__), Path("src/stock_momentum.py"),
        Path("src/stock_panel.py"), Path("src/fama_french.py"),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices", type=Path,
        default=Path("data/research_inputs/quantrocket_usstock_learn_2007_2011.csv.gz"),
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--liquidity", type=Path, default=None)
    parser.add_argument(
        "--risk-free", type=Path,
        default=Path("data/research_inputs/fama_french_3_factors_daily_202608.zip"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/stock_momentum_development.json")
    )
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose a new --report path to preserve each run.")
    metadata_path = args.metadata or args.prices.with_name(
        args.prices.name[:-7] + "_security_master.csv.gz"
        if args.prices.name.endswith(".csv.gz")
        else args.prices.stem + "_security_master" + args.prices.suffix
    )
    liquidity_path = args.liquidity or args.prices.with_name(
        args.prices.name[:-7] + "_monthly_liquidity.csv.gz"
        if args.prices.name.endswith(".csv.gz")
        else args.prices.stem + "_monthly_liquidity" + args.prices.suffix
    )
    panel, price_source = load_stock_panel(args.prices)
    metadata, metadata_source = load_security_metadata(metadata_path, set(panel["sid"].astype(str)))
    spy_sids = metadata.loc[metadata["symbol"].eq("SPY"), "sid"].astype(str).tolist()
    if len(spy_sids) != 1:
        parser.error(f"Expected exactly one SPY Sid in security metadata, found {len(spy_sids)}.")
    price_source["market_session_coverage"] = validate_learning_sample_coverage(panel, spy_sids[0])
    liquidity, liquidity_source = load_liquidity_panel(liquidity_path)
    risk_free, risk_free_source = load_daily_risk_free(args.risk_free)
    report = run_experiment(panel, metadata, liquidity, cash_returns=risk_free)
    report["sources"] = {
        "prices": price_source,
        "security_metadata": metadata_source,
        "monthly_liquidity": liquidity_source,
        "risk_free_returns": risk_free_source,
    }
    report["code_fingerprint"] = code_fingerprint()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    scores = pd.DataFrame(report["results"])
    print(scores.loc[
        (scores.cost_bps_per_side == 30) & (scores.delisting_scenario == "zero_recovery"),
        ["strategy", "net_return", "excess_net_return_vs_spy", "max_drawdown",
         "within_research_drawdown_budget", "sharpe_excess_rf"],
    ].to_string(index=False))
    print(f"Saved historical-development-only comparison: {args.report}")


if __name__ == "__main__":
    main()
