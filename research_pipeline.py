"""Reproducible multi-stock feature comparison, with validation-only defaults."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess

import pandas as pd

from src.data_loader import get_market_data
from src.research import (
    build_research_panel, chronological_split, dataset_fingerprint, record_experiment,
)
from src.research_experiment import compare_feature_sets, ensure_test_unseen
from src.research_features import FEATURE_SETS, add_panel_features
from src.research_model import build_multiclass_models, evaluate_locked_test


def model_configuration(models):
    # XGBoost uses NaN as its missing-value sentinel; encode that parameter as
    # text while keeping metric serialization strict (no NaN scores in reports).
    return {
        name: {
            key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in model.get_params(deep=True).items()
        }
        for name, model in models.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tickers", nargs="+")
    source.add_argument("--input", help="Reuse a saved panel CSV without downloading data.")
    parser.add_argument("--output", default="data/research_panel.csv")
    parser.add_argument("--round-trip-cost-bps", type=float)
    parser.add_argument("--minimum-edge-bps", type=float)
    parser.add_argument("--train-end", help="Last training date (YYYY-MM-DD).")
    parser.add_argument("--validation-end", help="Last validation date (YYYY-MM-DD).")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="technical")
    parser.add_argument("--compare-features", action="store_true",
                        help="Compare technical, price-relative, and market-context groups.")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Record and evaluate an unseen test period after selection.")
    parser.add_argument("--ledger", default="experiments/research.jsonl")
    args = parser.parse_args(argv)
    if bool(args.train_end) != bool(args.validation_end):
        parser.error("Provide both --train-end and --validation-end.")
    output = Path(args.output)
    source_metadata = {}
    if args.input:
        snapshot = Path(args.input)
        if snapshot.resolve() == output.resolve():
            parser.error("Use a separate --output path to preserve the input snapshot.")
        if args.round_trip_cost_bps is not None or args.minimum_edge_bps is not None:
            parser.error("Snapshot targets are frozen; cost options apply only to downloads.")
        panel = pd.read_csv(snapshot, parse_dates=["Date", "Target_Available_At"])
        panel = panel.set_index(["Date", "Ticker"]).sort_index()
        metadata_path = snapshot.with_suffix(".manifest.json")
        if metadata_path.exists():
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        train_end = args.train_end or source_metadata.get("train_end")
        validation_end = args.validation_end or source_metadata.get("validation_end")
        if not train_end or not validation_end:
            parser.error("A snapshot requires saved split dates or explicit date arguments.")
        target_config = source_metadata.get("target", {"source": "precomputed; costs unknown"})
        source_hash = sha256(snapshot.read_bytes()).hexdigest()
    else:
        cost = 30 if args.round_trip_cost_bps is None else args.round_trip_cost_bps
        edge = 0 if args.minimum_edge_bps is None else args.minimum_edge_bps
        panel = build_research_panel(
            get_market_data(args.tickers), round_trip_cost_bps=cost, minimum_edge_bps=edge,
        )
        target_config = {"horizon": 5, "round_trip_cost_bps": cost, "minimum_edge_bps": edge}
        train_end, validation_end = args.train_end, args.validation_end
        source_hash = dataset_fingerprint(panel)

    panel = add_panel_features(panel)
    split = chronological_split(panel, train_end=train_end, validation_end=validation_end)
    names = list(FEATURE_SETS) if args.compare_features else [args.feature_set]
    tickers = sorted(panel.index.get_level_values("Ticker").unique())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    models = build_multiclass_models()
    root = Path(__file__).resolve().parent
    source_digest = sha256()
    for path in sorted([root / "research_pipeline.py", *root.glob("src/*.py")]):
        source_digest.update(str(path.relative_to(root)).encode())
        source_digest.update(path.read_bytes())
    config = {
        "tickers": tickers, "target": target_config, "source_hash": source_hash,
        "train_end": str(split.train_end.date()),
        "validation_end": str(split.validation_end.date()),
        "test_end": str(panel.index.get_level_values("Date").max().date()),
        "feature_sets": {name: FEATURE_SETS[name] for name in names},
        "model_parameters": model_configuration(models),
        "revision": revision, "working_tree_dirty": dirty,
        "code_fingerprint": source_digest.hexdigest(),
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]},
    }
    record_experiment(args.ledger, {**config, "stage": "validation_started"}, {}, panel)
    try:
        comparison, chosen, selection, development = compare_feature_sets(split, names, models)
    except Exception as exc:
        record_experiment(args.ledger, {**config, "stage": "validation_failed"},
                          {"error": str(exc)}, panel)
        raise
    manifest = {
        **config, "rows": len(panel), "fingerprint": dataset_fingerprint(panel),
        "train_rows": len(development.train), "validation_rows": len(development.validation),
        "test_rows": len(split.test), "selected_model": selection.selected_name,
        "selected_feature_set": chosen,
        "beats_baseline_on_validation": selection.promoted_over_baseline,
        "validation_log_loss_improvement": selection.log_loss_improvement,
        "validation": comparison.to_dict("records"),
        "limitations": "Supplied universe only; no historical membership or delisted coverage. "
                        "Validation superiority is not evidence of profitability or deployment readiness.",
    }
    record_experiment(args.ledger, {**config, "stage": "validation"},
                      {"validation": manifest["validation"]}, panel)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output)
    if args.evaluate_test:
        ensure_test_unseen(args.ledger, tickers, split.validation_end, config["test_end"])
        # Reserve before fitting so a failed run cannot silently reopen the period.
        test_config = {**config, "stage": "test_reserved", "selected_model": selection.selected_name,
                       "selected_feature_set": chosen}
        record_experiment(args.ledger, test_config, {}, panel)
        _, predictions, test_metrics = evaluate_locked_test(
            selection, development, features=FEATURE_SETS[chosen]
        )
        manifest["test"] = test_metrics
        record_experiment(args.ledger, {**test_config, "stage": "test_complete"},
                          {"test": test_metrics}, panel)
        predictions.to_csv(output.with_suffix(".test_predictions.csv"))
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(comparison[["Feature_set", "Model", "log_loss", "brier", "macro_f1"]].to_string(index=False))
    print(f"Selected: {chosen} / {selection.selected_name}")
    print(f"Saved report: {output.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
