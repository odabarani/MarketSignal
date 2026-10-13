"""Run chronological calibration experiments on a saved development snapshot."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import pandas as pd

from research_pipeline import model_configuration
from src.research import record_experiment
from src.research_features import FEATURE_SETS, add_panel_features
from src.research_model import build_multiclass_models
from src.research_walkforward import walk_forward
from src.research_report import render_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/walk_forward.json")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="technical")
    parser.add_argument("--min-train-dates", type=int, default=504)
    parser.add_argument("--calibration-dates", type=int, default=126)
    parser.add_argument("--evaluation-dates", type=int, default=126)
    parser.add_argument("--ledger", default="experiments/research.jsonl")
    args = parser.parse_args(argv)
    source, output = Path(args.input), Path(args.output)
    destinations = [output, output.with_suffix(".html"), output.with_suffix(".predictions.csv")]
    if len(set(p.resolve() for p in destinations)) != 3 or any(
        p.resolve() in {source.resolve(), source.with_suffix(".manifest.json").resolve()}
        for p in destinations
    ):
        parser.error("Use a separate .json output path to preserve the source snapshot.")
    if output.suffix != ".json":
        parser.error("Output must be a .json report.")
    manifest_path = source.with_suffix(".manifest.json")
    if not manifest_path.exists():
        parser.error("The snapshot needs its saved .manifest.json with validation_end.")
    metadata = json.loads(manifest_path.read_text())
    cutoff = metadata.get("validation_end")
    if not cutoff:
        parser.error("The snapshot manifest must contain validation_end.")
    panel = pd.read_csv(source, parse_dates=["Date", "Target_Available_At"])
    # Slice before computing any feature or metric; later test rows stay excluded.
    panel = panel.loc[panel.Date <= pd.Timestamp(cutoff)].set_index(["Date", "Ticker"]).sort_index()
    panel = add_panel_features(panel)
    models = build_multiclass_models()
    root = Path(__file__).resolve().parent
    code = sha256()
    for path in sorted([root / "walk_forward_pipeline.py", root / "research_pipeline.py",
                        *root.glob("src/*.py")]):
        code.update(str(path.relative_to(root)).encode())
        code.update(path.read_bytes())
    config = {"source_hash": sha256(source.read_bytes()).hexdigest(),
              "code_fingerprint": code.hexdigest(), "development_end": cutoff,
              "tickers": sorted(panel.index.get_level_values("Ticker").unique()),
              "feature_set": args.feature_set, "target": metadata.get("target", "precomputed"),
              "min_train_dates": args.min_train_dates, "calibration_dates": args.calibration_dates,
              "evaluation_dates": args.evaluation_dates, "model_parameters": model_configuration(models)}
    config["versions"] = {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]}
    record_experiment(args.ledger, {**config, "stage": "walk_forward_started"}, {}, panel)
    try:
        report, predictions = walk_forward(
            panel, FEATURE_SETS[args.feature_set], cutoff, args.min_train_dates,
            args.calibration_dates, args.evaluation_dates, models,
        )
        report["config"] = config
        report["experiment"] = record_experiment(
            args.ledger, {**config, "stage": "walk_forward_complete"},
            {"summary": report["summary"]}, panel,
            notes="Development-only comparison. Previously inspected data; no untouched-test claim.",
        )
        page = render_report(report)
    except Exception as exc:
        record_experiment(args.ledger, {**config, "stage": "walk_forward_failed"},
                          {"error": str(exc)}, panel)
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output.with_suffix(".predictions.csv"), index=False)
    output.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    output.with_suffix(".html").write_text(page)
    print(pd.DataFrame(report["summary"])[["candidate", "log_loss", "brier", "macro_f1"]].to_string(index=False))
    print(f"Saved interactive report: {output.with_suffix('.html')}")


if __name__ == "__main__":
    main()
