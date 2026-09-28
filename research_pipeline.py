"""Build a reproducible multi-stock research dataset from the command line."""

import argparse
import json
from pathlib import Path

from src.data_loader import get_market_data
from src.research import (
    build_research_panel, chronological_split, dataset_fingerprint,
    record_experiment,
)
from src.research_model import evaluate_locked_test, select_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--output", default="data/research_panel.csv")
    parser.add_argument("--round-trip-cost-bps", type=float, default=30)
    parser.add_argument("--minimum-edge-bps", type=float, default=0)
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Open the locked test period after validation selects a model.",
    )
    parser.add_argument("--ledger", default="experiments/research.jsonl")
    args = parser.parse_args()

    market_data = get_market_data(args.tickers)
    panel = build_research_panel(
        market_data,
        round_trip_cost_bps=args.round_trip_cost_bps,
        minimum_edge_bps=args.minimum_edge_bps,
    )
    split = chronological_split(panel)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output)
    manifest = {
        "tickers": sorted(market_data),
        "rows": len(panel),
        "fingerprint": dataset_fingerprint(panel),
        "train_end": str(split.train_end.date()),
        "validation_end": str(split.validation_end.date()),
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "survivorship_warning": (
            "Ticker membership is supplied by the researcher. Use dated universe "
            "snapshots before drawing historical conclusions."
        ),
    }
    selection = select_model(split)
    manifest["selected_model"] = selection.selected_name
    manifest["promoted_over_baseline"] = selection.promoted_over_baseline
    manifest["validation_log_loss_improvement"] = selection.log_loss_improvement
    manifest["validation"] = selection.validation_results.to_dict("records")
    if args.evaluate_test:
        _, predictions, test_metrics = evaluate_locked_test(selection, split)
        manifest["test"] = test_metrics
        record_experiment(
            args.ledger,
            config={
                "tickers": sorted(market_data),
                "round_trip_cost_bps": args.round_trip_cost_bps,
                "minimum_edge_bps": args.minimum_edge_bps,
                "selected_model": selection.selected_name,
                "train_end": str(split.train_end.date()),
                "validation_end": str(split.validation_end.date()),
            },
            metrics={"validation": manifest["validation"], "test": test_metrics},
            panel=panel,
            notes="Locked test opened by explicit --evaluate-test flag.",
        )
        predictions.to_csv(output.with_suffix(".test_predictions.csv"))
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
