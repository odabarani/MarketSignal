"""Build a reproducible multi-stock research dataset from the command line."""

import argparse
import json
from pathlib import Path

from src.data_loader import get_market_data
from src.research import build_research_panel, chronological_split, dataset_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--output", default="data/research_panel.csv")
    parser.add_argument("--round-trip-cost-bps", type=float, default=30)
    parser.add_argument("--minimum-edge-bps", type=float, default=0)
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
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
