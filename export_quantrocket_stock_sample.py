"""Export QuantRocket's free 2007–2011 daily US stock sample to a research CSV.

Run this inside a QuantRocket Python/Jupyter environment after ingesting the
``usstock-learn-1d`` daily bundle. It is not a dependency of the MarketSignal app.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd

from src.stock_panel import (
    file_sha256,
    validate_security_metadata,
    validate_stock_panel,
)

BUNDLE = "usstock-learn-1d"
START_DATE = "2007-01-01"
END_DATE = "2011-12-31"
MASTER_FIELDS = [
    "Sid", "usstock_DateDelisted", "usstock_FirstPriceDate",
    "usstock_LastPriceDate", "usstock_SecurityType2", "usstock_Sector",
]


def bundle_prices_to_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert QuantRocket's Field/Date-by-Sid frame to a tidy daily panel."""
    if not isinstance(prices.index, pd.MultiIndex) or "Date" not in prices.index.names:
        raise ValueError("Expected QuantRocket get_prices output indexed by Field and Date.")
    if "Field" not in prices.index.names:
        raise ValueError("QuantRocket price rows must include a Field index level.")
    fields = set(prices.index.get_level_values("Field"))
    if not {"Close", "Volume"}.issubset(fields):
        raise ValueError("The QuantRocket bundle export must include Close and Volume.")

    blocks = []
    for field in ("Close", "Volume"):
        wide = prices.xs(field, level="Field")
        long = (wide.rename_axis(index="date", columns="sid").stack(future_stack=True)
                .dropna().rename(field.lower()))
        blocks.append(long)
    panel = pd.concat(blocks, axis=1).reset_index()
    return validate_stock_panel(panel)


def master_to_metadata(securities: pd.DataFrame, panel_sids: set[str]) -> pd.DataFrame:
    """Keep delisting and classification labels for exported Sids only."""
    master = securities.copy()
    if "Sid" not in master.columns:
        master = master.reset_index()
    master = master.rename(columns={
        "Sid": "sid",
        "usstock_DateDelisted": "date_delisted",
        "usstock_FirstPriceDate": "first_price_date",
        "usstock_LastPriceDate": "last_price_date",
        "usstock_SecurityType2": "security_type",
        "usstock_Sector": "sector",
    })
    metadata = validate_security_metadata(master, panel_sids)
    return metadata.loc[metadata["sid"].astype(str).isin(panel_sids)].reset_index(drop=True)


def export(output: Path, force: bool = False) -> dict[str, object]:
    """Retrieve the bundle, validate it, and atomically write panel and manifest."""
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force to replace it.")
    try:
        from quantrocket import get_prices
        from quantrocket.master import get_securities
    except ImportError as exc:
        raise RuntimeError(
            "Run this exporter from a QuantRocket Python/Jupyter environment after "
            "creating and ingesting the usstock-learn-1d bundle."
        ) from exc

    prices = get_prices(
        BUNDLE, fields=["Close", "Volume"], start_date=START_DATE, end_date=END_DATE
    )
    panel = bundle_prices_to_panel(prices)
    securities = get_securities(vendors="usstock", fields=MASTER_FIELDS)
    master = master_to_metadata(securities, set(panel["sid"].astype(str)))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    metadata_name = (
        output.name[:-7] + "_security_master.csv.gz"
        if output.name.endswith(".csv.gz")
        else output.stem + "_security_master" + output.suffix
    )
    metadata_path = output.with_name(metadata_name)
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --force to replace it.")
    if metadata_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {metadata_path}; pass --force to replace it.")

    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".csv.gz", delete=False) as stream:
        temporary_panel = Path(stream.name)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".csv.gz", delete=False) as stream:
        temporary_metadata = Path(stream.name)
    temporary_manifest = output.parent / (temporary_panel.name + ".manifest.json")
    try:
        panel.to_csv(temporary_panel, index=False, compression="gzip", date_format="%Y-%m-%d")
        master.to_csv(temporary_metadata, index=False, compression="gzip", date_format="%Y-%m-%d")
        manifest = {
            "source": "QuantRocket US Stock learning bundle",
            "bundle": BUNDLE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "fields": ["Close", "Volume"],
            "price_basis": "Bundle Close, adjusted by QuantRocket as of the query end date.",
            "identifier": "Stable QuantRocket Sid; ticker symbols are intentionally omitted.",
            "security_metadata_file": metadata_path.name,
            "security_metadata_sha256": file_sha256(temporary_metadata),
            "security_metadata_fields": [
                "Sid", "DateDelisted", "FirstPriceDate", "LastPriceDate",
                "SecurityType2", "Sector",
            ],
            "security_metadata_role": (
                "Latest securities-master snapshot for event labeling only; never use these "
                "fields as historical signal inputs or to define past universe membership."
            ),
            "rows": int(len(panel)),
            "securities": int(panel["sid"].nunique()),
            "first_date": str(panel["date"].min().date()),
            "last_date": str(panel["date"].max().date()),
            "csv_sha256": file_sha256(temporary_panel),
            "limitations": [
                "Delisted securities are included in the source, but this export does not contain delisting cash or stock consideration.",
                "An absent later price is not itself evidence of a total loss; terminal returns require a documented delisting policy.",
                "Bundle prices are adjusted as of the query end date; use returns, not historical nominal price levels, as features.",
            ],
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary_panel.replace(output)
        temporary_metadata.replace(metadata_path)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_panel.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/research_inputs/quantrocket_usstock_learn_2007_2011.csv.gz"),
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing export and manifest.")
    args = parser.parse_args()
    manifest = export(args.output, force=args.force)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
