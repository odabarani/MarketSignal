"""Export QuantRocket's free 2007–2011 daily US stock sample to a research CSV.

Run this inside a QuantRocket Python/Jupyter environment after ingesting the
``usstock-learn-1d`` daily bundle. It is not a dependency of the MarketSignal app.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.stock_panel import (
    file_sha256,
    validate_liquidity_panel,
    validate_security_metadata,
    validate_stock_panel,
)

BUNDLE = "usstock-learn-1d"
START_DATE = "2007-01-01"
END_DATE = "2011-12-31"
MASTER_FIELDS = [
    "Sid", "usstock_Symbol", "usstock_DateDelisted", "usstock_FirstPriceDate",
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
        "usstock_Symbol": "symbol",
        "usstock_DateDelisted": "date_delisted",
        "usstock_FirstPriceDate": "first_price_date",
        "usstock_LastPriceDate": "last_price_date",
        "usstock_SecurityType2": "security_type",
        "usstock_Sector": "sector",
    })
    metadata = validate_security_metadata(master, panel_sids)
    return metadata.loc[metadata["sid"].astype(str).isin(panel_sids)].reset_index(drop=True)


def link_adjusted_price_chunk(
    previous_panel: pd.DataFrame, current_panel: pd.DataFrame, bridge_prices: pd.DataFrame
) -> pd.DataFrame:
    """Rebase this year's adjusted closes to preserve the return across the year boundary."""
    if previous_panel.empty or current_panel.empty:
        raise ValueError("Cannot link empty yearly price chunks.")
    if not isinstance(bridge_prices.index, pd.MultiIndex) or not {"Field", "Date"}.issubset(bridge_prices.index.names):
        raise ValueError("Expected QuantRocket bridge prices indexed by Field and Date.")
    if "Close" not in set(bridge_prices.index.get_level_values("Field")):
        raise ValueError("QuantRocket bridge query is missing Close.")

    prior_date = previous_panel["date"].max()
    current_date = current_panel["date"].min()
    bridge = bridge_prices.xs("Close", level="Field")
    bridge.index = pd.to_datetime(bridge.index)
    if prior_date not in bridge.index or current_date not in bridge.index:
        raise ValueError("QuantRocket bridge query must include both year-boundary sessions.")

    prior_close = previous_panel.loc[previous_panel.date.eq(prior_date)].set_index("sid")["close"]
    first_close = current_panel.loc[current_panel.date.eq(current_date)].set_index("sid")["close"]
    as_of_prior = bridge.loc[prior_date]
    as_of_current = bridge.loc[current_date]
    common = prior_close.index.intersection(first_close.index).intersection(as_of_prior.dropna().index).intersection(as_of_current.dropna().index)
    if common.empty:
        return current_panel

    boundary_return = as_of_current.loc[common] / as_of_prior.loc[common]
    scales = (prior_close.loc[common] * boundary_return / first_close.loc[common]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    result = current_panel.copy()
    result["close"] = result["close"] * result["sid"].map(scales).fillna(1.0)
    return result


def point_in_time_monthly_liquidity(get_prices, decision_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Measure 20-session dollar volume with each query ending at its decision date."""
    chunks = []
    for decision in decision_dates:
        decision = pd.Timestamp(decision)
        prices = get_prices(
            BUNDLE,
            fields=["Close", "Volume"],
            start_date=str(max(pd.Timestamp(START_DATE), decision - pd.Timedelta(days=45)).date()),
            end_date=str(decision.date()),
        )
        if not isinstance(prices.index, pd.MultiIndex) or not {"Field", "Date"}.issubset(prices.index.names):
            raise ValueError("Expected QuantRocket prices indexed by Field and Date.")
        fields = set(prices.index.get_level_values("Field"))
        if not {"Close", "Volume"}.issubset(fields):
            raise ValueError("Point-in-time liquidity query is missing Close or Volume.")
        close = prices.xs("Close", level="Field")
        volume = prices.xs("Volume", level="Field")
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        if decision not in close.index:
            raise ValueError(f"QuantRocket returned no prices on decision date {decision.date()}.")
        dollar_volume = (close * volume).replace([np.inf, -np.inf], np.nan)
        observations = close.notna().rolling(20, min_periods=1).sum().loc[decision]
        average = dollar_volume.fillna(0.0).rolling(20, min_periods=20).mean().loc[decision]
        valid = average.loc[(observations >= 15) & average.notna()].dropna()
        if not valid.empty:
            chunks.append(pd.DataFrame({
                "sid": valid.index.astype(str),
                "date": decision,
                "average_dollar_volume": valid.to_numpy(dtype=float),
            }))
    if not chunks:
        raise ValueError("QuantRocket returned no valid monthly liquidity observations.")
    return validate_liquidity_panel(pd.concat(chunks, ignore_index=True))


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

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    metadata_name = (
        output.name[:-7] + "_security_master.csv.gz"
        if output.name.endswith(".csv.gz")
        else output.stem + "_security_master" + output.suffix
    )
    metadata_path = output.with_name(metadata_name)
    liquidity_name = (
        output.name[:-7] + "_monthly_liquidity.csv.gz"
        if output.name.endswith(".csv.gz")
        else output.stem + "_monthly_liquidity" + output.suffix
    )
    liquidity_path = output.with_name(liquidity_name)
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --force to replace it.")
    if metadata_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {metadata_path}; pass --force to replace it.")
    if liquidity_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {liquidity_path}; pass --force to replace it.")

    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".csv.gz", delete=False) as stream:
        temporary_panel = Path(stream.name)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".csv.gz", delete=False) as stream:
        temporary_metadata = Path(stream.name)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".csv.gz", delete=False) as stream:
        temporary_liquidity = Path(stream.name)
    temporary_manifest = output.parent / (temporary_panel.name + ".manifest.json")
    try:
        security_ids: set[str] = set()
        market_dates: set[pd.Timestamp] = set()
        chunk_summaries = []
        first_date, last_date = None, None
        previous_panel = None
        row_count = 0
        # Query one year at a time so the wide all-stock matrices stay manageable.
        with gzip.open(temporary_panel, "wt", encoding="utf-8", newline="") as output_stream:
            for year in range(2007, 2012):
                start_date, end_date = f"{year}-01-01", f"{year}-12-31"
                prices = get_prices(
                    BUNDLE, fields=["Close", "Volume"],
                    start_date=start_date, end_date=end_date,
                )
                panel = bundle_prices_to_panel(prices)
                if panel.empty:
                    raise ValueError(f"QuantRocket returned no observations for {year}.")
                if not panel["date"].between(start_date, end_date).all():
                    raise ValueError(f"QuantRocket returned observations outside {year}.")
                if previous_panel is not None:
                    prior_date, current_date = previous_panel["date"].max(), panel["date"].min()
                    bridge = get_prices(
                        BUNDLE, fields=["Close"],
                        start_date=str(prior_date.date()), end_date=str(current_date.date()),
                    )
                    panel = link_adjusted_price_chunk(previous_panel, panel, bridge)
                panel.to_csv(
                    output_stream, index=False, header=(year == 2007), date_format="%Y-%m-%d"
                )
                year_first, year_last = panel["date"].min(), panel["date"].max()
                first_date = year_first if first_date is None else min(first_date, year_first)
                last_date = year_last if last_date is None else max(last_date, year_last)
                row_count += len(panel)
                security_ids.update(panel["sid"].astype(str))
                market_dates.update(pd.Timestamp(date) for date in panel["date"].unique())
                previous_panel = panel
                chunk_summaries.append({
                    "year": year, "rows": int(len(panel)),
                    "first_date": str(year_first.date()), "last_date": str(year_last.date()),
                })

        calendar = pd.DatetimeIndex(sorted(market_dates))
        decision_dates = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).last().tolist()
        liquidity = point_in_time_monthly_liquidity(get_prices, decision_dates)
        liquidity.to_csv(
            temporary_liquidity, index=False, compression="gzip", date_format="%Y-%m-%d"
        )

        securities = get_securities(vendors="usstock", fields=MASTER_FIELDS)
        master = master_to_metadata(securities, security_ids)
        master.to_csv(temporary_metadata, index=False, compression="gzip", date_format="%Y-%m-%d")
        manifest = {
            "source": "QuantRocket US Stock learning bundle",
            "bundle": BUNDLE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "fields": ["Close", "Volume"],
            "price_basis": "Bundle Close, adjusted by QuantRocket as of the query end date.",
            "identifier": "Stable QuantRocket Sid; the companion master snapshot includes current ticker labels.",
            "security_metadata_file": metadata_path.name,
            "security_metadata_sha256": file_sha256(temporary_metadata),
            "monthly_liquidity_file": liquidity_path.name,
            "monthly_liquidity_sha256": file_sha256(temporary_liquidity),
            "monthly_liquidity_measure": (
                "20-session mean Close*Volume queried at each month-end with the query "
                "ending on that date, avoiding later corporate-action adjustments."
            ),
            "security_metadata_fields": [
                "Sid", "Symbol", "DateDelisted", "FirstPriceDate", "LastPriceDate",
                "SecurityType2", "Sector",
            ],
            "security_metadata_role": (
                "Latest securities-master snapshot for event labeling only; never use these "
                "fields as historical signal inputs or to define past universe membership."
            ),
            "rows": row_count,
            "monthly_liquidity_rows": int(len(liquidity)),
            "securities": len(security_ids),
            "first_date": str(first_date.date()),
            "last_date": str(last_date.date()),
            "yearly_chunks": chunk_summaries,
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
        temporary_liquidity.replace(liquidity_path)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_panel.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        temporary_liquidity.unlink(missing_ok=True)
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
