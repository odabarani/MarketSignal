"""Load and validate the QuantRocket free daily US stock learning sample."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import numpy as np

STOCK_PANEL_COLUMNS = ["sid", "date", "close", "volume"]
LIQUIDITY_PANEL_COLUMNS = ["sid", "date", "average_adjusted_dollar_volume"]
SECURITY_METADATA_COLUMNS = [
    "sid", "symbol", "date_delisted", "first_price_date", "last_price_date",
    "security_type", "sector",
]
LEARNING_SAMPLE_START = pd.Timestamp("2007-01-01")
LEARNING_SAMPLE_END = pd.Timestamp("2011-12-31")


def validate_stock_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized panel or raise for data that could corrupt a study."""
    missing = set(STOCK_PANEL_COLUMNS).difference(panel.columns)
    if missing:
        raise ValueError(f"Stock panel is missing required columns: {sorted(missing)}")

    result = panel.loc[:, STOCK_PANEL_COLUMNS].copy()
    result["sid"] = result["sid"].astype("string").str.strip()
    if result["sid"].isna().any() or result["sid"].eq("").any():
        raise ValueError("Stock panel contains a missing or blank Sid.")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("Stock panel contains an invalid date.")
    if ((result["date"] < LEARNING_SAMPLE_START) |
            (result["date"] > LEARNING_SAMPLE_END)).any():
        raise ValueError("Dates must fall within the 2007–2011 learning sample.")
    if result.duplicated(["sid", "date"]).any():
        raise ValueError("Stock panel contains duplicate Sid/date observations.")

    for field in ("close", "volume"):
        result[field] = pd.to_numeric(result[field], errors="coerce")
        if not result[field].map(pd.notna).all():
            raise ValueError(f"Stock panel contains a missing or nonnumeric {field}.")
        if not np.isfinite(result[field].to_numpy(dtype=float)).all():
            raise ValueError(f"Stock panel contains a non-finite {field}.")
    if (result["close"] <= 0).any():
        raise ValueError("Close prices must be positive.")
    if (result["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    return result.sort_values(["date", "sid"], kind="stable").reset_index(drop=True)


def validate_liquidity_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate point-in-time monthly adjusted-dollar-volume proxy observations."""
    missing = set(LIQUIDITY_PANEL_COLUMNS).difference(panel.columns)
    if missing:
        raise ValueError(f"Liquidity panel is missing required columns: {sorted(missing)}")
    result = panel.loc[:, LIQUIDITY_PANEL_COLUMNS].copy()
    result["sid"] = result["sid"].astype("string").str.strip()
    if result["sid"].isna().any() or result["sid"].eq("").any():
        raise ValueError("Liquidity panel contains a missing or blank Sid.")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("Liquidity panel contains an invalid date.")
    if ((result["date"] < LEARNING_SAMPLE_START) |
            (result["date"] > LEARNING_SAMPLE_END)).any():
        raise ValueError("Liquidity dates must fall within the 2007–2011 learning sample.")
    if result.duplicated(["sid", "date"]).any():
        raise ValueError("Liquidity panel contains duplicate Sid/date observations.")
    result["average_adjusted_dollar_volume"] = pd.to_numeric(
        result["average_adjusted_dollar_volume"], errors="coerce"
    )
    values = result["average_adjusted_dollar_volume"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Liquidity panel contains a missing or non-finite adjusted dollar-volume proxy.")
    if (values < 0).any():
        raise ValueError("Adjusted dollar-volume proxy cannot be negative.")
    return result.sort_values(["date", "sid"], kind="stable").reset_index(drop=True)


def validate_security_metadata(
    metadata: pd.DataFrame, panel_sids: set[str] | list[str]
) -> pd.DataFrame:
    """Validate a current master snapshot used only to label panel securities."""
    missing = set(SECURITY_METADATA_COLUMNS).difference(metadata.columns)
    if missing:
        raise ValueError(f"Security metadata is missing columns: {sorted(missing)}")
    result = metadata.loc[:, SECURITY_METADATA_COLUMNS].copy()
    result["sid"] = result["sid"].astype("string").str.strip()
    if result["sid"].isna().any() or result["sid"].eq("").any():
        raise ValueError("Security metadata contains a missing or blank Sid.")
    if result["sid"].duplicated().any():
        raise ValueError("Security metadata contains duplicate Sid values.")
    result["symbol"] = result["symbol"].astype("string").str.strip().str.upper()
    result["security_type"] = result["security_type"].astype("string").str.strip()
    panel_sids = {str(sid).strip() for sid in panel_sids}
    unmatched = panel_sids.difference(set(result["sid"].astype(str)))
    if unmatched:
        raise ValueError(f"Security metadata is missing {len(unmatched)} panel Sids.")
    for field in ("date_delisted", "first_price_date", "last_price_date"):
        raw = result[field]
        parsed = pd.to_datetime(raw, errors="coerce")
        if (raw.notna() & parsed.isna()).any():
            raise ValueError(f"Security metadata contains an invalid {field}.")
        result[field] = parsed
    return result.sort_values("sid", kind="stable").reset_index(drop=True)


def file_sha256(path: str | Path) -> str:
    """Hash a file in chunks so large compressed panels need little extra memory."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stock_panel(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a gzipped CSV export and return rows plus reproducibility metadata."""
    path = Path(path)
    panel = validate_stock_panel(pd.read_csv(path, compression="infer"))
    metadata = {
        "source": "QuantRocket US Stock learning bundle (usstock-learn-1d)",
        "file": path.name,
        "sha256": file_sha256(path),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
        "rows": int(len(panel)),
        "securities": int(panel["sid"].nunique()),
        "columns": STOCK_PANEL_COLUMNS,
    }
    return panel, metadata


def load_security_metadata(
    path: str | Path, panel_sids: set[str] | list[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate the companion master snapshot against panel Sids."""
    path = Path(path)
    metadata = validate_security_metadata(pd.read_csv(path, compression="infer"), panel_sids)
    return metadata, {
        "file": path.name,
        "sha256": file_sha256(path),
        "rows": int(len(metadata)),
    }


def load_liquidity_panel(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and hash the point-in-time monthly liquidity export."""
    path = Path(path)
    panel = validate_liquidity_panel(pd.read_csv(path, compression="infer"))
    return panel, {
        "file": path.name,
        "sha256": file_sha256(path),
        "rows": int(len(panel)),
        "securities": int(panel["sid"].nunique()),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
    }
