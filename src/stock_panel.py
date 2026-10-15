"""Load and validate the QuantRocket free daily US stock learning sample."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import numpy as np

STOCK_PANEL_COLUMNS = ["sid", "date", "close", "volume"]
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
