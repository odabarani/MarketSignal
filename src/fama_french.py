"""Readers for frozen public Fama/French daily factor files."""

from __future__ import annotations

from hashlib import sha256
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def load_daily_risk_free(path: str | Path) -> tuple[pd.Series, dict[str, object]]:
    """Load the daily RF factor as a decimal return with source metadata."""
    path = Path(path)
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV file in {path.name}.")
        text = archive.read(members[0]).decode("utf-8-sig")

    lines = text.splitlines()
    marker = next((i for i, line in enumerate(lines) if "Mkt-RF,SMB,HML,RF" in line), None)
    if marker is None:
        raise ValueError(f"Could not find the daily Fama/French factors in {path.name}.")
    header = next((i for i in range(marker, len(lines))
                   if lines[i].startswith(",") or lines[i].startswith("Date,")), None)
    if header is None:
        raise ValueError(f"Could not find the factor CSV header in {path.name}.")
    end = next((i for i in range(header + 1, len(lines)) if not lines[i].strip()), len(lines))
    frame = pd.read_csv(StringIO("\n".join(lines[header:end])), skipinitialspace=True)
    frame = frame.rename(columns={frame.columns[0]: "Date"})
    if "RF" not in frame.columns:
        raise ValueError(f"Daily RF column is missing from {path.name}.")
    frame["Date"] = pd.to_datetime(
        frame["Date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce"
    )
    frame["RF"] = pd.to_numeric(frame["RF"], errors="coerce")
    frame.loc[frame["RF"] <= -99.99, "RF"] = np.nan
    frame = frame.loc[frame["Date"].notna()].set_index("Date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Daily RF source contains duplicate dates.")
    if frame["RF"].isna().any():
        raise ValueError("Daily RF source contains missing or sentinel values.")
    series = (frame["RF"] / 100.0).astype(float).rename("risk_free_return")
    if not np.isfinite(series.to_numpy()).all() or (series <= -1).any():
        raise ValueError("Daily RF returns must be finite and greater than -100%.")
    return series, {
        "file": path.name,
        "member": members[0],
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "first_date": str(series.index.min().date()),
        "last_date": str(series.index.max().date()),
        "sessions": int(len(series)),
        "field": "RF",
        "units": "decimal daily return (source percent divided by 100)",
    }
