import pandas as pd
import pytest

from export_quantrocket_stock_sample import bundle_prices_to_panel
from src.stock_panel import load_stock_panel, validate_stock_panel


def valid_panel():
    return pd.DataFrame({
        "sid": ["SID-B", "SID-A"],
        "date": ["2007-01-03", "2007-01-03"],
        "close": [10.0, 20.0],
        "volume": [100, 0],
    })


def test_validation_normalizes_and_sorts_stock_panel():
    result = validate_stock_panel(valid_panel())
    assert result.sid.tolist() == ["SID-A", "SID-B"]
    assert result.date.is_monotonic_increasing


@pytest.mark.parametrize("change, message", [
    (lambda df: df.assign(close=0), "positive"),
    (lambda df: df.assign(volume=-1), "negative"),
    (lambda df: df.assign(date="2012-01-03"), "2007–2011"),
    (lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True), "duplicate"),
    (lambda df: df.assign(sid=" "), "blank Sid"),
    (lambda df: df.assign(close=float("inf")), "non-finite close"),
])
def test_validation_rejects_bad_observations(change, message):
    with pytest.raises(ValueError, match=message):
        validate_stock_panel(change(valid_panel()))


def test_load_panel_records_hash_and_dimensions(tmp_path):
    path = tmp_path / "sample.csv.gz"
    valid_panel().to_csv(path, index=False, compression="gzip")
    panel, metadata = load_stock_panel(path)
    assert len(panel) == 2
    assert metadata["sha256"]
    assert metadata["securities"] == 2
    assert metadata["first_date"] == "2007-01-03"


def test_quantrocket_wide_prices_become_tidy_sid_date_panel():
    index = pd.MultiIndex.from_tuples(
        [("Close", "2007-01-03"), ("Volume", "2007-01-03"),
         ("Close", "2007-01-04"), ("Volume", "2007-01-04")],
        names=["Field", "Date"],
    )
    prices = pd.DataFrame({"SID-A": [10.0, 100, 11.0, 120], "SID-B": [20.0, 0, 19.0, 50]}, index=index)
    panel = bundle_prices_to_panel(prices)
    assert list(panel.columns) == ["sid", "date", "close", "volume"]
    assert len(panel) == 4
    selected = panel.loc[(panel.sid == "SID-A") & (panel.date == "2007-01-04"), "close"]
    assert selected.iloc[0] == 11


def test_quantrocket_export_rejects_missing_fields():
    index = pd.MultiIndex.from_tuples([("Close", "2007-01-03")], names=["Field", "Date"])
    with pytest.raises(ValueError, match="Close and Volume"):
        bundle_prices_to_panel(pd.DataFrame({"SID-A": [10.0]}, index=index))
