import pandas as pd
import pytest
import json
import sys
import types

from export_quantrocket_stock_sample import bundle_prices_to_panel, export
from src.stock_panel import (
    load_stock_panel,
    validate_security_metadata,
    validate_stock_panel,
)


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


def test_quantrocket_export_drops_unobserved_security_dates():
    index = pd.MultiIndex.from_tuples(
        [("Close", "2007-01-03"), ("Volume", "2007-01-03")], names=["Field", "Date"]
    )
    prices = pd.DataFrame({"SID-A": [10.0, 100], "SID-B": [float("nan"), float("nan")]}, index=index)
    panel = bundle_prices_to_panel(prices)
    assert panel.sid.tolist() == ["SID-A"]


def test_quantrocket_master_export_preserves_delisting_labels_by_sid():
    from export_quantrocket_stock_sample import master_to_metadata

    securities = pd.DataFrame({
        "usstock_DateDelisted": ["2009-03-12", None],
        "usstock_FirstPriceDate": ["2007-01-03", "2007-01-03"],
        "usstock_LastPriceDate": ["2009-03-10", "2011-12-30"],
        "usstock_SecurityType2": ["Common Stock", "Common Stock"],
        "usstock_Sector": ["Financials", "Technology"],
    }, index=pd.Index(["SID-DELISTED", "SID-ACTIVE"], name="Sid"))
    result = master_to_metadata(securities, {"SID-DELISTED", "SID-ACTIVE"})
    row = result.loc[result.sid == "SID-DELISTED"].iloc[0]
    assert row.date_delisted == pd.Timestamp("2009-03-12")
    assert row.sector == "Financials"


def test_security_metadata_rejects_missing_panel_sid():
    metadata = pd.DataFrame({
        "sid": ["SID-A"], "date_delisted": [None],
        "first_price_date": ["2007-01-03"], "last_price_date": [None],
        "security_type": ["Common Stock"], "sector": ["Technology"],
    })
    with pytest.raises(ValueError, match="missing 1 panel Sids"):
        validate_security_metadata(metadata, {"SID-A", "SID-B"})


def test_export_writes_panel_master_snapshot_and_hash_manifest(tmp_path, monkeypatch):
    index = pd.MultiIndex.from_tuples(
        [("Close", "2007-01-03"), ("Volume", "2007-01-03")], names=["Field", "Date"]
    )
    prices = pd.DataFrame({"SID-A": [10.0, 100]}, index=index)
    securities = pd.DataFrame({
        "usstock_DateDelisted": [None],
        "usstock_FirstPriceDate": ["2007-01-03"],
        "usstock_LastPriceDate": ["2011-12-30"],
        "usstock_SecurityType2": ["Common Stock"],
        "usstock_Sector": ["Technology"],
    }, index=pd.Index(["SID-A"], name="Sid"))
    quantrocket = types.ModuleType("quantrocket")
    quantrocket.get_prices = lambda *args, **kwargs: prices
    master = types.ModuleType("quantrocket.master")
    master.get_securities = lambda *args, **kwargs: securities
    monkeypatch.setitem(sys.modules, "quantrocket", quantrocket)
    monkeypatch.setitem(sys.modules, "quantrocket.master", master)

    output = tmp_path / "sample.csv.gz"
    manifest = export(output)
    metadata_path = tmp_path / "sample_security_master.csv.gz"
    manifest_path = tmp_path / "sample.csv.gz.manifest.json"
    assert output.exists() and metadata_path.exists() and manifest_path.exists()
    assert manifest["security_metadata_file"] == metadata_path.name
    assert json.loads(manifest_path.read_text())["security_metadata_sha256"]
    assert pd.read_csv(metadata_path).loc[0, "sid"] == "SID-A"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export(output)
