import pandas as pd
import pytest
import json
import sys
import types

from export_quantrocket_stock_sample import (
    bundle_prices_to_panel,
    export,
    link_adjusted_price_chunk,
    point_in_time_monthly_liquidity,
)
from src.stock_panel import (
    load_stock_panel,
    validate_liquidity_panel,
    validate_learning_sample_coverage,
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


def test_learning_sample_coverage_rejects_missing_year_and_partial_year():
    dates = pd.bdate_range("2007-01-01", "2011-12-31")
    panel = pd.DataFrame({"sid": "SPY-SID", "date": dates})
    coverage = validate_learning_sample_coverage(panel, "SPY-SID")
    assert set(coverage["by_year"]) == {"2007", "2008", "2009", "2010", "2011"}
    missing_year = panel.loc[panel.date.dt.year.ne(2009)]
    with pytest.raises(ValueError, match="Incomplete benchmark coverage for 2009"):
        validate_learning_sample_coverage(missing_year, "SPY-SID")
    partial_year = panel.loc[~((panel.date.dt.year == 2008) & (panel.date.dt.month > 6))]
    with pytest.raises(ValueError, match="Incomplete benchmark coverage for 2008"):
        validate_learning_sample_coverage(partial_year, "SPY-SID")
    sparse_year = panel.loc[~((panel.date.dt.year == 2010) & (panel.date.dt.month.isin([3, 4])))]
    with pytest.raises(ValueError, match="Incomplete benchmark coverage for 2010"):
        validate_learning_sample_coverage(sparse_year, "SPY-SID")


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
        "usstock_Symbol": ["OLD", "LIVE"],
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
        "sid": ["SID-A"], "symbol": ["A"], "date_delisted": [None],
        "first_price_date": ["2007-01-03"], "last_price_date": [None],
        "security_type": ["Common Stock"], "sector": ["Technology"],
    })
    with pytest.raises(ValueError, match="missing 1 panel Sids"):
        validate_security_metadata(metadata, {"SID-A", "SID-B"})


def test_liquidity_panel_rejects_duplicate_or_invalid_observations():
    row = {"sid": "SID-A", "date": "2007-01-03", "average_adjusted_dollar_volume": 10_000_000}
    valid = validate_liquidity_panel(pd.DataFrame([row]))
    assert valid.loc[0, "average_adjusted_dollar_volume"] == 10_000_000
    with pytest.raises(ValueError, match="duplicate"):
        validate_liquidity_panel(pd.DataFrame([row, row]))
    with pytest.raises(ValueError, match="non-finite"):
        validate_liquidity_panel(pd.DataFrame([{**row, "average_adjusted_dollar_volume": float("inf")}]))


def test_monthly_liquidity_query_ends_on_signal_date():
    decision = pd.Timestamp("2007-02-01")
    dates = pd.bdate_range(end=decision, periods=20)
    index = pd.MultiIndex.from_tuples(
        [(field, date) for date in dates for field in ("Close", "Volume")],
        names=["Field", "Date"],
    )
    prices = pd.DataFrame({
        "SID-A": [value for _ in dates for value in (10.0, 1_000_000)]
    }, index=index)
    calls = []

    def get_prices(*args, **kwargs):
        calls.append(kwargs)
        return prices

    result = point_in_time_monthly_liquidity(get_prices, [decision])
    assert calls[0]["end_date"] == "2007-02-01"
    assert result.loc[0, "average_adjusted_dollar_volume"] == 10_000_000


def test_yearly_adjustment_basis_is_linked_at_the_boundary():
    prior_date, current_date = pd.Timestamp("2007-12-31"), pd.Timestamp("2008-12-31")
    previous = pd.DataFrame({"sid": ["SID-A"], "date": [prior_date], "close": [100.0], "volume": [1000]})
    current = pd.DataFrame({"sid": ["SID-A"], "date": [current_date], "close": [40.0], "volume": [1000]})
    index = pd.MultiIndex.from_tuples(
        [("Close", prior_date), ("Close", current_date)], names=["Field", "Date"]
    )
    bridge = pd.DataFrame({"SID-A": [50.0, 55.0]}, index=index)
    linked = link_adjusted_price_chunk(previous, current, bridge)
    assert linked.loc[0, "close"] == pytest.approx(110.0)


def test_export_writes_panel_master_snapshot_and_hash_manifest(tmp_path, monkeypatch):
    def get_prices(*args, **kwargs):
        start_date, end_date = kwargs["start_date"], kwargs["end_date"]
        if start_date.endswith("01-01") and end_date.endswith("12-31"):
            date = pd.bdate_range(end=end_date, periods=1)[0]
            index = pd.MultiIndex.from_tuples(
                [("Close", date), ("Volume", date)], names=["Field", "Date"]
            )
            return pd.DataFrame({"SID-A": [10.0, 1_000_000]}, index=index)
        if start_date[:4] != end_date[:4] and start_date.endswith("12-31"):
            dates = pd.to_datetime([start_date, end_date])
            index = pd.MultiIndex.from_tuples(
                [("Close", date) for date in dates], names=["Field", "Date"]
            )
            return pd.DataFrame({"SID-A": [10.0, 11.0]}, index=index)
        dates = pd.bdate_range(end=kwargs["end_date"], periods=20)
        index = pd.MultiIndex.from_tuples(
            [(field, date) for date in dates for field in ("Close", "Volume")],
            names=["Field", "Date"],
        )
        return pd.DataFrame({
            "SID-A": [value for _ in dates for value in (10.0, 1_000_000)]
        }, index=index)

    securities = pd.DataFrame({
        "usstock_Symbol": ["AAA"],
        "usstock_DateDelisted": [None],
        "usstock_FirstPriceDate": ["2007-01-03"],
        "usstock_LastPriceDate": ["2011-12-30"],
        "usstock_SecurityType2": ["Common Stock"],
        "usstock_Sector": ["Technology"],
    }, index=pd.Index(["SID-A"], name="Sid"))
    quantrocket = types.ModuleType("quantrocket")
    quantrocket.get_prices = get_prices
    master = types.ModuleType("quantrocket.master")
    master_calls = []

    def get_securities(*args, **kwargs):
        master_calls.append(kwargs)
        return securities

    master.get_securities = get_securities
    monkeypatch.setitem(sys.modules, "quantrocket", quantrocket)
    monkeypatch.setitem(sys.modules, "quantrocket.master", master)

    output = tmp_path / "sample.csv.gz"
    manifest = export(output)
    assert master_calls[0]["exclude_delisted"] is False
    metadata_path = tmp_path / "sample_security_master.csv.gz"
    liquidity_path = tmp_path / "sample_monthly_liquidity.csv.gz"
    manifest_path = tmp_path / "sample.csv.gz.manifest.json"
    assert output.exists() and metadata_path.exists() and liquidity_path.exists() and manifest_path.exists()
    assert manifest["security_metadata_file"] == metadata_path.name
    assert manifest["rows"] == 5
    assert [chunk["year"] for chunk in manifest["yearly_chunks"]] == list(range(2007, 2012))
    assert manifest["monthly_liquidity_rows"] == 5
    assert manifest["monthly_liquidity_sha256"]
    assert json.loads(manifest_path.read_text())["security_metadata_sha256"]
    exported_panel = pd.read_csv(output, compression="gzip")
    assert exported_panel.sid.nunique() == 1
    year_end_closes = exported_panel.groupby(exported_panel.date.str[:4]).tail(1).close.tolist()
    assert year_end_closes == pytest.approx([10.0, 11.0, 12.1, 13.31, 14.641])
    assert pd.read_csv(metadata_path, compression="gzip").loc[0, "sid"] == "SID-A"
    assert pd.read_csv(liquidity_path, compression="gzip").date.nunique() == 5
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export(output)
