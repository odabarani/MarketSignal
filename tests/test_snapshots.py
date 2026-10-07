from datetime import datetime, timedelta, timezone
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.snapshots import SnapshotStore, normalize_symbol, LEGACY_POLICY, POLICY, REQUIRED


def market():
    return pd.DataFrame({
        "Open": [10., 11., 12.], "High": [12., 13., 14.],
        "Low": [9., 10., 11.], "Close": [11., 12., 13.],
        "Volume": [100., 200., 300.], "VIX": [20., 21., 22.],
        "SPY_Open": [100., 101., 102.], "SPY_Close": [101., 102., 103.],
        "Dividends": 0., "Stock_Splits": 0., "SPY_Dividends": 0., "SPY_Stock_Splits": 0.,
    }, index=pd.date_range("2025-01-01", periods=3))


def test_expiry_revisions_and_restart(tmp_path):
    path = tmp_path / "snapshots.sqlite"
    store = SnapshotStore(path, ttl_seconds=60)
    now = datetime(2025, 1, 5, tzinfo=timezone.utc)
    calls = []
    def download(symbol):
        calls.append(symbol)
        frame = market()
        frame.loc[frame.index[-1], "Close"] += len(calls) - 1
        return frame
    first = store.get("aapl", download, now)
    first_id = first.attrs["snapshot"]["id"]
    # A new process/store reuses the persisted snapshot during its freshness window.
    second = SnapshotStore(path, 60).get("AAPL", download, now + timedelta(seconds=59))
    assert len(calls) == 1
    assert second.attrs["snapshot"]["id"] == first_id
    revised = store.get("AAPL", download, now + timedelta(seconds=60))
    assert len(calls) == 2
    assert revised.attrs["snapshot"]["id"] != first_id
    assert store.load(first_id)["Close"].iloc[-1] == 13


def test_unchanged_refresh_keeps_identity(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots.sqlite", 60)
    now = datetime.now(timezone.utc)
    first = store.get("AAPL", lambda symbol: market(), now)
    second = store.get("AAPL", lambda symbol: market(), now + timedelta(seconds=60))
    assert first.attrs == second.attrs


def test_failed_refresh_does_not_silently_use_expired_data(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots.sqlite", 60)
    now = datetime.now(timezone.utc)
    first = store.get("AAPL", lambda symbol: market(), now)
    def fail(symbol):
        raise ConnectionError("provider unavailable")
    with pytest.raises(ConnectionError):
        store.get("AAPL", fail, now + timedelta(seconds=60))
    assert not store.load(first.attrs["snapshot"]["id"]).empty


@pytest.mark.parametrize("failure", ["duplicate", "missing", "nonfinite", "negative", "ohlc"])
def test_rejects_invalid_market_data(tmp_path, failure):
    frame = market()
    if failure == "duplicate":
        frame.index = [frame.index[0]] * 3
    elif failure == "missing":
        frame = frame.drop(columns="SPY_Open")
    elif failure == "nonfinite":
        frame.iloc[0, 0] = np.inf
    elif failure == "negative":
        frame.iloc[0, 0] = -1
    else:
        frame.iloc[0, 1] = 1
    with pytest.raises(ValueError):
        SnapshotStore(tmp_path / "snapshots.sqlite").get("AAPL", lambda symbol: frame)


def test_integrity_check(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots.sqlite")
    frame = store.get("AAPL", lambda symbol: market())
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE snapshots SET payload = '{}' ")
    with pytest.raises(ValueError, match="integrity"):
        store.load(frame.attrs["snapshot"]["id"])


@pytest.mark.parametrize("symbol", ["", "../AAPL", "AAPL;", "A" * 16])
def test_rejects_invalid_tickers(symbol):
    with pytest.raises(ValueError):
        normalize_symbol(symbol)


def test_legacy_snapshot_migration_preserves_identity_and_requires_refresh(tmp_path):
    path = tmp_path / "snapshots.sqlite"
    old_frame = market()[REQUIRED]
    payload = old_frame.to_json(orient="split", date_format="iso", double_precision=15)
    snapshot_id = SnapshotStore.identity("AAPL", payload, LEGACY_POLICY)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE snapshots (id TEXT PRIMARY KEY, symbol TEXT, payload TEXT, acquired_at TEXT)")
        connection.execute("CREATE TABLE latest (symbol TEXT PRIMARY KEY, snapshot_id TEXT, checked_at TEXT)")
        connection.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?)", (snapshot_id, "AAPL", payload, now))
        connection.execute("INSERT INTO latest VALUES (?, ?, ?)", ("AAPL", snapshot_id, now))
    store = SnapshotStore(path)
    before = path.read_bytes()
    readonly = store.load_readonly(snapshot_id)
    assert readonly.attrs["snapshot"]["price_policy"] == LEGACY_POLICY
    assert path.read_bytes() == before
    old = store.load(snapshot_id)
    assert old.attrs["snapshot"]["price_policy"] == LEGACY_POLICY
    calls = []
    def download(symbol):
        calls.append(symbol)
        return market()
    refreshed = store.get("AAPL", download)
    assert calls == ["AAPL"]
    assert refreshed.attrs["snapshot"]["price_policy"] == POLICY
    assert refreshed.attrs["snapshot"]["id"] != snapshot_id
    assert store.load(snapshot_id).attrs["snapshot"]["id"] == snapshot_id
