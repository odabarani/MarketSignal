import sqlite3

import numpy as np
import pytest

from src.reports import ReportStore, json_safe
from src.snapshots import SnapshotStore


def test_report_survives_restart_and_is_not_mutated(tmp_path):
    path = tmp_path / "snapshots.sqlite"
    report = {"settings": {"ticker": "AAPL", "threshold": .6}, "metric": .25}
    store = ReportStore(SnapshotStore(path))
    identity = store.save(report)
    report["settings"]["threshold"] = .8
    retained = ReportStore(SnapshotStore(path)).load(identity)
    assert retained["settings"]["threshold"] == .6
    assert store.save(retained) == identity


def test_report_integrity_check(tmp_path):
    snapshots = SnapshotStore(tmp_path / "snapshots.sqlite")
    store = ReportStore(snapshots)
    identity = store.save({"metric": .25})
    with sqlite3.connect(snapshots.path) as connection:
        connection.execute("UPDATE reports SET payload = '{}' ")
    with pytest.raises(ValueError, match="integrity"):
        store.load(identity)


def test_undefined_metrics_use_json_null():
    assert json_safe({"metrics": [np.float64(.5), np.nan, np.inf, -np.inf],
                      "sample": np.int64(3)}) == {
        "metrics": [.5, None, None, None], "sample": 3,
    }
