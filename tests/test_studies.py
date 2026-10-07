import json
import sqlite3

import pytest

from src import studies
from src.snapshots import SnapshotStore
from test_app import make_market_data


@pytest.fixture
def prepared(tmp_path):
    path = tmp_path / "research.sqlite"
    snapshots = SnapshotStore(path)
    raw = snapshots.get("AAPL", lambda symbol: make_market_data())
    store = studies.StudyStore(path)
    protocol = store.prepare(raw.attrs["snapshot"]["id"])
    return store, protocol["study_id"]


def test_real_validation_and_final_workflow_is_once_only(prepared, monkeypatch):
    store, study_id = prepared
    with pytest.raises(ValueError, match="requires frozen"):
        store.finalize(study_id)
    selected = store.validate(study_id)
    protocol = store.read(study_id)["protocol"]
    assert selected["state"] == "selected"
    assert len(selected["selection"]["trials"]) == 9
    for trial in selected["selection"]["trials"]:
        evaluation = trial["evaluation"]
        assert evaluation["prediction_start"] >= protocol["validation_start"]
        assert evaluation["last_label_available"] <= protocol["validation_end"]
        assert evaluation["prediction_end"] < protocol["final_start"]
    final = store.finalize(study_id)
    assert final["evaluation"]["prediction_start"] == protocol["final_start"]
    assert final["final_period_consumed"] is True
    assert final["selection"] == selected["selection"]
    def unexpected(*args):
        raise AssertionError("Completed final test must not run again")
    monkeypatch.setattr(studies, "evaluate", unexpected)
    assert store.finalize(study_id) == final
    assert store.validate(study_id)["selection"] == selected["selection"]


def test_validation_never_receives_final_period_features(prepared, monkeypatch):
    store, study_id = prepared
    protocol = store.read(study_id)["protocol"]
    calls = []
    def evaluate(features, protocol, candidate, start, **kwargs):
        calls.append(features)
        assert str(features.index.max().date()) <= protocol["validation_end"]
        assert str(features["Target_Available_At"].dropna().max().date()) <= protocol["validation_end"]
        return {"metrics": {"strategy": {"sharpe": 0.}}}
    monkeypatch.setattr(studies, "evaluate", evaluate)
    selected = store.validate(study_id)
    assert len(calls) == 9
    assert selected["selection"]["candidate_index"] == 0  # deterministic tie break


def test_same_snapshot_cannot_start_a_second_selection(prepared):
    store, study_id = prepared
    snapshot_id = store.read(study_id)["protocol"]["dataset"]["id"]
    with pytest.raises(ValueError, match="already has a study"):
        store.prepare(snapshot_id)


def test_failed_final_stage_is_consumed(prepared, monkeypatch):
    store, study_id = prepared
    monkeypatch.setattr(studies, "evaluate", lambda *args, **kwargs: {"metrics": {"strategy": {"sharpe": 0.}}})
    store.validate(study_id)
    def fail(*args):
        raise RuntimeError("worker failed after exposure")
    monkeypatch.setattr(studies, "evaluate", fail)
    with pytest.raises(RuntimeError):
        store.finalize(study_id)
    assert store.read(study_id)["state"] == "failed_final"
    with pytest.raises(ValueError, match="unconsumed"):
        store.finalize(study_id)


def test_frozen_protocol_and_selection_are_integrity_checked(prepared, monkeypatch):
    store, study_id = prepared
    monkeypatch.setattr(studies, "evaluate", lambda *args, **kwargs: {"metrics": {"strategy": {"sharpe": 0.}}})
    store.validate(study_id)
    with sqlite3.connect(store.snapshots.path) as connection:
        connection.execute("UPDATE studies SET selection = ? WHERE id = ?", (json.dumps({"settings": {"threshold": .8}}), study_id))
    with pytest.raises(ValueError, match="integrity"):
        store.finalize(study_id)


def test_stage_claim_is_atomic(prepared):
    store, study_id = prepared
    store.claim(study_id, "prepared", "validating")
    with pytest.raises(ValueError, match="already running"):
        store.claim(study_id, "prepared", "validating")


def test_validation_failure_can_retry_only_the_same_protocol(prepared, monkeypatch):
    store, study_id = prepared
    original = store.read(study_id)["protocol"]
    def fail(*args, **kwargs):
        raise RuntimeError("validation infrastructure failure")
    monkeypatch.setattr(studies, "evaluate", fail)
    with pytest.raises(RuntimeError):
        store.validate(study_id)
    assert store.read(study_id)["state"] == "failed_validation"
    monkeypatch.setattr(studies, "evaluate", lambda *args, **kwargs: {"metrics": {"strategy": {"sharpe": 0.}}})
    assert store.validate(study_id)["state"] == "selected"
    assert store.read(study_id)["protocol"] == original
