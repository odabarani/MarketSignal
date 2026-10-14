import json
from hashlib import sha256

import pandas as pd
import pytest

from paper_signal import reconcile_signals
from src.return_research import AVAILABLE, POLICY, RETURN_TARGET


def prepared_panel(expiry_position=6, periods=8):
    dates = pd.bdate_range("2026-01-05", periods=periods)
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["Date", "Ticker"])
    panel = pd.DataFrame(index=index)
    panel[RETURN_TARGET] = float("nan")
    panel[AVAILABLE] = pd.NaT
    panel.loc[(dates[0], "AAA"), RETURN_TARGET] = 0.10
    panel.loc[(dates[0], "AAA"), AVAILABLE] = dates[expiry_position]
    return panel, dates


def saved_signal(signal_date):
    return {"signal_id": "signal-1", "as_of": str(signal_date.date()),
            "forecasts": [{"candidate": "Ridge returns", "ticker": "AAA",
                           "expected_return": 0.05}]}


def reconcile(panel, signal, reconciled, ledger, snapshot_dir):
    source_text = panel.reset_index().to_csv(index=False)
    source_hash = sha256(source_text.encode("utf-8")).hexdigest()
    return reconcile_signals(panel, signal, reconciled, source_hash, ledger,
                             source_text, snapshot_dir)


def test_reconcile_waits_for_the_sixth_subsequent_open(tmp_path):
    panel, dates = prepared_panel()
    signal = saved_signal(dates[0])
    reconciled = set()
    ledger = tmp_path / "signals.jsonl"
    snapshots = tmp_path / "snapshots"

    assert POLICY["horizon"] == 5
    assert reconcile(panel.loc[:dates[5]], [signal], reconciled, ledger, snapshots) == 0
    assert not ledger.exists()
    assert not snapshots.exists()
    assert reconcile(panel.loc[:dates[6]], [signal], reconciled, ledger, snapshots) == 1

    event = json.loads(ledger.read_text().splitlines()[0])
    assert event["exit_open_date"] == str(dates[6].date())
    assert event["outcomes"][0]["actual_return"] == pytest.approx(0.10)
    assert event["forecast_errors"]["Ridge returns"]["mae"] == pytest.approx(0.05)
    snapshot = tmp_path / event["source_snapshot"]
    assert snapshot.is_file()
    assert sha256(snapshot.read_bytes()).hexdigest() == event["source_snapshot_hash"]
    assert event["reconciliation_code_fingerprint"]


def test_reconcile_rejects_a_target_before_its_saved_availability(tmp_path):
    panel, dates = prepared_panel(expiry_position=7)

    with pytest.raises(ValueError, match="marked mature before its target was observable"):
        reconcile(panel.loc[:dates[6]], [saved_signal(dates[0])], set(),
                  tmp_path / "signals.jsonl", tmp_path / "snapshots")


def test_reconcile_is_idempotent_after_an_event_is_written(tmp_path):
    panel, dates = prepared_panel()
    signal = saved_signal(dates[0])
    reconciled = set()
    ledger = tmp_path / "signals.jsonl"
    snapshots = tmp_path / "snapshots"

    assert reconcile(panel.loc[:dates[6]], [signal], reconciled, ledger, snapshots) == 1
    before = ledger.read_bytes()
    snapshots_before = sorted(path.read_bytes() for path in snapshots.glob("*.csv.gz"))
    assert reconcile(panel.loc[:dates[6]], [signal], reconciled, ledger, snapshots) == 0
    assert ledger.read_bytes() == before
    assert sorted(path.read_bytes() for path in snapshots.glob("*.csv.gz")) == snapshots_before


def test_reconcile_rejects_source_text_that_does_not_match_hash(tmp_path):
    panel, dates = prepared_panel()

    with pytest.raises(ValueError, match="must match its saved hash"):
        reconcile_signals(panel.loc[:dates[6]], [saved_signal(dates[0])], set(), "bad-hash",
                          tmp_path / "signals.jsonl", "source", tmp_path / "snapshots")
