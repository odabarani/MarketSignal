import json
from hashlib import sha256
from pathlib import Path
from datetime import datetime as RealDateTime

import pandas as pd
import pytest

from paper_signal import load_universe_config, main, reconcile_signals
from src.return_research import POLICY


def prepared_panel(periods=8):
    dates = pd.bdate_range("2026-01-05", periods=periods)
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["Date", "Ticker"])
    panel = pd.DataFrame(index=index)
    panel["Open"] = [100.0, 100.0, 102.0, 106.0, 104.0, 105.0, 110.0, 112.0][:periods]
    return panel, dates


def saved_signal(signal_date, horizon=5):
    return {"signal_id": "signal-1", "as_of": str(signal_date.date()),
            "horizon_sessions": horizon, "policy": {"horizon": horizon},
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


def test_reconcile_waits_for_the_horizon_saved_with_the_signal(tmp_path):
    panel, dates = prepared_panel()

    assert reconcile(panel.loc[:dates[6]], [saved_signal(dates[0], horizon=6)], set(),
                     tmp_path / "signals.jsonl", tmp_path / "snapshots") == 0


def test_reconcile_uses_saved_horizon_if_current_policy_differs(tmp_path):
    panel, dates = prepared_panel()
    signal = saved_signal(dates[0], horizon=2)
    ledger = tmp_path / "signals.jsonl"
    snapshots = tmp_path / "snapshots"

    assert reconcile(panel.loc[:dates[2]], [signal], set(), ledger, snapshots) == 0
    assert reconcile(panel.loc[:dates[3]], [signal], set(), ledger, snapshots) == 1
    event = json.loads(ledger.read_text().splitlines()[0])

    assert event["exit_open_date"] == str(dates[3].date())
    assert event["outcomes"][0]["actual_return"] == pytest.approx(0.06)


def test_reconcile_rejects_inconsistent_saved_horizons(tmp_path):
    panel, dates = prepared_panel()
    signal = saved_signal(dates[0], horizon=2)
    signal["policy"]["horizon"] = 5

    with pytest.raises(ValueError, match="invalid or inconsistent forecast horizon"):
        reconcile(panel, [signal], set(), tmp_path / "signals.jsonl", tmp_path / "snapshots")


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


def test_universe_config_returns_fixed_roster_and_content_hash(tmp_path):
    config = tmp_path / "universe.json"
    raw = ('{"universe_id":"sample-v1","selected_on":"2026-10-14",'
           '"selection_method":"fixed prospective sample","tickers":["AAA","BBB"],'
           '"sector_groups":{"Sector A":["AAA"],"Sector B":["BBB"]}}')
    config.write_text(raw)

    tickers, metadata = load_universe_config(config)

    assert tickers == ["AAA", "BBB"]
    assert metadata["id"] == "sample-v1"
    assert metadata["selected_on"] == "2026-10-14"
    assert metadata["sector_count"] == 2
    assert metadata["sha256"] == sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize("tickers", [["AAA", "AAA"], ["aaa"], ["AAA", ""]])
def test_universe_config_rejects_mutable_or_malformed_rosters(tmp_path, tickers):
    config = tmp_path / "universe.json"
    config.write_text(json.dumps({"universe_id": "sample-v1", "selected_on": "2026-10-14",
                                  "selection_method": "fixed prospective sample",
                                  "tickers": tickers,
                                  "sector_groups": {"Sector": tickers}}))

    with pytest.raises(ValueError, match="ticker"):
        load_universe_config(config)


def test_universe_config_rejects_sector_partition_mismatch(tmp_path):
    config = tmp_path / "universe.json"
    config.write_text(json.dumps({"universe_id": "sample-v1", "selected_on": "2026-10-14",
                                  "selection_method": "fixed prospective sample",
                                  "tickers": ["AAA", "BBB"],
                                  "sector_groups": {"Sector A": ["AAA"]}}))

    with pytest.raises(ValueError, match="partition"):
        load_universe_config(config)


def test_paper_signal_reports_public_data_failure_without_writing_a_signal(
        tmp_path, monkeypatch, capsys):
    import paper_signal

    def fail_download(_tickers):
        raise ValueError("No price data was returned for SPY.")

    monkeypatch.setattr(paper_signal, "get_research_market_data", fail_download)
    root = Path(__file__).resolve().parents[1]
    ledger = tmp_path / "stock-paper" / "signals.jsonl"

    with pytest.raises(SystemExit) as error:
        main(["--universe-file", str(root / "configs/paper-stock-universe-v1.json"),
              "--ledger", str(ledger), "--snapshot-dir", str(tmp_path / "snapshots"),
              "--reconciliation-snapshot-dir", str(tmp_path / "reconciliations")])

    assert error.value.code == 2
    assert "Could not load complete public price data" in capsys.readouterr().err
    assert not ledger.exists()


def test_late_or_stale_latest_session_does_not_create_a_backdated_signal(
        tmp_path, monkeypatch, capsys):
    import paper_signal

    class FixedDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            current = RealDateTime(2026, 10, 14, 21, 5)
            return current.replace(tzinfo=tz)

    monkeypatch.setattr(paper_signal, "datetime", FixedDateTime)
    stale_date = pd.Timestamp("2026-10-13")
    frame = pd.DataFrame({
        "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0],
        "Volume": [1000.0], "VIX": [20.0], "SPY_Open": [500.0], "SPY_Close": [501.0],
    }, index=pd.DatetimeIndex([stale_date], name="Date"))
    monkeypatch.setattr(paper_signal, "get_research_market_data", lambda _tickers: {"AAA": frame})
    monkeypatch.setattr(paper_signal, "prepare_return_panel", lambda panel, *_args: panel)
    root = Path(__file__).resolve().parents[1]
    ledger = tmp_path / "stock-paper" / "signals.jsonl"
    snapshots = tmp_path / "snapshots"

    main(["--universe-file", str(root / "configs/paper-stock-universe-v1.json"),
          "--ledger", str(ledger), "--snapshot-dir", str(snapshots),
          "--reconciliation-snapshot-dir", str(tmp_path / "reconciliations")])

    output = capsys.readouterr().out
    assert "Latest complete session: 2026-10-13" in output
    assert "latest complete session (2026-10-13) is not today (2026-10-14)" in output
    assert "refusing to backdate a next-open signal" in output
    assert not ledger.exists()
    assert not snapshots.exists()
