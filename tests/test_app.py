import numpy as np
import pandas as pd
import pytest
from src.snapshots import SnapshotStore
from datetime import date
from datetime import timedelta
import json
import time
import copy
from hashlib import sha256
import subprocess
import sys
from dash import no_update
from src.reports import ReportStore
from src.jobs import JobManager, execute_research

import app as app_module
from src.market_calendar import calendar
from src.replay import replay


def make_market_data(n=280):
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date()
    idx = calendar("2024-01-01", str(end)).sessions[-n:]
    close = pd.Series(
        100 + np.arange(n) * 0.05 + np.sin(np.arange(n) / 3) * 4,
        index=idx,
    )
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]) + 0.2,
        "High": np.maximum(close, close.shift(1).fillna(close.iloc[0]) + 0.2) + 1,
        "Low": np.minimum(close, close.shift(1).fillna(close.iloc[0]) + 0.2) - 1,
        "Close": close,
        "Volume": 1_000_000 + (np.arange(n) % 20) * 10_000,
        "VIX": 20 + np.sin(np.arange(n) / 8),
        "SPY_Open": 200 + np.arange(n) * 0.1,
        "SPY_Close": 200 + np.arange(n) * 0.1,
        "Dividends": 0., "Stock_Splits": 0., "SPY_Dividends": 0., "SPY_Stock_Splits": 0.,
    }, index=idx)


def test_dash_server_and_research_dashboard(monkeypatch):
    monkeypatch.setattr(
        app_module, "get_stock_data", lambda ticker: make_market_data()
    )
    app_module.fit_snapshot.cache_clear()

    response = app_module.server.test_client().get("/")
    dashboard = app_module.build_research_dashboard("AAPL", .6, 21, 10, 5)

    assert response.status_code == 200
    assert b"MarketSignal Research" in response.data
    assert dashboard is not None
    assert "Current research snapshot" in str(dashboard)
    assert "Evidence for this run" in str(dashboard)
    assert "Evaluated predictions" in str(dashboard)


def test_provider_failure_does_not_create_a_result(monkeypatch):
    monkeypatch.setattr(
        app_module, "get_stock_data",
        lambda ticker: (_ for _ in ()).throw(ValueError("No price data")),
    )
    app_module.fit_snapshot.cache_clear()

    with pytest.raises(ValueError, match="No price data"):
        app_module.build_research_dashboard("BAD", .6, 21, 10, 5)


def test_model_fit_is_reused_for_backtest_setting_changes(monkeypatch):
    monkeypatch.setattr(
        app_module, "get_stock_data", lambda ticker: make_market_data()
    )
    app_module.fit_snapshot.cache_clear()

    first = app_module.build_research_dashboard("AAPL", .60, 21, 10, 5)
    second = app_module.build_research_dashboard("AAPL", .70, 21, 20, 10)

    assert first is not None
    assert second is not None
    assert app_module.fit_snapshot.cache_info().hits == 1


def test_initial_page_does_not_download_or_fit(monkeypatch):
    def unexpected(*args):
        raise AssertionError("Initial page must not start research")

    monkeypatch.setattr(app_module, "run_model_research", unexpected)
    client = app_module.server.test_client()
    layout = client.get("/_dash-layout")
    assert layout.status_code == 200
    assert b"Explore the evidence" in layout.data
    dependencies = client.get("/_dash-dependencies").get_json()
    callback = next(item for item in dependencies
                    if "research-job.data" in item["output"])
    assert callback["prevent_initial_call"] is True
    assert app_module.submit_research(0, "AAPL", .6, 21, 10, 5) == (no_update, no_update)


def test_evidence_reports_sample_and_stale_data():
    raw = make_market_data()
    predictions = raw.tail(30)
    last_date = raw.index[-1].date()
    evidence = str(app_module.research_evidence(
        raw, predictions, raw.iloc[-1], as_of=last_date + timedelta(days=10)
    ))
    assert str(last_date) in evidence
    assert str(predictions.index[0].date()) in evidence
    assert "30" in evidence
    assert "not independent trials" in evidence
    assert "These results do not describe today's market" in evidence
    recent = str(app_module.research_evidence(
        raw, predictions, raw.iloc[-1], as_of=last_date
    ))
    assert "not a real-time quote" in recent
    assert "These results do not describe today's market" not in recent


@pytest.fixture(autouse=True)
def isolated_snapshots(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MARKET_STORE", SnapshotStore(tmp_path / "snapshots.sqlite"))
    app_module.fit_snapshot.cache_clear()
    comparison_cache = app_module.run_model_comparison
    comparison_cache.cache_clear()
    yield
    app_module.fit_snapshot.cache_clear()
    comparison_cache.cache_clear()


@pytest.mark.parametrize("settings", [
    (.6, 21, -1, 5), (.6, 21, 10, float("nan")),
    (.9, 21, 10, 5), (.6, 22, 10, 5), (.6, 21, None, 5),
])
def test_invalid_settings_do_not_download(monkeypatch, settings):
    def unexpected(*args):
        raise AssertionError("Invalid settings reached provider")
    monkeypatch.setattr(app_module, "get_stock_data", unexpected)
    with pytest.raises(ValueError):
        app_module.build_research_dashboard("AAPL", *settings)


def test_comparison_uses_displayed_snapshot(monkeypatch):
    monkeypatch.setattr(app_module, "get_stock_data", lambda symbol: make_market_data())
    dashboard = app_module.build_research_dashboard("AAPL", .6, 21, 10, 5)
    context = dashboard.children[0].data
    seen = []
    class RecordingJobs:
        def submit(self, key, task):
            seen.append(task["args"])
            return "comparison-id"
    monkeypatch.setattr(app_module, "JOBS", RecordingJobs())
    result = app_module.submit_comparison(1, dict(context, ticker="MSFT", retrain_days=63))
    assert seen == [(context["snapshot_id"], 21)]
    assert result == ("comparison-id", "")


def test_revised_snapshot_refits_model(monkeypatch):
    calls = []
    def download(symbol):
        frame = make_market_data()
        frame["Volume"] += len(calls)
        calls.append(symbol)
        return frame
    monkeypatch.setattr(app_module, "get_stock_data", download)
    app_module.MARKET_STORE.ttl_seconds = 0
    first = app_module.run_model_research("AAPL", 21)
    second = app_module.run_model_research("AAPL", 21)
    assert first[0].attrs["snapshot"]["id"] != second[0].attrs["snapshot"]["id"]
    assert app_module.fit_snapshot.cache_info().misses == 2
    assert app_module.fit_snapshot.cache_info().hits == 0


def test_download_retains_completed_run_without_provider_or_refit(monkeypatch):
    monkeypatch.setattr(app_module, "get_stock_data", lambda symbol: make_market_data())
    dashboard = app_module.build_research_dashboard("AAPL", .6, 21, 10, 5)
    context = dashboard.children[0].data
    original = ReportStore(app_module.MARKET_STORE).load(context["report_id"])
    def unexpected(*args):
        raise AssertionError("Export must not download or refit")
    monkeypatch.setattr(app_module, "get_stock_data", unexpected)
    monkeypatch.setattr(app_module, "run_model_research", unexpected)
    # Browser state is untrusted; only its report ID selects the retained report.
    edited_context = dict(context, ticker="MSFT", retrain_days=63)
    download, status = app_module.download_report(1, edited_context)
    payload = json.loads(download["content"])
    assert payload["report"] == original
    assert payload["report"]["settings"]["ticker"] == "AAPL"
    assert payload["report"]["settings"]["transaction_cost_bps"] == 10
    assert payload["report"]["settings"]["threshold"] == .6
    assert download["filename"].startswith("MarketSignal-AAPL-")
    assert status == "Report downloaded."
    assert payload["report"]["runtime"]["source_files_sha256"]["app.py"]
    assert "xgboost" in payload["report"]["runtime"]["dependencies"]
    assert "stock_buy_hold" in payload["report"]["metrics"]
    # Exercise the actual Dash transport, not just the Python handler.
    response = app_module.server.test_client().post("/_dash-update-component", json={
        "output": "..report-download.data...export-status.children..",
        "outputs": [{"id": "report-download", "property": "data"},
                    {"id": "export-status", "property": "children"}],
        "inputs": [{"id": "export-button", "property": "n_clicks", "value": 1}],
        "state": [{"id": "research-context", "property": "data", "value": edited_context}],
        "changedPropIds": ["export-button.n_clicks"],
    })
    assert response.status_code == 200
    transported = response.get_json()["response"]["report-download"]["data"]
    assert json.loads(transported["content"]) == payload


@pytest.mark.parametrize("context", [None, {}, {"report_id": "../secret"}, {"report_id": "a" * 64}])
def test_download_missing_report_is_actionable(context):
    download, status = app_module.download_report(1, context)
    assert download is no_update
    assert "research" in str(status)


def test_background_research_through_dash_http(monkeypatch):
    # A current retained snapshot lets the spawned worker run offline without mocks.
    app_module.MARKET_STORE.get("AAPL", lambda symbol: make_market_data())
    manager = JobManager(execute_research, timeout=30)
    monkeypatch.setattr(app_module, "JOBS", manager)
    client = app_module.server.test_client()
    try:
        response = client.post("/_dash-update-component", json={
            "output": "..research-job.data...research-submit-status.children..",
            "outputs": [{"id": "research-job", "property": "data"},
                        {"id": "research-submit-status", "property": "children"}],
            "inputs": [{"id": "run-button", "property": "n_clicks", "value": 1}],
            "state": [{"id": component, "property": "value", "value": value}
                      for component, value in [("ticker", "AAPL"), ("threshold", .6),
                                               ("retrain-days", 21), ("cost-bps", 10), ("slippage-bps", 5)]],
            "changedPropIds": ["run-button.n_clicks"],
        })
        assert response.status_code == 200
        job_id = response.get_json()["response"]["research-job"]["data"]
        deadline = time.monotonic() + 30
        while manager.status(job_id)["state"] in ("queued", "running"):
            assert time.monotonic() < deadline
            time.sleep(.05)
        assert manager.status(job_id)["state"] == "completed"
        response = client.post("/_dash-update-component", json={
            "output": "..research-results.children...research-job-status.children...research-poll.disabled...run-button.disabled..",
            "outputs": [{"id": "research-results", "property": "children"},
                        {"id": "research-job-status", "property": "children"},
                        {"id": "research-poll", "property": "disabled"},
                        {"id": "run-button", "property": "disabled"}],
            "inputs": [{"id": "research-job", "property": "data", "value": job_id},
                       {"id": "research-poll", "property": "n_intervals", "value": 1}],
            "state": [], "changedPropIds": ["research-poll.n_intervals"],
        })
        assert response.status_code == 200
        result = response.get_json()["response"]
        dashboard = result["research-results"]["children"]
        context = dashboard["props"]["children"][0]["props"]["data"]
        assert ReportStore(app_module.MARKET_STORE).load(context["report_id"])["settings"]["ticker"] == "AAPL"
        assert result["research-poll"]["disabled"] is True
        assert result["run-button"]["disabled"] is False
    finally:
        manager.close()


def test_poll_failure_preserves_previous_results_and_reenables_controls(monkeypatch):
    class FailedJobs:
        def status(self, job_id):
            return {"state": "timed_out", "result": "Research exceeded its time limit."}
    monkeypatch.setattr(app_module, "JOBS", FailedJobs())
    result, status, poll_disabled, button_disabled = app_module.poll_research("job", 1)
    assert result is no_update
    assert "time limit" in str(status)
    assert poll_disabled is True and button_disabled is False


def test_stale_retained_history_refreshes_before_model_fit(monkeypatch):
    old = make_market_data()
    old.index = calendar("2020-01-02", "2022-01-01").sessions[:len(old)]
    fresh = make_market_data()
    responses = iter([old, fresh])
    calls = []
    def download(symbol):
        calls.append(symbol)
        return next(responses)
    monkeypatch.setattr(app_module, "get_stock_data", download)
    raw, features = app_module.load_market("AAPL")
    assert calls == ["AAPL", "AAPL"]
    assert raw.index[-1] == fresh.index[-1]


def test_stale_provider_history_cannot_start_research(monkeypatch):
    frame = make_market_data()
    frame.index = calendar("2020-01-02", "2022-01-01").sessions[:len(frame)]
    monkeypatch.setattr(app_module, "get_stock_data", lambda symbol: frame)
    with pytest.raises(ValueError, match="more than two completed"):
        app_module.run_model_research("AAPL", 21)
    assert app_module.fit_snapshot.cache_info().misses == 0


@pytest.fixture
def completed_report(monkeypatch):
    monkeypatch.setattr(app_module, "get_stock_data", lambda symbol: make_market_data())
    dashboard = app_module.build_research_dashboard("AAPL", .6, 21, 10, 5)
    context = dashboard.children[0].data
    download, _ = app_module.download_report(1, context)
    return json.loads(download["content"])


def test_offline_replay_reproduces_export_without_network_or_writes(monkeypatch, completed_report):
    import socket
    def unexpected(*args, **kwargs):
        raise AssertionError("Replay must not access the network")
    monkeypatch.setattr(socket, "create_connection", unexpected)
    monkeypatch.setattr(app_module, "get_stock_data", unexpected)
    before = app_module.MARKET_STORE.path.read_bytes()
    result = replay(completed_report, app_module.MARKET_STORE.path)
    assert result["status"] == "verified"
    assert result["report_id"] == completed_report["report_id"]
    assert app_module.MARKET_STORE.path.read_bytes() == before


@pytest.mark.parametrize("failure", ["checksum", "runtime", "metric", "dataset", "protocol"])
def test_replay_rejects_tampering_and_incompatible_runtime(completed_report, failure):
    envelope = copy.deepcopy(completed_report)
    expected_error = "checksum"
    if failure == "checksum":
        envelope["report"]["settings"]["threshold"] = .7
    else:
        if failure == "runtime":
            envelope["report"]["runtime"]["python_version"] = "0.0.0"
            expected_error = "Runtime or source differs"
        elif failure == "metric":
            envelope["report"]["metrics"]["strategy"]["total_return"] += .1
            expected_error = "numerical mismatch"
        elif failure == "dataset":
            envelope["report"]["dataset"]["id"] = "a" * 64
            expected_error = "snapshot is unavailable"
        else:
            envelope["report"]["evaluation_protocol"]["selection_locked"] = True
            expected_error = "evaluation_protocol"
        envelope["report_id"] = sha256(json.dumps(envelope["report"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match=expected_error):
        replay(envelope, app_module.MARKET_STORE.path)


def test_replay_cli_runs_in_separate_process(completed_report, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(completed_report))
    result = subprocess.run([sys.executable, "-m", "src.replay", "--report", str(path),
                             "--store", str(app_module.MARKET_STORE.path)],
                            cwd=app_module.BASE_DIR, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "verified"
