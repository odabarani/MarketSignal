import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from src.research import RESEARCH_TARGET
from src.research_walkforward import fit_temperature, temperature_scale, walk_forward
from src.research_report import render_report


def panel():
    dates = pd.bdate_range("2020-01-01", periods=190)
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB", "CCC"]], names=["Date", "Ticker"])
    n = np.arange(len(index))
    available = pd.Series(dates, index=dates).shift(-5)
    return pd.DataFrame({"x": np.sin(n / 4), RESEARCH_TARGET: (n // 2) % 3 - 1,
                         "Target_Available_At": available.reindex(index.get_level_values("Date")).to_numpy()}, index=index)


def models():
    return {"Prior baseline": DummyClassifier(strategy="prior"),
            "Tree": DecisionTreeClassifier(max_depth=2, min_samples_leaf=8, random_state=42)}


def run(data=None):
    return walk_forward(panel() if data is None else data, ["x"], "2020-08-25", 40, 20, 20, models())


def test_purging_and_complete_nonoverlapping_windows():
    report, predictions = run()
    assert len({f["fold"] for f in report["folds"]}) >= 3
    for fold in report["folds"]:
        assert fold["train_label_end"] < fold["calibration_start"]
        assert fold["calibration_label_end"] < fold["evaluation_start"]
        assert fold["evaluation_end"] <= report["development_end"]
    assert not predictions.duplicated(["candidate", "Date", "Ticker"]).any()
    assert predictions.groupby("candidate").size().nunique() == 1


def test_future_data_cannot_change_development_results():
    data = panel()
    original, before = run(data)
    after_cutoff = data.index.get_level_values("Date") > pd.Timestamp("2020-08-25")
    data.loc[after_cutoff, "x"] = np.inf
    data.loc[after_cutoff, RESEARCH_TARGET] = 99
    changed, after = run(data)
    assert original == changed
    pd.testing.assert_frame_equal(before, after)


def test_evaluation_labels_cannot_change_same_fold_predictions_or_temperature():
    data = panel()
    report, before = run(data)
    fold = report["folds"][0]
    dates = data.index.get_level_values("Date")
    mask = (dates >= fold["evaluation_start"]) & (dates <= fold["evaluation_end"])
    data.loc[mask, RESEARCH_TARGET] = 1
    changed, after = run(data)
    columns = ["P_DOWN", "P_NO_TRADE", "P_UP"]
    pd.testing.assert_frame_equal(before.loc[before.fold == 1, columns], after.loc[after.fold == 1, columns])
    assert [f["temperature"] for f in report["folds"] if f["fold"] == 1] == [
        f["temperature"] for f in changed["folds"] if f["fold"] == 1]


def test_temperature_calms_overconfidence_and_preserves_top_class():
    p = np.tile([.98, .01, .01], (30, 1))
    y = np.tile([-1, 0, 1], 10)
    t = fit_temperature(p, y)
    assert t > 1
    calibrated = temperature_scale(p, t)
    assert np.all(calibrated[:, 0] < p[:, 0])
    np.testing.assert_allclose(calibrated.sum(axis=1), 1)
    np.testing.assert_array_equal(calibrated.argmax(axis=1), p.argmax(axis=1))
    np.testing.assert_allclose(temperature_scale(p, 1), p)
    with pytest.raises(ValueError):
        temperature_scale(p, 0)
    with pytest.raises(ValueError):
        temperature_scale(np.array([[np.nan, .5, .5]]), 1)


def test_incomplete_windows_fail_clearly():
    with pytest.raises(ValueError, match="Not enough"):
        walk_forward(panel(), ["x"], "2020-02-01", 40, 20, 20, models())
    data = panel()
    data.iloc[0, data.columns.get_loc("Target_Available_At")] = data.index[0][0]
    with pytest.raises(ValueError, match="after"):
        run(data)


def test_report_scores_match_predictions_and_escape_labels():
    report, predictions = run()
    from sklearn.metrics import log_loss
    for score in report["summary"]:
        group = predictions.loc[predictions.candidate == score["candidate"]]
        assert score["log_loss"] == pytest.approx(log_loss(
            group[RESEARCH_TARGET] + 1, group[["P_DOWN", "P_NO_TRADE", "P_UP"]], labels=[0, 1, 2]))
        for cls in ["P_DOWN", "P_NO_TRADE", "P_UP"]:
            assert sum(p["count"] for p in report["reliability"]
                       if p["candidate"] == score["candidate"] and p["class"] == cls) == len(group)
    report["config"] = {"tickers": ["<script>bad()</script>"]}
    page = render_report(report)
    assert "<script>bad()</script>" not in page
    assert "&lt;script&gt;" in page
    assert "prefers-reduced-motion" in page
    assert "Plotly.newPlot" in page


def test_lab_routes_without_market_download(monkeypatch, tmp_path):
    import app
    path = tmp_path / "report.json"
    monkeypatch.setenv("MARKETSIGNAL_RESEARCH_REPORT", str(path))
    monkeypatch.setattr(app, "get_stock_data", lambda *_: pytest.fail("Lab must not download prices"))
    client = app.server.test_client()
    assert b"No experiment yet" in client.get("/research-lab").data
    path.write_text(json.dumps(run()[0]))
    response = client.get("/research-lab")
    assert response.status_code == 200
    assert b"Does the signal hold up?" in response.data
    path.write_text("bad JSON")
    assert client.get("/research-lab").status_code == 503


def test_snapshot_cli_is_offline_and_honors_saved_cutoff(monkeypatch, tmp_path):
    import walk_forward_pipeline as cli
    source, output = tmp_path / "source.csv", tmp_path / "result.json"
    panel().to_csv(source)
    source.with_suffix(".manifest.json").write_text(json.dumps({"validation_end": "2020-08-25"}))
    original = source.read_bytes()
    monkeypatch.setattr(cli, "add_panel_features", lambda x: x)
    monkeypatch.setattr(cli, "build_multiclass_models", models)
    monkeypatch.setitem(cli.FEATURE_SETS, "technical", ["x"])
    ledger = tmp_path / "ledger.jsonl"
    cli.main(["--input", str(source), "--output", str(output), "--ledger", str(ledger),
              "--min-train-dates", "40", "--calibration-dates", "20", "--evaluation-dates", "20"])
    result = json.loads(output.read_text())
    assert result["development_end"] == "2020-08-25"
    assert output.with_suffix(".html").exists()
    assert pd.read_csv(output.with_suffix(".predictions.csv")).Date.max() <= "2020-08-25"
    assert source.read_bytes() == original
    assert [json.loads(line)["config"]["stage"] for line in ledger.read_text().splitlines()] == [
        "walk_forward_started", "walk_forward_complete"]
    with pytest.raises(SystemExit):
        cli.main(["--input", str(source), "--output", str(source.with_suffix(".manifest.json"))])


def test_lab_includes_a_saved_example_for_fresh_installs(monkeypatch):
    import app
    monkeypatch.delenv("MARKETSIGNAL_RESEARCH_REPORT", raising=False)
    saved = app.BASE_DIR / "docs/research-lab.json"
    report = json.loads(saved.read_text())
    assert report["development_end"] == "2024-05-23"
    assert len({f["fold"] for f in report["folds"]}) == 13
    assert "Saved experiment" in render_report(report)
