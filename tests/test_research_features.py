from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

import research_pipeline
from src.research import build_research_panel, chronological_split
from src.research_experiment import compare_feature_sets, ensure_test_unseen
from src.research_features import CONTEXT_FEATURES, FEATURE_SETS, add_panel_features


def market(n=350, phase=0):
    t = np.arange(n)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(100 + t * .03 + 3 * np.sin((t + phase) / 4), index=index)
    return pd.DataFrame({
        "Open": close.shift().fillna(close.iloc[0]), "High": close + 1,
        "Low": close - 1, "Close": close, "Volume": 1e6 + (t % 17) * 1000,
        "VIX": 20 + np.sin(t / 11), "SPY_Open": 200 + t * .03,
        "SPY_Close": 200 + t * .03,
    }, index=index)


def panel():
    return build_research_panel({"AAA": market(), "BBB": market(phase=6)})


def test_context_is_causal_under_future_data_changes():
    original = panel()
    cutoff = original.index.get_level_values("Date").unique()[150]
    altered = original.copy()
    future = altered.index.get_level_values("Date") > cutoff
    altered.loc[future, ["Close", "MA_5", "MA_20", "SPY_Close", "VIX"]] *= 2
    result = add_panel_features(original)
    mutated = add_panel_features(altered)
    before = result.index.get_level_values("Date") <= cutoff
    pd.testing.assert_frame_equal(result.loc[before, CONTEXT_FEATURES],
                                  mutated.loc[before, CONTEXT_FEATURES])
    prefix = add_panel_features(original.loc[before])
    pd.testing.assert_frame_equal(result.loc[before, CONTEXT_FEATURES], prefix[CONTEXT_FEATURES])


def test_features_invariant_to_asset_price_units():
    raw = market()
    scaled = raw.copy()
    scaled[["Open", "High", "Low", "Close"]] *= 10
    original = add_panel_features(build_research_panel({"AAA": raw}))
    result = add_panel_features(build_research_panel({"AAA": scaled}))
    np.testing.assert_allclose(original[CONTEXT_FEATURES], result[CONTEXT_FEATURES],
                               atol=1e-10, equal_nan=True)


def test_rankings_and_breadth_use_same_date_only():
    raw = panel()
    result = add_panel_features(raw)
    date = result.index.get_level_values("Date").unique()[100]
    day = result.loc[date]
    expected = day["Return_20"].rank(pct=True)
    pd.testing.assert_series_equal(day["Momentum_Rank"], expected, check_names=False)
    assert day["Universe_Breadth"].eq((day.Close > day.MA_20).mean()).all()
    assert day["SPY_Trend_60"].nunique() == 1
    inconsistent = raw.copy()
    inconsistent.loc[(date, "AAA"), "VIX"] += 1
    with pytest.raises(ValueError, match="same SPY and VIX"):
        add_panel_features(inconsistent)


def test_comparisons_use_same_rows_and_ignore_test_values():
    split = chronological_split(add_panel_features(panel()))
    models = {"Prior baseline": DummyClassifier(strategy="prior")}
    comparison, _, _, development = compare_feature_sets(split, list(FEATURE_SETS), models)
    # A poison test frame would fail any feature/target access, but selection never reads it.
    poisoned = replace(split, test=pd.DataFrame({"forbidden": [np.inf]}))
    other, _, _, _ = compare_feature_sets(poisoned, list(FEATURE_SETS), models)
    pd.testing.assert_frame_equal(comparison, other)
    assert comparison.train_rows.nunique() == comparison.validation_rows.nunique() == 1
    assert len(development.train) < len(split.train)
    assert comparison.log_loss.nunique() == 1


def test_fixed_dates_survive_additional_future_rows():
    original = panel()
    split = chronological_split(original)
    expanded = build_research_panel({"AAA": market(380), "BBB": market(380, 6)})
    fixed = chronological_split(expanded, train_end=split.train_end,
                                validation_end=split.validation_end)
    pd.testing.assert_frame_equal(split.train, fixed.train)
    pd.testing.assert_frame_equal(split.validation, fixed.validation)


def test_target_uses_original_sessions_when_a_feature_row_is_dropped():
    raw = market()
    # Missing volume removes feature rows but must not extend the label horizon.
    raw.loc[raw.index[102], "Volume"] = np.nan
    result = build_research_panel({"AAA": raw})
    assert result.loc[(raw.index[100], "AAA"), "Target_Available_At"] == raw.index[105]


@pytest.mark.parametrize("stage,metrics", [("test_reserved", {}),
                                            ("legacy", {"test": {"accuracy": .4}})])
def test_test_ledger_blocks_overlap_across_feature_versions(tmp_path, stage, metrics):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"config": {
        "tickers": ["AAA"], "validation_end": "2024-01-01", "stage": stage,
        "test_end": "2025-01-01",
    }, "metrics": metrics}) + "\n")
    with pytest.raises(ValueError, match="overlapping test"):
        ensure_test_unseen(ledger, ["AAA", "BBB"], "2024-02-01", "2026-01-01")
    ensure_test_unseen(ledger, ["AAA"], "2025-01-01", "2026-01-01")


def test_snapshot_cli_records_validation_without_opening_test(tmp_path, monkeypatch):
    snapshot = tmp_path / "panel.csv"
    original = panel()
    original.to_csv(snapshot)
    split = chronological_split(original)
    snapshot.with_suffix(".manifest.json").write_text(json.dumps({
        "train_end": str(split.train_end.date()),
        "validation_end": str(split.validation_end.date()),
    }))
    def forbidden(*args, **kwargs):
        raise AssertionError("No download or test evaluation allowed")
    monkeypatch.setattr(research_pipeline, "get_market_data", forbidden)
    monkeypatch.setattr(research_pipeline, "evaluate_locked_test", forbidden)
    monkeypatch.setattr(research_pipeline, "build_multiclass_models",
                        lambda: {"Prior baseline": DummyClassifier(strategy="prior")})
    output, ledger = tmp_path / "run.csv", tmp_path / "ledger.jsonl"
    research_pipeline.main(["--input", str(snapshot), "--output", str(output),
                            "--ledger", str(ledger), "--compare-features"])
    report = json.loads(output.with_suffix(".manifest.json").read_text())
    assert "test" not in report
    assert len(report["validation"]) == 3
    assert report["validation_end"] == str(split.validation_end.date())
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [record["config"]["stage"] for record in records] == ["validation_started", "validation"]
    assert records[0]["config"]["source_hash"]
    assert records[0]["config"]["code_fingerprint"]
    assert snapshot.read_text() == original.to_csv()


def test_all_model_parameters_are_valid_json():
    config = research_pipeline.model_configuration(research_pipeline.build_multiclass_models())
    serialized = json.dumps(config, default=str, allow_nan=False)
    assert json.loads(serialized)["XGBoost"]["missing"] == "nan"


def test_production_models_fit_and_score_context_features():
    split = chronological_split(add_panel_features(panel()))
    comparison, chosen, selection, _ = compare_feature_sets(split, ["context"])
    assert chosen == "context"
    assert len(comparison) == 4
    assert np.isfinite(comparison[["log_loss", "brier", "macro_f1"]]).all().all()
    assert isinstance(selection.promoted_over_baseline, bool)
