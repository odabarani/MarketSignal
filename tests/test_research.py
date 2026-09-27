import json

import numpy as np
import pandas as pd
import pytest

import src.data_loader as loader
from src.features import add_features
from src.research import (
    DOWN,
    NO_TRADE,
    RESEARCH_TARGET,
    UP,
    add_cost_aware_target,
    build_research_panel,
    chronological_split,
    dataset_fingerprint,
    record_experiment,
)


def make_market_data(n=140, offset=0):
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(
        100 + offset + np.arange(n) * .08 + np.sin(np.arange(n) / 4),
        index=index,
    )
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1_000_000 + np.arange(n) * 1_000,
        "VIX": 20 + np.sin(np.arange(n) / 9),
        "SPY_Open": 200 + np.arange(n) * .1,
        "SPY_Close": 200 + np.arange(n) * .1,
    }, index=index)


def test_cost_aware_target_has_up_down_and_no_trade():
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    frame = pd.DataFrame({
        "Open": 100.0,
        "Close": [100, 102, 98, 100.1, 100],
        "SPY_Open": 100.0,
        "SPY_Close": 100.0,
    }, index=index)

    result = add_cost_aware_target(frame, horizon=1, round_trip_cost_bps=30)

    assert result.loc[index[0], RESEARCH_TARGET] == UP
    assert result.loc[index[1], RESEARCH_TARGET] == DOWN
    assert result.loc[index[2], RESEARCH_TARGET] == NO_TRADE
    assert pd.isna(result.loc[index[-1], RESEARCH_TARGET])
    assert result.loc[index[0], "Target_Available_At"] == index[1]


def test_panel_builds_features_per_ticker_without_cross_asset_rolling():
    markets = {"aaa": make_market_data(offset=0), "BBB": make_market_data(offset=500)}
    panel = build_research_panel(markets, sectors={"AAA": "Tech"})
    expected = add_features(markets["aaa"])

    assert panel.index.names == ["Date", "Ticker"]
    assert set(panel.index.get_level_values("Ticker")) == {"AAA", "BBB"}
    assert panel.xs("AAA", level="Ticker")["MA_20"].iloc[0] == pytest.approx(
        expected["MA_20"].iloc[0]
    )
    assert panel.xs("AAA", level="Ticker")["Sector"].eq("Tech").all()


def test_chronological_split_keeps_dates_disjoint_and_embargoed():
    panel = build_research_panel({
        "AAA": make_market_data(offset=0),
        "BBB": make_market_data(offset=20),
    })
    split = chronological_split(panel)
    train_dates = split.train.index.get_level_values("Date")
    validation_dates = split.validation.index.get_level_values("Date")
    test_dates = split.test.index.get_level_values("Date")

    assert train_dates.max() < validation_dates.min()
    assert validation_dates.max() < test_dates.min()
    assert split.train["Target_Available_At"].max() <= split.train_end
    assert split.validation["Target_Available_At"].max() <= split.validation_end


def test_fingerprint_and_experiment_ledger_are_reproducible(tmp_path):
    panel = build_research_panel({"AAA": make_market_data()})
    fingerprint = dataset_fingerprint(panel)
    changed = panel.copy()
    changed.iloc[0, changed.columns.get_loc("Close")] += 1

    assert dataset_fingerprint(panel) == fingerprint
    assert dataset_fingerprint(changed) != fingerprint

    ledger = tmp_path / "experiments.jsonl"
    record = record_experiment(
        ledger, {"model": "baseline"}, {"accuracy": .5}, panel, "first trial"
    )
    saved = json.loads(ledger.read_text().strip())

    assert saved["experiment_id"] == record["experiment_id"]
    assert saved["dataset_fingerprint"] == fingerprint


def test_multi_ticker_loader_reuses_benchmarks(monkeypatch):
    calls = []

    def fake_download(symbol):
        calls.append(symbol)
        frame = make_market_data(40)
        return frame[["Open", "High", "Low", "Close", "Volume"]]

    monkeypatch.setattr(loader, "_download_close", fake_download)
    result = loader.get_market_data(["aaa", "BBB", "AAA"])

    assert set(result) == {"AAA", "BBB"}
    assert calls.count("^VIX") == 1
    assert calls.count("SPY") == 1
    assert calls.count("AAA") == 1
    assert calls.count("BBB") == 1
