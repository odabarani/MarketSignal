import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from src.portfolio import BENCHMARK, portfolio_backtest
from src.rebalancing import rebalance_backtest
from src.research import build_research_panel
from src.return_research import (AVAILABLE, POLICY, RETURN_TARGET, allocation_rules,
                                 prepare_return_panel, return_walk_forward)


def prices(periods=200):
    dates = pd.bdate_range("2020-01-01", periods=periods)
    n = np.arange(periods)
    result = {}
    for offset, symbol in enumerate(["AAA", "BBB", "CCC"]):
        close = 100 * np.exp(.001 * n + .03 * np.sin(n / 6 + offset))
        result[symbol] = pd.DataFrame({"Open": close * (1 + .001 * np.cos(n)),
            "Close": close, "High": close * 1.02, "Low": close * .98,
            "Volume": 100000 + 1000 * np.sin(n / 9), "VIX": 20 + np.sin(n / 8),
            "SPY_Open": 100 * np.exp(.0005 * n), "SPY_Close": 100 * np.exp(.0005 * n + .001)}, index=dates)
    return build_research_panel(result)


def flat():
    dates = pd.bdate_range("2020-01-01", periods=12)
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["Date", "Ticker"])
    return pd.DataFrame({"Open": 100., "Close": 100., "SPY_Open": 100., "SPY_Close": 100.}, index=index)


def models():
    from src.return_research import StableRidgeRegressor
    return {"Historical mean": DummyRegressor(), "Tiny ridge": StableRidgeRegressor(alpha=100.)}


def run(data=None):
    data = prices() if data is None else data
    cutoff = data.index.get_level_values("Date").unique()[-1]
    return return_walk_forward(data, cutoff, 40, 20, 20, models(), ["Momentum"])


def test_retaining_positions_pays_only_initial_and_final_costs():
    data = flat()
    dates = data.index.get_level_values("Date").unique()
    weights = pd.DataFrame({"AAA": [1., 1., 1.]}, index=dates[[0, 3, 6]])
    result = rebalance_backtest(data, weights, dates[10], cost_bps=15)
    assert result.metrics["orders"] == 2
    assert result.trades.side.tolist() == ["BUY", "SELL"]
    assert result.equity.iloc[-1] == pytest.approx(10000 / 1.0015 * .9985)
    assert result.equity.iloc[0] - result.equity.iloc[-1] == pytest.approx(result.trades.fee.sum())
    repeated = portfolio_backtest(data, weights, horizon=3, cost_bps=15)
    assert result.equity.iloc[-1] > repeated.equity.iloc[-1]


def test_overnight_gain_and_final_open_are_preserved():
    data = flat()
    dates = data.index.get_level_values("Date").unique()
    data.loc[(dates[3], "AAA"), ["Open", "Close"]] = [110, 120]
    data.loc[(dates[4], "AAA"), "Open"] = 130
    weights = pd.DataFrame({"AAA": [1.]}, index=dates[:1])
    result = rebalance_backtest(data, weights, dates[4], cost_bps=0)
    assert result.equity.loc[dates[3]] == pytest.approx(12000)
    assert result.equity.iloc[-1] == pytest.approx(13000)
    assert result.trades.iloc[-1].reference_open == 130


def test_switches_and_partial_positions_never_borrow_cash():
    data = flat()
    dates = data.index.get_level_values("Date").unique()
    weights = pd.DataFrame({"AAA": [.8, .2, 0.], "BBB": [.2, .8, 0.]}, index=dates[[0, 3, 6]])
    result = rebalance_backtest(data, weights, dates[10], cost_bps=50)
    assert (result.trades.cash_after_rebalance >= 0).all()
    assert result.equity.iloc[-1] == pytest.approx(10000 - result.trades.fee.sum())
    assert (result.equity <= 10000).all()


def test_missing_held_prices_invalid_weights_and_bad_terminal_dates_fail():
    data = flat()
    dates = data.index.get_level_values("Date").unique()
    weights = pd.DataFrame({"AAA": [1.]}, index=dates[:1])
    data.loc[(dates[3], "AAA"), "Close"] = np.nan
    with pytest.raises(ValueError, match="held asset"):
        rebalance_backtest(data, weights, dates[6])
    with pytest.raises(ValueError, match="sum to at most one"):
        rebalance_backtest(data, weights * 1.1, dates[6])
    with pytest.raises(ValueError, match="before final liquidation"):
        rebalance_backtest(data, weights, dates[1])


def test_open_to_open_target_uses_next_open_and_sixth_open_availability():
    data = prices()
    cutoff = data.index.get_level_values("Date").unique()[-1]
    prepared = prepare_return_panel(data, cutoff)
    asset = prepared.xs("AAA", level="Ticker")
    assert asset[RETURN_TARGET].iloc[0] == pytest.approx(asset.Open.iloc[6] / asset.Open.iloc[1] - 1)
    assert asset[AVAILABLE].iloc[0] == asset.index[6]
    assert asset[RETURN_TARGET].tail(6).isna().all()


def test_future_prices_are_sliced_before_feature_and_target_computation():
    data = prices()
    cutoff = data.index.get_level_values("Date").unique()[-21]
    before = prepare_return_panel(data, cutoff)
    data.loc[data.index.get_level_values("Date") > cutoff, ["Open", "Close", "SPY_Open", "SPY_Close"]] = np.inf
    pd.testing.assert_frame_equal(before, prepare_return_panel(data, cutoff))


def test_stored_feature_columns_are_not_trusted_as_point_in_time_inputs():
    from src.features import TECHNICAL_FEATURES
    from src.return_research import RETURN_FEATURES
    data = prices()
    cutoff = data.index.get_level_values("Date").unique()[-1]
    before = prepare_return_panel(data, cutoff)
    data[list(set(TECHNICAL_FEATURES).difference({"VIX"}))] = np.inf
    after = prepare_return_panel(data, cutoff)
    pd.testing.assert_frame_equal(before[RETURN_FEATURES], after[RETURN_FEATURES])


def test_stable_ridge_matches_standard_ridge_and_rejects_numerical_errors():
    import warnings
    from src.return_research import StableRidgeRegressor
    x = np.array([[1, 2], [2, 1], [3, 5], [5, 4], [6, 8]], dtype=float)
    y = np.array([.1, .3, .5, .8, .9])
    reference = Ridge(alpha=100, solver="svd").fit(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        stable = StableRidgeRegressor().fit(x, y)
        predicted = stable.predict(x)
    np.testing.assert_allclose(stable.coef_, reference.coef_, atol=1e-12)
    assert stable.intercept_ == pytest.approx(reference.intercept_, abs=1e-12)
    np.testing.assert_allclose(predicted, reference.predict(x), atol=1e-12)
    with pytest.raises(ValueError, match="feature count"):
        stable.predict(x[:, :1])
    with pytest.raises(ValueError):
        StableRidgeRegressor().fit(x, [float("nan")] * len(y))


def test_selection_and_refitting_purge_labels_at_both_boundaries():
    report, forecasts, results = run()
    assert len(report["folds"]) > 2
    for fold in report["folds"]:
        assert fold["train_label_end"] < fold["selection_start"]
        assert fold["selection_liquidation"] < fold["evaluation_start"]
        assert fold["refit_label_end"] < fold["evaluation_start"]
    assert not forecasts.duplicated(["candidate", "Date", "Ticker"]).any()
    reference = next(iter(results.values())).equity.index
    assert all(result.equity.index.equals(reference) for result in results.values())
    assert report["end"] <= report["folds"][-1]["evaluation_end"]


def test_evaluation_returns_cannot_choose_their_own_model():
    data = prices()
    report, forecasts, _ = run(data)
    fold = report["folds"][0]
    dates = data.index.get_level_values("Date")
    mask = (dates >= fold["evaluation_start"]) & (dates <= fold["evaluation_end"])
    # Future opening prices change outcomes, while this fixture's chosen feature
    # is the close-based momentum and stays observable and unchanged.
    data.loc[mask, "Open"] *= 1.5
    changed, after, _ = run(data)
    assert report["folds"][0] == changed["folds"][0]
    pd.testing.assert_frame_equal(forecasts.loc[forecasts.fold == 1], after.loc[after.fold == 1])


def test_hysteresis_holds_small_positive_forecast_and_exits_negative_forecast():
    panel = prepare_return_panel(prices(), "2020-10-06")
    dates = panel.index.get_level_values("Date").unique()[[45, 50, 55]]
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB", "CCC"]], names=["Date", "Ticker"])
    forecasts = pd.DataFrame({"expected_return": [0.01, -1, -1, .001, -1, -1, -.001, -1, -1],
                             "candidate": "Example"}, index=index).reset_index()
    allocation = allocation_rules(panel, forecasts, dates)["Example"]
    assert 0 < allocation.loc[dates[0], "AAA"] <= POLICY["max_weight"]
    assert allocation.loc[dates[1], "AAA"] > 0
    assert allocation.loc[dates[2]].sum() == 0
    for date in dates:
        risk = (allocation.loc[date].drop(BENCHMARK) * panel.loc[date].Risk_Volatility_20).sum()
        assert risk <= POLICY["volatility_target"] + 1e-12


def test_future_outcome_features_and_incomplete_universe_are_rejected():
    data = prices()
    cutoff = data.index.get_level_values("Date").unique()[-1]
    with pytest.raises(ValueError, match="Future outcomes"):
        return_walk_forward(data, cutoff, 40, 20, 20, models(), [RETURN_TARGET])
    with pytest.raises(ValueError, match="complete fixed-universe"):
        prepare_return_panel(data.iloc[1:], cutoff)


def test_invalid_risk_policy_is_rejected_and_cost_stress_keeps_rules_fixed():
    data = prices()
    cutoff = data.index.get_level_values("Date").unique()[-1]
    for field, value in [("max_weight", 1.5), ("horizon", 2.5), ("volatility_target", float("nan"))]:
        with pytest.raises(ValueError):
            return_walk_forward(data, cutoff, 40, 20, 20, models(), ["Momentum"], {field: value})
    report, _, results = run(data)
    for name in {row["candidate"] for row in report["summary"]}:
        low, high = results[(name, 15)], results[(name, 50)]
        assert high.metrics["total_return"] <= low.metrics["total_return"] + 1e-12
        assert high.trades[["date", "symbol", "side"]].equals(low.trades[["date", "symbol", "side"]])


def test_offline_cli_preserves_inputs_records_trials_and_checks_paths(monkeypatch, tmp_path):
    import json
    import return_pipeline as cli
    data = prices()
    source, output = tmp_path / "panel.csv", tmp_path / "result.json"
    ledger = tmp_path / "ledger.jsonl"
    data.to_csv(source)
    cutoff = str(data.index.get_level_values("Date").unique()[-1].date())
    source.with_suffix(".manifest.json").write_text(json.dumps({"validation_end": cutoff}))
    original = source.read_bytes()
    monkeypatch.setattr(cli, "build_return_models", models)
    cli.main(["--input", str(source), "--output", str(output), "--ledger", str(ledger),
              "--min-train-dates", "80", "--selection-dates", "20", "--evaluation-dates", "20"])
    report = json.loads(output.read_text())
    assert report["development_end"] == cutoff
    assert report["config"]["windows"]["min_train_dates"] == 80
    assert report["end"] <= cutoff
    assert len(report["summary"]) == 24
    assert output.with_suffix(".predictions.csv").exists()
    assert output.with_suffix(".trades.csv").exists()
    assert "no live-profitability claim" in output.with_suffix(".html").read_text()
    assert source.read_bytes() == original
    assert [json.loads(line)["config"]["stage"] for line in ledger.read_text().splitlines()] == [
        "return_started", "return_complete"]
    with pytest.raises(SystemExit):
        cli.main(["--input", str(source), "--ledger", str(source)])
    with pytest.raises(SystemExit):
        cli.main(["--input", str(source), "--output", str(source.with_suffix(".manifest.json"))])
    assert source.read_bytes() == original
