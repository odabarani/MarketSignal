import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from src.portfolio import BENCHMARK, compare_portfolios, portfolio_backtest, strategy_allocations


def market(horizon=2):
    dates = pd.bdate_range("2020-01-01", periods=22)
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["Date", "Ticker"])
    available = pd.Series(dates, index=dates).shift(-horizon)
    return pd.DataFrame({"Open": 100., "Close": 100., "SPY_Open": 100., "SPY_Close": 100.,
                         "Return_20": np.tile([.2, .1], len(dates)),
                         "Target_Available_At": available.reindex(index.get_level_values("Date")).to_numpy(),
                         "Research_Target": np.tile([1, -1], len(dates))}, index=index)


def signals(panel=None):
    panel = market() if panel is None else panel
    rows = panel.reset_index().iloc[:22][["Date", "Ticker", "Research_Target"]]
    rows["P_DOWN"], rows["P_NO_TRADE"], rows["P_UP"] = .15, .15, .7
    rows["candidate"] = "Fixed model"
    return rows


def test_next_open_and_exact_horizon_with_both_execution_fees():
    data = market()
    dates = data.index.get_level_values("Date").unique()
    data.loc[(dates[0], "AAA"), "Close"] = 900  # Decision close is never the entry fill.
    data.loc[(dates[1], "AAA"), "Open"] = 200
    data.loc[(dates[1], "AAA"), "Close"] = 180
    data.loc[(dates[2], "AAA"), "Close"] = 240
    weights = pd.DataFrame({"AAA": [1.]}, index=dates[:1])
    result = portfolio_backtest(data, weights, horizon=2, cost_bps=100)
    shares = 10000 / (200 * 1.01)
    assert result.equity.iloc[0] == 10000
    assert result.equity.iloc[1] == pytest.approx(shares * 180)
    assert result.equity.iloc[2] == pytest.approx(shares * 240 * .99)
    assert result.metrics["max_drawdown"] < -.1  # An intra-trade decline is retained.
    assert result.trades.entry_date.iloc[0] == dates[1]
    assert result.trades.exit_date.iloc[0] == dates[2]
    assert result.equity.iloc[-1] - 10000 == pytest.approx(result.trades.net_pnl.sum())


def test_cash_and_partial_allocation_are_self_financing():
    data = market()
    dates = data.index.get_level_values("Date").unique()
    weights = pd.DataFrame({"AAA": [.5, 0.]}, index=dates[[0, 2]])
    result = portfolio_backtest(data, weights, horizon=2, cost_bps=100)
    expected = 5000 + 5000 / 1.01 * .99
    np.testing.assert_allclose(result.equity.iloc[2:], expected)
    assert result.metrics["trades"] == 1
    assert result.metrics["invested_cohorts"] == 1


def test_reinvestment_uses_realized_nav_and_closes_final_position():
    data = market()
    dates = data.index.get_level_values("Date").unique()
    result = portfolio_backtest(data, pd.DataFrame({"AAA": [1., 1.]}, index=dates[[0, 2]]), 2, 100)
    factor = .99 / 1.01
    assert result.equity.iloc[-1] == pytest.approx(10000 * factor ** 2)
    assert result.trades.entry_nav.iloc[1] == pytest.approx(10000 * factor)
    assert result.equity.iloc[-1] - 10000 == pytest.approx(result.trades.net_pnl.sum())


@pytest.mark.parametrize("weight", [-.1, 1.1, float("nan"), float("inf")])
def test_leverage_shorting_and_invalid_allocations_are_rejected(weight):
    data = market()
    date = data.index.get_level_values("Date").unique()[:1]
    with pytest.raises(ValueError, match="Allocations"):
        portfolio_backtest(data, pd.DataFrame({"AAA": [weight]}, index=date))


def test_missing_prices_and_overlapping_or_incomplete_trades_fail():
    data = market()
    dates = data.index.get_level_values("Date").unique()
    with pytest.raises(ValueError, match="overlap"):
        portfolio_backtest(data, pd.DataFrame({"AAA": [1., 1.]}, index=dates[:2]), 2)
    with pytest.raises(ValueError, match="complete"):
        portfolio_backtest(data, pd.DataFrame({"AAA": [1.]}, index=dates[-1:]), 2)
    data.loc[(dates[2], "AAA"), "Close"] = np.nan
    with pytest.raises(ValueError, match="held asset"):
        portfolio_backtest(data, pd.DataFrame({"AAA": [1.]}, index=dates[:1]), 2)
    data.loc[(dates[2], "AAA"), "Close"] = 100
    data.loc[(dates[0], "AAA"), "SPY_Close"] = 101
    with pytest.raises(ValueError, match="agree"):
        portfolio_backtest(data, pd.DataFrame({BENCHMARK: [1.]}, index=dates[:1]), 2)


def test_weight_decisions_cannot_see_future_outcomes():
    data = market()
    pred = signals(data)
    _, before = strategy_allocations(data, pred, "2020-01-30", horizon=2)
    # Change every outcome and an exit price without changing current features.
    data["Research_Target"] *= -1
    pred["Research_Target"] *= -1
    data.loc[(pred.Date.min() + pd.offsets.BDay(2), "AAA"), "Close"] = 10000
    _, after = strategy_allocations(data, pred, "2020-01-30", horizon=2)
    for name in before:
        pd.testing.assert_frame_equal(before[name], after[name])


def test_single_eligible_stock_is_capped_and_prediction_gaps_stay_cash():
    data = market()
    pred = signals(data)
    pred.loc[pred.Ticker == "BBB", ["P_DOWN", "P_NO_TRADE", "P_UP"]] = [.7, .15, .15]
    dates = pred.Date.unique()
    pred = pred.loc[pred.Date != dates[2]]
    _, allocations = strategy_allocations(data, pred, "2020-01-30", horizon=2)
    model = allocations["Fixed model / fixed rule"]
    assert model.iloc[0].to_dict() == {"AAA": .5, "BBB": 0.}
    assert model.loc[dates[2]].sum() == 0


def test_mismatched_candidates_probabilities_or_horizon_are_rejected():
    data, pred = market(), signals()
    extra = pred.iloc[:-1].assign(candidate="Incomplete model")
    with pytest.raises(ValueError, match="same prediction"):
        strategy_allocations(data, pd.concat([pred, extra]), "2020-01-30", horizon=2)
    pred.loc[0, "P_UP"] = .9
    with pytest.raises(ValueError, match="normalized"):
        strategy_allocations(data, pred, "2020-01-30", horizon=2)
    with pytest.raises(ValueError, match="horizon does not match"):
        strategy_allocations(data, signals(), "2020-01-30", horizon=3)


def test_every_strategy_has_identical_dates_and_cost_stress_degrades_returns():
    summary, results = compare_portfolios(market(), signals(), "2020-01-30", horizon=2)
    assert summary.start.nunique() == summary.end.nunique() == 1
    reference = next(iter(results.values())).equity.index
    for result in results.values():
        assert result.equity.index.equals(reference)
    for name, rows in summary.groupby("strategy"):
        values = rows.sort_values("cost_bps_per_side").total_return.to_numpy()
        assert np.all(np.diff(values) <= 0)
    spy = summary.query("strategy == 'SPY buy & hold' and cost_bps_per_side == 15").iloc[0]
    churn = summary.query("strategy == 'SPY / 2 sessions' and cost_bps_per_side == 15").iloc[0]
    assert spy.total_return > churn.total_return
    assert spy.trades == 1
    assert spy.return_vs_allocation_matched_spy == pytest.approx(0)
    assert churn.return_vs_allocation_matched_spy == pytest.approx(0)


def test_allocation_matched_benchmark_preserves_cash_and_trade_timing():
    data = market()
    data[["SPY_Open", "SPY_Close"]] = data[["Open", "Close"]].to_numpy()
    pred = signals()
    pred.loc[pred.Ticker == "BBB", ["P_DOWN", "P_NO_TRADE", "P_UP"]] = [.7, .15, .15]
    summary, _ = compare_portfolios(data, pred, "2020-01-30", horizon=2)
    model = summary.loc[summary.strategy == "Fixed model / fixed rule"]
    assert (model.mean_allocated_weight == .5).all()
    np.testing.assert_allclose(model.return_vs_allocation_matched_spy, 0, atol=1e-12)


def test_post_cutoff_prices_cannot_change_development_results():
    data, pred = market(), signals()
    cutoff = "2020-01-20"
    before, _ = compare_portfolios(data, pred, cutoff, horizon=2)
    data.loc[data.index.get_level_values("Date") > cutoff, ["Open", "Close", "SPY_Open", "SPY_Close"]] = np.inf
    after, _ = compare_portfolios(data, pred, cutoff, horizon=2)
    pd.testing.assert_frame_equal(before, after)


def test_report_provenance_prevents_mixing_snapshots_and_predictions(tmp_path):
    from sklearn.metrics import log_loss
    from strategy_pipeline import load_verified_inputs
    data, pred = market(), signals()
    source, report = tmp_path / "panel.csv", tmp_path / "walk.json"
    data.to_csv(source)
    source.with_suffix(".manifest.json").write_text(json.dumps({"validation_end": "2020-01-30"}))
    pred.to_csv(report.with_suffix(".predictions.csv"), index=False)
    payload = {"development_end": "2020-01-30", "config": {"source_hash": sha256(source.read_bytes()).hexdigest()},
               "summary": [{"candidate": "Fixed model", "rows": len(pred),
                            "log_loss": log_loss(pred.Research_Target + 1, pred[["P_DOWN", "P_NO_TRADE", "P_UP"]], labels=[0, 1, 2])}]}
    report.write_text(json.dumps(payload))
    load_verified_inputs(source, report)
    pred.loc[0, ["P_DOWN", "P_UP"]] = [.7, .15]
    pred.to_csv(report.with_suffix(".predictions.csv"), index=False)
    with pytest.raises(ValueError, match="saved scores"):
        load_verified_inputs(source, report)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="different price snapshot"):
        load_verified_inputs(source, report)


def test_offline_cli_records_trials_and_preserves_inputs(tmp_path):
    from sklearn.metrics import log_loss
    from strategy_pipeline import main
    data = market(horizon=5)
    pred = signals(data)
    name = "</script><script>alert(1)</script>"
    pred["candidate"] = name
    source, report = tmp_path / "panel.csv", tmp_path / "walk.json"
    output, ledger = tmp_path / "result.json", tmp_path / "ledger.jsonl"
    data.to_csv(source)
    source.with_suffix(".manifest.json").write_text(json.dumps({"validation_end": "2020-01-30"}))
    pred.to_csv(report.with_suffix(".predictions.csv"), index=False)
    report.write_text(json.dumps({"development_end": "2020-01-30",
        "config": {"source_hash": sha256(source.read_bytes()).hexdigest()},
        "summary": [{"candidate": name, "rows": len(pred), "log_loss": log_loss(
            pred.Research_Target + 1, pred[["P_DOWN", "P_NO_TRADE", "P_UP"]], labels=[0, 1, 2])}]}))
    before = source.read_bytes(), report.read_bytes()
    main(["--input", str(source), "--report", str(report), "--output", str(output), "--ledger", str(ledger)])
    saved = json.loads(output.read_text())
    assert saved["config"]["horizon"] == 5
    assert saved["config"]["cost_scenarios"] == [15, 30, 50]
    assert len(saved["summary"]) == 18
    assert "Previously inspected" in output.with_suffix(".html").read_text()
    assert name not in output.with_suffix(".html").read_text()
    assert output.with_suffix(".equity.csv").exists()
    assert output.with_suffix(".trades.csv").exists()
    assert before == (source.read_bytes(), report.read_bytes())
    assert [json.loads(row)["config"]["stage"] for row in ledger.read_text().splitlines()] == [
        "portfolio_started", "portfolio_complete"]
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--report", str(report), "--output", str(report)])
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--report", str(report), "--ledger", str(source)])
    assert before == (source.read_bytes(), report.read_bytes())
