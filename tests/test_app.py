import numpy as np
import pandas as pd

import app as app_module


def make_market_data(n=280):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(
        100 + np.arange(n) * 0.05 + np.sin(np.arange(n) / 3) * 4,
        index=idx,
    )
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]) + 0.2,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1_000_000 + (np.arange(n) % 20) * 10_000,
        "VIX": 20 + np.sin(np.arange(n) / 8),
        "SPY_Open": 200 + np.arange(n) * 0.1,
        "SPY_Close": 200 + np.arange(n) * 0.1,
    }, index=idx)


def test_dash_server_and_research_dashboard(monkeypatch):
    monkeypatch.setattr(
        app_module, "get_stock_data", lambda ticker: make_market_data()
    )
    app_module.load_market.cache_clear()
    app_module.run_model_research.cache_clear()

    response = app_module.server.test_client().get("/")
    dashboard = app_module.build_research_dashboard("AAPL", .6, 21, 10, 5)

    assert response.status_code == 200
    assert b"MarketSignal Research" in response.data
    assert dashboard is not None
    assert "Current research snapshot" in str(dashboard)


def test_dash_error_card_for_bad_ticker(monkeypatch):
    monkeypatch.setattr(
        app_module, "get_stock_data",
        lambda ticker: (_ for _ in ()).throw(ValueError("No price data")),
    )
    app_module.load_market.cache_clear()
    app_module.run_model_research.cache_clear()

    result = app_module.update_research(1, "BAD", .6, 21, 10, 5)

    assert "Research run stopped" in str(result)


def test_model_fit_is_reused_for_backtest_setting_changes(monkeypatch):
    monkeypatch.setattr(
        app_module, "get_stock_data", lambda ticker: make_market_data()
    )
    app_module.load_market.cache_clear()
    app_module.run_model_research.cache_clear()

    first = app_module.build_research_dashboard("AAPL", .60, 21, 10, 5)
    second = app_module.build_research_dashboard("AAPL", .70, 21, 20, 10)

    assert first is not None
    assert second is not None
    assert app_module.run_model_research.cache_info().hits == 1


def test_company_name_is_displayed_and_only_symbol_reaches_model(monkeypatch):
    symbols = []
    def prices(symbol):
        symbols.append(symbol)
        return make_market_data()
    monkeypatch.setattr(app_module, "get_stock_data", prices)
    app_module.load_market.cache_clear()
    app_module.run_model_research.cache_clear()
    dashboard = app_module.build_research_dashboard("NVDA|NVIDIA Corporation", .6, 21, 10, 5)
    text = str(dashboard)
    assert symbols == ["NVDA"]
    assert "NVIDIA Corporation" in text
    assert "You are viewing NVIDIA Corporation (NVDA)" in text
    assert "analyst-results" in text


def test_invalid_settings_stop_before_fetch(monkeypatch):
    monkeypatch.setattr(app_module, "get_stock_data", lambda *_: (_ for _ in ()).throw(AssertionError("No fetch expected")))
    for threshold, retrain, cost, slippage in [(None,21,10,5), (.6,21,-1,5), (.6,21,10,float('nan')), (.6,0,10,5)]:
        result = app_module.update_research(1,"NVDA|NVIDIA Corporation",threshold,retrain,cost,slippage)
        assert "Research run stopped" in str(result)
        assert "No fetch expected" not in str(result)


def test_autocomplete_retains_selected_stock_and_finds_nvidia():
    options = app_module.update_stock_options("nv", "AAPL|Apple Inc.")
    assert any(option["value"] == "NVDA|NVIDIA Corporation" for option in options)
    assert any(option["value"] == "AAPL|Apple Inc." for option in options)


def test_analyst_panel_keeps_missing_data_distinct_from_neutral_opinion(monkeypatch):
    monkeypatch.setattr(app_module, "load_analysts", lambda *_: {
        "symbol":"NVDA", "votes": None, "price_targets": None, "currency":None,
        "checked_at":"2026-10-13T12:00:00+00:00", "errors":{"recommendations":"unavailable"},
    })
    result, currency = app_module.update_analysts({"symbol":"NVDA","company":"NVIDIA Corporation"})
    assert "Missing coverage is not a neutral or bearish opinion" in str(result)
    assert "NVIDIA Corporation" in str(result)
    assert currency == "Currency unavailable"


def test_analyst_opinions_are_separate_and_dated():
    result = app_module.analyst_panel({
        "symbol":"NVDA","votes":{"buy":8,"hold":2,"sell":0},
        "price_targets":{"low":100,"mean":140,"high":180},"currency":"USD",
        "checked_at":"2026-10-13T12:00:00+00:00","errors":{},
    }, "NVIDIA Corporation")
    text = str(result)
    assert "Most common view: Buy / strong buy" in text
    assert "not used to calculate the model signal" in text
    assert "Retrieval time is not the date each analyst issued" in text
    assert "Oct 13, 2026 at 12:00 UTC" in text


def test_comparison_uses_the_displayed_stock_and_settings(monkeypatch):
    called = []
    def compare(symbol, days):
        called.append((symbol, days))
        return pd.DataFrame([dict(Model="Example",accuracy=.6,majority_accuracy=.5,roc_auc=.55,brier=.2,log_loss=.6)])
    monkeypatch.setattr(app_module, "run_model_comparison", compare)
    result = app_module.update_comparison(1, {"symbol":"NVDA","company":"NVIDIA Corporation","retrain_days":42})
    assert called == [("NVDA",42)]
    assert "NVIDIA Corporation (NVDA)" in str(result)


def test_cache_expires_in_running_app(monkeypatch):
    from src import cache
    current = [1]
    calls = []
    monkeypatch.setattr(cache.time, "monotonic", lambda: current[0])
    @cache.ttl_cache(seconds=10, maxsize=2)
    def read(symbol):
        calls.append(symbol)
        return len(calls)
    assert read("NVDA") == read("NVDA") == 1
    current[0] = 11
    assert read("NVDA") == 2
