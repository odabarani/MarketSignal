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
