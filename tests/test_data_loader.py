import pandas as pd
import pytest

from src import data_loader as loader


def prices():
    return pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100.,
                         "Volume": 1000., "Dividends": 0., "Stock Splits": 0.},
                        index=pd.bdate_range("2024-03-04", periods=10))


def test_loader_requests_and_retains_actions(monkeypatch):
    calls = []
    def download(symbol, **kwargs):
        calls.append((symbol, kwargs))
        data = prices()
        if symbol == "AAPL":
            data.loc[data.index[4], "Dividends"] = .25
            data.loc[data.index[6], "Stock Splits"] = 4
        if symbol == "SPY":
            data.loc[data.index[3], "Dividends"] = 1
        return data
    monkeypatch.setattr(loader.yf, "download", download)
    data = loader.get_stock_data("AAPL", as_of="2024-03-15 21:00Z")
    assert all(kwargs["actions"] and kwargs["auto_adjust"] is False for _, kwargs in calls)
    assert data["Dividends"].sum() == .25
    assert data["Stock_Splits"].sum() == 4
    assert data["SPY_Dividends"].sum() == 1


def test_missing_benchmark_session_is_not_silently_dropped(monkeypatch):
    def download(symbol):
        data = prices()
        return data.drop(data.index[4]) if symbol == "SPY" else data
    monkeypatch.setattr(loader, "_download_close", download)
    with pytest.raises(ValueError, match="missing stock sessions"):
        loader.get_stock_data("AAPL")


def test_missing_provider_actions_stops_download(monkeypatch):
    monkeypatch.setattr(loader.yf, "download", lambda *args, **kwargs: prices().drop(columns="Dividends"))
    with pytest.raises(ValueError, match="Corporate-action"):
        loader._download_close("AAPL")
