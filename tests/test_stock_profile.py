from types import SimpleNamespace

import pandas as pd

from src import stock_profile as profile


def test_short_company_search_is_local_and_decodes_name(monkeypatch):
    monkeypatch.setattr(profile.yf, "Search", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("A popular company should not trigger remote search")
    ))
    for query in ("nv", "nvidia", "NVDA"):
        matches = profile.search_options(query)
        assert len(matches) == 1
        option = matches[0]
        assert option == {
            "label": "NVIDIA Corporation (NVDA)",
            "value": "NVDA|NVIDIA Corporation",
        }
        assert profile.decode_selection(option["value"]) == ("NVDA", "NVIDIA Corporation")
    assert profile.search_options("", selected="NVDA|NVIDIA Corporation")[0]["value"] == (
        "NVDA|NVIDIA Corporation"
    )


def test_remote_search_filters_non_stocks_and_caches(monkeypatch):
    calls = []

    class FakeSearch:
        def __init__(self, query, **kwargs):
            calls.append((query, kwargs))
            self.quotes = [
                {"symbol": "ANET", "longname": "Arista Networks, Inc.", "quoteType": "EQUITY"},
                {"symbol": "ANET", "longname": "Duplicate", "quoteType": "EQUITY"},
                {"symbol": "BTC-USD", "longname": "Bitcoin", "quoteType": "CRYPTOCURRENCY"},
                {"symbol": "BAD/INPUT", "longname": "Bad", "quoteType": "EQUITY"},
            ]

    monkeypatch.setattr(profile.yf, "Search", FakeSearch)
    monkeypatch.setattr(profile.time, "time", lambda: 1_000_000)
    profile._remote_search.cache_clear()
    try:
        options = profile.search_options("Arista")
        assert options == [{"label": "Arista Networks, Inc. (ANET)",
                            "value": "ANET|Arista Networks, Inc."}]
        assert profile.search_options("Arista") == options
        assert len(calls) == 1
        assert calls[0][1]["timeout"] == 5
    finally:
        profile._remote_search.cache_clear()


def test_decode_selection_rejects_unsafe_symbol_without_inventing_name():
    assert profile.decode_selection("AAPL") == ("AAPL", "Apple Inc.")
    assert profile.decode_selection("XYZ") == ("XYZ", "")
    assert profile.decode_selection("NVDA/../|") == ("", "")
    assert profile.decode_selection(None) == ("", "")


def _summary(period="0m", strong_buy=3, buy=5, hold=2, sell=1, strong_sell=0):
    return pd.DataFrame([{
        "period": period, "strongBuy": strong_buy, "buy": buy,
        "hold": hold, "sell": sell, "strongSell": strong_sell,
    }])


def test_analyst_snapshot_aggregates_current_month_and_verifies_currency(monkeypatch):
    fake = SimpleNamespace(
        recommendations_summary=pd.concat([
            _summary("-1m", 20, 20, 0, 0, 0), _summary("0m")
        ], ignore_index=True),
        analyst_price_targets={"low": 100, "mean": 140, "high": 180},
        fast_info=SimpleNamespace(currency="usd"),
    )
    monkeypatch.setattr(profile.yf, "Ticker", lambda symbol: fake)
    snapshot = profile.get_analyst_snapshot(" nvda ")
    assert snapshot["symbol"] == "NVDA"
    assert snapshot["period"] == "0m"
    assert snapshot["votes"] == {"buy": 8, "hold": 2, "sell": 1}
    assert snapshot["price_targets"] == {"low": 100.0, "mean": 140.0, "high": 180.0}
    assert snapshot["currency"] == "USD"
    assert snapshot["errors"] == {}
    assert snapshot["checked_at"].endswith("+00:00")


def test_missing_current_month_does_not_hide_independent_targets(monkeypatch):
    fake = SimpleNamespace(
        recommendations_summary=_summary("-1m"),
        analyst_price_targets={"low": 90, "mean": 100, "high": 120},
        fast_info=SimpleNamespace(currency="EUR"),
    )
    monkeypatch.setattr(profile.yf, "Ticker", lambda symbol: fake)
    snapshot = profile.get_analyst_snapshot("SAP")
    assert snapshot["votes"] is None
    assert snapshot["period"] is None
    assert "recommendations" in snapshot["errors"]
    assert snapshot["price_targets"]["mean"] == 100.0
    assert snapshot["currency"] == "EUR"


def test_missing_targets_do_not_hide_current_votes_or_assume_currency(monkeypatch):
    class BrokenTargets:
        recommendations_summary = _summary()

        @property
        def analyst_price_targets(self):
            raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(profile.yf, "Ticker", lambda symbol: BrokenTargets())
    snapshot = profile.get_analyst_snapshot("NVDA")
    assert snapshot["votes"] == {"buy": 8, "hold": 2, "sell": 1}
    assert snapshot["price_targets"] is None
    assert snapshot["currency"] is None
    assert "price_targets" in snapshot["errors"]


def test_invalid_counts_and_inconsistent_targets_are_not_displayed(monkeypatch):
    fake = SimpleNamespace(
        recommendations_summary=_summary(strong_buy=float("nan")),
        analyst_price_targets={"low": 180, "mean": 140, "high": 100},
    )
    monkeypatch.setattr(profile.yf, "Ticker", lambda symbol: fake)
    snapshot = profile.get_analyst_snapshot("NVDA")
    assert snapshot["votes"] is None
    assert snapshot["price_targets"] is None
    assert snapshot["currency"] is None
    assert set(snapshot["errors"]) == {"recommendations", "price_targets"}


def test_analyst_provider_failure_returns_unavailable_data(monkeypatch):
    def unavailable(symbol):
        raise ConnectionError("Provider offline")

    monkeypatch.setattr(profile.yf, "Ticker", unavailable)
    snapshot = profile.get_analyst_snapshot("NVDA")
    assert snapshot["votes"] is None
    assert snapshot["price_targets"] is None
    assert snapshot["currency"] is None
    assert set(snapshot["errors"]) == {"recommendations", "price_targets"}
