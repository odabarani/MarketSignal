import pandas as pd
import pytest

import src.data_loader as data_loader


def prices(dates):
    return pd.DataFrame(
        {
            "Open": [100.0] * len(dates),
            "High": [101.0] * len(dates),
            "Low": [99.0] * len(dates),
            "Close": [100.0] * len(dates),
            "Volume": [1000.0] * len(dates),
        },
        index=pd.DatetimeIndex(dates),
    )


def test_research_panel_names_symbols_with_stale_last_bars(monkeypatch):
    dates = pd.bdate_range("2026-10-12", periods=3)
    frames = {
        "SPY": prices(dates),
        "^VIX": prices(dates),
        "AAA": prices(dates),
        "BBB": prices(dates[:-1]),
    }

    monkeypatch.setattr(
        data_loader,
        "_download_close",
        lambda symbol, **_kwargs: frames[symbol].copy(),
    )

    with pytest.raises(ValueError, match=r"SPY has data through 2026-10-14.*BBB \(2026-10-13\)"):
        data_loader.get_research_market_data(["AAA", "BBB"], end="2026-10-15")


def test_research_panel_keeps_a_complete_shared_session_calendar(monkeypatch):
    dates = pd.bdate_range("2026-10-12", periods=3)
    frames = {symbol: prices(dates) for symbol in ["SPY", "^VIX", "AAA", "BBB"]}

    monkeypatch.setattr(
        data_loader,
        "_download_close",
        lambda symbol, **_kwargs: frames[symbol].copy(),
    )

    result = data_loader.get_research_market_data(["AAA", "BBB"], end="2026-10-15")

    assert list(result) == ["AAA", "BBB"]
    assert all(frame.index.equals(dates) for frame in result.values())
    assert all(frame[["Open", "Close", "VIX", "SPY_Open", "SPY_Close"]].notna().all().all()
               for frame in result.values())
