import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import src.data_loader


def test_dashboard_renders_with_research_explanations(monkeypatch):
    n = 280
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(
        100 + np.arange(n) * 0.05 + np.sin(np.arange(n) / 3) * 4,
        index=idx,
    )
    raw = pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]) + 0.2,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1_000_000 + (np.arange(n) % 20) * 10_000,
        "VIX": 20 + np.sin(np.arange(n) / 8),
        "SPY_Open": 200 + np.arange(n) * 0.1,
        "SPY_Close": 200 + np.arange(n) * 0.1,
    }, index=idx)
    monkeypatch.setattr(src.data_loader, "get_stock_data", lambda ticker: raw)

    app = AppTest.from_file("app.py", default_timeout=60).run()

    assert not app.exception
    assert not app.error
    assert app.title[0].value == "MarketSignal"
    assert any("Current research snapshot" in item.value for item in app.markdown)
    assert [tab.label for tab in app.tabs] == ["Overview", "Validation", "How it works"]
