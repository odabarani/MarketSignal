from zipfile import ZipFile

import pandas as pd
import pytest

from src.fama_french import load_daily_risk_free


def make_factor_zip(path, rf_rows):
    text = (
        "Daily Factors\n\n"
        ",Mkt-RF,SMB,HML,RF\n"
        + "\n".join(rf_rows)
        + "\n\nCopyright\n"
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors_daily.csv", text)


def test_daily_rf_loader_converts_percent_to_decimal_and_hashes_source(tmp_path):
    path = tmp_path / "factors.zip"
    make_factor_zip(path, [
        "20070102,0.10,0.20,0.30,0.01",
        "20070103,-0.10,0.00,0.05,0.02",
    ])
    returns, metadata = load_daily_risk_free(path)
    assert returns.iloc[0] == pytest.approx(0.0001)
    assert returns.iloc[1] == pytest.approx(0.0002)
    assert metadata["sha256"]
    assert metadata["sessions"] == 2


def test_daily_rf_loader_rejects_missing_sentinels(tmp_path):
    path = tmp_path / "factors.zip"
    make_factor_zip(path, ["20070102,0.10,0.20,0.30,-99.99"])
    with pytest.raises(ValueError, match="missing or sentinel"):
        load_daily_risk_free(path)
