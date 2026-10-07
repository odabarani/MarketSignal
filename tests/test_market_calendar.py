import pandas as pd
import pytest

from src.market_calendar import session_quality, validate_us_sessions


def test_holidays_and_weekends_do_not_count_as_missing_or_stale():
    index = pd.DatetimeIndex(["2024-07-03", "2024-07-05"])
    assert validate_us_sessions(index, "2024-07-07 20:00Z")["lag_sessions"] == 0
    assert session_quality("2024-07-03", "2024-07-04 23:00Z")["lag_sessions"] == 0


def test_early_close_counts_only_after_actual_close():
    # July 3 closes at 13:00 New York, or 17:00 UTC.
    assert session_quality("2024-07-02", "2024-07-03 16:59Z")["lag_sessions"] == 0
    assert session_quality("2024-07-02", "2024-07-03 17:00Z")["lag_sessions"] == 1


def test_dst_changes_close_time():
    assert session_quality("2024-03-08", "2024-03-11 19:59Z")["lag_sessions"] == 0
    assert session_quality("2024-03-08", "2024-03-11 20:00Z")["lag_sessions"] == 1


def test_missing_stock_session_and_holiday_rows_are_rejected():
    with pytest.raises(ValueError, match="missing or unexpected"):
        validate_us_sessions(pd.DatetimeIndex(["2024-07-01", "2024-07-03"]), "2024-07-03 20:00Z")
    with pytest.raises(ValueError, match="missing or unexpected"):
        validate_us_sessions(pd.DatetimeIndex(["2024-07-03", "2024-07-04", "2024-07-05"]), "2024-07-05 21:00Z")


def test_stale_and_unfinished_data_are_rejected():
    with pytest.raises(ValueError, match="3 completed sessions"):
        validate_us_sessions(pd.DatetimeIndex(["2024-07-05"]), "2024-07-10 21:00Z")
    with pytest.raises(ValueError, match="unfinished"):
        validate_us_sessions(pd.DatetimeIndex(["2024-07-05"]), "2024-07-05 19:00Z")
