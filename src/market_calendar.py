"""Session checks for the beta's regular US equity-market workflow."""

from functools import lru_cache

import exchange_calendars as xc
import pandas as pd
from src.errors import DataQualityError


@lru_cache(maxsize=16)
def calendar(start, end):
    return xc.get_calendar("XNYS", start=start, end=end)


def session_quality(last_date, as_of=None):
    now = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz="UTC"))
    now = now.tz_localize("UTC") if now.tz is None else now.tz_convert("UTC")
    last = pd.Timestamp(last_date).normalize()
    if last.tz is not None:
        last = last.tz_localize(None)
    if last > now.tz_localize(None).normalize():
        raise DataQualityError("Market history contains future sessions.")
    cal = calendar(str((last - pd.Timedelta(days=10)).date()), str(now.date()))
    completed = cal.schedule.index[cal.schedule["close"] <= now]
    if completed.empty or last > completed[-1]:
        raise DataQualityError("Market history includes an unfinished session.")
    return {"calendar": "XNYS", "latest_completed_session": str(completed[-1].date()),
            "lag_sessions": int((completed > last).sum())}


def validate_us_sessions(index, as_of=None, max_lag=2):
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise DataQualityError("US equity history needs dated sessions.")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise DataQualityError("US equity dates must be valid, unique and increasing.")
    if index.tz is not None or not index.equals(index.normalize()):
        raise DataQualityError("US equity session dates must be timezone-free midnight dates.")
    cal = calendar(str(index.min().date()), str((index.max() + pd.Timedelta(days=1)).date()))
    expected = cal.sessions_in_range(index.min(), index.max())
    missing = expected.difference(index)
    unexpected = index.difference(expected)
    if len(missing) or len(unexpected):
        raise DataQualityError("Stock history has missing or unexpected US market sessions. "
                         "Suspended or differently scheduled assets are unsupported in this beta.")
    quality = session_quality(index.max(), as_of)
    if quality["lag_sessions"] > max_lag:
        raise DataQualityError(f"Market history is {quality['lag_sessions']} completed sessions behind. "
                         "Research stopped until fresh provider data is available.")
    return quality
