"""Atomic, content-addressed local market snapshots. No credentials are stored."""

from datetime import datetime, timezone
from contextlib import contextmanager
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

SOURCE = "Yahoo Finance via yfinance"
LEGACY_POLICY = "auto_adjust=False; aligned stock/SPY/VIX daily history"
POLICY = "Yahoo split-normalized OHLC; auto_adjust=False; explicit cash dividends; actions=True"
REQUIRED = ["Open", "High", "Low", "Close", "Volume", "VIX", "SPY_Open", "SPY_Close"]
ACTION_COLUMNS = ["Dividends", "Stock_Splits", "SPY_Dividends", "SPY_Stock_Splits"]


def normalize_symbol(symbol):
    symbol = str(symbol or "").strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
        raise ValueError("Enter a stock ticker such as AAPL or BRK-B (up to 15 characters).")
    return symbol


def validate_market(frame):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        raise ValueError("Market data needs a nonempty dated history.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Market dates must be unique and increasing.")
    if frame.index.hasnans or frame.index.tz is not None:
        raise ValueError("Market dates must be valid timezone-free daily sessions.")
    if set(REQUIRED).difference(frame.columns):
        raise ValueError("Market data is missing stock or benchmark columns.")
    values = frame[REQUIRED].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Market data contains missing or nonfinite prices.")
    if (frame[[c for c in REQUIRED if c != "Volume"]] <= 0).any().any():
        raise ValueError("Market prices must be positive.")
    if (frame["Volume"] < 0).any():
        raise ValueError("Market volume cannot be negative.")
    if ((frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1))
            | (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("Market OHLC prices are inconsistent.")
    present = set(ACTION_COLUMNS).intersection(frame.columns)
    if present and present != set(ACTION_COLUMNS):
        raise ValueError("Corporate-action history is incomplete.")
    if present:
        actions = frame[ACTION_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(actions).all() or (actions < 0).any():
            raise ValueError("Corporate actions must contain finite nonnegative values.")


class SnapshotStore:
    def __init__(self, path, ttl_seconds=3600):
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds

    @contextmanager
    def connect(self, timeout=30):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=timeout)
        try:
            with connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY, symbol TEXT NOT NULL, payload TEXT NOT NULL,
                    acquired_at TEXT NOT NULL, policy TEXT)""")
                columns = {row[1] for row in connection.execute("PRAGMA table_info(snapshots)")}
                if "policy" not in columns:
                    connection.execute("ALTER TABLE snapshots ADD COLUMN policy TEXT")
                    connection.execute("UPDATE snapshots SET policy = ? WHERE policy IS NULL", (LEGACY_POLICY,))
                connection.execute("""CREATE TABLE IF NOT EXISTS latest (
                    symbol TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, checked_at TEXT NOT NULL)""")
                yield connection
        finally:
            connection.close()

    def load(self, snapshot_id):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT symbol, payload, acquired_at, policy FROM snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()
        return self._decode(snapshot_id, row)

    def load_readonly(self, snapshot_id):
        """Replay existing stores without initialization, migration or writes."""
        if not self.path.is_file():
            raise ValueError("Retained dataset store is unavailable.")
        connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(snapshots)")}
            if "policy" in columns:
                row = connection.execute(
                    "SELECT symbol, payload, acquired_at, policy FROM snapshots WHERE id = ?", (snapshot_id,)
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT symbol, payload, acquired_at, ? FROM snapshots WHERE id = ?",
                    (LEGACY_POLICY, snapshot_id),
                ).fetchone()
        finally:
            connection.close()
        return self._decode(snapshot_id, row)

    def _decode(self, snapshot_id, row):
        if row is None:
            raise ValueError("Research snapshot is unavailable. Run research again.")
        symbol, payload, acquired_at, policy = row
        if self.identity(symbol, payload, policy) != snapshot_id:
            raise ValueError("Research snapshot failed its integrity check.")
        frame = pd.read_json(StringIO(payload), orient="split")
        validate_market(frame)
        frame.attrs["snapshot"] = {
            "id": snapshot_id, "symbol": symbol, "source": SOURCE,
            "price_policy": policy, "acquired_at": acquired_at,
        }
        return frame

    def check_writable(self):
        # A committed write tests the actual SQLite store, not just file permissions.
        with self.connect(timeout=1) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS readiness_probe (id INTEGER PRIMARY KEY)")
            connection.execute("INSERT OR REPLACE INTO readiness_probe VALUES (1)")
            connection.execute("DELETE FROM readiness_probe WHERE id = 1")

    @staticmethod
    def identity(symbol, payload, policy=POLICY):
        return sha256(json.dumps([1, symbol, SOURCE, policy, payload]).encode()).hexdigest()

    def get(self, symbol, download, now=None, force_refresh=False):
        symbol = normalize_symbol(symbol)
        now = now or datetime.now(timezone.utc)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT latest.snapshot_id, checked_at, policy FROM latest JOIN snapshots "
                "ON snapshots.id = latest.snapshot_id WHERE latest.symbol = ?", (symbol,)
            ).fetchone()
        if row:
            age = (now - datetime.fromisoformat(row[1])).total_seconds()
            if not force_refresh and 0 <= age < self.ttl_seconds and row[2] == POLICY:
                return self.load(row[0])
        # Fail visibly on refresh errors: do not silently serve an expired snapshot.
        frame = download(symbol)
        validate_market(frame)
        # Provider-backed downloads must supply actions; legacy snapshots remain readable.
        if not set(ACTION_COLUMNS) <= set(frame.columns):
            raise ValueError("Download must include stock and benchmark corporate actions.")
        payload = frame[REQUIRED + ACTION_COLUMNS].astype(float).to_json(
            orient="split", date_format="iso", double_precision=15,
        )
        snapshot_id = self.identity(symbol, payload)
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO snapshots (id, symbol, payload, acquired_at, policy) VALUES (?, ?, ?, ?, ?)",
                               (snapshot_id, symbol, payload, now.isoformat(), POLICY))
            connection.execute("INSERT OR REPLACE INTO latest VALUES (?, ?, ?)",
                               (symbol, snapshot_id, now.isoformat()))
        return self.load(snapshot_id)
