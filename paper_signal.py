"""Record five-session forecasts before the next open and reconcile them later."""

import argparse
from datetime import datetime, time, timezone
from gzip import open as gzip_open
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from zoneinfo import ZoneInfo

from research_pipeline import model_configuration
from src.data_loader import get_research_market_data
from src.rebalancing import rebalance_backtest
from src.return_research import (
    AVAILABLE, POLICY, RETURN_FEATURES, RETURN_TARGET, _forecast,
    allocation_rules, build_return_models, prepare_return_panel,
)

DEFAULT_TICKERS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD",
]


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_event(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, allow_nan=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def raw_panel_from_market(market):
    frames = []
    for ticker, frame in market.items():
        item = frame.copy()
        item["Ticker"] = ticker
        frames.append(item.reset_index(names="Date"))
    return pd.concat(frames, ignore_index=True).set_index(["Date", "Ticker"]).sort_index()


def code_fingerprint():
    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted([root / "paper_signal.py", *root.glob("src/*.py"),
                        root / "research_pipeline.py"]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reconcile_signals(prepared, signal_events, reconciled_ids, source_hash, ledger,
                      source_text, snapshot_dir):
    if not isinstance(source_text, str) or sha256(source_text.encode("utf-8")).hexdigest() != source_hash:
        raise ValueError("Reconciliation source text must match its saved hash.")
    dates = prepared.index.get_level_values("Date").unique().sort_values()
    latest = dates[-1]
    new_count = 0
    for signal in signal_events:
        if signal["signal_id"] in reconciled_ids:
            continue
        signal_date = pd.Timestamp(signal["as_of"])
        if signal_date not in dates:
            raise ValueError(f"Saved signal date {signal_date.date()} is absent from the updated history.")
        date_position = dates.get_loc(signal_date)
        expiry_position = date_position + POLICY["horizon"] + 1
        if expiry_position >= len(dates):
            continue
        expiry = dates[expiry_position]
        outcomes = []
        errors = {}
        for forecast in signal["forecasts"]:
            key = (signal_date, forecast["ticker"])
            row = prepared.loc[key]
            if pd.isna(row[AVAILABLE]) or pd.Timestamp(row[AVAILABLE]) > latest:
                raise ValueError("A signal was marked mature before its target was observable.")
            actual = float(row[RETURN_TARGET])
            expected = float(forecast["expected_return"])
            outcomes.append({"candidate": forecast["candidate"], "ticker": forecast["ticker"],
                             "expected_return": expected, "actual_return": actual})
            bucket = errors.setdefault(forecast["candidate"], [])
            bucket.append(actual - expected)
        for candidate, values in errors.items():
            values = np.asarray(values, dtype=float)
            errors[candidate] = {"assets": len(values), "mae": float(np.abs(values).mean()),
                                 "rmse": float(np.sqrt(np.mean(values ** 2)))}
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"{latest:%Y-%m-%d}-{source_hash[:12]}.csv.gz"
        source_bytes = source_text.encode("utf-8")
        if snapshot.exists():
            with gzip_open(snapshot, "rb") as handle:
                if handle.read() != source_bytes:
                    raise ValueError("A different reconciliation source already exists for this hash.")
        else:
            with gzip_open(snapshot, "wb") as handle:
                handle.write(source_bytes)
        snapshot_hash = sha256(snapshot.read_bytes()).hexdigest()
        append_event(ledger, {"event": "paper_reconciled", "signal_id": signal["signal_id"],
                              "signal_date": str(signal_date.date()), "exit_open_date": str(expiry.date()),
                              "observed_through": str(latest.date()), "source_hash": source_hash,
                              "source_snapshot": str(snapshot), "source_snapshot_hash": snapshot_hash,
                              "reconciliation_code_fingerprint": code_fingerprint(),
                              "forecast_errors": errors, "outcomes": outcomes,
                              "created_at_utc": datetime.now(timezone.utc).isoformat()})
        reconciled_ids.add(signal["signal_id"])
        new_count += 1
    return new_count


def portfolio_mark(prepared, signals, as_of):
    """Mark executed paper targets at the latest close without forced liquidation."""
    dates = prepared.index.get_level_values("Date").unique().sort_values()
    eligible = [record for record in signals if pd.Timestamp(record["as_of"]) < as_of]
    if not eligible:
        return None
    candidates = sorted(eligible[0]["allocations"])
    rows = []
    for record in eligible:
        for candidate in candidates:
            weights = record["allocations"].get(candidate)
            if weights is None:
                raise ValueError("Paper allocation candidates changed within one ledger.")
            rows.append((pd.Timestamp(record["as_of"]), candidate, weights))
    metrics = {}
    for candidate in candidates:
        dates_for_candidate = [date for date, name, _ in rows if name == candidate]
        matrices = [weights for _, name, weights in rows if name == candidate]
        frame = pd.DataFrame(matrices, index=pd.DatetimeIndex(dates_for_candidate)).sort_index()
        for cost in [15, 30, 50]:
            result = rebalance_backtest(prepared, frame, liquidation_date=None, cost_bps=cost)
            metrics[f"{candidate}|{cost}bps"] = result.metrics
    return {"event": "paper_portfolio_mark", "observed_through": str(dates[-1].date()),
            "first_decision": str(pd.Timestamp(eligible[0]["as_of"]).date()),
            "last_filled_decision": str(pd.Timestamp(eligible[-1]["as_of"]).date()),
            "liquidation": "none; holdings marked to close",
            "metrics": metrics, "created_at_utc": datetime.now(timezone.utc).isoformat()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--ledger", default="data/paper/signals.jsonl")
    parser.add_argument("--snapshot-dir", default="data/paper/snapshots")
    parser.add_argument("--reconciliation-snapshot-dir", default="data/paper/reconciliation-snapshots")
    parser.add_argument("--input", help="Use a saved current-session CSV snapshot instead of downloading.")
    args = parser.parse_args(argv)
    tickers = list(dict.fromkeys(symbol.strip().upper() for symbol in args.tickers))
    if not tickers or any(not symbol for symbol in tickers):
        parser.error("Provide a fixed, nonempty ticker universe.")

    ledger = Path(args.ledger)
    snapshot_dir = Path(args.snapshot_dir)
    events = read_events(ledger)
    signals = [event for event in events if event.get("event") == "paper_signal"]
    reconciled_ids = {event["signal_id"] for event in events
                      if event.get("event") == "paper_reconciled"}
    if signals and signals[0].get("tickers") != tickers:
        parser.error("Keep the same ticker list for a paper ledger; start a new ledger for another universe.")

    if args.input:
        raw = pd.read_csv(args.input, parse_dates=["Date"]).set_index(["Date", "Ticker"]).sort_index()
        if sorted(raw.index.get_level_values("Ticker").unique()) != sorted(tickers):
            parser.error("Input snapshot tickers must match --tickers.")
    else:
        raw = raw_panel_from_market(get_research_market_data(tickers))
    dates = raw.index.get_level_values("Date").unique().sort_values()
    as_of = dates[-1]
    eastern = datetime.now(ZoneInfo("America/New_York"))
    commit_window = as_of.date() == eastern.date() and eastern.time() >= time(16, 15)
    prepared = prepare_return_panel(raw, as_of, POLICY["horizon"])
    source_text = raw.reset_index().to_csv(index=False, float_format="%.12g")
    source_hash = sha256(source_text.encode("utf-8")).hexdigest()
    reconciled = reconcile_signals(
        prepared, signals, reconciled_ids, source_hash, ledger, source_text,
        args.reconciliation_snapshot_dir,
    )

    if signals:
        last_date = pd.Timestamp(signals[-1]["as_of"])
        if as_of < last_date:
            parser.error("The latest market data predates the last committed paper signal.")
        elapsed_sessions = dates.get_loc(as_of) - dates.get_loc(last_date)
    else:
        elapsed_sessions = POLICY["horizon"]
    created = False
    due = not signals or (as_of > pd.Timestamp(signals[-1]["as_of"])
                          and elapsed_sessions >= POLICY["horizon"])
    if due and commit_window:
        feature_dates = prepared.index.get_level_values("Date")
        complete = np.isfinite(prepared[RETURN_FEATURES].to_numpy(dtype=float)).all(axis=1)
        train_mask = (complete & prepared[RETURN_TARGET].notna() & prepared[AVAILABLE].notna()
                      & (prepared[AVAILABLE] <= as_of) & (feature_dates <= as_of))
        training = prepared.loc[train_mask]
        current = prepared.loc[complete & (feature_dates == as_of)]
        if len(current) != len(tickers) or training.empty:
            parser.error("The latest session needs complete features for every asset and enough mature training history.")

        models = build_return_models()
        forecasts = []
        with threadpool_limits(limits=2):
            for name, model in models.items():
                fitted = clone(model).fit(training[RETURN_FEATURES], training[RETURN_TARGET])
                forecasts.append(_forecast(fitted, current, RETURN_FEATURES, name, "paper"))
        forecasts = pd.concat(forecasts, ignore_index=True)
        prior_holdings = {}
        if signals:
            for candidate, weights in signals[-1]["allocations"].items():
                prior_holdings[candidate] = {symbol for symbol, weight in weights.items()
                                             if symbol in tickers and float(weight) > 0}
        allocations = allocation_rules(prepared, forecasts, pd.DatetimeIndex([as_of]),
                                       policy=POLICY, initial_holdings=prior_holdings)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"{as_of:%Y-%m-%d}.csv.gz"
        if snapshot.exists():
            with gzip_open(snapshot, "rb") as handle:
                if handle.read() != source_text.encode("utf-8"):
                    parser.error("A different data snapshot already exists for this signal date.")
        else:
            with gzip_open(snapshot, "wb") as handle:
                handle.write(source_text.encode("utf-8"))
        snapshot_hash = sha256(snapshot.read_bytes()).hexdigest()
        model_hash = code_fingerprint()
        signal_id = sha256(f"{as_of.date()}|{source_hash}|{model_hash}".encode()).hexdigest()
        rows = [{"candidate": row.candidate, "ticker": row.Ticker,
                 "expected_return": float(row.expected_return)}
                for row in forecasts.itertuples(index=False)]
        weights_json = {candidate: {str(symbol): float(weight) for symbol, weight in frame.loc[as_of].items()}
                        for candidate, frame in allocations.items()}
        event = {"event": "paper_signal", "signal_id": signal_id,
                 "as_of": str(as_of.date()), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                 "execute": "next available session open", "horizon_sessions": POLICY["horizon"],
                 "target_exit": "sixth subsequent session open", "tickers": tickers,
                 "sessions_since_prior_signal": elapsed_sessions if signals else None,
                 "source_hash": source_hash, "snapshot": str(snapshot),
                 "snapshot_hash": snapshot_hash, "code_fingerprint": model_hash,
                 "policy": POLICY,
                 "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]},
                 "model_parameters": model_configuration(models),
                 "forecasts": rows, "allocations": weights_json}
        append_event(ledger, event)
        signals.append(event)
        created = True

    mark = portfolio_mark(prepared, signals, as_of)
    if mark and not any(event.get("event") == "paper_portfolio_mark"
                        and event.get("observed_through") == str(as_of.date()) for event in events):
        append_event(ledger, {**mark, "source_hash": source_hash})
    print(f"Latest complete session: {as_of.date()}")
    print(f"Mature forecasts reconciled this run: {reconciled}")
    if created:
        print("New five-session signal committed after the session close.")
    elif due and not commit_window:
        print("No signal committed; wait until after the latest session close to preserve next-open timing.")
    else:
        print("No new signal was due.")
    if created:
        print(f"Append-only ledger: {ledger}")
        print(f"Exact input snapshot: {snapshot}")
    if mark:
        print(f"Paper portfolio mark saved through {as_of.date()}.")


if __name__ == "__main__":
    main()
