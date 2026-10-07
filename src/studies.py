"""Operator workflow: validation-only selection, frozen settings, one final exposure."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

import pandas as pd
from threadpoolctl import threadpool_limits

from src.backtest import backtest, calculate_metrics, equity_from_returns, trading_metrics
from src.features import TECHNICAL_FEATURES, add_features
from src.model import walk_forward_evaluate
from src.reports import json_safe, runtime_provenance
from src.snapshots import POLICY, SnapshotStore

ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return json.dumps(json_safe(value), sort_keys=True, allow_nan=False)


class StudyStore:
    def __init__(self, path):
        self.snapshots = SnapshotStore(path)

    def connect(self):
        return self.snapshots.connect()

    def prepare(self, snapshot_id):
        raw = self.snapshots.load_readonly(snapshot_id)
        if raw.attrs["snapshot"]["price_policy"] != POLICY:
            raise ValueError("Refresh the legacy dataset before preparing a study.")
        featured = add_features(raw)
        dates = featured.index[featured["Target"].notna()]
        train_stop, validation_stop = int(len(dates) * .6), int(len(dates) * .8)
        if train_stop < 60 or validation_stop - train_stop - 5 < 30 or len(dates) - validation_stop < 30:
            raise ValueError("Study requires 60 training, 30 embargoed validation and 30 final predictions.")
        protocol = {
            "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dict(raw.attrs["snapshot"]), "runtime": runtime_provenance(ROOT),
            "train_end": str(dates[train_stop - 1].date()),
            "validation_start": str(dates[train_stop].date()),
            "validation_end": str(dates[validation_stop - 1].date()),
            "final_start": str(dates[validation_stop].date()),
            "history_end": str(raw.index.max().date()),
            "candidates": [{"threshold": threshold, "retrain_days": retrain}
                           for retrain in (21, 42, 63) for threshold in (.55, .60, .65)],
            "selection_rule": "highest validation strategy Sharpe; first candidate breaks ties",
            "model": "XGBoost", "transaction_cost_bps": 10, "slippage_bps": 5,
            "starting_capital_usd": 10000, "liquidate_at_end": True,
            "refit_rule": "expanding walk-forward; only labels known at each after-close fit",
            "limitations": [
                "Historical holdout discipline cannot make previously viewed data unseen.",
                "One study per snapshot in this store; copies/new stores can bypass this ledger.",
                "Retrospective data, overlapping targets and simplified costs remain limitations.",
                "Selection uses Sharpe on nine candidates; this is not evidence of profitability.",
            ],
        }
        payload = encoded(protocol)
        study_id = uuid4().hex
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS studies (
                id TEXT PRIMARY KEY, snapshot_id TEXT UNIQUE NOT NULL,
                state TEXT NOT NULL, protocol TEXT NOT NULL, digest TEXT NOT NULL,
                selection TEXT, selection_digest TEXT, result TEXT, result_digest TEXT)""")
            try:
                connection.execute("INSERT INTO studies (id, snapshot_id, state, protocol, digest) VALUES (?, ?, 'prepared', ?, ?)",
                                   (study_id, snapshot_id, payload, sha256(payload.encode()).hexdigest()))
            except sqlite3.IntegrityError:
                raise ValueError("This snapshot already has a study. Continue that study; do not reselect on its final period.") from None
        return {"study_id": study_id, "state": "prepared", "protocol": protocol}

    def prepare_or_resume(self, snapshot_id):
        # Preparation is serialized by the dashboard worker. The unique database
        # constraint still protects against concurrent CLI preparation.
        try:
            return self.prepare(snapshot_id)
        except ValueError:
            with self.connect() as connection:
                exists = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='studies'").fetchone()
                row = connection.execute("SELECT id FROM studies WHERE snapshot_id = ?", (snapshot_id,)).fetchone() if exists else None
            if row is None:
                raise
            return self.read(row[0])

    def read(self, study_id):
        with self.connect() as connection:
            row = connection.execute("SELECT state, protocol, digest, selection, selection_digest, result, result_digest FROM studies WHERE id = ?", (study_id,)).fetchone()
        if row is None:
            raise ValueError("Study is unavailable.")
        state, payload, digest, selection, selection_digest, result, result_digest = row
        if sha256(payload.encode()).hexdigest() != digest:
            raise ValueError("Study protocol integrity check failed.")
        for text, checksum in ((selection, selection_digest), (result, result_digest)):
            if text is not None and sha256(text.encode()).hexdigest() != checksum:
                raise ValueError("Study result or frozen selection integrity check failed.")
        return {"study_id": study_id, "state": state, "protocol": json.loads(payload),
                "selection": json.loads(selection) if selection else None,
                "result": json.loads(result) if result else None}

    def inputs(self, study):
        protocol = study["protocol"]
        if protocol["runtime"] != runtime_provenance(ROOT):
            raise ValueError("Restore the study's recorded source and runtime before continuing.")
        raw = self.snapshots.load_readonly(protocol["dataset"]["id"])
        if raw.attrs["snapshot"] != protocol["dataset"]:
            raise ValueError("Study dataset provenance differs.")
        return raw

    def claim(self, study_id, expected, new):
        with self.connect() as connection:
            changed = connection.execute("UPDATE studies SET state = ? WHERE id = ? AND state = ?",
                                         (new, study_id, expected)).rowcount
        if changed != 1:
            raise ValueError("Study stage is already running, consumed or incompatible.")

    def validate(self, study_id):
        study = self.read(study_id)
        if study["state"] in ("selected", "testing", "completed", "failed_final"):
            return {"study_id": study_id, "state": study["state"], "selection": study["selection"]}
        if study["state"] not in ("prepared", "failed_validation"):
            raise ValueError("Validation is already running. Preserve the study ledger and inspect the worker.")
        raw = self.inputs(study)
        protocol = study["protocol"]
        # Slice BEFORE features/labels: validation cannot access final-period prices.
        features = add_features(raw.loc[:protocol["validation_end"]])
        self.claim(study_id, study["state"], "validating")
        try:
            trials = []
            fits = {}
            with threadpool_limits(limits=2):
                for index, candidate in enumerate(protocol["candidates"]):
                    trials.append({"candidate_index": index, "settings": candidate,
                                   "evaluation": evaluate(features, protocol, candidate, protocol["validation_start"], cache=fits)})
            winner = max(trials, key=lambda trial: trial["evaluation"]["metrics"]["strategy"]["sharpe"])
            selection = {"selected_at": datetime.now(timezone.utc).isoformat(),
                         "candidate_index": winner["candidate_index"], "settings": winner["settings"],
                         "trials": trials, "final_period_observed": False}
            payload = encoded(selection)
            with self.connect() as connection:
                connection.execute("UPDATE studies SET state = 'selected', selection = ?, selection_digest = ? WHERE id = ? AND state = 'validating'",
                                   (payload, sha256(payload.encode()).hexdigest(), study_id))
        except Exception:
            self.claim(study_id, "validating", "failed_validation")
            raise
        return {"study_id": study_id, "state": "selected", "selection": selection}

    def finalize(self, study_id):
        study = self.read(study_id)
        if study["state"] == "completed":
            return study["result"]  # Never retest or replace a completed result.
        if study["state"] != "selected":
            raise ValueError("Final evaluation requires frozen validation settings and an unconsumed final period.")
        raw = self.inputs(study)
        self.claim(study_id, "selected", "testing")
        try:
            with threadpool_limits(limits=2):
                evaluation = evaluate(add_features(raw), study["protocol"], study["selection"]["settings"],
                                      study["protocol"]["final_start"])
            result = {"study_id": study_id, "kind": "locked_historical_final_evaluation",
                      "completed_at": datetime.now(timezone.utc).isoformat(),
                      "protocol": study["protocol"], "selection": study["selection"],
                      "final_period_consumed": True, "evaluation": evaluation}
            payload = encoded(result)
            with self.connect() as connection:
                connection.execute("UPDATE studies SET state = 'completed', result = ?, result_digest = ? WHERE id = ? AND state = 'testing'",
                                   (payload, sha256(payload.encode()).hexdigest(), study_id))
            return result
        except Exception:
            self.claim(study_id, "testing", "failed_final")
            raise


def evaluate(features, protocol, settings, start, cache=None):
    key = (settings["retrain_days"], start)
    if cache is None or key not in cache:
        _, predictions, classification = walk_forward_evaluate(
            features, TECHNICAL_FEATURES, retrain_every=settings["retrain_days"], evaluation_start=start,
        )
        if cache is not None:
            cache[key] = (predictions, classification)
    else:
        predictions, classification = cache[key]
    equity, detail = backtest(predictions, transaction_cost_bps=protocol["transaction_cost_bps"],
                              slippage_bps=protocol["slippage_bps"], confidence_threshold=settings["threshold"])
    strategy = calculate_metrics(equity)
    strategy.update(trading_metrics(detail))
    benchmarks = {}
    for name, column in (("stock_buy_hold", "market_return"), ("spy_buy_hold", "benchmark_return")):
        curve = equity_from_returns(pd.Series(detail[column].to_numpy(), index=pd.DatetimeIndex(detail["RealizationDate"])),
                                    10000, equity.index[0])
        benchmarks[name] = calculate_metrics(curve)
    return json_safe({"prediction_start": str(predictions.index.min().date()),
                      "prediction_end": str(predictions.index.max().date()),
                      "last_label_available": str(predictions["Target_Available_At"].max().date()),
                      "evaluated_predictions": len(predictions),
                      "metrics": {"classification": classification, "strategy": strategy, **benchmarks}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "validate", "finalize", "status"))
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--study-id")
    args = parser.parse_args()
    if (args.stage == "prepare" and not args.snapshot_id) or (args.stage != "prepare" and not args.study_id):
        parser.error("Prepare needs --snapshot-id; other stages need --study-id.")
    store = StudyStore(args.store)
    try:
        result = store.prepare(args.snapshot_id) if args.stage == "prepare" else getattr(store, "read" if args.stage == "status" else args.stage)(args.study_id)
    except (ValueError, KeyError, OSError, sqlite3.Error) as error:
        print(f"Study failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
