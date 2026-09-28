"""Compare fixed feature groups on a common validation sample."""

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from src.research_features import FEATURE_SETS, common_feature_rows
from src.research_model import select_model


def compare_feature_sets(split, names, models=None):
    feature_sets = {name: FEATURE_SETS[name] for name in names}
    # No access to split.test: its completeness cannot influence selection.
    development = replace(
        split,
        train=common_feature_rows(split.train, feature_sets),
        validation=common_feature_rows(split.validation, feature_sets),
    )
    selections, rows = {}, []
    for name, features in feature_sets.items():
        selection = select_model(development, features=features, models=models)
        selections[name] = selection
        result = selection.validation_results.copy()
        result.insert(0, "Feature_set", name)
        result["train_rows"] = len(development.train)
        result["validation_rows"] = len(development.validation)
        rows.append(result)
    comparison = pd.concat(rows, ignore_index=True).sort_values(
        ["log_loss", "brier", "macro_f1", "Feature_set", "Model"],
        ascending=[True, True, False, True, True], kind="stable",
    ).reset_index(drop=True)
    chosen_features = comparison.iloc[0]["Feature_set"]
    return comparison, chosen_features, selections[chosen_features], development


def ensure_test_unseen(ledger, tickers, validation_end, test_end):
    """Refuse overlapping test reuse in this ledger, including legacy records.

    This is a local safeguard, not access control. Deleting or changing ledgers
    defeats it; new results on an exposed period remain exploratory.
    """
    ledger = Path(ledger)
    if not ledger.exists():
        return
    for line in ledger.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        config = record["config"]
        if "test" not in record["metrics"] and config.get("stage") != "test_reserved":
            continue
        if not set(tickers).intersection(config.get("tickers", [])):
            continue
        start, end = config.get("validation_end"), config.get("test_end")
        if start is None or (
            pd.Timestamp(start) < pd.Timestamp(test_end)
            and (end is None or pd.Timestamp(end) > pd.Timestamp(validation_end))
        ):
            raise ValueError(
                "This ledger already contains an overlapping test period. "
                "Continue with validation only; this period is no longer untouched."
            )
