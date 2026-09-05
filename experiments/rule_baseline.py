"""Evaluate a frozen naive rule baseline on the held-out test split.

The rule thresholds were derived from training-distribution observations and
are NOT tuned on validation or test labels. Labels are used only for offline
evaluation metrics after predictions are formed.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.baseline import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST, assemble_dataset, load_data

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"

# Frozen heuristic thresholds — derived from training-distribution observations only.
FROZEN_RULE = {
    "customer_txn_count_30m": {"operator": ">=", "threshold": 2},
    "customer_txn_count_1h": {"operator": ">=", "threshold": 3},
    "device_account_count_before": {"operator": ">=", "threshold": 10},
    "instrument_account_count_before": {"operator": ">=", "threshold": 6},
}
RULE_DESCRIPTION = (
    "FLAG if ANY of: customer_txn_count_30m >= 2, customer_txn_count_1h >= 3, "
    "device_account_count_before >= 10, instrument_account_count_before >= 6"
)


def apply_frozen_rule(features: pd.DataFrame) -> pd.Series:
    """Return boolean flags from the frozen rule. Does not read abuse labels."""
    missing = set(FROZEN_RULE) - set(features.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    flagged = pd.Series(False, index=features.index)
    for column, spec in FROZEN_RULE.items():
        flagged |= features[column] >= spec["threshold"]
    return flagged


def evaluate_predictions(y_true: pd.Series, predictions: pd.Series) -> dict[str, Any]:
    """Compute classification and simulated business-cost metrics."""
    y_pred = predictions.astype(bool)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0
    total_cost = fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST
    n = len(y_true)
    flagged = int(y_pred.sum())
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "flagged": flagged,
        "flag_rate": float(flagged / n) if n else 0.0,
        "test_transactions": int(n),
        "simulated_cost": float(total_cost),
        "false_positive_cost": FALSE_POSITIVE_COST,
        "false_negative_cost": FALSE_NEGATIVE_COST,
    }


def load_production_lr_metrics() -> dict[str, Any]:
    """Load existing production Logistic Regression test metrics without retraining."""
    metrics_path = ARTIFACT_DIR / "test_metrics.json"
    config_path = ARTIFACT_DIR / "baseline_config.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing production metrics artifact: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return {
        "model": "behavioral_logistic_regression",
        "threshold": config.get("threshold", metrics.get("threshold")),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_positive_rate": metrics["false_positive_rate"],
        "false_negatives": metrics["false_negatives"],
        "false_positives": metrics["false_positives"],
        "simulated_cost": metrics["total_cost"],
        "flagged": metrics["false_positives"] + metrics["abuse_detected"],
        "source_artifact": str(metrics_path.relative_to(PROJECT_ROOT).as_posix()),
    }


def main() -> None:
    features, labels, splits = load_data()
    dataset = assemble_dataset(features, labels, splits)
    test = dataset[dataset["split"] == "test"].copy()
    if test.empty:
        raise ValueError("Test split is empty")

    predictions = apply_frozen_rule(test)
    naive_metrics = evaluate_predictions(test["is_abuse"], predictions)
    production_metrics = load_production_lr_metrics()

    split_counts = splits["split"].value_counts().to_dict()
    summary = {
        "experiment": "naive_rule_baseline",
        "dataset": {
            "total_transactions": int(len(splits)),
            "split_counts": {k: int(v) for k, v in split_counts.items()},
            "split_method": "chronological 70/15/15",
            "features_source": str((PROCESSED_DIR / "features.csv").relative_to(PROJECT_ROOT).as_posix()),
        },
        "frozen_rule": {
            "description": RULE_DESCRIPTION,
            "conditions": FROZEN_RULE,
            "threshold_derivation": "training-distribution observations; not tuned on validation or test labels",
        },
        "cost_assumptions": {
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "note": "Synthetic test-set evaluation costs; not production fraud savings",
        },
        "naive_rule_test": {
            "model": "naive_rule",
            **naive_metrics,
        },
        "production_lr_test": production_metrics,
        "comparison": {
            "precision_delta": naive_metrics["precision"] - production_metrics["precision"],
            "recall_delta": naive_metrics["recall"] - production_metrics["recall"],
            "f1_delta": naive_metrics["f1"] - production_metrics["f1"],
            "simulated_cost_delta": naive_metrics["simulated_cost"] - production_metrics["simulated_cost"],
            "false_positives_delta": naive_metrics["false_positives"] - production_metrics["false_positives"],
        },
    }

    comparison_rows = [
        {
            "model": "naive_rule",
            "threshold": "frozen_rule",
            **{k: naive_metrics[k] for k in [
                "precision", "recall", "f1", "false_positive_rate", "false_positives",
                "false_negatives", "flagged", "flag_rate", "simulated_cost",
            ]},
        },
        {
            "model": "behavioral_logistic_regression",
            "threshold": production_metrics["threshold"],
            "precision": production_metrics["precision"],
            "recall": production_metrics["recall"],
            "f1": production_metrics["f1"],
            "false_positive_rate": production_metrics["false_positive_rate"],
            "false_positives": production_metrics["false_positives"],
            "false_negatives": production_metrics["false_negatives"],
            "flagged": production_metrics["flagged"],
            "flag_rate": production_metrics["flagged"] / naive_metrics["test_transactions"],
            "simulated_cost": production_metrics["simulated_cost"],
        },
    ]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(ARTIFACT_DIR / "rule_baseline_comparison.csv", index=False)
    with (ARTIFACT_DIR / "rule_baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Frozen naive rule baseline - held-out test split")
    print(f"Rule: {RULE_DESCRIPTION}")
    print(f"Test size: {naive_metrics['test_transactions']}")
    for name in [
        "precision", "recall", "f1", "false_positive_rate", "false_positives",
        "false_negatives", "flagged", "flag_rate", "simulated_cost",
    ]:
        print(f"  naive_rule {name}: {naive_metrics[name]}")
    print("Production LR comparison (from test_metrics.json):")
    print(f"  precision={production_metrics['precision']}, recall={production_metrics['recall']}, "
          f"f1={production_metrics['f1']}, cost={production_metrics['simulated_cost']}")
    print(f"Wrote {ARTIFACT_DIR / 'rule_baseline_comparison.csv'}")
    print(f"Wrote {ARTIFACT_DIR / 'rule_baseline_summary.json'}")


if __name__ == "__main__":
    main()
