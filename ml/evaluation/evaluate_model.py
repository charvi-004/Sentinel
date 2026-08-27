from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"
FALSE_POSITIVE_COST = 5.0
FALSE_NEGATIVE_COST = 100.0


def load_test_set() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    labels = pd.read_csv(PROCESSED_DIR / "labels.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits.csv")
    test_ids = splits.loc[splits["split"] == "test", "transaction_id"]
    return features[features["transaction_id"].isin(test_ids)].copy(), labels[labels["transaction_id"].isin(test_ids)].copy(), splits[splits["transaction_id"].isin(test_ids)].copy()


def evaluate_model_with_threshold(threshold: float = 0.5) -> dict:
    features, labels, splits = load_test_set()
    merged = features.merge(labels[["transaction_id", "is_abuse"]], on="transaction_id", how="inner", validate="one_to_one")
    model = joblib.load(ARTIFACT_DIR / "baseline_logistic_regression.joblib")
    scoring_features = [column for column in features.columns if column != "transaction_id"]
    scores = model.predict_proba(merged[scoring_features])[:, 1]
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(merged["is_abuse"], predictions, labels=[0, 1]).ravel()

    metrics = {
        "test_transactions": int(len(merged)),
        "abuse_transactions": int(merged["is_abuse"].sum()),
        "normal_transactions": int((merged["is_abuse"] == 0).sum()),
        "precision": float(precision_score(merged["is_abuse"], predictions, zero_division=0)),
        "recall": float(recall_score(merged["is_abuse"], predictions, zero_division=0)),
        "f1": float(f1_score(merged["is_abuse"], predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(merged["is_abuse"], scores)),
        "roc_auc": float(roc_auc_score(merged["is_abuse"], scores)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "false_positive_cost": float(fp * FALSE_POSITIVE_COST),
        "false_negative_cost": float(fn * FALSE_NEGATIVE_COST),
        "total_cost": float(fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST),
        "threshold": float(threshold),
    }
    return metrics


def main() -> None:
    config = json.loads((ARTIFACT_DIR / "baseline_config.json").read_text(encoding="utf-8"))
    result = evaluate_model_with_threshold(float(config["threshold"]))
    (ARTIFACT_DIR / "final_evaluation.json").write_text(json.dumps({"dataset": {}, "model": {}, "detection_metrics": result, "business_cost": {}, "operations": {}, "evidence": {}, "ai_safety": {}}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
