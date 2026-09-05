"""Recreate the fair graph ablation for Sentinel.

Compares:
  Arm A: Behavioral-only (32 features)
  Arm B: Behavioral + Unique Graph features (32 + 23 = 55 features)

Both arms use the exact same:
  - chronological 70/15/15 train/validation/test split
  - StandardScaler fit on train only
  - LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42) fit on train only
  - validation-only threshold tuning across candidate thresholds [0.10, ..., 0.90]
  - cost minimization criterion (FP=$5, FN=$100), breaking ties by higher validation F1
  - single evaluation on held-out test set with frozen threshold
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"

# Synthetic business-cost assumptions (evaluation benchmark only)
FALSE_POSITIVE_COST = 5.0
FALSE_NEGATIVE_COST = 100.0

# Candidate thresholds evaluated on validation split only
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Known duplicate graph signals matching the 32 behavioral feature set
EXCLUDED_DUPLICATE_GRAPH_FEATURES = [
    "customer_merchant_count_before",
    "customer_device_count_before",
    "customer_instrument_count_before",
    "device_customer_count_before",
    "device_merchant_count_before",
    "instrument_customer_count_before",
    "instrument_merchant_count_before",
    "other_customers_on_device",
    "other_customers_on_instrument",
]


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Load precomputed behavioral features, graph features, labels, and splits."""
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    graph = pd.read_csv(PROCESSED_DIR / "graph_features.csv")
    labels = pd.read_csv(PROCESSED_DIR / "labels.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits.csv")

    behavioral_features = [col for col in features.columns if col != "transaction_id"]
    if len(behavioral_features) != 32:
        raise ValueError(f"Expected 32 behavioral features, found {len(behavioral_features)}")

    all_graph_cols = [col for col in graph.columns if col != "transaction_id"]
    missing_dupes = set(EXCLUDED_DUPLICATE_GRAPH_FEATURES) - set(all_graph_cols)
    if missing_dupes:
        raise ValueError(f"Excluded duplicate columns missing from graph features: {sorted(missing_dupes)}")

    unique_graph_features = [col for col in all_graph_cols if col not in EXCLUDED_DUPLICATE_GRAPH_FEATURES]
    if len(unique_graph_features) != 23:
        raise ValueError(f"Expected 23 unique graph features, found {len(unique_graph_features)}")

    # Merge behavioral arm
    df_behavioral = (
        features
        .merge(labels[["transaction_id", "is_abuse"]], on="transaction_id", validate="one_to_one")
        .merge(splits[["transaction_id", "split"]], on="transaction_id", validate="one_to_one")
    )
    df_behavioral["is_abuse"] = df_behavioral["is_abuse"].astype(int)

    # Merge combined arm (behavioral + unique graph)
    graph_subset = graph[["transaction_id"] + unique_graph_features]
    df_combined = (
        features
        .merge(graph_subset, on="transaction_id", validate="one_to_one")
        .merge(labels[["transaction_id", "is_abuse"]], on="transaction_id", validate="one_to_one")
        .merge(splits[["transaction_id", "split"]], on="transaction_id", validate="one_to_one")
    )
    df_combined["is_abuse"] = df_combined["is_abuse"].astype(int)

    return df_behavioral, df_combined, behavioral_features, unique_graph_features


def evaluate_thresholds_on_validation(
    y_val: pd.Series, val_scores: np.ndarray
) -> tuple[float, list[dict[str, Any]]]:
    """Select the threshold that minimizes simulated cost on validation, breaking ties with F1."""
    records = []
    for threshold in THRESHOLDS:
        preds = val_scores >= threshold
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        cost = fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST
        f1 = float(f1_score(y_val, preds, zero_division=0))
        prec = float(precision_score(y_val, preds, zero_division=0))
        rec = float(recall_score(y_val, preds, zero_division=0))
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        records.append({
            "threshold": threshold,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_positive_rate": fpr,
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "simulated_cost": float(cost),
        })

    val_df = pd.DataFrame(records)
    selected = val_df.sort_values(["simulated_cost", "f1", "threshold"], ascending=[True, False, True]).iloc[0]
    return float(selected["threshold"]), records


def train_and_evaluate_arm(
    arm_name: str,
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Train pipeline on train split, tune on validation split, evaluate on test split."""
    train = dataset[dataset["split"] == "train"]
    validation = dataset[dataset["split"] == "validation"]
    test = dataset[dataset["split"] == "test"]

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Train, validation, or test split is empty")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=42)),
        ]
    )

    # 1. Fit on train split only
    pipeline.fit(train[feature_columns], train["is_abuse"])

    # 2. Score validation split only
    val_scores = pipeline.predict_proba(validation[feature_columns])[:, 1]
    selected_threshold, val_records = evaluate_thresholds_on_validation(validation["is_abuse"], val_scores)

    # 3. Freeze threshold, score test split only
    test_scores = pipeline.predict_proba(test[feature_columns])[:, 1]
    test_preds = test_scores >= selected_threshold

    tn, fp, fn, tp = confusion_matrix(test["is_abuse"], test_preds, labels=[0, 1]).ravel()
    prec = float(precision_score(test["is_abuse"], test_preds, zero_division=0))
    rec = float(recall_score(test["is_abuse"], test_preds, zero_division=0))
    f1 = float(f1_score(test["is_abuse"], test_preds, zero_division=0))
    roc_auc = float(roc_auc_score(test["is_abuse"], test_scores))
    pr_auc = float(average_precision_score(test["is_abuse"], test_scores))
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    cost = float(fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST)

    return {
        "model": arm_name,
        "feature_count": len(feature_columns),
        "threshold": selected_threshold,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "flagged": int(fp + tp),
        "flag_rate": float((fp + tp) / len(test)),
        "simulated_cost": cost,
        "validation_selection_table": val_records,
    }


def main() -> None:
    df_behavioral, df_combined, behavioral_features, unique_graph_features = load_datasets()
    combined_features = behavioral_features + unique_graph_features

    arm_behavioral = train_and_evaluate_arm(
        arm_name="behavioral_only",
        dataset=df_behavioral,
        feature_columns=behavioral_features,
    )

    arm_graph_fair = train_and_evaluate_arm(
        arm_name="behavioral_plus_unique_graph",
        dataset=df_combined,
        feature_columns=combined_features,
    )

    splits_meta = df_behavioral["split"].value_counts().to_dict()

    summary: dict[str, Any] = {
        "experiment": "fair_graph_ablation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "total_transactions": len(df_behavioral),
            "split_counts": {k: int(v) for k, v in splits_meta.items()},
            "split_method": "chronological 70/15/15",
            "features_source": (PROCESSED_DIR / "features.csv").relative_to(PROJECT_ROOT).as_posix(),
            "graph_features_source": (PROCESSED_DIR / "graph_features.csv").relative_to(PROJECT_ROOT).as_posix(),
        },
        "model_configuration": {
            "model_family": "LogisticRegression",
            "scaler": "StandardScaler",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42,
        },
        "threshold_selection": {
            "candidates": THRESHOLDS,
            "split_used": "validation",
            "criterion": "minimize simulated cost; tie-breaker: maximum F1",
        },
        "cost_assumptions": {
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "note": "Synthetic benchmark evaluation assumptions; not real merchant loss reductions",
        },
        "feature_sets": {
            "behavioral_feature_count": len(behavioral_features),
            "behavioral_features": behavioral_features,
            "excluded_duplicate_graph_features_count": len(EXCLUDED_DUPLICATE_GRAPH_FEATURES),
            "excluded_duplicate_graph_features": EXCLUDED_DUPLICATE_GRAPH_FEATURES,
            "unique_graph_features_count": len(unique_graph_features),
            "unique_graph_features": unique_graph_features,
            "combined_fair_feature_count": len(combined_features),
            "combined_fair_features": combined_features,
        },
        "results": {
            "behavioral_only": {k: v for k, v in arm_behavioral.items() if k != "validation_selection_table"},
            "behavioral_plus_unique_graph": {k: v for k, v in arm_graph_fair.items() if k != "validation_selection_table"},
        },
        "comparison_deltas": {
            "precision_delta": arm_graph_fair["precision"] - arm_behavioral["precision"],
            "recall_delta": arm_graph_fair["recall"] - arm_behavioral["recall"],
            "f1_delta": arm_graph_fair["f1"] - arm_behavioral["f1"],
            "roc_auc_delta": arm_graph_fair["roc_auc"] - arm_behavioral["roc_auc"],
            "pr_auc_delta": arm_graph_fair["pr_auc"] - arm_behavioral["pr_auc"],
            "false_positive_rate_delta": arm_graph_fair["false_positive_rate"] - arm_behavioral["false_positive_rate"],
            "false_positives_delta": arm_graph_fair["false_positives"] - arm_behavioral["false_positives"],
            "false_negatives_delta": arm_graph_fair["false_negatives"] - arm_behavioral["false_negatives"],
            "simulated_cost_delta": arm_graph_fair["simulated_cost"] - arm_behavioral["simulated_cost"],
        },
        "interpretation": (
            "Graph features improve precision, F1, PR-AUC and reduce FPR/false positives, "
            "but reduce recall and increase simulated cost under independently validation-selected thresholds. "
            "This means graph evidence may improve the quality of prioritization/explainability while not "
            "necessarily minimizing the current simulated business cost."
        ),
    }

    comparison_rows = [
        {
            "model": "behavioral_only",
            "feature_count": arm_behavioral["feature_count"],
            "threshold": arm_behavioral["threshold"],
            "precision": arm_behavioral["precision"],
            "recall": arm_behavioral["recall"],
            "f1": arm_behavioral["f1"],
            "roc_auc": arm_behavioral["roc_auc"],
            "pr_auc": arm_behavioral["pr_auc"],
            "false_positive_rate": arm_behavioral["false_positive_rate"],
            "false_positives": arm_behavioral["false_positives"],
            "false_negatives": arm_behavioral["false_negatives"],
            "simulated_cost": arm_behavioral["simulated_cost"],
        },
        {
            "model": "behavioral_plus_unique_graph",
            "feature_count": arm_graph_fair["feature_count"],
            "threshold": arm_graph_fair["threshold"],
            "precision": arm_graph_fair["precision"],
            "recall": arm_graph_fair["recall"],
            "f1": arm_graph_fair["f1"],
            "roc_auc": arm_graph_fair["roc_auc"],
            "pr_auc": arm_graph_fair["pr_auc"],
            "false_positive_rate": arm_graph_fair["false_positive_rate"],
            "false_positives": arm_graph_fair["false_positives"],
            "false_negatives": arm_graph_fair["false_negatives"],
            "simulated_cost": arm_graph_fair["simulated_cost"],
        },
    ]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = ARTIFACT_DIR / "graph_ablation_comparison.csv"
    summary_json_path = ARTIFACT_DIR / "graph_ablation_summary.json"

    pd.DataFrame(comparison_rows).to_csv(comparison_csv_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== FAIR GRAPH ABLATION RESULTS ===")
    print(f"Features: Behavioral={len(behavioral_features)}, Unique Graph={len(unique_graph_features)}, Combined={len(combined_features)}")
    print(f"Excluded duplicate graph features ({len(EXCLUDED_DUPLICATE_GRAPH_FEATURES)}): {EXCLUDED_DUPLICATE_GRAPH_FEATURES}")
    print("\n[Arm A: Behavioral-only (32 features)]")
    print(f"  Selected threshold: {arm_behavioral['threshold']:.2f}")
    print(f"  Precision: {arm_behavioral['precision']:.4f}")
    print(f"  Recall:    {arm_behavioral['recall']:.4f}")
    print(f"  F1:        {arm_behavioral['f1']:.4f}")
    print(f"  ROC-AUC:   {arm_behavioral['roc_auc']:.4f}")
    print(f"  PR-AUC:    {arm_behavioral['pr_auc']:.4f}")
    print(f"  FPR:       {arm_behavioral['false_positive_rate'] * 100:.2f}%")
    print(f"  FP:        {arm_behavioral['false_positives']}")
    print(f"  FN:        {arm_behavioral['false_negatives']}")
    print(f"  Cost:      ${arm_behavioral['simulated_cost']:.0f}")

    print("\n[Arm B: Behavioral + 23 Unique Graph (55 features)]")
    print(f"  Selected threshold: {arm_graph_fair['threshold']:.2f}")
    print(f"  Precision: {arm_graph_fair['precision']:.4f}")
    print(f"  Recall:    {arm_graph_fair['recall']:.4f}")
    print(f"  F1:        {arm_graph_fair['f1']:.4f}")
    print(f"  ROC-AUC:   {arm_graph_fair['roc_auc']:.4f}")
    print(f"  PR-AUC:    {arm_graph_fair['pr_auc']:.4f}")
    print(f"  FPR:       {arm_graph_fair['false_positive_rate'] * 100:.2f}%")
    print(f"  FP:        {arm_graph_fair['false_positives']}")
    print(f"  FN:        {arm_graph_fair['false_negatives']}")
    print(f"  Cost:      ${arm_graph_fair['simulated_cost']:.0f}")

    print(f"\nWrote artifacts:\n  {comparison_csv_path}\n  {summary_json_path}")


if __name__ == "__main__":
    main()
