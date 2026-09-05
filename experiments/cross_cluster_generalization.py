"""Isolated leave-one-abuse-cluster-out robustness experiment.

This script never writes production artifacts or changes the chronological
production split. It regenerates behavioral features per fold so held-out
cluster transactions cannot influence training-row historical features.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
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

from ml.features.build_features import build_point_in_time_features


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_cluster_generalization"
THRESHOLD = 0.50
FALSE_POSITIVE_COST = 5.0
FALSE_NEGATIVE_COST = 100.0
EXPECTED_FEATURE_COUNT = 32
FORBIDDEN_FEATURES = {
    "is_abuse",
    "abuse_type",
    "abuse_cluster_id",
    "customer_id",
    "device_id",
    "merchant_id",
    "payment_instrument_id",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_csv(RAW_DIR / "transactions.csv")
    labels = pd.read_csv(PROCESSED_DIR / "labels.csv")
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], utc=True, errors="raise")
    dataset = transactions.merge(
        labels[["transaction_id", "is_abuse", "abuse_type", "abuse_cluster_id"]],
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )
    if len(dataset) != len(transactions):
        raise ValueError("Transactions and labels do not map exactly")
    dataset["is_abuse"] = dataset["is_abuse"].astype(bool)
    return transactions, dataset


def feature_columns() -> list[str]:
    features = pd.read_csv(PROCESSED_DIR / "features.csv", nrows=1)
    columns = [column for column in features.columns if column != "transaction_id"]
    if len(columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} behavioral features, found {len(columns)}")
    forbidden = set(columns) & FORBIDDEN_FEATURES
    if forbidden:
        raise ValueError(f"Forbidden model features: {sorted(forbidden)}")
    return columns


def build_features(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    transaction_columns = [
        "transaction_id", "timestamp", "merchant_id", "customer_id", "device_id",
        "payment_instrument_id", "amount", "currency", "status",
    ]
    features = build_point_in_time_features(rows[transaction_columns].copy())
    return features[["transaction_id", *columns]]


def metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, Any]:
    predictions = scores >= THRESHOLD
    true_values = y_true.astype(int).to_numpy()
    positives = int(true_values.sum())
    negatives = int(len(true_values) - positives)
    false_positives = int(((predictions == 1) & (true_values == 0)).sum())
    false_negatives = int(((predictions == 0) & (true_values == 1)).sum())
    result: dict[str, Any] = {
        "precision": float(precision_score(true_values, predictions, zero_division=0)),
        "recall": float(recall_score(true_values, predictions, zero_division=0)),
        "f1": float(f1_score(true_values, predictions, zero_division=0)),
        "fpr": float(false_positives / negatives) if negatives else 0.0,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "simulated_cost": float(false_positives * FALSE_POSITIVE_COST + false_negatives * FALSE_NEGATIVE_COST),
    }
    result["roc_auc"] = float(roc_auc_score(true_values, scores)) if len(np.unique(true_values)) > 1 else None
    result["pr_auc"] = float(average_precision_score(true_values, scores)) if positives else None
    return result


def normal_comparison(dataset: pd.DataFrame, cluster_rows: pd.DataFrame) -> pd.DataFrame:
    start = cluster_rows["timestamp"].min()
    end = cluster_rows["timestamp"].max()
    normals = dataset[(~dataset["is_abuse"]) & dataset["timestamp"].between(start, end, inclusive="both")].copy()
    if normals.empty:
        raise ValueError("No normal transactions in the held-out cluster time window")
    return normals.sort_values(["timestamp", "transaction_id"], kind="mergesort")


def run_fold(dataset: pd.DataFrame, cluster_id: str, columns: list[str]) -> dict[str, Any]:
    cluster_rows = dataset[dataset["abuse_cluster_id"] == cluster_id].copy()
    normals = normal_comparison(dataset, cluster_rows)
    eval_rows = pd.concat([cluster_rows, normals], ignore_index=True).sort_values(
        ["timestamp", "transaction_id"], kind="mergesort"
    )
    train_rows = dataset[
        ~dataset["abuse_cluster_id"].eq(cluster_id)
        & ~dataset["transaction_id"].isin(set(normals["transaction_id"]))
    ].copy().sort_values(["timestamp", "transaction_id"], kind="mergesort")
    heldout_positive_training = int(
        train_rows["is_abuse"].astype(bool).sum() * 0
        + train_rows["abuse_cluster_id"].eq(cluster_id).sum()
    )
    if heldout_positive_training != 0:
        raise AssertionError(f"Held-out cluster {cluster_id} leaked into training")
    if train_rows["is_abuse"].nunique() != 2:
        raise ValueError(f"Training fold for {cluster_id} lacks both classes")

    # Build training features in isolation, then build evaluation features with
    # training history available. Evaluation rows cannot affect training state.
    train_features = build_features(train_rows, columns)
    eval_features = build_features(pd.concat([train_rows, eval_rows], ignore_index=True), columns)
    eval_features = eval_features[eval_features["transaction_id"].isin(set(eval_rows["transaction_id"]))]
    eval_features = eval_features.merge(eval_rows[["transaction_id", "is_abuse"]], on="transaction_id", validate="one_to_one")
    train_features = train_features.merge(train_rows[["transaction_id", "is_abuse"]], on="transaction_id", validate="one_to_one")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=42)),
    ])
    model.fit(train_features[columns], train_features["is_abuse"].astype(int))
    scores = model.predict_proba(eval_features[columns])[:, 1]
    result = metrics(eval_features["is_abuse"], scores)
    result.update({
        "held_out_cluster": cluster_id,
        "mechanism": str(cluster_rows["abuse_type"].iloc[0]),
        "positive_examples": int(len(cluster_rows)),
        "normal_comparison_size": int(len(normals)),
        "training_rows": int(len(train_rows)),
        "training_positive_examples": int(train_rows["is_abuse"].astype(bool).sum()),
        "held_out_positive_training_examples": heldout_positive_training,
        "evaluation_start": str(cluster_rows["timestamp"].min()),
        "evaluation_end": str(cluster_rows["timestamp"].max()),
        "threshold": THRESHOLD,
        "scaler_fit_rows": int(len(train_features)),
        "model_fit_rows": int(len(train_features)),
        "_y_true": eval_features["is_abuse"].astype(int).tolist(),
        "_scores": scores.tolist(),
    })
    return result


def markdown_table(results: pd.DataFrame, columns: list[str]) -> str:
    display_columns = [
        "held_out_cluster", "mechanism", "positive_examples", "normal_comparison_size",
        "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr",
        "false_positives", "false_negatives", "simulated_cost",
    ]
    header = "| " + " | ".join(display_columns) + " |"
    separator = "| " + " | ".join("---" for _ in display_columns) + " |"
    rows = [header, separator]
    for _, row in results.iterrows():
        values = []
        for column in display_columns:
            value = row[column]
            values.append("NA" if pd.isna(value) else f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_summary(results: pd.DataFrame, columns: list[str], aggregate: dict[str, Any]) -> None:
    lines = [
        "# Cross-Cluster Generalization Experiment",
        "",
        "Supplementary robustness experiment only; production evaluation and artifacts were not changed.",
        "",
        "## Design",
        "",
        "Leave-one-abuse-cluster-out evaluation across all four generated abuse clusters. Each held-out cluster is excluded entirely from training. Normal comparison rows are all normal transactions whose timestamps fall within the held-out cluster's inclusive timestamp range, selected deterministically without random sampling.",
        "",
        "Holding out a cluster also holds out its abuse mechanism. This measures a combination of unseen-cluster and unseen-mechanism generalization, not proof of real-world generalization.",
        "",
        f"Features: {len(columns)} behavioral features from point-in-time feature generation; no labels, cluster IDs, or entity identifiers. StandardScaler and LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42) are fit only on each fold's training rows. Primary threshold: {THRESHOLD:.2f}; no held-out-label threshold tuning. Simulated costs: FP=${FALSE_POSITIVE_COST:.0f}, FN=${FALSE_NEGATIVE_COST:.0f}.",
        "",
        "## Per-cluster results",
        "",
        markdown_table(results, columns),
        "",
        "## Aggregate results",
        "",
        "Pooled across all held-out clusters and their deterministic normal comparison cohorts:",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        *[f"| {key} | {value if value is None else f'{value:.4f}' if isinstance(value, float) else value} |" for key, value in aggregate.items()],
        "",
        "## Limitations",
        "",
        "- Only four clusters exist, and each cluster is tied to one abuse mechanism.",
        "- The 45-transaction clusters produce noisy estimates.",
        "- Shared entities across generated activity may make the shift less independent than a true new-entity deployment case.",
        "- The normal comparison is a time-window cohort, so fold sizes and class balance differ.",
        "- Results are synthetic evaluation results and do not establish production robustness.",
    ]
    (RESULTS_DIR / "cross_cluster_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _, dataset = load_inputs()
    columns = feature_columns()
    clusters = sorted(dataset.loc[dataset["is_abuse"], "abuse_cluster_id"].unique())
    expected = ["abuse_01", "abuse_02", "abuse_03", "abuse_04"]
    if clusters != expected:
        raise ValueError(f"Expected clusters {expected}, found {clusters}")
    fold_results = [run_fold(dataset, cluster_id, columns) for cluster_id in clusters]
    pooled_y_true = np.concatenate([np.asarray(result.pop("_y_true"), dtype=int) for result in fold_results])
    pooled_scores = np.concatenate([np.asarray(result.pop("_scores"), dtype=float) for result in fold_results])
    aggregate = metrics(pd.Series(pooled_y_true), pooled_scores)
    results = pd.DataFrame(fold_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "cross_cluster_results.csv", index=False)
    write_summary(results, columns, aggregate)
    print(results.to_string(index=False))
    print(f"Wrote {RESULTS_DIR / 'cross_cluster_results.csv'}")
    print(f"Wrote {RESULTS_DIR / 'cross_cluster_summary.md'}")


if __name__ == "__main__":
    main()
