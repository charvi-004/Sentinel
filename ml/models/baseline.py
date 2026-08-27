"""Train and evaluate Sentinel's interpretable Logistic Regression baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"
FEATURE_KEY = "transaction_id"
LABEL_COLUMNS = ["transaction_id", "is_abuse", "abuse_type", "abuse_cluster_id"]
SPLITS = {"train", "validation", "test"}
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Synthetic comparison assumptions, not Razorpay financial estimates.
FALSE_POSITIVE_COST = 5.0
FALSE_NEGATIVE_COST = 100.0


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the precomputed feature, label, and chronological split tables."""
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    labels = pd.read_csv(PROCESSED_DIR / "labels.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits.csv")
    required = {FEATURE_KEY}
    if not required.issubset(features.columns):
        raise ValueError("features.csv must contain transaction_id")
    if set(LABEL_COLUMNS) - set(labels.columns):
        raise ValueError("labels.csv is missing required columns")
    if {"transaction_id", "split"} - set(splits.columns):
        raise ValueError("splits.csv must contain transaction_id and split")
    if features[FEATURE_KEY].duplicated().any() or labels[FEATURE_KEY].duplicated().any() or splits[FEATURE_KEY].duplicated().any():
        raise ValueError("Input tables contain duplicate transaction IDs")
    if not set(splits["split"]).issubset(SPLITS):
        raise ValueError(f"Unexpected split values: {sorted(set(splits['split']) - SPLITS)}")
    return features, labels, splits


def assemble_dataset(features: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Join tables by transaction ID, retaining labels only as the target column."""
    feature_columns = [column for column in features.columns if column != FEATURE_KEY]
    forbidden = {"is_abuse", "abuse_type", "abuse_cluster_id", "customer_id", "device_id", "merchant_id", "payment_instrument_id"}
    forbidden_features = set(feature_columns) & forbidden
    if forbidden_features:
        raise ValueError(f"Forbidden feature columns: {sorted(forbidden_features)}")
    dataset = features.merge(labels[[FEATURE_KEY, "is_abuse"]], on=FEATURE_KEY, how="inner", validate="one_to_one")
    dataset = dataset.merge(splits, on=FEATURE_KEY, how="inner", validate="one_to_one")
    if len(dataset) != len(features) or set(dataset[FEATURE_KEY]) != set(features[FEATURE_KEY]):
        raise ValueError("Feature, label, and split tables do not map exactly")
    dataset["is_abuse"] = dataset["is_abuse"].astype(int)
    return dataset


def metrics_at_threshold(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate classification and synthetic business-cost metrics at one threshold."""
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0
    total_cost = fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "abuse_detected": int(tp),
        "abuse_missed": int(fn),
        "total_cost": float(total_cost),
    }


def select_threshold(y_validation: pd.Series, validation_scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Select the lowest validation cost, breaking ties with F1 only."""
    rows = [metrics_at_threshold(y_validation, validation_scores, threshold) for threshold in THRESHOLDS]
    table = pd.DataFrame(rows)
    selected = table.sort_values(["total_cost", "f1", "threshold"], ascending=[True, False, True]).iloc[0]
    return float(selected["threshold"]), table


def save_coefficients(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    coefficients = model.named_steps["model"].coef_[0]
    result = pd.DataFrame({"feature": feature_columns, "coefficient": coefficients})
    result["absolute_coefficient"] = result["coefficient"].abs()
    result["direction"] = np.where(result["coefficient"] >= 0, "positive", "negative")
    return result.sort_values("absolute_coefficient", ascending=False)


def main() -> None:
    features, labels, splits = load_data()
    dataset = assemble_dataset(features, labels, splits)
    feature_columns = [column for column in features.columns if column != FEATURE_KEY]
    train = dataset[dataset["split"] == "train"]
    validation = dataset[dataset["split"] == "validation"]
    test = dataset[dataset["split"] == "test"]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Every chronological split must contain rows")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=42)),
        ]
    )
    model.fit(train[feature_columns], train["is_abuse"])
    validation_scores = model.predict_proba(validation[feature_columns])[:, 1]
    selected_threshold, validation_table = select_threshold(validation["is_abuse"], validation_scores)

    # The threshold and costs are frozen before test probabilities are requested.
    test_scores = model.predict_proba(test[feature_columns])[:, 1]
    test_metrics = metrics_at_threshold(test["is_abuse"], test_scores, selected_threshold)
    test_metrics.update(
        {
            "pr_auc": float(average_precision_score(test["is_abuse"], test_scores)),
            "roc_auc": float(roc_auc_score(test["is_abuse"], test_scores)),
            "accuracy": float(accuracy_score(test["is_abuse"], test_scores >= selected_threshold)),
            "selection_split": "validation",
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": FALSE_NEGATIVE_COST,
        }
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "baseline_logistic_regression.joblib")
    validation_table.to_csv(ARTIFACT_DIR / "validation_thresholds.csv", index=False)
    save_coefficients(model, feature_columns).to_csv(ARTIFACT_DIR / "baseline_coefficients.csv", index=False)
    with (ARTIFACT_DIR / "baseline_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "model": "logistic_regression",
                "threshold": selected_threshold,
                "selection_split": "validation",
                "selection_method": "minimum validation total simulated cost; ties broken by validation F1",
                "class_weight": "balanced",
                "false_positive_cost": FALSE_POSITIVE_COST,
                "false_negative_cost": FALSE_NEGATIVE_COST,
                "random_state": 42,
            },
            config_file,
            indent=2,
        )
    with (ARTIFACT_DIR / "test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(test_metrics, metrics_file, indent=2)

    print("model: Logistic Regression (StandardScaler, class_weight='balanced')")
    print(f"features: {len(feature_columns)}")
    print(f"sizes: train={len(train)}, validation={len(validation)}, test={len(test)}")
    print(f"class counts: train={train['is_abuse'].value_counts().sort_index().to_dict()}, validation={validation['is_abuse'].value_counts().sort_index().to_dict()}, test={test['is_abuse'].value_counts().sort_index().to_dict()}")
    print(f"selected validation threshold: {selected_threshold:.2f}")
    print("validation threshold table:")
    print(validation_table[["threshold", "precision", "recall", "f1", "false_positive_rate", "total_cost"]].round(4).to_string(index=False))
    print("held-out test metrics:")
    for name in ["precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate", "false_negative_rate", "false_positives", "false_negatives", "total_cost"]:
        print(f"  {name}: {test_metrics[name]}")
    coefficients = save_coefficients(model, feature_columns)
    print(f"strongest positive features: {coefficients[coefficients['direction'] == 'positive'].head(5)['feature'].tolist()}")
    print(f"strongest negative features: {coefficients[coefficients['direction'] == 'negative'].head(5)['feature'].tolist()}")


if __name__ == "__main__":
    main()