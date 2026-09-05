"""Compare behavioral, graph, and fused Sentinel risk models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"
ID_COLUMN = "transaction_id"
LABEL_COLUMN = "is_abuse"
FORBIDDEN_FEATURES = {
    "transaction_id",
    "customer_id",
    "device_id",
    "merchant_id",
    "payment_instrument_id",
    "is_abuse",
    "abuse_type",
    "abuse_cluster_id",
}
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
FALSE_POSITIVE_COST = 5.0
FALSE_NEGATIVE_COST = 100.0


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and assemble precomputed features without regenerating splits."""
    behavioral = pd.read_csv(PROCESSED_DIR / "features.csv")
    graph = pd.read_csv(PROCESSED_DIR / "graph_features.csv")
    labels = pd.read_csv(PROCESSED_DIR / "labels.csv")
    splits = pd.read_csv(PROCESSED_DIR / "splits.csv")
    for name, table in [("behavioral", behavioral), ("graph", graph), ("labels", labels), ("splits", splits)]:
        if table[ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} contains duplicate transaction IDs")
    if set(behavioral[ID_COLUMN]) != set(graph[ID_COLUMN]) or set(behavioral[ID_COLUMN]) != set(labels[ID_COLUMN]) or set(behavioral[ID_COLUMN]) != set(splits[ID_COLUMN]):
        raise ValueError("Input tables do not contain the same transaction IDs")
    behavioral_columns = [column for column in behavioral.columns if column != ID_COLUMN]
    graph_columns = [column for column in graph.columns if column != ID_COLUMN]
    forbidden = (set(behavioral_columns) | set(graph_columns)) & FORBIDDEN_FEATURES
    if forbidden:
        raise ValueError(f"Forbidden feature columns: {sorted(forbidden)}")
    graph = graph.rename(columns={column: f"graph__{column}" for column in graph_columns})
    dataset = behavioral.merge(graph, on=ID_COLUMN, validate="one_to_one")
    dataset = dataset.merge(labels[[ID_COLUMN, LABEL_COLUMN]], on=ID_COLUMN, validate="one_to_one")
    dataset = dataset.merge(splits, on=ID_COLUMN, validate="one_to_one")
    if not set(dataset["split"]) == {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test splits")
    return dataset, behavioral_columns, [f"graph__{column}" for column in graph_columns]


def build_model(model_kind: str) -> Pipeline | HistGradientBoostingClassifier:
    """Build either the exact Phase 4 baseline or the fixed nonlinear model."""
    if model_kind == "behavioral_baseline":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=42)),
            ]
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )


def balanced_sample_weights(target: pd.Series) -> np.ndarray:
    """Return inverse-frequency weights for classifiers without class_weight."""
    counts = target.value_counts().to_dict()
    total = len(target)
    class_count = len(counts)
    return target.map({label: total / (class_count * count) for label, count in counts.items()}).to_numpy()


def threshold_metrics(target: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = scores >= threshold
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    false_positive_rate = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
    false_negative_rate = false_negative / (false_negative + true_positive) if false_negative + true_positive else 0.0
    return {
        "threshold": threshold,
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "confusion_matrix": [[int(true_negative), int(false_positive)], [int(false_negative), int(true_positive)]],
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "total_cost": float(false_positive * FALSE_POSITIVE_COST + false_negative * FALSE_NEGATIVE_COST),
    }


def choose_threshold(target: pd.Series, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = [threshold_metrics(target, scores, threshold) for threshold in THRESHOLDS]
    table = pd.DataFrame(rows)
    selected = table.sort_values(["total_cost", "f1", "threshold"], ascending=[True, False, True]).iloc[0]
    return float(selected["threshold"]), table


def fit_and_score(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_kind: str,
) -> tuple[Pipeline | HistGradientBoostingClassifier, float, pd.DataFrame]:
    """Fit on train and select a threshold without requesting test predictions."""
    train = dataset[dataset["split"] == "train"]
    validation = dataset[dataset["split"] == "validation"]
    test = dataset[dataset["split"] == "test"]
    model = build_model(model_kind)
    fit_kwargs = {"sample_weight": balanced_sample_weights(train[LABEL_COLUMN])} if model_kind != "behavioral_baseline" else {}
    model.fit(train[feature_columns], train[LABEL_COLUMN], **fit_kwargs)
    validation_scores = model.predict_proba(validation[feature_columns])[:, 1]
    selected_threshold, validation_table = choose_threshold(validation[LABEL_COLUMN], validation_scores)
    return model, selected_threshold, validation_table


def evaluate_test(target: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    result = threshold_metrics(target, scores, threshold)
    result.update(
        {
            "pr_auc": float(average_precision_score(target, scores)),
            "roc_auc": float(roc_auc_score(target, scores)),
        }
    )
    return result


def feature_family(feature: str, graph_columns: list[str]) -> str:
    if "same_device_" in feature or "same_instrument_" in feature:
        return "coordination"
    if feature in graph_columns:
        return "graph"
    if "txn_count_" in feature:
        return "velocity"
    if feature.startswith("seconds_since_"):
        return "recency"
    if "device_count" in feature or "instrument_count" in feature or "merchant_count" in feature or "other_customer" in feature:
        return "entity_reuse"
    return "behavioral"


def save_importance(
    model: HistGradientBoostingClassifier,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    feature_columns: list[str],
    graph_columns: list[str],
) -> pd.DataFrame:
    """Estimate importance on validation data because HGB has no native coefficients."""
    permutation = permutation_importance(
        model,
        validation_features[feature_columns],
        validation_target,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": permutation.importances_mean,
        }
    )
    result["feature_family"] = result["feature"].map(lambda feature: feature_family(feature, graph_columns))
    return result.sort_values("importance", ascending=False)


def main() -> None:
    dataset, behavioral_columns, graph_columns = load_dataset()
    combined_columns = behavioral_columns + graph_columns
    model_specs = {
        "behavioral_baseline": behavioral_columns,
        "graph_only": graph_columns,
        "behavioral_graph": combined_columns,
    }
    test = dataset[dataset["split"] == "test"]
    validation_results: dict[str, pd.DataFrame] = {}
    test_results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    thresholds: dict[str, float] = {}

    for model_name, feature_columns in model_specs.items():
        model, threshold, validation_table = fit_and_score(dataset, feature_columns, model_name)
        fitted_models[model_name] = model
        thresholds[model_name] = threshold
        validation_results[model_name] = validation_table

    # All model choices and thresholds are frozen before any held-out test scores are requested.
    for model_name, feature_columns in model_specs.items():
        test_scores = fitted_models[model_name].predict_proba(test[feature_columns])[:, 1]
        test_results[model_name] = evaluate_test(test[LABEL_COLUMN], test_scores, thresholds[model_name])

    comparison_rows = []
    for model_name, result in test_results.items():
        comparison_rows.append(
            {
                "model": model_name,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "pr_auc": result["pr_auc"],
                "roc_auc": result["roc_auc"],
                "false_positives": result["false_positives"],
                "false_negatives": result["false_negatives"],
                "simulated_cost": result["total_cost"],
                "threshold": thresholds[model_name],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    baseline_result = test_results["behavioral_baseline"]
    fused_result = test_results["behavioral_graph"]
    comparison["pr_auc_absolute_vs_behavioral"] = comparison["pr_auc"] - baseline_result["pr_auc"]
    comparison["pr_auc_relative_vs_behavioral"] = comparison["pr_auc_absolute_vs_behavioral"] / baseline_result["pr_auc"]
    comparison["precision_improvement_vs_behavioral"] = comparison["precision"] - baseline_result["precision"]
    comparison["f1_improvement_vs_behavioral"] = comparison["f1"] - baseline_result["f1"]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": fitted_models["behavioral_graph"], "feature_columns": combined_columns, "risk_score_definition": "100 * P(abuse), not calibrated probability"},
        ARTIFACT_DIR / "sentinel_risk_model.joblib",
    )
    importance = save_importance(
        fitted_models["behavioral_graph"],
        dataset[dataset["split"] == "validation"],
        dataset.loc[dataset["split"] == "validation", LABEL_COLUMN],
        combined_columns,
        graph_columns,
    )
    importance.to_csv(ARTIFACT_DIR / "sentinel_feature_importance.csv", index=False)
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    pd.concat([table.assign(model=model_name) for model_name, table in validation_results.items()], ignore_index=True).to_csv(ARTIFACT_DIR / "ablation_results.csv", index=False)
    with (ARTIFACT_DIR / "sentinel_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "model": "HistGradientBoostingClassifier",
                "feature_count": len(combined_columns),
                "behavioral_feature_count": len(behavioral_columns),
                "graph_feature_count": len(graph_columns),
                "thresholds": thresholds,
                "selected_threshold": thresholds["behavioral_graph"],
                "selection_split": "validation",
                "selection_method": "minimum validation total simulated cost; ties broken by validation F1",
                "false_positive_cost": FALSE_POSITIVE_COST,
                "false_negative_cost": FALSE_NEGATIVE_COST,
                "risk_score": "100 * model probability; not calibrated unless separately calibrated",
                "random_state": 42,
            },
            config_file,
            indent=2,
        )

    print("models: behavioral Logistic Regression, graph-only HistGradientBoosting, fused HistGradientBoosting")
    print(f"feature counts: behavioral={len(behavioral_columns)}, graph={len(graph_columns)}, combined={len(combined_columns)}")
    print(f"sizes: train={(dataset['split'] == 'train').sum()}, validation={(dataset['split'] == 'validation').sum()}, test={(dataset['split'] == 'test').sum()}")
    print(f"class counts: train={dataset[dataset['split'] == 'train'][LABEL_COLUMN].value_counts().sort_index().to_dict()}, validation={dataset[dataset['split'] == 'validation'][LABEL_COLUMN].value_counts().sort_index().to_dict()}, test={test[LABEL_COLUMN].value_counts().sort_index().to_dict()}")
    print(f"thresholds: {thresholds}")
    print("test comparison:")
    print(comparison[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "false_positives", "false_negatives", "simulated_cost"]].round(4).to_string(index=False))
    print(f"fused PR-AUC improvement: absolute={fused_result['pr_auc'] - baseline_result['pr_auc']:.4f}, relative={(fused_result['pr_auc'] - baseline_result['pr_auc']) / baseline_result['pr_auc']:.2%}")
    print("top 15 fused features:")
    print(importance.head(15).to_string(index=False))
    print("test set was scored only after all validation thresholds were frozen")


if __name__ == "__main__":
    main()