from __future__ import annotations

from typing import Any

import pandas as pd


RISK_LEVEL_BANDS = {
    "LOW": (0.0, 39.0, "ALLOW"),
    "MEDIUM": (40.0, 59.0, "MONITOR"),
    "HIGH": (60.0, 79.0, "REVIEW"),
    "CRITICAL": (80.0, 100.0, "ESCALATE"),
}


def _ensure_numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def classify_risk(risk_score: float, risk_threshold: float = 50.0) -> tuple[str, str]:
    """Map a scaled risk score into a prototype operating band and action."""
    score = max(0.0, min(100.0, float(risk_score)))
    threshold = max(0.0, min(100.0, float(risk_threshold)))

    if score >= max(80.0, threshold + 30.0):
        return "CRITICAL", "ESCALATE"
    if score >= threshold:
        return "HIGH", "REVIEW"
    if score >= max(25.0, threshold * 0.6):
        return "MEDIUM", "MONITOR"
    return "LOW", "ALLOW"


def derive_reason_thresholds(
    features_df: pd.DataFrame,
    graph_df: pd.DataFrame | None = None,
    quantile: float = 0.9,
) -> dict[str, float]:
    """Derive reason thresholds from historical training-set distributions only."""
    thresholds: dict[str, float] = {
        "customer_txn_count_30m": float(features_df["customer_txn_count_30m"].quantile(quantile)) if "customer_txn_count_30m" in features_df.columns else 8.0,
        "other_customer_device_count_before": float(features_df["other_customer_device_count_before"].quantile(quantile)) if "other_customer_device_count_before" in features_df.columns else 3.0,
        "other_customer_instrument_count_before": float(features_df["other_customer_instrument_count_before"].quantile(quantile)) if "other_customer_instrument_count_before" in features_df.columns else 3.0,
        "customer_merchant_count_before": float(features_df["customer_merchant_count_before"].quantile(quantile)) if "customer_merchant_count_before" in features_df.columns else 5.0,
        "seconds_since_customer_last_transaction": float(features_df["seconds_since_customer_last_transaction"].quantile(1.0 - quantile)) if "seconds_since_customer_last_transaction" in features_df.columns else 900.0,
    }

    if graph_df is not None:
        for column in [
            "other_customers_on_device",
            "other_customers_on_instrument",
            "shared_device_count_for_customer",
            "shared_instrument_count_for_customer",
            "customers_connected_via_device",
            "customers_connected_via_instrument",
            "unique_customer_neighbors",
        ]:
            if column in graph_df.columns:
                thresholds[column] = float(graph_df[column].quantile(quantile))
    return thresholds
