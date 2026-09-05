from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .risk_policy import classify_risk, derive_reason_thresholds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "artifacts" / "baseline_logistic_regression.joblib"
CONFIG_PATH = PROJECT_ROOT / "ml" / "models" / "artifacts" / "baseline_config.json"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "graph_features.csv"
TRANSACTIONS_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits.csv"


def _load_model() -> Any:
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing baseline model artifact: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _load_training_thresholds() -> dict[str, float]:
    features = pd.read_csv(FEATURES_PATH)
    graph = pd.read_csv(GRAPH_PATH)
    splits = pd.read_csv(SPLITS_PATH)
    train_txns = splits.loc[splits["split"] == "train", "transaction_id"]
    train_features = features[features["transaction_id"].isin(train_txns)].copy()
    train_graph = graph[graph["transaction_id"].isin(train_txns)].copy()
    return derive_reason_thresholds(train_features, train_graph)


def _row_for_transaction(transaction_id: str, df: pd.DataFrame) -> pd.Series:
    matches = df.loc[df["transaction_id"] == transaction_id]
    if matches.empty:
        raise ValueError(f"No record for transaction_id={transaction_id}")
    return matches.iloc[0]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _summarize_network(graph_row: pd.Series) -> dict[str, int]:
    connected_customers = int(
        graph_row.get("unique_customer_neighbors", 0)
        or graph_row.get("customers_connected_via_device", 0)
        + graph_row.get("customers_connected_via_instrument", 0)
    )
    connected_merchants = int(graph_row.get("unique_merchant_neighbors", 0) or graph_row.get("merchant_degree_before", 0))
    connected_devices = int(graph_row.get("customers_connected_via_device", 0) or graph_row.get("device_customer_count_before", 0))
    connected_instruments = int(graph_row.get("customers_connected_via_instrument", 0) or graph_row.get("instrument_customer_count_before", 0))
    return {
        "connected_customers": connected_customers,
        "connected_merchants": connected_merchants,
        "connected_devices": connected_devices,
        "connected_instruments": connected_instruments,
    }


def _make_reason(
    reason_type: str,
    description: str,
    evidence: dict[str, Any],
    observed_value: Any,
    threshold: float,
) -> dict[str, Any]:
    value = _safe_float(observed_value, 0.0)
    threshold_value = _safe_float(threshold, 0.0)
    severity = "HIGH" if value >= threshold_value * 1.5 else "MEDIUM" if value >= threshold_value else "LOW"
    return {
        "type": reason_type,
        "severity": severity,
        "description": description,
        "evidence": evidence,
    }


def _generate_reasons(feature_row: pd.Series, graph_row: pd.Series, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []

    def add_reason(reason_type: str, observed: Any, column: str, threshold: float, template: str):
        if observed is None:
            return
        value = _safe_float(observed, 0.0)
        if value > threshold:
            reasons.append(
                _make_reason(
                    reason_type,
                    template.format(value=value, threshold=threshold),
                    {column: int(round(value)) if isinstance(value, float) and value.is_integer() else value},
                    observed,
                    threshold,
                )
            )

    add_reason(
        "HIGH VELOCITY",
        feature_row.get("customer_txn_count_30m", 0),
        "customer_txn_count_30m",
        thresholds.get("customer_txn_count_30m", 8.0),
        "{value:.0f} transactions from this customer were observed within the previous 30 minutes.",
    )
    add_reason(
        "SHARED_DEVICE",
        feature_row.get("other_customer_device_count_before", 0),
        "other_customer_device_count_before",
        thresholds.get("other_customer_device_count_before", 3.0),
        "This device was previously associated with {value:.0f} other customers.",
    )
    add_reason(
        "SHARED_INSTRUMENT",
        feature_row.get("other_customer_instrument_count_before", 0),
        "other_customer_instrument_count_before",
        thresholds.get("other_customer_instrument_count_before", 3.0),
        "This payment instrument was previously associated with {value:.0f} other customers.",
    )
    add_reason(
        "CROSS_MERCHANT",
        feature_row.get("customer_merchant_count_before", 0),
        "customer_merchant_count_before",
        thresholds.get("customer_merchant_count_before", 5.0),
        "This customer has transacted with {value:.0f} merchants in the observed history.",
    )
    recent_seconds = _safe_float(feature_row.get("seconds_since_customer_last_transaction", -1.0), -1.0)
    recent_threshold = _safe_float(thresholds.get("seconds_since_customer_last_transaction", 900.0), 900.0)
    if recent_seconds >= 0 and recent_seconds < recent_threshold:
        reasons.append(
            _make_reason(
                "RECENT_ACTIVITY",
                f"The customer last transacted {recent_seconds:.0f} seconds ago, which is unusually recent.",
                {"seconds_since_customer_last_transaction": int(round(recent_seconds))},
                recent_seconds,
                recent_threshold,
            )
        )

    other_customers_on_device = _safe_float(graph_row.get("other_customers_on_device", 0), 0.0)
    if other_customers_on_device > thresholds.get("other_customers_on_device", 0.0):
        reasons.append(
            _make_reason(
                "DEVICE_CONNECTION",
                f"This device is connected to {other_customers_on_device:.0f} other customers in the historical graph.",
                {"other_customers_on_device": int(round(other_customers_on_device))},
                other_customers_on_device,
                thresholds.get("other_customers_on_device", 0.0),
            )
        )

    other_customers_on_instrument = _safe_float(graph_row.get("other_customers_on_instrument", 0), 0.0)
    if other_customers_on_instrument > thresholds.get("other_customers_on_instrument", 0.0):
        reasons.append(
            _make_reason(
                "INSTRUMENT_CONNECTION",
                f"This payment instrument is linked to {other_customers_on_instrument:.0f} other customers in the historical graph.",
                {"other_customers_on_instrument": int(round(other_customers_on_instrument))},
                other_customers_on_instrument,
                thresholds.get("other_customers_on_instrument", 0.0),
            )
        )

    return sorted(reasons, key=lambda item: ("HIGH" if item["severity"] == "HIGH" else "MEDIUM" if item["severity"] == "MEDIUM" else "LOW", item["type"]), reverse=True)


def assess_transaction(
    transaction_id: str,
    model: Any,
    risk_threshold: float,
    features_df: pd.DataFrame,
    graph_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    reason_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, investigator-friendly risk assessment for one transaction."""
    feature_row = _row_for_transaction(transaction_id, features_df)
    graph_row = _row_for_transaction(transaction_id, graph_df) if not graph_df.empty else pd.Series(dtype=object)
    transaction_row = _row_for_transaction(transaction_id, transactions_df)

    feature_df = pd.DataFrame([feature_row]).drop(columns=["transaction_id"], errors="ignore")
    probability = float(model.predict_proba(feature_df)[0, 1])
    risk_score = probability * 100.0
    risk_level, recommended_action = classify_risk(risk_score, risk_threshold)

    thresholds = reason_thresholds or _load_training_thresholds()
    reasons = _generate_reasons(feature_row, graph_row, thresholds)
    if risk_level in {"HIGH", "CRITICAL"} and not reasons:
        reasons = [
            _make_reason(
                "BEHAVIORAL_SIGNAL",
                f"The model produced a {risk_score:.1f} risk score, which exceeds the validated operating threshold.",
                {"risk_score": round(risk_score, 1), "risk_threshold": risk_threshold},
                risk_score,
                risk_threshold,
            )
        ]

    network_context = _summarize_network(graph_row)
    amount = _safe_float(transaction_row.get("amount"), 0.0)
    transaction_context = {
        "amount": amount,
        "currency": str(transaction_row.get("currency", "USD")),
        "timestamp": str(transaction_row.get("timestamp", "")),
    }
    connected_exposure = amount * max(0.0, min(1.0, network_context["connected_customers"] / 10.0))
    transaction_context["estimated_connected_exposure"] = round(connected_exposure, 2)
    transaction_context["exposure_note"] = "simulated historical connected exposure estimate; not a financial loss estimate"

    assessment = {
        "transaction_id": str(transaction_id),
        "risk_score": round(risk_score, 1),
        "risk_probability": round(probability, 6),
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "reasons": reasons,
        "network_context": network_context,
        "transaction": transaction_context,
    }
    return assessment


def score_all_transactions(
    features_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    model: Any,
    risk_threshold: float,
) -> pd.DataFrame:
    """Score every row in a feature table in one vectorized model call and return transaction metadata."""
    if "transaction_id" not in features_df.columns:
        raise ValueError("features_df must include a transaction_id column")

    model_features = features_df.drop(columns=["transaction_id"], errors="ignore")
    probabilities = model.predict_proba(model_features)
    probability_array = np.asarray(probabilities)
    if probability_array.ndim == 1:
        risk_probabilities = probability_array.astype(float)
    else:
        risk_probabilities = probability_array[:, 1].astype(float)

    scored_df = pd.DataFrame({
        "transaction_id": features_df["transaction_id"],
        "risk_score": risk_probabilities * 100.0,
    })

    classification_rows = [classify_risk(float(score), risk_threshold) for score in scored_df["risk_score"]]
    scored_df["risk_level"] = [entry[0] for entry in classification_rows]
    scored_df["recommended_action"] = [entry[1] for entry in classification_rows]

    transaction_metadata = transactions_df[["transaction_id", "amount", "currency", "timestamp"]].copy()
    merged_df = scored_df.merge(transaction_metadata, on="transaction_id", how="left")

    merged_df["amount"] = merged_df["amount"].fillna(0.0).apply(lambda value: _safe_float(value, 0.0))
    merged_df["currency"] = merged_df["currency"].fillna("USD").astype(str)
    merged_df["timestamp"] = merged_df["timestamp"].fillna("").astype(str)

    return merged_df[["transaction_id", "amount", "currency", "timestamp", "risk_score", "risk_level", "recommended_action"]]


@lru_cache(maxsize=1)
def load_default_engine() -> tuple[Any, float, dict[str, Any]]:
    model = _load_model()
    config = _load_config()
    threshold = float(config.get("threshold", 0.5)) * 100.0
    return model, threshold, config
