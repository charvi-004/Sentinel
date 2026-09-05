"""Build point-in-time-safe risk features from Sentinel transactions."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRANSACTION_COLUMNS = [
    "transaction_id",
    "timestamp",
    "merchant_id",
    "customer_id",
    "device_id",
    "payment_instrument_id",
    "amount",
    "currency",
    "status",
]
LABEL_COLUMNS = ["transaction_id", "is_abuse", "abuse_type", "abuse_cluster_id"]
ENTITY_COLUMNS = {"transaction_id", "customer_id", "device_id", "merchant_id", "payment_instrument_id"}
WINDOWS = {
    "customer": {"5m": 300, "30m": 1_800, "1h": 3_600, "24h": 86_400},
    "device": {"5m": 300, "1h": 3_600},
    "instrument": {"5m": 300, "1h": 3_600},
}


@dataclass
class CustomerHistory:
    amounts: list[float]
    merchants: set[str]
    devices: set[str]
    instruments: set[str]
    last_timestamp: pd.Timestamp | None = None


@dataclass
class RelationshipHistory:
    accounts: set[str]
    merchants: set[str]
    last_timestamp: pd.Timestamp | None = None


def _empty_row() -> dict[str, float]:
    return {
        "customer_transaction_count_before": 0,
        "customer_avg_amount_before": 0.0,
        "customer_median_amount_before": 0.0,
        "customer_amount_std_before": 0.0,
        "customer_merchant_count_before": 0,
        "customer_device_count_before": 0,
        "customer_instrument_count_before": 0,
        "customer_txn_count_5m": 0,
        "customer_txn_count_30m": 0,
        "customer_txn_count_1h": 0,
        "customer_txn_count_24h": 0,
        "device_txn_count_5m": 0,
        "device_txn_count_1h": 0,
        "instrument_txn_count_5m": 0,
        "instrument_txn_count_1h": 0,
        "device_account_count_before": 0,
        "device_merchant_count_before": 0,
        "instrument_account_count_before": 0,
        "instrument_merchant_count_before": 0,
        "device_merchant_count_before": 0,
        "instrument_merchant_count_before": 0,
        "other_customer_device_count_before": 0,
        "other_customer_instrument_count_before": 0,
        "seconds_since_customer_last_transaction": -1.0,
        "seconds_since_device_last_transaction": -1.0,
        "seconds_since_instrument_last_transaction": -1.0,
        "amount_vs_customer_mean": 0.0,
        "amount_vs_customer_median": 0.0,
        "customer_amount_zscore": 0.0,
    }


def _prune(queue: deque[pd.Timestamp], timestamp: pd.Timestamp, window_seconds: int) -> None:
    cutoff = timestamp - pd.Timedelta(seconds=window_seconds)
    while queue and queue[0] <= cutoff:
        queue.popleft()


def _count_before(queue: deque[pd.Timestamp], timestamp: pd.Timestamp, window_seconds: int) -> int:
    _prune(queue, timestamp, window_seconds)
    return len(queue)


def _seconds_since(last_timestamp: pd.Timestamp | None, timestamp: pd.Timestamp) -> float:
    if last_timestamp is None:
        return -1.0
    return (timestamp - last_timestamp).total_seconds()


def _history_features(
    transaction: pd.Series,
    customer_history: CustomerHistory,
    device_history: RelationshipHistory,
    instrument_history: RelationshipHistory,
    customer_windows: dict[str, deque[pd.Timestamp]],
    device_windows: dict[str, deque[pd.Timestamp]],
    instrument_windows: dict[str, deque[pd.Timestamp]],
) -> dict[str, float]:
    """Calculate one row using state containing only strictly earlier timestamps."""
    features = _empty_row()
    amount = float(transaction["amount"])
    if customer_history.amounts:
        history = np.asarray(customer_history.amounts, dtype=float)
        mean = float(history.mean())
        median = float(np.median(history))
        std = float(history.std(ddof=1)) if len(history) > 1 else 0.0
        features.update(
            customer_transaction_count_before=len(history),
            customer_avg_amount_before=mean,
            customer_median_amount_before=median,
            customer_amount_std_before=std,
            customer_merchant_count_before=len(customer_history.merchants),
            customer_device_count_before=len(customer_history.devices),
            customer_instrument_count_before=len(customer_history.instruments),
            amount_vs_customer_mean=amount - mean,
            amount_vs_customer_median=amount - median,
            customer_amount_zscore=(amount - mean) / std if std > 0 else 0.0,
        )

    customer_id = str(transaction["customer_id"])
    device_id = str(transaction["device_id"])
    instrument_id = str(transaction["payment_instrument_id"])
    merchant_id = str(transaction["merchant_id"])
    timestamp = transaction["timestamp"]
    features.update(
        customer_txn_count_5m=_count_before(customer_windows[customer_id + "|5m"], timestamp, WINDOWS["customer"]["5m"]),
        customer_txn_count_30m=_count_before(customer_windows[customer_id + "|30m"], timestamp, WINDOWS["customer"]["30m"]),
        customer_txn_count_1h=_count_before(customer_windows[customer_id + "|1h"], timestamp, WINDOWS["customer"]["1h"]),
        customer_txn_count_24h=_count_before(customer_windows[customer_id + "|24h"], timestamp, WINDOWS["customer"]["24h"]),
        device_txn_count_5m=_count_before(device_windows[device_id + "|5m"], timestamp, WINDOWS["device"]["5m"]),
        device_txn_count_1h=_count_before(device_windows[device_id + "|1h"], timestamp, WINDOWS["device"]["1h"]),
        instrument_txn_count_5m=_count_before(instrument_windows[instrument_id + "|5m"], timestamp, WINDOWS["instrument"]["5m"]),
        instrument_txn_count_1h=_count_before(instrument_windows[instrument_id + "|1h"], timestamp, WINDOWS["instrument"]["1h"]),
        device_account_count_before=len(device_history.accounts),
        device_merchant_count_before=len(device_history.merchants),
        instrument_account_count_before=len(instrument_history.accounts),
        instrument_merchant_count_before=len(instrument_history.merchants),
        other_customer_device_count_before=max(0, len(device_history.accounts - {customer_id})),
        other_customer_instrument_count_before=max(0, len(instrument_history.accounts - {customer_id})),
        seconds_since_customer_last_transaction=_seconds_since(customer_history.last_timestamp, timestamp),
        seconds_since_device_last_transaction=_seconds_since(device_history.last_timestamp, timestamp),
        seconds_since_instrument_last_transaction=_seconds_since(instrument_history.last_timestamp, timestamp),
    )
    return features


def _update_history(
    transaction: pd.Series,
    customer_history: CustomerHistory,
    device_history: RelationshipHistory,
    instrument_history: RelationshipHistory,
    customer_windows: dict[str, deque[pd.Timestamp]],
    device_windows: dict[str, deque[pd.Timestamp]],
    instrument_windows: dict[str, deque[pd.Timestamp]],
) -> None:
    """Add a transaction to state after all rows at its timestamp are scored."""
    customer_id = str(transaction["customer_id"])
    device_id = str(transaction["device_id"])
    instrument_id = str(transaction["payment_instrument_id"])
    merchant_id = str(transaction["merchant_id"])
    timestamp = transaction["timestamp"]
    customer_history.amounts.append(float(transaction["amount"]))
    customer_history.merchants.add(merchant_id)
    customer_history.devices.add(device_id)
    customer_history.instruments.add(instrument_id)
    customer_history.last_timestamp = timestamp
    device_history.accounts.add(customer_id)
    device_history.merchants.add(merchant_id)
    device_history.last_timestamp = timestamp
    instrument_history.accounts.add(customer_id)
    instrument_history.merchants.add(merchant_id)
    instrument_history.last_timestamp = timestamp
    for window_name in WINDOWS["customer"]:
        customer_windows[customer_id + "|" + window_name].append(timestamp)
    for window_name in WINDOWS["device"]:
        device_windows[device_id + "|" + window_name].append(timestamp)
    for window_name in WINDOWS["instrument"]:
        instrument_windows[instrument_id + "|" + window_name].append(timestamp)


def build_point_in_time_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build numeric, historical features in chronological order."""
    ordered = transactions.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="raise")
    ordered = ordered.sort_values(["timestamp", "transaction_id"], kind="mergesort")
    customer_histories: defaultdict[str, CustomerHistory] = defaultdict(lambda: CustomerHistory([], set(), set(), set()))
    device_histories: defaultdict[str, RelationshipHistory] = defaultdict(lambda: RelationshipHistory(set(), set()))
    instrument_histories: defaultdict[str, RelationshipHistory] = defaultdict(lambda: RelationshipHistory(set(), set()))
    customer_windows: defaultdict[str, deque[pd.Timestamp]] = defaultdict(deque)
    device_windows: defaultdict[str, deque[pd.Timestamp]] = defaultdict(deque)
    instrument_windows: defaultdict[str, deque[pd.Timestamp]] = defaultdict(deque)
    feature_rows: dict[str, dict[str, float]] = {}

    for _, timestamp_group in ordered.groupby("timestamp", sort=False):
        pending: list[tuple[pd.Series, dict[str, float]]] = []
        for _, transaction in timestamp_group.iterrows():
            customer_id = str(transaction["customer_id"])
            device_id = str(transaction["device_id"])
            instrument_id = str(transaction["payment_instrument_id"])
            features = _history_features(
                transaction,
                customer_histories[customer_id],
                device_histories[device_id],
                instrument_histories[instrument_id],
                customer_windows,
                device_windows,
                instrument_windows,
            )
            features.update(
                amount=float(transaction["amount"]),
                log_amount=float(np.log1p(transaction["amount"])),
                hour=int(transaction["timestamp"].hour),
                day_of_week=int(transaction["timestamp"].dayofweek),
                is_weekend=int(transaction["timestamp"].dayofweek >= 5),
            )
            pending.append((transaction, features))
        for transaction, features in pending:
            feature_rows[str(transaction["transaction_id"])] = features
            _update_history(
                transaction,
                customer_histories[str(transaction["customer_id"])],
                device_histories[str(transaction["device_id"])],
                instrument_histories[str(transaction["payment_instrument_id"])],
                customer_windows,
                device_windows,
                instrument_windows,
            )

    features = pd.DataFrame.from_dict(feature_rows, orient="index")
    features.index.name = "transaction_id"
    return features.reset_index()


def _metadata() -> list[dict[str, Any]]:
    families: dict[str, tuple[str, str]] = {}
    for name in ["amount", "log_amount", "hour", "day_of_week", "is_weekend"]:
        families[name] = ("basic_transaction", "Observable transaction amount or calendar component.")
    for name in [column for column in _empty_row() if column.startswith("customer_") or column.startswith("amount_vs_")]:
        families[name] = ("customer_behavior", "Historical customer behavior available before the transaction.")
    for name in [column for column in _empty_row() if "txn_count_" in column]:
        families[name] = ("velocity", "Count of strictly earlier transactions in a bounded time window.")
    for name in [column for column in _empty_row() if "account_count" in column or "merchant_count" in column or "other_customer" in column]:
        families[name] = ("entity_reuse", "Historical entity relationship count before the transaction.")
    for name in [column for column in _empty_row() if column.startswith("seconds_since_")]:
        families[name] = ("recency", "Seconds since the entity's previous transaction; -1 for first observation.")
    return [
        {
            "feature_name": name,
            "feature_family": family,
            "description": description,
            "point_in_time_safe": True,
            "expected_data_type": "integer" if name in {"hour", "day_of_week", "is_weekend"} or name.endswith("_count_before") or "txn_count_" in name or "account_count" in name or "merchant_count" in name or "other_customer" in name else "float",
        }
        for name, (family, description) in families.items()
    ]


def _build_splits(transactions: pd.DataFrame) -> pd.DataFrame:
    ordered = transactions.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:validation_end] = "validation"
    return pd.DataFrame({"transaction_id": ordered["transaction_id"], "split": split})


def _validate_outputs(features: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame, source: pd.DataFrame) -> None:
    model_columns = [column for column in features.columns if column != "transaction_id"]
    forbidden = (set(model_columns) & (set(LABEL_COLUMNS) | ENTITY_COLUMNS))
    if forbidden:
        raise ValueError(f"Forbidden model columns: {sorted(forbidden)}")
    if features[model_columns].isna().any().any():
        raise ValueError("Feature matrix contains NaN values")
    if not np.isfinite(features[model_columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains infinite values")
    source_ids = set(source["transaction_id"])
    if set(features["transaction_id"]) != source_ids or features["transaction_id"].duplicated().any():
        raise ValueError("Feature rows do not map exactly once to source transactions")
    if set(labels["transaction_id"]) != source_ids or set(splits["transaction_id"]) != source_ids:
        raise ValueError("Labels or splits do not map exactly to source transactions")
    if splits["transaction_id"].duplicated().any():
        raise ValueError("Splits contain duplicate transaction IDs")
    if set(splits["split"]) != {"train", "validation", "test"}:
        raise ValueError("Splits must contain train, validation, and test")


def _manual_rolling_check(transactions: pd.DataFrame, features: pd.DataFrame) -> None:
    """Compare selected five-minute counts with a direct historical calculation."""
    sample = transactions.sort_values("timestamp").head(min(50, len(transactions)))
    feature_by_id = features.set_index("transaction_id")
    for _, row in sample.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        reference = transactions[
            (transactions["customer_id"] == row["customer_id"])
            & (pd.to_datetime(transactions["timestamp"], utc=True) < timestamp)
            & (pd.to_datetime(transactions["timestamp"], utc=True) > timestamp - pd.Timedelta(minutes=5))
        ]
        expected = len(reference)
        actual = int(feature_by_id.loc[row["transaction_id"], "customer_txn_count_5m"])
        if expected != actual:
            raise ValueError(f"Rolling reference mismatch for {row['transaction_id']}: expected={expected}, actual={actual}")


def main() -> None:
    transaction_path = RAW_DIR / "transactions.csv"
    ground_truth_path = RAW_DIR / "ground_truth.csv"
    if not transaction_path.exists() or not ground_truth_path.exists():
        raise FileNotFoundError("Expected data/raw/transactions.csv and data/raw/ground_truth.csv")
    transactions = pd.read_csv(transaction_path)
    ground_truth = pd.read_csv(ground_truth_path)
    missing = sorted(set(TRANSACTION_COLUMNS) - set(transactions.columns))
    if missing:
        raise ValueError(f"Missing transaction columns: {missing}")
    if set(LABEL_COLUMNS) - set(ground_truth.columns):
        raise ValueError("Ground truth is missing required label columns")
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], utc=True, errors="raise")
    features = build_point_in_time_features(transactions)
    labels = ground_truth[LABEL_COLUMNS].copy()
    splits = _build_splits(transactions)
    _validate_outputs(features, labels, splits, transactions)
    _manual_rolling_check(transactions, features)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_DIR / "features.csv", index=False)
    labels.to_csv(PROCESSED_DIR / "labels.csv", index=False)
    splits.to_csv(PROCESSED_DIR / "splits.csv", index=False)
    with (PROCESSED_DIR / "feature_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(_metadata(), metadata_file, indent=2)

    model_columns = [column for column in features.columns if column != "transaction_id"]
    print(f"rows: {len(features)}")
    print(f"features: {len(model_columns)}")
    print(f"feature families: {sorted({entry['feature_family'] for entry in _metadata()})}")
    print(f"splits: {splits['split'].value_counts().sort_index().to_dict()}")
    print(f"missing values: {int(features[model_columns].isna().sum().sum())}")
    print(f"infinite values: {int((~np.isfinite(features[model_columns].to_numpy(dtype=float))).sum())}")
    print("point-in-time checks: passed")


if __name__ == "__main__":
    main()