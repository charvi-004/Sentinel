"""Validate Sentinel's generated payment data without building model features."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_TRANSACTION_COLUMNS = [
    "transaction_id",
    "timestamp",
    "merchant_id",
    "customer_id",
    "device_id",
    "payment_instrument_id",
    "ip_hash",
    "amount",
    "currency",
    "status",
]
EXPECTED_GROUND_TRUTH_COLUMNS = ["transaction_id", "is_abuse", "abuse_type", "abuse_cluster_id"]
EXPECTED_START = pd.Timestamp("2025-01-01", tz="UTC")
EXPECTED_END = pd.Timestamp("2025-05-01", tz="UTC")


class ValidationReport:
    """Collect checks, warnings, and failures while printing a readable report."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.failures: list[str] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}{': ' + detail if detail else ''}")
        if not passed:
            self.failures.append(f"{name}: {detail}" if detail else name)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _format_number(value: float) -> str:
    return f"{value:,.2f}"


def _print_distribution(title: str, series: pd.Series) -> None:
    print(f"{title}: {series.value_counts(dropna=False).to_dict()}")


def _reuse_report(name: str, series: pd.Series) -> None:
    counts = series.value_counts()
    distribution = counts.value_counts().sort_index().to_dict()
    values = counts.to_numpy()
    print(
        f"{name}: entities={len(counts)}, mean={values.mean():.2f}, "
        f"median={np.median(values):.2f}, max={values.max()}, "
        f"p95={np.percentile(values, 95):.2f}, p99={np.percentile(values, 99):.2f}"
    )
    print(f"  transactions per entity -> entity_count: {distribution}")


def validate_schema(transactions: pd.DataFrame, ground_truth: pd.DataFrame, report: ValidationReport) -> None:
    missing_transactions = sorted(set(EXPECTED_TRANSACTION_COLUMNS) - set(transactions.columns))
    unexpected_transactions = sorted(set(transactions.columns) - set(EXPECTED_TRANSACTION_COLUMNS))
    missing_ground_truth = sorted(set(EXPECTED_GROUND_TRUTH_COLUMNS) - set(ground_truth.columns))
    unexpected_ground_truth = sorted(set(ground_truth.columns) - set(EXPECTED_GROUND_TRUTH_COLUMNS))
    report.check("Schema", not missing_transactions and not missing_ground_truth, f"missing transaction={missing_transactions}, missing ground_truth={missing_ground_truth}")
    if unexpected_transactions:
        report.warn(f"Unexpected transaction columns: {unexpected_transactions}")
    if unexpected_ground_truth:
        report.warn(f"Unexpected ground-truth columns: {unexpected_ground_truth}")
    if missing_transactions or missing_ground_truth:
        return
    leaked_labels = {"is_abuse", "abuse_type", "abuse_cluster_id"}.intersection(transactions.columns)
    if leaked_labels:
        report.failures.append("Ground-truth fields are present in transactions")
        print(f"[FAIL] Ground-truth leakage: label fields found in transactions: {sorted(leaked_labels)}")
    else:
        report.check("Ground-truth leakage boundary", True, "no label fields in transactions")


def validate_basic_integrity(transactions: pd.DataFrame, ground_truth: pd.DataFrame, report: ValidationReport) -> None:
    transaction_duplicates = int(transactions["transaction_id"].duplicated().sum())
    ground_truth_duplicates = int(ground_truth["transaction_id"].duplicated().sum())
    ip_duplicates = int(transactions["ip_hash"].duplicated().sum())
    report.check("Transaction IDs", transaction_duplicates == 0, f"duplicates={transaction_duplicates}")
    report.check("Ground-truth transaction IDs", ground_truth_duplicates == 0, f"duplicates={ground_truth_duplicates}")
    report.check("Transaction count", len(transactions) > 0, f"count={len(transactions)}")
    if ip_duplicates:
        report.warn(f"IP hashes are not unique: duplicate rows={ip_duplicates}; repeated IPs are not automatically fraud")
    else:
        report.check("IP hash uniqueness", True, "all hashes unique")

    missing = transactions.isna().sum()
    empty = transactions.astype("string").apply(lambda column: column.str.strip().eq("").sum())
    print("Missing values by transaction column:", missing[missing > 0].to_dict() or "none")
    print("Empty strings by transaction column:", empty[empty > 0].to_dict() or "none")
    report.check("Missing transaction values", int(missing.sum()) == 0, f"total={int(missing.sum())}")
    report.check("Empty transaction strings", int(empty.sum()) == 0, f"total={int(empty.sum())}")


def validate_references(transactions: pd.DataFrame, report: ValidationReport) -> None:
    master_files = {
        "merchant_id": "merchants.csv",
        "customer_id": "customers.csv",
        "device_id": "devices.csv",
        "payment_instrument_id": "instruments.csv",
    }
    for key, filename in master_files.items():
        path = RAW_DIR / filename
        if path.exists():
            master = pd.read_csv(path)
            master_key = "instrument_id" if key == "payment_instrument_id" else key
            valid = set(master[master_key].dropna()) if master_key in master else set()
            source = f"{filename}.{master_key}"
        else:
            valid = set(transactions[key].dropna())
            source = "observed transaction IDs (master table unavailable)"
            report.warn(f"No {filename}; {key} references were checked only against observed transaction IDs")
        invalid = sorted(set(transactions[key].dropna()) - valid)
        report.check(f"References: {key}", not invalid, f"invalid={len(invalid)}, source={source}")


def validate_amounts_and_time(transactions: pd.DataFrame, report: ValidationReport) -> None:
    amounts = pd.to_numeric(transactions["amount"], errors="coerce")
    amount_invalid = int(amounts.isna().sum() + (amounts <= 0).sum())
    report.check("Amount validity", amount_invalid == 0, f"invalid={amount_invalid}")
    if amounts.notna().any():
        print(
            "Amount statistics: "
            f"min={_format_number(amounts.min())}, max={_format_number(amounts.max())}, "
            f"mean={_format_number(amounts.mean())}, median={_format_number(amounts.median())}, "
            f"std={_format_number(amounts.std())}, p01={_format_number(amounts.quantile(.01))}, "
            f"p25={_format_number(amounts.quantile(.25))}, p75={_format_number(amounts.quantile(.75))}, "
            f"p99={_format_number(amounts.quantile(.99))}"
        )

    timestamps = pd.to_datetime(transactions["timestamp"], errors="coerce", utc=True)
    invalid_timestamps = int(timestamps.isna().sum())
    in_period = timestamps.between(EXPECTED_START, EXPECTED_END, inclusive="left")
    report.check("Timestamp parsing", invalid_timestamps == 0, f"invalid={invalid_timestamps}")
    report.check("Timestamp period", bool(in_period.all()), f"outside_expected_period={int((~in_period).sum())}")
    if timestamps.notna().any():
        duration = timestamps.max() - timestamps.min()
        daily = transactions.assign(_timestamp=timestamps).set_index("_timestamp").resample("D").size()
        hourly = transactions.assign(_timestamp=timestamps).set_index("_timestamp").resample("h").size()
        print(f"Timestamp statistics: earliest={timestamps.min()}, latest={timestamps.max()}, duration_days={duration.total_seconds() / 86400:.2f}")
        print(f"  transactions per day: mean={daily.mean():.2f}, median={daily.median():.2f}, max={daily.max()}")
        print(f"  transactions per hour: mean={hourly.mean():.2f}, median={hourly.median():.2f}, max={hourly.max()}")


def validate_reuse_and_categories(transactions: pd.DataFrame) -> None:
    print("\nCATEGORICAL DISTRIBUTIONS")
    _print_distribution("Currency", transactions["currency"])
    _print_distribution("Status", transactions["status"])
    print("\nENTITY REUSE")
    _reuse_report("Customers", transactions["customer_id"])
    _reuse_report("Devices", transactions["device_id"])
    _reuse_report("Instruments", transactions["payment_instrument_id"])
    _reuse_report("Merchants", transactions["merchant_id"])


def validate_ground_truth(transactions: pd.DataFrame, ground_truth: pd.DataFrame, report: ValidationReport) -> None:
    transaction_ids = set(transactions["transaction_id"])
    ground_truth_ids = set(ground_truth["transaction_id"])
    counts = ground_truth["transaction_id"].value_counts()
    missing_records = len(transaction_ids - ground_truth_ids)
    extra_records = len(ground_truth_ids - transaction_ids)
    wrong_cardinality = int((counts.reindex(transactions["transaction_id"], fill_value=0) != 1).sum())
    report.check("Ground-truth coverage", missing_records == 0 and extra_records == 0 and wrong_cardinality == 0, f"missing={missing_records}, extra={extra_records}, not_exactly_one={wrong_cardinality}")

    abuse_values = ground_truth["is_abuse"].map({True: True, False: True, "true": True, "false": True, "True": True, "False": True}).notna()
    report.check("Ground-truth boolean values", bool(abuse_values.all()), f"invalid={int((~abuse_values).sum())}")
    normal = ground_truth["is_abuse"].eq(False)
    abuse = ground_truth["is_abuse"].eq(True)
    type_consistent = bool((ground_truth.loc[normal, "abuse_type"] == "none").all() and (ground_truth.loc[abuse, "abuse_type"] != "none").all())
    cluster_consistent = bool((ground_truth.loc[normal, "abuse_cluster_id"] == "none").all() and (ground_truth.loc[abuse, "abuse_cluster_id"] != "none").all())
    report.check("Abuse type semantics", type_consistent)
    report.check("Abuse cluster semantics", cluster_consistent)


def validate_abuse_sanity(transactions: pd.DataFrame, ground_truth: pd.DataFrame) -> None:
    joined = transactions.merge(ground_truth, on="transaction_id", how="inner", validate="one_to_one")
    abuse = joined[joined["is_abuse"] == True]  # noqa: E712 - explicit ground-truth section
    print("\nABUSE SANITY")
    print(f"Total abuse transactions: {len(abuse)} ({len(abuse) / len(joined):.2%})")
    print(f"Abuse clusters: {abuse['abuse_cluster_id'].nunique()}")
    print(f"Transactions by abuse type: {abuse['abuse_type'].value_counts().to_dict()}")
    print(f"Cluster sizes: {abuse['abuse_cluster_id'].value_counts().to_dict()}")
    for label, column in [("merchants", "merchant_id"), ("customers", "customer_id"), ("devices", "device_id"), ("instruments", "payment_instrument_id")]:
        print(f"Abuse {label}: {abuse[column].nunique()}")


def validate_duplicates(transactions: pd.DataFrame) -> None:
    keys = ["customer_id", "merchant_id", "amount", "timestamp"]
    duplicates = int(transactions.duplicated(keys, keep=False).sum())
    groups = int(transactions.loc[transactions.duplicated(keys, keep=False)].groupby(keys, dropna=False).ngroups)
    print(f"\nExact transaction signatures ({', '.join(keys)}): rows={duplicates}, groups={groups}")
    if duplicates:
        print("  These are reported for inspection and are not classified as fraud.")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    transaction_path = RAW_DIR / "transactions.csv"
    ground_truth_path = RAW_DIR / "ground_truth.csv"
    if not transaction_path.exists() or not ground_truth_path.exists():
        missing = [str(path) for path in (transaction_path, ground_truth_path) if not path.exists()]
        raise FileNotFoundError(f"Required input files are missing: {', '.join(missing)}")
    return pd.read_csv(transaction_path), pd.read_csv(ground_truth_path)


def main() -> int:
    print("====================================")
    print("SENTINEL DATA VALIDATION REPORT")
    print("====================================")
    try:
        transactions, ground_truth = load_inputs()
    except (OSError, ValueError) as error:
        print(f"[FAIL] Input loading: {error}")
        return 1

    report = ValidationReport()
    validate_schema(transactions, ground_truth, report)
    required_available = set(EXPECTED_TRANSACTION_COLUMNS).issubset(transactions.columns) and set(EXPECTED_GROUND_TRUTH_COLUMNS).issubset(ground_truth.columns)
    if not required_available:
        print("\nFINAL RESULT: FAIL")
        return 1

    validate_basic_integrity(transactions, ground_truth, report)
    validate_references(transactions, report)
    validate_amounts_and_time(transactions, report)
    validate_reuse_and_categories(transactions)
    validate_ground_truth(transactions, ground_truth, report)
    validate_abuse_sanity(transactions, ground_truth)
    validate_duplicates(transactions)

    normal_count = int((ground_truth["is_abuse"] == False).sum())  # noqa: E712 - explicit ground-truth section
    abuse_count = int((ground_truth["is_abuse"] == True).sum())  # noqa: E712 - explicit ground-truth section
    print("\nCLASS BALANCE")
    print(f"Normal: {normal_count}; abuse: {abuse_count}; abuse percentage: {abuse_count / len(ground_truth):.2%}; normal:abuse ratio: {normal_count}:{abuse_count}")
    print("Interpretation: abuse is a minority class; accuracy alone would be misleading.")

    if report.warnings:
        print("\nWARNINGS")
        for warning in report.warnings:
            print(f"[WARN] {warning}")
    if report.failures:
        print("\nFAILURES")
        for failure in report.failures:
            print(f"[FAIL] {failure}")
    result = "FAIL" if report.failures else "PASS WITH WARNINGS" if report.warnings else "PASS"
    print(f"\nFINAL RESULT: {result}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())