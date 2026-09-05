"""Generate a reproducible synthetic payment ecosystem for Sentinel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULTS = {
    "merchants": 100,
    "customers": 2_000,
    "devices": 500,
    "instruments": 1_000,
    "transactions": 10_000,
    "seed": 42,
}

CATEGORIES = np.array(["grocery", "fashion", "electronics", "travel", "food", "services"])
REGIONS = np.array(["north", "south", "east", "west", "central"])
DEVICE_TYPES = np.array(["android", "ios", "web", "tablet"])
INSTRUMENT_TYPES = np.array(["card_token", "bank_token", "wallet_token"])
CATEGORY_MEDIANS = {
    "grocery": 650.0,
    "fashion": 1_800.0,
    "electronics": 7_500.0,
    "travel": 12_000.0,
    "food": 450.0,
    "services": 2_200.0,
}


def _ids(prefix: str, count: int) -> np.ndarray:
    return np.array([f"{prefix}_{number:05d}" for number in range(1, count + 1)])


def _choose_unique(rng: np.random.Generator, values: np.ndarray, count: int) -> np.ndarray:
    return rng.choice(values, size=min(count, len(values)), replace=False)


def _choose_across_categories(
    rng: np.random.Generator,
    table: pd.DataFrame,
    id_column: str,
    category_column: str,
    count: int,
) -> np.ndarray:
    """Sample entities across categories before filling the remaining slots."""
    groups = [group[id_column].to_numpy() for _, group in table.groupby(category_column, sort=False)]
    rng.shuffle(groups)
    selected: list[str] = []
    for group in groups:
        if len(selected) == count:
            break
        selected.extend(rng.choice(group, size=1, replace=False).tolist())
    remaining = np.setdiff1d(table[id_column].to_numpy(), selected)
    if len(selected) < count:
        selected.extend(rng.choice(remaining, size=count - len(selected), replace=False).tolist())
    return np.array(selected)


def build_entities(
    rng: np.random.Generator,
    merchant_count: int,
    customer_count: int,
    device_count: int,
    instrument_count: int,
) -> dict[str, pd.DataFrame]:
    """Create synthetic entity tables without real-world identifiers or PII."""
    merchants = pd.DataFrame(
        {
            "merchant_id": _ids("mrc", merchant_count),
            "merchant_category": rng.choice(CATEGORIES, merchant_count),
            "region": rng.choice(REGIONS, merchant_count),
            "merchant_age_days": rng.integers(30, 3_000, merchant_count),
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": _ids("cus", customer_count),
            "account_age_days": rng.integers(15, 2_500, customer_count),
            "region": rng.choice(REGIONS, customer_count),
        }
    )
    devices = pd.DataFrame(
        {
            "device_id": _ids("dev", device_count),
            "device_type": rng.choice(DEVICE_TYPES, device_count, p=[0.42, 0.32, 0.18, 0.08]),
            "first_seen_day": rng.integers(0, 100, device_count),
        }
    )
    instruments = pd.DataFrame(
        {
            "instrument_id": _ids("ins", instrument_count),
            "instrument_type": rng.choice(INSTRUMENT_TYPES, instrument_count, p=[0.58, 0.27, 0.15]),
        }
    )
    return {
        "merchants": merchants,
        "customers": customers,
        "devices": devices,
        "instruments": instruments,
    }


def _normal_transactions(
    rng: np.random.Generator,
    entities: dict[str, pd.DataFrame],
    transaction_count: int,
) -> pd.DataFrame:
    """Create ordinary activity using stable, varied customer preferences."""
    merchants = entities["merchants"]
    customers = entities["customers"]
    devices = entities["devices"]
    instruments = entities["instruments"]

    merchant_weights = rng.dirichlet(np.full(len(merchants), 1.7))
    customer_positions = rng.integers(0, len(customers), transaction_count)
    merchant_positions = rng.choice(len(merchants), transaction_count, p=merchant_weights)

    customer_devices = [
        _choose_unique(rng, devices["device_id"].to_numpy(), int(rng.integers(1, 4)))
        for _ in customers.itertuples()
    ]
    customer_instruments = [
        _choose_unique(rng, instruments["instrument_id"].to_numpy(), int(rng.integers(1, 4)))
        for _ in customers.itertuples()
    ]
    selected_devices = [rng.choice(customer_devices[position]) for position in customer_positions]
    selected_instruments = [rng.choice(customer_instruments[position]) for position in customer_positions]
    selected_categories = merchants.iloc[merchant_positions]["merchant_category"].to_numpy()
    amounts = np.array(
        [rng.lognormal(np.log(CATEGORY_MEDIANS[category]), 0.62) for category in selected_categories]
    ).round(2)
    timestamps = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        rng.integers(0, 120 * 24 * 60 * 60, transaction_count), unit="s"
    )
    status = rng.choice(["succeeded", "declined", "failed", "reversed"], transaction_count, p=[0.91, 0.055, 0.025, 0.01])

    return pd.DataFrame(
        {
            "transaction_id": _ids("txn", transaction_count),
            "timestamp": timestamps,
            "merchant_id": merchants.iloc[merchant_positions]["merchant_id"].to_numpy(),
            "customer_id": customers.iloc[customer_positions]["customer_id"].to_numpy(),
            "device_id": selected_devices,
            "payment_instrument_id": selected_instruments,
            "ip_hash": [f"ip_{value:08x}" for value in rng.integers(0, 2**32, transaction_count, dtype=np.uint64)],
            "amount": amounts,
            "currency": rng.choice(["INR", "USD", "GBP"], transaction_count, p=[0.82, 0.12, 0.06]),
            "status": status,
        }
    )


def _unused_rows(rng: np.random.Generator, total: int, used: set[int], count: int) -> np.ndarray:
    available = np.array(list(set(range(total)) - used), dtype=int)
    if count > len(available):
        raise ValueError("Not enough transactions for the requested abuse scenarios")
    return rng.choice(available, count, replace=False)


def _event_start(rng: np.random.Generator) -> pd.Timestamp:
    """Choose a reproducible event date and hour across the development period."""
    day_offset = int(rng.integers(3, 117))
    hour = int(rng.integers(0, 24))
    return pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=day_offset, hours=hour)


def _inject_cluster(
    rng: np.random.Generator,
    transactions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    rows: Iterable[int],
    cluster_id: str,
    abuse_type: str,
    customers: np.ndarray,
    merchants: np.ndarray,
    shared_device: str | np.ndarray | None = None,
    shared_instrument: str | np.ndarray | None = None,
    burst: bool = False,
    event_start: pd.Timestamp | list[pd.Timestamp] | None = None,
) -> None:
    """Modify observable relationships while recording labels in a separate table."""
    rows = np.asarray(list(rows), dtype=int)
    customer_values = rng.choice(customers, len(rows), replace=True)
    merchant_values = rng.choice(merchants, len(rows), replace=True)
    transactions.loc[rows, "customer_id"] = customer_values
    transactions.loc[rows, "merchant_id"] = merchant_values
    if shared_device is not None:
        transactions.loc[rows, "device_id"] = rng.choice(shared_device, len(rows), replace=True)
    if shared_instrument is not None:
        transactions.loc[rows, "payment_instrument_id"] = rng.choice(shared_instrument, len(rows), replace=True)
    if burst:
        if event_start is None:
            raise ValueError("A burst cluster requires an event start")
        starts = [event_start] if isinstance(event_start, pd.Timestamp) else event_start
        row_starts = np.resize(np.asarray(starts, dtype=object), len(rows))
        transactions.loc[rows, "timestamp"] = [
            start + pd.Timedelta(seconds=int(offset))
            for start, offset in zip(row_starts, rng.integers(0, 90 * 60, len(rows)))
        ]
    ground_truth.loc[rows, "is_abuse"] = True
    ground_truth.loc[rows, "abuse_type"] = abuse_type
    ground_truth.loc[rows, "abuse_cluster_id"] = cluster_id


def inject_abuse(rng: np.random.Generator, transactions: pd.DataFrame, entities: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inject several overlapping-in-appearance but separately labelled patterns."""
    total = len(transactions)
    ground_truth = pd.DataFrame(
        {"transaction_id": transactions["transaction_id"], "is_abuse": False, "abuse_type": "none", "abuse_cluster_id": "none"}
    )
    used: set[int] = set()
    customers = entities["customers"]["customer_id"].to_numpy()
    merchants = entities["merchants"]["merchant_id"].to_numpy()
    devices = entities["devices"]
    instruments = entities["instruments"]

    scenarios = [
        ("shared_device", 8, 5, 45, {"shared_device": True}),
        ("shared_payment_instrument", 8, 5, 45, {"shared_instrument": True}),
        ("velocity_burst", 18, 3, 115, {"burst": True}),
        ("cross_merchant_ring", 20, 7, 150, {"shared_device": True, "shared_instrument": True, "burst": True}),
    ]
    for number, (abuse_type, min_customers, merchant_count, row_count, options) in enumerate(scenarios, start=1):
        rows = _unused_rows(rng, total, used, row_count)
        used.update(rows.tolist())
        cluster_customers = _choose_unique(rng, customers, int(rng.integers(min_customers, min_customers + 8)))
        cluster_merchants = _choose_across_categories(rng, entities["merchants"], "merchant_id", "merchant_category", merchant_count)
        shared_device = None
        shared_instrument = None
        if options.pop("shared_device", False):
            shared_device = _choose_across_categories(rng, devices, "device_id", "device_type", 4)
        if options.pop("shared_instrument", False):
            shared_instrument = _choose_across_categories(rng, instruments, "instrument_id", "instrument_type", 3)
        if options.get("burst"):
            options["event_start"] = [_event_start(rng) for _ in range(4)]
        _inject_cluster(
            rng, transactions, ground_truth, rows, f"abuse_{number:02d}", abuse_type,
            cluster_customers, cluster_merchants, shared_device=shared_device,
            shared_instrument=shared_instrument, **options
        )
    return ground_truth


def generate_dataset(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Generate entity, transaction, and hidden ground-truth tables."""
    rng = np.random.default_rng(args.seed)
    entities = build_entities(rng, args.merchants, args.customers, args.devices, args.instruments)
    transactions = _normal_transactions(rng, entities, args.transactions)
    ground_truth = inject_abuse(rng, transactions, entities)
    return {**entities, "transactions": transactions, "ground_truth": ground_truth}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(vars(args).values()) <= 0:
        raise ValueError("All generator parameters must be positive")
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = generate_dataset(args)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    ground_truth = tables["ground_truth"]
    abuse = ground_truth[ground_truth["is_abuse"]]
    print(f"merchants: {len(tables['merchants'])}")
    print(f"customers: {len(tables['customers'])}")
    print(f"devices: {len(tables['devices'])}")
    print(f"instruments: {len(tables['instruments'])}")
    print(f"transactions: {len(tables['transactions'])}")
    print(f"abuse transactions: {len(abuse)} ({len(abuse) / len(ground_truth):.2%})")
    print(f"abuse clusters: {abuse['abuse_cluster_id'].nunique()}")
    print("abuse breakdown:")
    for abuse_type, count in abuse["abuse_type"].value_counts().sort_index().items():
        print(f"  {abuse_type}: {count}")


if __name__ == "__main__":
    main()