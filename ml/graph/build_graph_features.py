"""Build point-in-time NetworkX relationship features for Sentinel transactions."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABEL_NAMES = {"is_abuse", "abuse_type", "abuse_cluster_id"}
ENTITY_NAMES = {"customer_id", "device_id", "merchant_id", "payment_instrument_id"}
WINDOWS = {"5m": 300, "30m": 1_800, "1h": 3_600}


def node_id(node_type: str, value: str) -> str:
    """Return a typed, inspectable graph node identifier."""
    return f"{node_type}:{value}"


def _prune(events: deque[tuple[pd.Timestamp, str]], timestamp: pd.Timestamp, seconds: int) -> None:
    cutoff = timestamp - pd.Timedelta(seconds=seconds)
    while events and events[0][0] <= cutoff:
        events.popleft()


def _event_count(events: deque[tuple[pd.Timestamp, str]], timestamp: pd.Timestamp, seconds: int) -> int:
    _prune(events, timestamp, seconds)
    return len(events)


def _new_features() -> dict[str, float]:
    return {
        "customer_degree_before": 0,
        "device_degree_before": 0,
        "instrument_degree_before": 0,
        "merchant_degree_before": 0,
        "customer_device_count_before": 0,
        "customer_instrument_count_before": 0,
        "customer_merchant_count_before": 0,
        "device_customer_count_before": 0,
        "device_merchant_count_before": 0,
        "instrument_customer_count_before": 0,
        "instrument_merchant_count_before": 0,
        "other_customers_on_device": 0,
        "other_customers_on_instrument": 0,
        "other_merchants_on_device": 0,
        "other_merchants_on_instrument": 0,
        "shared_device_count_for_customer": 0,
        "shared_instrument_count_for_customer": 0,
        "customers_connected_via_device": 0,
        "customers_connected_via_instrument": 0,
        "merchants_connected_via_device": 0,
        "merchants_connected_via_instrument": 0,
        "connected_entity_count": 0,
        "local_degree": 0,
        "neighbor_count": 0,
        "unique_customer_neighbors": 0,
        "unique_merchant_neighbors": 0,
        "other_customers_same_device_5m": 0,
        "other_customers_same_device_30m": 0,
        "other_customers_same_device_1h": 0,
        "other_customers_same_instrument_5m": 0,
        "other_customers_same_instrument_30m": 0,
        "other_customers_same_instrument_1h": 0,
    }


def _historical_features(
    transaction: pd.Series,
    graph: nx.Graph,
    customer_devices: defaultdict[str, set[str]],
    customer_instruments: defaultdict[str, set[str]],
    customer_merchants: defaultdict[str, set[str]],
    device_customers: defaultdict[str, set[str]],
    device_merchants: defaultdict[str, set[str]],
    instrument_customers: defaultdict[str, set[str]],
    instrument_merchants: defaultdict[str, set[str]],
    device_events: defaultdict[str, deque[tuple[pd.Timestamp, str]]],
    instrument_events: defaultdict[str, deque[tuple[pd.Timestamp, str]]],
) -> dict[str, float]:
    """Calculate one transaction's graph features before adding its edges."""
    features = _new_features()
    customer = str(transaction["customer_id"])
    device = str(transaction["device_id"])
    instrument = str(transaction["payment_instrument_id"])
    merchant = str(transaction["merchant_id"])
    timestamp = transaction["timestamp"]
    customer_node = node_id("customer", customer)
    device_node = node_id("device", device)
    instrument_node = node_id("instrument", instrument)
    merchant_node = node_id("merchant", merchant)

    features.update(
        customer_degree_before=graph.degree(customer_node),
        device_degree_before=graph.degree(device_node),
        instrument_degree_before=graph.degree(instrument_node),
        merchant_degree_before=graph.degree(merchant_node),
        customer_device_count_before=len(customer_devices[customer]),
        customer_instrument_count_before=len(customer_instruments[customer]),
        customer_merchant_count_before=len(customer_merchants[customer]),
        device_customer_count_before=len(device_customers[device]),
        device_merchant_count_before=len(device_merchants[device]),
        instrument_customer_count_before=len(instrument_customers[instrument]),
        instrument_merchant_count_before=len(instrument_merchants[instrument]),
        other_customers_on_device=len(device_customers[device] - {customer}),
        other_customers_on_instrument=len(instrument_customers[instrument] - {customer}),
        other_merchants_on_device=len(device_merchants[device] - {merchant}),
        other_merchants_on_instrument=len(instrument_merchants[instrument] - {merchant}),
        shared_device_count_for_customer=sum(
            bool(device_customers[other_device] - {customer}) for other_device in customer_devices[customer]
        ),
        shared_instrument_count_for_customer=sum(
            bool(instrument_customers[other_instrument] - {customer}) for other_instrument in customer_instruments[customer]
        ),
    )

    connected_via_device = set().union(*(device_customers[other_device] for other_device in customer_devices[customer])) - {customer}
    connected_via_instrument = set().union(*(instrument_customers[other_instrument] for other_instrument in customer_instruments[customer])) - {customer}
    merchants_via_device = device_merchants[device] - {merchant}
    merchants_via_instrument = instrument_merchants[instrument] - {merchant}
    local_neighbors: set[str] = set()
    for current_node in [customer_node, device_node, instrument_node, merchant_node]:
        local_neighbors.update(graph.neighbors(current_node))
    customer_neighbors = {
        neighbor for neighbor in local_neighbors if neighbor.startswith("customer:") and neighbor != customer_node
    }
    merchant_neighbors = {neighbor for neighbor in local_neighbors if neighbor.startswith("merchant:")}
    features.update(
        customers_connected_via_device=len(connected_via_device),
        customers_connected_via_instrument=len(connected_via_instrument),
        merchants_connected_via_device=len(merchants_via_device),
        merchants_connected_via_instrument=len(merchants_via_instrument),
        connected_entity_count=len(local_neighbors),
        local_degree=sum(graph.degree(current_node) for current_node in [customer_node, device_node, instrument_node, merchant_node]),
        neighbor_count=len(local_neighbors),
        unique_customer_neighbors=len(customer_neighbors),
        unique_merchant_neighbors=len(merchant_neighbors),
    )
    for window_name, seconds in WINDOWS.items():
        recent_device = device_events[device]
        recent_instrument = instrument_events[instrument]
        _prune(recent_device, timestamp, seconds)
        _prune(recent_instrument, timestamp, seconds)
        features[f"other_customers_same_device_{window_name}"] = len({value for _, value in recent_device if value != customer})
        features[f"other_customers_same_instrument_{window_name}"] = len({value for _, value in recent_instrument if value != customer})
    return features


def _add_transaction(
    transaction: pd.Series,
    graph: nx.Graph,
    customer_devices: defaultdict[str, set[str]],
    customer_instruments: defaultdict[str, set[str]],
    customer_merchants: defaultdict[str, set[str]],
    device_customers: defaultdict[str, set[str]],
    device_merchants: defaultdict[str, set[str]],
    instrument_customers: defaultdict[str, set[str]],
    instrument_merchants: defaultdict[str, set[str]],
    device_events: defaultdict[str, deque[tuple[pd.Timestamp, str]]],
    instrument_events: defaultdict[str, deque[tuple[pd.Timestamp, str]]],
) -> None:
    """Add one observed transaction to graph and relationship indexes."""
    customer = str(transaction["customer_id"])
    device = str(transaction["device_id"])
    instrument = str(transaction["payment_instrument_id"])
    merchant = str(transaction["merchant_id"])
    timestamp = transaction["timestamp"]
    relationships = [
        (node_id("customer", customer), node_id("device", device), "customer_device"),
        (node_id("customer", customer), node_id("instrument", instrument), "customer_instrument"),
        (node_id("customer", customer), node_id("merchant", merchant), "customer_merchant"),
        (node_id("device", device), node_id("merchant", merchant), "device_merchant"),
        (node_id("instrument", instrument), node_id("merchant", merchant), "instrument_merchant"),
    ]
    for left, right, relationship in relationships:
        graph.add_edge(left, right, relationship=relationship)
    customer_devices[customer].add(device)
    customer_instruments[customer].add(instrument)
    customer_merchants[customer].add(merchant)
    device_customers[device].add(customer)
    device_merchants[device].add(merchant)
    instrument_customers[instrument].add(customer)
    instrument_merchants[instrument].add(merchant)
    device_events[device].append((timestamp, customer))
    instrument_events[instrument].append((timestamp, customer))


def _initialize_graph(transactions: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for column, node_type in [
        ("customer_id", "customer"),
        ("device_id", "device"),
        ("payment_instrument_id", "instrument"),
        ("merchant_id", "merchant"),
    ]:
        for value in transactions[column].dropna().unique():
            graph.add_node(node_id(node_type, str(value)), node_type=node_type)
    return graph


def build_graph_features(transactions: pd.DataFrame) -> tuple[pd.DataFrame, nx.Graph]:
    """Build features from an incrementally updated graph in timestamp order."""
    ordered = transactions.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="raise")
    ordered = ordered.sort_values(["timestamp", "transaction_id"], kind="mergesort")
    graph = _initialize_graph(ordered)
    customer_devices: defaultdict[str, set[str]] = defaultdict(set)
    customer_instruments: defaultdict[str, set[str]] = defaultdict(set)
    customer_merchants: defaultdict[str, set[str]] = defaultdict(set)
    device_customers: defaultdict[str, set[str]] = defaultdict(set)
    device_merchants: defaultdict[str, set[str]] = defaultdict(set)
    instrument_customers: defaultdict[str, set[str]] = defaultdict(set)
    instrument_merchants: defaultdict[str, set[str]] = defaultdict(set)
    device_events: defaultdict[str, deque[tuple[pd.Timestamp, str]]] = defaultdict(deque)
    instrument_events: defaultdict[str, deque[tuple[pd.Timestamp, str]]] = defaultdict(deque)
    feature_rows: dict[str, dict[str, float]] = {}

    for _, timestamp_group in ordered.groupby("timestamp", sort=False):
        pending: list[tuple[pd.Series, dict[str, float]]] = []
        for _, transaction in timestamp_group.iterrows():
            pending.append(
                (
                    transaction,
                    _historical_features(
                        transaction,
                        graph,
                        customer_devices,
                        customer_instruments,
                        customer_merchants,
                        device_customers,
                        device_merchants,
                        instrument_customers,
                        instrument_merchants,
                        device_events,
                        instrument_events,
                    ),
                )
            )
        for transaction, features in pending:
            feature_rows[str(transaction["transaction_id"])] = features
            _add_transaction(
                transaction,
                graph,
                customer_devices,
                customer_instruments,
                customer_merchants,
                device_customers,
                device_merchants,
                instrument_customers,
                instrument_merchants,
                device_events,
                instrument_events,
            )
    result = pd.DataFrame.from_dict(feature_rows, orient="index")
    result.index.name = "transaction_id"
    return result.reset_index(), graph


def _manual_reference_check(transactions: pd.DataFrame, features: pd.DataFrame) -> None:
    """Compare selected historical device-neighbor counts with direct references."""
    ordered = transactions.sort_values("timestamp").head(min(100, len(transactions)))
    indexed = features.set_index("transaction_id")
    timestamps = pd.to_datetime(transactions["timestamp"], utc=True)
    for _, row in ordered.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        prior = transactions[
            (transactions["device_id"] == row["device_id"])
            & (timestamps < timestamp)
        ]
        expected = prior.loc[prior["customer_id"] != row["customer_id"], "customer_id"].nunique()
        actual = int(indexed.loc[row["transaction_id"], "other_customers_on_device"])
        if expected != actual:
            raise ValueError(f"Graph reference mismatch for {row['transaction_id']}: expected={expected}, actual={actual}")


def _validate_outputs(transactions: pd.DataFrame, features: pd.DataFrame, graph: nx.Graph) -> None:
    model_columns = [column for column in features.columns if column != "transaction_id"]
    forbidden = (set(model_columns) & (LABEL_NAMES | ENTITY_NAMES))
    if forbidden:
        raise ValueError(f"Forbidden graph feature columns: {sorted(forbidden)}")
    if features["transaction_id"].duplicated().any() or set(features["transaction_id"]) != set(transactions["transaction_id"]):
        raise ValueError("Graph features do not map exactly once to transactions")
    values = features[model_columns].to_numpy(dtype=float)
    if features[model_columns].isna().any().any() or not np.isfinite(values).all():
        raise ValueError("Graph features contain NaN or infinite values")
    expected_types = {"customer", "device", "instrument", "merchant"}
    actual_types = {data["node_type"] for _, data in graph.nodes(data=True)}
    if actual_types != expected_types:
        raise ValueError(f"Graph node types mismatch: expected={expected_types}, actual={actual_types}")


def main() -> None:
    started = time.perf_counter()
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing input: {RAW_PATH}")
    transactions = pd.read_csv(RAW_PATH)
    required = {"transaction_id", "timestamp", "customer_id", "device_id", "merchant_id", "payment_instrument_id"}
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"Missing transaction columns: {missing}")
    features, graph = build_graph_features(transactions)
    _manual_reference_check(transactions, features)
    _validate_outputs(transactions, features, graph)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_DIR / "graph_features.csv", index=False)
    nx.write_graphml(graph, PROCESSED_DIR / "payment_graph.graphml")

    model_columns = [column for column in features.columns if column != "transaction_id"]
    node_counts = pd.Series([data["node_type"] for _, data in graph.nodes(data=True)]).value_counts().sort_index().to_dict()
    print(f"rows: {len(features)}")
    print(f"graph nodes: {graph.number_of_nodes()}")
    print(f"graph edges: {graph.number_of_edges()}")
    print(f"nodes by type: {node_counts}")
    print(f"graph features: {len(model_columns)}")
    print(f"feature groups: basic_degree=4, typed_relationship=7, shared_entity=6, two_hop=5, local_structure=4, coordination=6")
    print(f"runtime_seconds: {time.perf_counter() - started:.3f}")
    example = features.sort_values("other_customers_on_device", ascending=False).iloc[0]
    source = transactions.set_index("transaction_id").loc[example["transaction_id"]]
    example_nodes = {
        "customer": node_id("customer", str(source["customer_id"])),
        "device": node_id("device", str(source["device_id"])),
        "instrument": node_id("instrument", str(source["payment_instrument_id"])),
        "merchant": node_id("merchant", str(source["merchant_id"])),
    }
    print(
        "example neighborhood: "
        f"transaction={example['transaction_id']}, other_customers_on_device={int(example['other_customers_on_device'])}, "
        f"other_customers_on_instrument={int(example['other_customers_on_instrument'])}, "
        f"connected_entity_count={int(example['connected_entity_count'])}"
    )
    for neighborhood_type, current_node in example_nodes.items():
        neighbors = sorted(graph.neighbors(current_node))
        print(f"  {neighborhood_type}={current_node} neighbors={neighbors[:8]}{' ...' if len(neighbors) > 8 else ''}")
    print("validation: mappings, leakage boundary, finite values, node types, equal-timestamp batching, and naive references passed")


if __name__ == "__main__":
    main()