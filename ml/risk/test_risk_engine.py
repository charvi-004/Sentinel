from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ml.risk.risk_engine import assess_transaction


class DummyModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        score = float(features.iloc[0]["customer_txn_count_30m"])
        if score >= 10:
            return np.array([[0.02, 0.98]])
        if score >= 5:
            return np.array([[0.15, 0.85]])
        return np.array([[0.92, 0.08]])


def build_inputs(**overrides):
    base = {
        "transaction_id": "txn_00001",
        "customer_txn_count_5m": 0,
        "customer_txn_count_30m": 1,
        "customer_txn_count_1h": 2,
        "customer_merchant_count_before": 1,
        "customer_device_count_before": 1,
        "customer_instrument_count_before": 1,
        "other_customer_device_count_before": 0,
        "other_customer_instrument_count_before": 0,
        "seconds_since_customer_last_transaction": 120.0,
        "seconds_since_device_last_transaction": 1000.0,
        "seconds_since_instrument_last_transaction": 900.0,
        "amount": 250.00,
    }
    base.update(overrides)
    feature_row = pd.DataFrame([base])
    graph_row = pd.DataFrame([
        {
            "transaction_id": base["transaction_id"],
            "other_customers_on_device": 0,
            "other_customers_on_instrument": 0,
            "customers_connected_via_device": 0,
            "customers_connected_via_instrument": 0,
            "unique_customer_neighbors": 0,
            "local_degree": 0,
            "customer_device_count_before": 1,
            "customer_instrument_count_before": 1,
            "customer_merchant_count_before": 1,
        }
    ])
    txn_row = pd.DataFrame([
        {
            "transaction_id": base["transaction_id"],
            "amount": float(base["amount"]),
            "currency": "USD",
            "timestamp": "2025-04-13 06:41:00+00:00",
        }
    ])
    return feature_row, graph_row, txn_row


class RiskEngineTests(unittest.TestCase):
    def test_low_risk_transaction(self):
        feature_row, graph_row, txn_row = build_inputs()
        result = assess_transaction(
            transaction_id="txn_00001",
            model=DummyModel(),
            risk_threshold=50.0,
            features_df=feature_row,
            graph_df=graph_row,
            transactions_df=txn_row,
            reason_thresholds={
                "customer_txn_count_30m": 8.0,
                "other_customer_device_count_before": 3.0,
                "other_customer_instrument_count_before": 3.0,
                "customer_merchant_count_before": 5.0,
                "seconds_since_customer_last_transaction": 900.0,
            },
        )
        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(result["recommended_action"], "ALLOW")

    def test_high_risk_transaction(self):
        feature_row, graph_row, txn_row = build_inputs(customer_txn_count_30m=12, other_customer_device_count_before=8, amount=8000)
        result = assess_transaction(
            transaction_id="txn_00001",
            model=DummyModel(),
            risk_threshold=50.0,
            features_df=feature_row,
            graph_df=graph_row,
            transactions_df=txn_row,
            reason_thresholds={
                "customer_txn_count_30m": 8.0,
                "other_customer_device_count_before": 3.0,
                "other_customer_instrument_count_before": 3.0,
                "customer_merchant_count_before": 5.0,
                "seconds_since_customer_last_transaction": 900.0,
            },
        )
        self.assertGreaterEqual(result["risk_score"], 50.0)
        self.assertIn(result["risk_level"], {"HIGH", "CRITICAL"})
        self.assertTrue(any(reason["type"] in {"HIGH VELOCITY", "SHARED_DEVICE"} for reason in result["reasons"]))

    def test_shared_device_case(self):
        feature_row, graph_row, txn_row = build_inputs(other_customer_device_count_before=7)
        result = assess_transaction(
            transaction_id="txn_00001",
            model=DummyModel(),
            risk_threshold=50.0,
            features_df=feature_row,
            graph_df=graph_row,
            transactions_df=txn_row,
            reason_thresholds={
                "customer_txn_count_30m": 8.0,
                "other_customer_device_count_before": 3.0,
                "other_customer_instrument_count_before": 3.0,
                "customer_merchant_count_before": 5.0,
                "seconds_since_customer_last_transaction": 900.0,
            },
        )
        self.assertTrue(any(reason["type"] == "SHARED_DEVICE" for reason in result["reasons"]))

    def test_shared_instrument_case(self):
        feature_row, graph_row, txn_row = build_inputs(other_customer_instrument_count_before=7)
        result = assess_transaction(
            transaction_id="txn_00001",
            model=DummyModel(),
            risk_threshold=50.0,
            features_df=feature_row,
            graph_df=graph_row,
            transactions_df=txn_row,
            reason_thresholds={
                "customer_txn_count_30m": 8.0,
                "other_customer_device_count_before": 3.0,
                "other_customer_instrument_count_before": 3.0,
                "customer_merchant_count_before": 5.0,
                "seconds_since_customer_last_transaction": 900.0,
            },
        )
        self.assertTrue(any(reason["type"] == "SHARED_INSTRUMENT" for reason in result["reasons"]))

    def test_high_velocity_case(self):
        feature_row, graph_row, txn_row = build_inputs(customer_txn_count_30m=12)
        result = assess_transaction(
            transaction_id="txn_00001",
            model=DummyModel(),
            risk_threshold=50.0,
            features_df=feature_row,
            graph_df=graph_row,
            transactions_df=txn_row,
            reason_thresholds={
                "customer_txn_count_30m": 8.0,
                "other_customer_device_count_before": 3.0,
                "other_customer_instrument_count_before": 3.0,
                "customer_merchant_count_before": 5.0,
                "seconds_since_customer_last_transaction": 900.0,
            },
        )
        self.assertTrue(any(reason["type"] == "HIGH VELOCITY" for reason in result["reasons"]))

    def test_deterministic_output(self):
        feature_row, graph_row, txn_row = build_inputs(customer_txn_count_30m=12, other_customer_device_count_before=4)
        first = assess_transaction(
            transaction_id="txn_00001",
            model=DummyModel(),
            risk_threshold=50.0,
            features_df=feature_row,
            graph_df=graph_row,
            transactions_df=txn_row,
            reason_thresholds={
                "customer_txn_count_30m": 8.0,
                "other_customer_device_count_before": 3.0,
                "other_customer_instrument_count_before": 3.0,
                "customer_merchant_count_before": 5.0,
                "seconds_since_customer_last_transaction": 900.0,
            },
        )
        second = assess_transaction(
            transaction_id="txn_00001",
            model=DummyModel(),
            risk_threshold=50.0,
            features_df=feature_row,
            graph_df=graph_row,
            transactions_df=txn_row,
            reason_thresholds={
                "customer_txn_count_30m": 8.0,
                "other_customer_device_count_before": 3.0,
                "other_customer_instrument_count_before": 3.0,
                "customer_merchant_count_before": 5.0,
                "seconds_since_customer_last_transaction": 900.0,
            },
        )
        self.assertEqual(first, second)

    def test_no_future_information_leakage(self):
        feature_row, graph_row, txn_row = build_inputs()
        result = assess_transaction(
            transaction_id="txn_00001",
            model=DummyModel(),
            risk_threshold=50.0,
            features_df=feature_row,
            graph_df=graph_row,
            transactions_df=txn_row,
            reason_thresholds={
                "customer_txn_count_30m": 8.0,
                "other_customer_device_count_before": 3.0,
                "other_customer_instrument_count_before": 3.0,
                "customer_merchant_count_before": 5.0,
                "seconds_since_customer_last_transaction": 900.0,
            },
        )
        self.assertIn("transaction_id", result)
        self.assertIn("risk_probability", result)
        self.assertIn("risk_score", result)
        self.assertIn("reasons", result)
        self.assertIn("network_context", result)
        self.assertIn("transaction", result)
        self.assertIn("timestamp", result["transaction"])


if __name__ == "__main__":
    unittest.main()
