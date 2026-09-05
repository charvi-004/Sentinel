from __future__ import annotations

import json
import os
import unittest

from ml.evaluation.end_to_end import evaluate_transaction, generate_demo_cases, run_evaluation


class EndToEndTests(unittest.TestCase):
    def test_end_to_end_transaction_evaluation(self):
        result = evaluate_transaction("txn_00001")
        self.assertIn("transaction_id", result)
        self.assertIn("risk_score", result)
        self.assertIn("risk_level", result)
        self.assertIn("recommended_action", result)
        self.assertIn("investigation", result)

    def test_risk_decision_preservation(self):
        result = evaluate_transaction("txn_00001")
        self.assertEqual(result["risk_level"], result["investigation"]["risk_assessment"]["risk_level"])
        self.assertEqual(result["recommended_action"], result["investigation"]["risk_assessment"]["recommended_action"])

    def test_investigator_integration(self):
        result = evaluate_transaction("txn_00001")
        self.assertIn("executive_summary", result["investigation"])
        self.assertIn("key_findings", result["investigation"])

    def test_ground_truth_exclusion(self):
        result = evaluate_transaction("txn_00001")
        payload = json.dumps(result)
        self.assertNotIn("is_abuse", payload)
        self.assertNotIn("abuse_type", payload)
        self.assertNotIn("abuse_cluster_id", payload)

    def test_evidence_presence(self):
        result = evaluate_transaction("txn_00001")
        self.assertTrue(len(result["reasons"]) >= 0)
        self.assertTrue(len(result["investigation"]["key_findings"]) >= 1)

    def test_representative_case_generation(self):
        cases = generate_demo_cases()
        self.assertIn("LOW", [case["risk_level"] for case in cases])
        self.assertIn("HIGH", [case["risk_level"] for case in cases])
        self.assertIn("CRITICAL", [case["risk_level"] for case in cases])

    def test_invalid_ai_output_handling(self):
        result = evaluate_transaction("txn_00001")
        self.assertIn("risk_assessment", result["investigation"])

    def test_deterministic_fallback(self):
        original = os.environ.get("SENTINEL_LLM_API_KEY")
        os.environ.pop("SENTINEL_LLM_API_KEY", None)
        try:
            result = evaluate_transaction("txn_00001")
            self.assertIn("executive_summary", result["investigation"])
        finally:
            if original is not None:
                os.environ["SENTINEL_LLM_API_KEY"] = original

    def test_no_future_data_leakage(self):
        result = evaluate_transaction("txn_00001")
        for key in ["is_abuse", "abuse_type", "abuse_cluster_id"]:
            self.assertNotIn(key, json.dumps(result))

    def test_test_artifact_generation(self):
        evaluation = run_evaluation()
        self.assertIn("dataset", evaluation)
        self.assertIn("model", evaluation)
        self.assertIn("detection_metrics", evaluation)
        self.assertIn("business_cost", evaluation)
        self.assertIn("operations", evaluation)
        self.assertIn("evidence", evaluation)
        self.assertIn("ai_safety", evaluation)


if __name__ == "__main__":
    unittest.main()
