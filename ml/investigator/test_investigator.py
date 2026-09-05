from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from ml.investigator.case_builder import build_case
from ml.investigator.investigator import (
    AIProvider,
    FallbackInvestigator,
    Investigator,
    OpenAICompatibleProvider,
    render_investigation_report,
)
from ml.investigator.output_validator import validate_report


class DummyAI(AIProvider):
    def generate_investigation(self, case):
        return {
            "executive_summary": "Observed evidence suggests elevated risk.",
            "risk_assessment": {
                "risk_score": case["risk"]["score"],
                "risk_level": case["risk"]["level"],
                "recommended_action": case["risk"]["recommended_action"],
            },
            "key_findings": [
                {"finding": "Velocity is elevated.", "evidence": ["12 transactions in 30 minutes"]}
            ],
            "risk_pattern": "Velocity + device reuse is consistent with coordinated activity.",
            "network_analysis": "Network connections are concentrated on a small number of merchants and devices.",
            "recommended_investigation": [
                "Check merchant history",
                "Review device reuse",
                "Confirm customer context",
            ],
            "uncertainty": "The available evidence supports escalation for review but does not establish fraudulent intent.",
            "analyst_summary": "Observed evidence suggests a coordinated pattern that deserves analyst review.",
        }


def build_sample_case():
    return {
        "case_id": "case_001",
        "transaction": {
            "id": "txn_001",
            "amount": 8200.0,
            "currency": "USD",
            "timestamp": "2025-06-22T09:10:00Z",
        },
        "risk": {
            "score": 86.3,
            "level": "CRITICAL",
            "recommended_action": "ESCALATE",
        },
        "evidence": [
            {"signal": "HIGH_VELOCITY", "severity": "HIGH", "value": 12, "description": "12 transactions in 30 minutes."},
            {"signal": "SHARED_DEVICE", "severity": "HIGH", "value": 7, "description": "This device was previously associated with 7 other customers."},
            {"signal": "CONNECTED_MERCHANTS", "severity": "MEDIUM", "value": 3, "description": "3 connected merchants observed."},
        ],
        "network": {
            "connected_customers": 7,
            "connected_merchants": 3,
            "connected_devices": 2,
            "connected_instruments": 1,
        },
    }


class InvestigatorTests(unittest.TestCase):
    def test_case_construction(self):
        assessment = {
            "transaction_id": "txn_001",
            "risk_score": 86.3,
            "risk_probability": 0.863,
            "risk_level": "CRITICAL",
            "recommended_action": "ESCALATE",
            "reasons": [
                {"type": "HIGH VELOCITY", "severity": "HIGH", "description": "12 transactions in 30 minutes.", "evidence": {"customer_txn_count_30m": 12}},
                {"type": "SHARED_DEVICE", "severity": "HIGH", "description": "This device was previously associated with 7 other customers.", "evidence": {"other_customer_device_count_before": 7}},
            ],
            "network_context": {"connected_customers": 7, "connected_merchants": 3, "connected_devices": 2, "connected_instruments": 1},
            "transaction": {"amount": 8200.0, "currency": "USD", "timestamp": "2025-06-22T09:10:00Z"},
        }
        case = build_case(assessment)
        self.assertEqual(case["transaction"]["id"], "txn_001")
        self.assertEqual(case["risk"]["score"], 86.3)
        self.assertEqual(case["risk"]["level"], "CRITICAL")
        self.assertEqual(case["network"]["connected_customers"], 7)
        self.assertGreater(len(case["evidence"]), 0)

    def test_ground_truth_exclusion(self):
        case = build_case({
            "transaction_id": "txn_001",
            "risk_score": 75.0,
            "risk_level": "HIGH",
            "recommended_action": "REVIEW",
            "reasons": [{"type": "SHARED_DEVICE", "severity": "HIGH", "description": "Device reused by multiple customers.", "evidence": {"other_customer_device_count_before": 4}}],
            "network_context": {"connected_customers": 4},
            "transaction": {"amount": 100, "currency": "USD", "timestamp": "2025-06-22T09:10:00Z"},
        })
        case_text = json.dumps(case)
        self.assertNotIn("is_abuse", case_text)
        self.assertNotIn("abuse_type", case_text)
        self.assertNotIn("abuse_cluster_id", case_text)

    def test_required_evidence_fields(self):
        case = build_sample_case()
        evidence = case["evidence"][0]
        self.assertIn("signal", evidence)
        self.assertIn("severity", evidence)
        self.assertIn("value", evidence)
        self.assertIn("description", evidence)

    def test_deterministic_fallback(self):
        case = build_sample_case()
        report = FallbackInvestigator().generate_investigation(case)
        self.assertIn("executive_summary", report)
        self.assertEqual(report["risk_assessment"]["risk_score"], case["risk"]["score"])
        self.assertEqual(report["risk_assessment"]["risk_level"], case["risk"]["level"])
        self.assertEqual(report["risk_assessment"]["recommended_action"], case["risk"]["recommended_action"])

    def test_valid_ai_output(self):
        case = build_sample_case()
        investigator = Investigator(provider=DummyAI())
        report = investigator.generate_report(case)
        self.assertEqual(report["risk_assessment"]["risk_score"], 86.3)
        self.assertEqual(report["risk_assessment"]["risk_level"], "CRITICAL")
        self.assertEqual(report["risk_assessment"]["recommended_action"], "ESCALATE")

    def test_invalid_json_handling(self):
        case = build_sample_case()
        report = validate_report("not valid json", case)
        self.assertIn("executive_summary", report)
        self.assertEqual(report["risk_assessment"]["risk_level"], "CRITICAL")

    def test_risk_level_contradiction_handling(self):
        case = build_sample_case()
        invalid = {
            "executive_summary": "n/a",
            "risk_assessment": {"risk_score": 86.3, "risk_level": "LOW", "recommended_action": "ALLOW"},
            "key_findings": [{"finding": "x", "evidence": ["y"]}],
            "risk_pattern": "z",
            "network_analysis": "n",
            "recommended_investigation": ["Check history"],
            "uncertainty": "u",
            "analyst_summary": "s",
        }
        fixed = validate_report(invalid, case)
        self.assertEqual(fixed["risk_assessment"]["risk_level"], "CRITICAL")
        self.assertEqual(fixed["risk_assessment"]["recommended_action"], "ESCALATE")

    def test_recommended_action_contradiction_handling(self):
        case = build_sample_case()
        invalid = {
            "executive_summary": "n/a",
            "risk_assessment": {"risk_score": 86.3, "risk_level": "CRITICAL", "recommended_action": "MONITOR"},
            "key_findings": [{"finding": "x", "evidence": ["y"]}],
            "risk_pattern": "z",
            "network_analysis": "n",
            "recommended_investigation": ["Check history"],
            "uncertainty": "u",
            "analyst_summary": "s",
        }
        fixed = validate_report(invalid, case)
        self.assertEqual(fixed["risk_assessment"]["recommended_action"], "ESCALATE")

    def test_missing_evidence_handling(self):
        case = build_sample_case()
        invalid = {
            "executive_summary": "n/a",
            "risk_assessment": {"risk_score": 86.3, "risk_level": "CRITICAL", "recommended_action": "ESCALATE"},
            "risk_pattern": "z",
            "network_analysis": "n",
            "recommended_investigation": ["Check history"],
            "uncertainty": "u",
            "analyst_summary": "s",
        }
        fixed = validate_report(invalid, case)
        self.assertIn("key_findings", fixed)
        self.assertTrue(len(fixed["key_findings"]) >= 1)

    def test_deterministic_risk_score_preservation(self):
        case = build_sample_case()
        invalid = {
            "executive_summary": "n/a",
            "risk_assessment": {"risk_score": 10.0, "risk_level": "CRITICAL", "recommended_action": "ESCALATE"},
            "key_findings": [{"finding": "x", "evidence": ["y"]}],
            "risk_pattern": "z",
            "network_analysis": "n",
            "recommended_investigation": ["Check history"],
            "uncertainty": "u",
            "analyst_summary": "s",
        }
        fixed = validate_report(invalid, case)
        self.assertEqual(fixed["risk_assessment"]["risk_score"], 86.3)

    def test_render_investigation_report(self):
        case = build_sample_case()
        report = FallbackInvestigator().generate_investigation(case)
        result = render_investigation_report(report, case)
        self.assertIn("SENTINEL", result)
        self.assertIn("Transaction: txn_001", result)
        self.assertIn("Risk Score:", result)
        self.assertIn("Risk Level:", result)
        self.assertIn("Action:", result)
        self.assertIn("WHY WAS IT FLAGGED?", result)
        self.assertIn("NETWORK CONTEXT", result)

    def test_investigator_without_api_key_uses_fallback(self):
        original = os.environ.get("SENTINEL_LLM_API_KEY")
        os.environ.pop("SENTINEL_LLM_API_KEY", None)
        try:
            investigator = Investigator()
            self.assertIsInstance(investigator.provider, FallbackInvestigator)
            case = build_sample_case()
            report = investigator.generate_report(case)
            self.assertIn("executive_summary", report)
            self.assertEqual(report["risk_assessment"]["risk_score"], case["risk"]["score"])
            self.assertEqual(report["risk_assessment"]["risk_level"], case["risk"]["level"])
            self.assertEqual(report["risk_assessment"]["recommended_action"], case["risk"]["recommended_action"])
        finally:
            if original is not None:
                os.environ["SENTINEL_LLM_API_KEY"] = original

    def test_investigator_with_dummy_api_key_falls_back_deterministically(self):
        original = os.environ.get("SENTINEL_LLM_API_KEY")
        os.environ["SENTINEL_LLM_API_KEY"] = "dummy_configured_key"
        try:
            investigator = Investigator()
            self.assertIsInstance(investigator.provider, OpenAICompatibleProvider)
            case = build_sample_case()
            # Confirm that provider itself raises NotImplementedError
            with self.assertRaises(NotImplementedError):
                investigator.provider.generate_investigation(case)
            # Confirm generate_report handles the provider failure and returns valid fallback report
            report = investigator.generate_report(case)
            self.assertIn("executive_summary", report)
            self.assertEqual(report["risk_assessment"]["risk_score"], case["risk"]["score"])
            self.assertEqual(report["risk_assessment"]["risk_level"], case["risk"]["level"])
            self.assertEqual(report["risk_assessment"]["recommended_action"], case["risk"]["recommended_action"])
            self.assertIn("key_findings", report)
            self.assertIn("network_analysis", report)
        finally:
            if original is not None:
                os.environ["SENTINEL_LLM_API_KEY"] = original
            else:
                os.environ.pop("SENTINEL_LLM_API_KEY", None)

    def test_investigator_non_dict_case_raises_value_error(self):
        investigator = Investigator()
        with self.assertRaises(ValueError):
            investigator.generate_report("not_a_dictionary")  # type: ignore[arg-type]

    def test_investigator_validation_error_propagates(self):
        investigator = Investigator(provider=FallbackInvestigator())
        case = build_sample_case()
        with patch("ml.investigator.investigator.validate_report", side_effect=RuntimeError("Validation crash")):
            with self.assertRaises(RuntimeError) as ctx:
                investigator.generate_report(case)
            self.assertIn("Validation crash", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
