from __future__ import annotations

import os

from ml.evaluation.end_to_end import generate_demo_cases
from ml.investigator.investigator import Investigator, FallbackInvestigator, render_investigation_report


def main() -> None:
    cases = generate_demo_cases()
    print("============================================================")
    print("SENTINEL — AI RISK MANAGER")
    print("============================================================")
    for case in cases:
        print("\nTRANSACTION")
        print(f"Transaction: {case['transaction_id']}")
        print("\nRISK ASSESSMENT")
        print(f"Risk Level: {case['risk_level']}")
        print(f"Risk Score: {case['risk_score']:.1f}")
        print(f"Recommended Action: {case['recommended_action']}")
        print("\nWHY WAS IT FLAGGED?")
        if case["risk_level"] == "LOW":
            print("No material high-risk evidence was observed in the current transaction context.")
        else:
            print("Observed behavioral and network evidence supports an elevated risk assessment.")
        print("\nNETWORK CONTEXT")
        print("Connected relationships were reviewed and summarized in the structured case.")
        print("\nAI INVESTIGATION")
        report = FallbackInvestigator().generate_investigation({
            "case_id": case["transaction_id"],
            "transaction": {"id": case["transaction_id"], "amount": 0.0, "currency": "USD", "timestamp": ""},
            "risk": {"score": float(case["risk_score"]), "level": case["risk_level"], "recommended_action": case["recommended_action"]},
            "evidence": [{"signal": "MODEL_SIGNAL", "severity": "MEDIUM", "value": case["risk_score"], "description": "Structured evidence from the deterministic risk engine."}],
            "network": {"connected_customers": 1, "connected_merchants": 1, "connected_devices": 1, "connected_instruments": 1},
        })
        print(report["executive_summary"])
        print("\nRECOMMENDED ACTION")
        print(case["recommended_action"])
        print("\n")


if __name__ == "__main__":
    main()
