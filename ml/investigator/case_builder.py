from __future__ import annotations

import uuid
from typing import Any


def build_case(assessment: dict[str, Any]) -> dict[str, Any]:
    """Convert a deterministic risk assessment into a structured investigation case."""
    evidence = []
    for reason in assessment.get("reasons", []):
        evidence.append(
            {
                "signal": str(reason.get("type", "UNKNOWN")).replace(" ", "_"),
                "severity": str(reason.get("severity", "MEDIUM")).upper(),
                "value": _coerce_value(reason.get("evidence", {})),
                "description": str(reason.get("description", "Observed evidence.")),
            }
        )

    network = assessment.get("network_context", {})
    transaction = assessment.get("transaction", {})
    case = {
        "case_id": f"case_{uuid.uuid4().hex[:12]}",
        "transaction": {
            "id": str(assessment.get("transaction_id", "unknown")),
            "amount": float(transaction.get("amount", 0.0)),
            "currency": str(transaction.get("currency", "USD")),
            "timestamp": str(transaction.get("timestamp", transaction.get("transaction_time", ""))),
        },
        "risk": {
            "score": float(assessment.get("risk_score", 0.0)),
            "level": str(assessment.get("risk_level", "LOW")).upper(),
            "recommended_action": str(assessment.get("recommended_action", "ALLOW")).upper(),
        },
        "evidence": evidence,
        "network": {
            "connected_customers": int(network.get("connected_customers", 0) or 0),
            "connected_merchants": int(network.get("connected_merchants", 0) or 0),
            "connected_devices": int(network.get("connected_devices", 0) or 0),
            "connected_instruments": int(network.get("connected_instruments", 0) or 0),
        },
    }
    return case


def _coerce_value(evidence: dict[str, Any]) -> float | int | str:
    if not evidence:
        return 0
    if len(evidence) == 1:
        value = next(iter(evidence.values()))
        return int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value
    return str(evidence)
