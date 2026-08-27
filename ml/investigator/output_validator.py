from __future__ import annotations

import json
from typing import Any


REQUIRED_FIELDS = {
    "executive_summary",
    "risk_assessment",
    "key_findings",
    "risk_pattern",
    "network_analysis",
    "recommended_investigation",
    "uncertainty",
    "analyst_summary",
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def validate_report(raw: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Validate generated AI output and enforce deterministic risk values."""
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
    elif isinstance(raw, dict):
        payload = raw
    else:
        payload = {}

    risk = _as_dict(payload.get("risk_assessment"))
    if not risk:
        risk = {"risk_score": case["risk"]["score"], "risk_level": case["risk"]["level"], "recommended_action": case["risk"]["recommended_action"]}

    risk_score = float(case["risk"]["score"])
    risk_level = str(case["risk"]["level"]).upper()
    recommended_action = str(case["risk"]["recommended_action"]).upper()

    fixed = {
        "executive_summary": str(payload.get("executive_summary") or "Observed evidence was reviewed and summarized without inventing new facts."),
        "risk_assessment": {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
        },
        "key_findings": payload.get("key_findings") if isinstance(payload.get("key_findings"), list) and payload.get("key_findings") else [
            {
                "finding": "The available evidence supports a high-priority review.",
                "evidence": [
                    f"risk_score={risk_score}",
                    f"risk_level={risk_level}",
                    f"recommended_action={recommended_action}",
                ],
            }
        ],
        "risk_pattern": str(payload.get("risk_pattern") or "The observed pattern is consistent with the structured evidence available from the risk model and graph context."),
        "network_analysis": str(payload.get("network_analysis") or "Network evidence was reviewed but no unsupported claims were added."),
        "recommended_investigation": payload.get("recommended_investigation") if isinstance(payload.get("recommended_investigation"), list) and payload.get("recommended_investigation") else [
            "Review the transaction timeline for abnormal velocity or repetition.",
            "Check device and instrument reuse patterns against recent customer activity.",
            "Confirm merchant and customer context before taking a final action.",
        ],
        "uncertainty": str(payload.get("uncertainty") or "The available evidence supports risk prioritization but does not independently establish fraudulent intent."),
        "analyst_summary": str(payload.get("analyst_summary") or "The risk engine remains authoritative; the investigation should focus on corroborating the observed evidence."),
    }

    for field in REQUIRED_FIELDS:
        if field not in fixed:
            return _fallback_report(case)

    if risk.get("risk_level") not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        fixed["risk_assessment"]["risk_level"] = risk_level
    if risk.get("recommended_action") not in {"ALLOW", "MONITOR", "REVIEW", "ESCALATE"}:
        fixed["risk_assessment"]["recommended_action"] = recommended_action

    fixed["risk_assessment"]["risk_score"] = risk_score
    fixed["risk_assessment"]["risk_level"] = risk_level
    fixed["risk_assessment"]["recommended_action"] = recommended_action

    if isinstance(raw, str):
        try:
            json.dumps(fixed)
        except TypeError:
            return _fallback_report(case)

    if "is_abuse" in json.dumps(fixed).lower():
        return _fallback_report(case)

    return fixed


def _fallback_report(case: dict[str, Any]) -> dict[str, Any]:
    risk = case["risk"]
    evidence = case.get("evidence", [])
    if not evidence:
        evidence = [{"signal": "MODEL_SIGNAL", "severity": "MEDIUM", "description": "Structured evidence available from the risk engine.", "value": risk["score"]}]
    return {
        "executive_summary": "Observed evidence suggests a risk pattern consistent with the deterministic risk signal; no unsupported speculation was introduced.",
        "risk_assessment": {
            "risk_score": float(risk["score"]),
            "risk_level": str(risk["level"]).upper(),
            "recommended_action": str(risk["recommended_action"]).upper(),
        },
        "key_findings": [{"finding": "Available structured evidence supports escalation based on the deterministic risk engine.", "evidence": [item.get("description", "Structured evidence") for item in evidence[:3]]}],
        "risk_pattern": "The observed pattern is consistent with the strongest available behavioral and network evidence.",
        "network_analysis": "Network relationships were reviewed without inventing unsupported connections or identifiers.",
        "recommended_investigation": [
            "Review the transaction timeline and recent customer velocity.",
            "Check whether the device or instrument was reused by other customers.",
            "Confirm the merchant and customer context before final action.",
        ],
        "uncertainty": "The available evidence supports risk prioritization, but it does not independently establish fraudulent intent.",
        "analyst_summary": "The deterministic risk engine remains authoritative; the investigation should focus on corroborating the observed evidence.",
    }
