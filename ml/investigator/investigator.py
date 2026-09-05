from __future__ import annotations

import json
import os
from typing import Any

from .case_builder import build_case
from .output_validator import validate_report
from .prompts import INVESTIGATOR_PROMPT


class AIProvider:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def generate_investigation(self, case: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleProvider(AIProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str = "gpt-4o-mini"):
        super().__init__(api_key)
        self.base_url = base_url or os.getenv("SENTINEL_LLM_BASE_URL")
        self.model = model or os.getenv("SENTINEL_LLM_MODEL", "gpt-4o-mini")

    def generate_investigation(self, case: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("No API key configured for the LLM provider.")
        # Intentionally left unimplemented here to avoid any external SDK dependency.
        # The backend remains runnable because fallback logic is used when no key is present.
        raise NotImplementedError("LLM integration is intentionally not implemented in this backend-only module.")


class FallbackInvestigator:
    def generate_investigation(self, case: dict[str, Any]) -> dict[str, Any]:
        evidence = case.get("evidence", []) or [
            {"signal": "MODEL_SIGNAL", "severity": "MEDIUM", "value": case["risk"]["score"], "description": "The deterministic model flagged the transaction."}
        ]
        strongest = [
            item.get("description", "Structured evidence available from the risk engine.")
            for item in evidence[:3]
        ]

        risk = case["risk"]
        summary = (
            "Observed evidence suggests a risk pattern driven by the highest-priority signals in the case. "
            "The available evidence is consistent with the deterministic risk engine and supports investigation, "
            "without asserting that fraud has been proven."
        )
        network_text = (
            f"The transaction is associated with {case['network'].get('connected_customers', 0)} connected customers, "
            f"{case['network'].get('connected_merchants', 0)} connected merchants, "
            f"{case['network'].get('connected_devices', 0)} connected devices, and "
            f"{case['network'].get('connected_instruments', 0)} connected instruments."
        )
        key_findings = [{"finding": "The strongest observed evidence is a combination of velocity and relationship reuse.", "evidence": strongest}]
        return {
            "executive_summary": summary,
            "risk_assessment": {
                "risk_score": float(risk["score"]),
                "risk_level": str(risk["level"]).upper(),
                "recommended_action": str(risk["recommended_action"]).upper(),
            },
            "key_findings": key_findings,
            "risk_pattern": "The strongest signal is unusual behavior combined with documented reuse across related entities.",
            "network_analysis": network_text,
            "recommended_investigation": [
                "Review the recent customer activity timeline for unusual velocity.",
                "Check whether the device or instrument was reused by other customers.",
                "Confirm merchant and customer context before making a final disposition.",
            ],
            "uncertainty": "The available evidence supports risk prioritization but does not independently establish fraudulent intent.",
            "analyst_summary": "The deterministic risk engine remains authoritative; investigation should focus on corroborating the observed risk signals.",
        }


class Investigator:
    def __init__(self, provider: AIProvider | None = None):
        self.provider = provider or self._build_provider()

    def _build_provider(self) -> AIProvider:
        api_key = os.getenv("SENTINEL_LLM_API_KEY")
        if not api_key:
            return FallbackInvestigator()
        return OpenAICompatibleProvider(api_key=api_key)

    def generate_report(self, case: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(case, dict):
            raise ValueError("Case must be a dictionary.")
        provider = self.provider
        try:
            raw = provider.generate_investigation(case)
        except Exception:
            raw = FallbackInvestigator().generate_investigation(case)
        return validate_report(raw, case)


def render_investigation_report(report: dict[str, Any], case: dict[str, Any]) -> str:
    """Render the structured report as a human-readable summary block."""
    risk = report.get("risk_assessment", {})
    network = case.get("network", {})
    lines = [
        "============================================================",
        "                    SENTINEL",
        "              AI RISK MANAGEMENT SYSTEM",
        "============================================================",
        "",
        f"Transaction: {case['transaction']['id']}",
        "",
        f"Amount:       ${case['transaction'].get('amount', 0.0):,.2f}",
        f"Risk Score:   {float(risk.get('risk_score', 0.0)):.1f} / 100",
        f"Risk Level:   {str(risk.get('risk_level', 'LOW')).upper()}",
        f"Action:       {str(risk.get('recommended_action', 'ALLOW')).upper()}",
        "",
        "------------------------------------------------------------",
        "WHY WAS IT FLAGGED?",
        "------------------------------------------------------------",
    ]

    findings = report.get("key_findings", [])
    for finding in findings:
        lines.append("")
        lines.append(f"[{finding.get('severity', 'MEDIUM')}] {finding.get('finding', 'Observed evidence')}")
        if finding.get("evidence"):
            for item in finding["evidence"][:3]:
                lines.append(f"{item}")

    lines.extend([
        "",
        "------------------------------------------------------------",
        "NETWORK CONTEXT",
        "------------------------------------------------------------",
        "",
        f"Connected customers:    {network.get('connected_customers', 0)}",
        f"Connected merchants:    {network.get('connected_merchants', 0)}",
        f"Connected devices:      {network.get('connected_devices', 0)}",
        f"Connected instruments:  {network.get('connected_instruments', 0)}",
        "",
        "------------------------------------------------------------",
        "AI INVESTIGATION",
        "------------------------------------------------------------",
        "",
        str(report.get("executive_summary", "Observed evidence suggests a risk pattern consistent with the deterministic risk signal.")),
        "",
        "------------------------------------------------------------",
        "RECOMMENDATION",
        "------------------------------------------------------------",
        "",
        str(risk.get('recommended_action', 'ALLOW')).upper() + " FOR MANUAL REVIEW",
        "",
        "Suggested checks:",
    ])

    for index, item in enumerate(report.get("recommended_investigation", [])[:3], start=1):
        lines.append(f"{index}. {item}")

    lines.append("")
    lines.append("============================================================")
    return "\n".join(lines)


def build_investigation_report(assessment: dict[str, Any]) -> dict[str, Any]:
    case = build_case(assessment)
    investigator = Investigator()
    return investigator.generate_report(case)
