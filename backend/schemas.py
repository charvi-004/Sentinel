from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RiskRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, description="Transaction identifier to analyze.")


class TransactionInfo(BaseModel):
    amount: float
    currency: str
    timestamp: str


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra='allow')

    score: float | None = None
    level: str | None = None
    recommended_action: str | None = None
    risk_score: float | None = None
    risk_level: str | None = None
    recommended_action_alias: str | None = None


class RiskReason(BaseModel):
    type: str
    severity: str
    description: str
    evidence: dict[str, Any]


class NetworkContext(BaseModel):
    connected_customers: int = 0
    connected_merchants: int = 0
    connected_devices: int = 0
    connected_instruments: int = 0


class InvestigationReport(BaseModel):
    executive_summary: str
    risk_assessment: RiskAssessment
    key_findings: list[dict[str, Any]]
    risk_pattern: str
    network_analysis: str
    recommended_investigation: list[str]
    uncertainty: str
    analyst_summary: str


class RiskResponse(BaseModel):
    transaction_id: str
    transaction: TransactionInfo
    risk: RiskAssessment
    reasons: list[RiskReason]
    network_context: NetworkContext
    investigation: InvestigationReport


class ErrorResponse(BaseModel):
    error: str
    message: str


class TransactionSummary(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    timestamp: str
    risk_score: float
    risk_level: str
    recommended_action: str
    top_reason: str | None = None


class TransactionListResponse(BaseModel):
    items: list[TransactionSummary]
    total: int
    page: int
    page_size: int
