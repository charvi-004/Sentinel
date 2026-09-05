from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml.evaluation.end_to_end import _cached_transaction_tables
from ml.investigator.case_builder import build_case
from ml.investigator.investigator import Investigator, FallbackInvestigator
from ml.risk.risk_engine import _generate_reasons, _load_training_thresholds, assess_transaction, load_default_engine, score_all_transactions
from backend.schemas import ErrorResponse, InvestigationReport, NetworkContext, RiskAssessment, RiskReason, RiskRequest, RiskResponse, TransactionInfo, TransactionListResponse, TransactionSummary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / 'ml' / 'models' / 'artifacts'



@lru_cache(maxsize=1)
def _stored_metrics() -> dict[str, Any]:
    path = ARTIFACT_DIR / 'final_evaluation.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))

    config = json.loads((ARTIFACT_DIR / 'baseline_config.json').read_text(encoding='utf-8'))
    from ml.evaluation.end_to_end import run_evaluation

    return run_evaluation()

@lru_cache(maxsize=1)
def _seeded_model_state() -> tuple[Any, float, dict[str, Any]]:
    return load_default_engine()


app = FastAPI(
    title='Sentinel Risk Engine API',
    description='Backend API for deterministic risk analysis and structured investigation reporting.',
    version='1.0.0',
    docs_url='/docs',
)

allowed_origins = [
    'http://localhost',
    'http://localhost:3000',
    'http://127.0.0.1',
    'http://127.0.0.1:3000',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'sentinel-risk-engine'}


@app.get('/metrics')
def metrics() -> dict[str, Any]:
    return _stored_metrics()


@app.get('/risk/{transaction_id}')
def get_risk(transaction_id: str) -> dict[str, Any]:
    return analyze_risk(request=RiskRequest(transaction_id=transaction_id))


@app.get('/transactions', response_model=TransactionListResponse)
def list_transactions(
    risk_level: str | None = Query(default=None, description='Optional risk level filter'),
    page: int = Query(default=1, ge=1, description='Page number starting at 1.'),
    page_size: int = Query(default=20, ge=1, le=100, description='Rows per page.'),
) -> dict[str, Any]:
    features, graph, _, _, transactions = _cached_transaction_tables()
    model, threshold, _ = _seeded_model_state()
    scored_df = score_all_transactions(features, transactions, model, threshold)

    if risk_level is not None:
        scored_df = scored_df[scored_df['risk_level'].str.upper() == risk_level.strip().upper()]

    scored_df = scored_df.sort_values('risk_score', ascending=False).reset_index(drop=True)
    total = len(scored_df)
    start = (page - 1) * page_size
    paged_df = scored_df.iloc[start:start + page_size].copy()

    items: list[TransactionSummary] = []
    for row in paged_df.to_dict(orient='records'):
        tx_id = str(row['transaction_id'])
        feature_row = features[features['transaction_id'] == tx_id].copy()
        graph_row = graph[graph['transaction_id'] == tx_id].copy()
        if feature_row.empty:
            feature_row = pd.DataFrame([{'transaction_id': tx_id}])
        if graph_row.empty:
            graph_row = pd.DataFrame([{'transaction_id': tx_id}])

        reasons = _generate_reasons(feature_row.iloc[0], graph_row.iloc[0] if not graph_row.empty else pd.Series(dtype=object), _load_training_thresholds())
        if str(row['risk_level']).upper() in {'HIGH', 'CRITICAL'} and not reasons:
            severity = 'HIGH' if str(row['risk_level']).upper() == 'HIGH' else 'HIGH'
            reasons = [{
                'type': 'BEHAVIORAL_SIGNAL',
                'severity': severity,
                'description': f"The model produced a {float(row['risk_score']):.1f} risk score, which exceeds the validated operating threshold.",
                'evidence': {'risk_score': round(float(row['risk_score']), 1), 'risk_threshold': threshold},
            }]
        top_reason = reasons[0]['description'] if reasons else None
        items.append(
            TransactionSummary(
                transaction_id=tx_id,
                amount=float(row.get('amount', 0.0)),
                currency=str(row.get('currency', 'USD')),
                timestamp=str(row.get('timestamp', '')),
                risk_score=float(row['risk_score']),
                risk_level=str(row['risk_level']),
                recommended_action=str(row['recommended_action']),
                top_reason=top_reason,
            )
        )

    return {
        'items': items,
        'total': total,
        'page': page,
        'page_size': page_size,
    }


@app.post('/risk/analyze', response_model=RiskResponse)
def analyze_risk(request: RiskRequest) -> dict[str, Any]:
    tx_id = request.transaction_id

    features, graph, _, _, transactions = _cached_transaction_tables()
    feature_row = features[features['transaction_id'] == tx_id].copy()
    if feature_row.empty:
        raise HTTPException(status_code=404, detail={'error': 'transaction_not_found', 'message': f'Transaction {tx_id} does not exist.'})

    model, threshold, _ = _seeded_model_state()
    graph_row = graph[graph['transaction_id'] == tx_id].copy()
    if graph_row.empty:
        graph_row = pd.DataFrame([{'transaction_id': tx_id, 'other_customers_on_device': 0, 'other_customers_on_instrument': 0, 'customers_connected_via_device': 0, 'customers_connected_via_instrument': 0, 'unique_customer_neighbors': 0, 'local_degree': 0}])
    txn_row = transactions[transactions['transaction_id'] == tx_id].copy()
    if txn_row.empty:
        txn_row = pd.DataFrame([{'transaction_id': tx_id, 'amount': 0.0, 'currency': 'USD', 'timestamp': ''}])

    try:
        assessment = assess_transaction(
            transaction_id=tx_id,
            model=model,
            risk_threshold=threshold,
            features_df=feature_row,
            graph_df=graph_row,
            transactions_df=txn_row,
            reason_thresholds=_load_training_thresholds(),
        )
        case = build_case(assessment)
        investigator = Investigator(provider=FallbackInvestigator()) if not os.getenv('SENTINEL_LLM_API_KEY') else Investigator()
        report = investigator.generate_report(case)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={'error': 'internal_processing_error', 'message': str(exc)}) from exc

    report = dict(report)
    risk_block = report.get('risk_assessment', {})
    risk_score = float(risk_block.get('risk_score', assessment['risk_score']))
    risk_level = str(risk_block.get('risk_level', assessment['risk_level'])).upper()
    recommended_action = str(risk_block.get('recommended_action', assessment['recommended_action'])).upper()
    report['risk_assessment'] = {
        'score': risk_score,
        'level': risk_level,
        'recommended_action': recommended_action,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'recommended_action_alias': recommended_action,
    }

    response = {
        'transaction_id': tx_id,
        'transaction': {
            'amount': float(assessment['transaction']['amount']),
            'currency': str(assessment['transaction']['currency']),
            'timestamp': str(assessment['transaction']['timestamp']),
        },
        'risk': {
            'score': float(assessment['risk_score']),
            'level': str(assessment['risk_level']),
            'recommended_action': str(assessment['recommended_action']),
        },
        'reasons': [
            {
                'type': reason['type'],
                'severity': reason['severity'],
                'description': reason['description'],
                'evidence': reason.get('evidence', {}),
            }
            for reason in assessment.get('reasons', [])
        ],
        'network_context': {
            'connected_customers': int(assessment['network_context'].get('connected_customers', 0)),
            'connected_merchants': int(assessment['network_context'].get('connected_merchants', 0)),
            'connected_devices': int(assessment['network_context'].get('connected_devices', 0)),
            'connected_instruments': int(assessment['network_context'].get('connected_instruments', 0)),
        },
        'investigation': report,
    }
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {'error': 'internal_processing_error', 'message': 'An unexpected error occurred.'}
    return JSONResponse(status_code=exc.status_code, content=detail)


@app.exception_handler(Exception)
async def general_exception_handler(_request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={'error': 'internal_processing_error', 'message': 'Internal processing failure.'})
