#!/usr/bin/env python3
"""
Standalone API health check script for Sentinel Risk Engine.
Tests all critical endpoints and reports pass/fail status.
Assumes the server is already running at http://127.0.0.1:8000
"""

import sys
import requests

BASE_URL = 'http://127.0.0.1:8000'
CHECKS = []


def run_check(check_name: str, check_func):
    """Run a check and record the result."""
    try:
        passed, reason = check_func()
        status = 'PASS' if passed else 'FAIL'
        print(f"{status}: {check_name}")
        if not passed and reason:
            print(f"       Reason: {reason}")
        CHECKS.append(passed)
        return passed
    except Exception as exc:
        print(f"FAIL: {check_name}")
        print(f"       Reason: {str(exc)}")
        CHECKS.append(False)
        return False


def check_health():
    """a. GET /health -> expect 200, JSON has "status": "ok"."""
    resp = requests.get(f'{BASE_URL}/health', timeout=5)
    if resp.status_code != 200:
        return False, f"Expected status 200, got {resp.status_code}"
    data = resp.json()
    if data.get('status') != 'ok':
        return False, f"Expected 'status': 'ok', got {data.get('status')}"
    return True, None


def check_docs():
    """b. GET /docs -> expect 200."""
    resp = requests.get(f'{BASE_URL}/docs', timeout=5)
    if resp.status_code != 200:
        return False, f"Expected status 200, got {resp.status_code}"
    return True, None


def check_metrics():
    """c. GET /metrics -> expect 200, JSON has "detection_metrics" key."""
    resp = requests.get(f'{BASE_URL}/metrics', timeout=5)
    if resp.status_code != 200:
        return False, f"Expected status 200, got {resp.status_code}"
    data = resp.json()
    if 'detection_metrics' not in data:
        return False, f"Expected 'detection_metrics' key in response, got keys: {list(data.keys())}"
    return True, None


def check_list_transactions():
    """d. GET /transactions?page=1&page_size=5 -> expect 200, items sorted by risk_score descending."""
    resp = requests.get(f'{BASE_URL}/transactions?page=1&page_size=5', timeout=5)
    if resp.status_code != 200:
        return False, f"Expected status 200, got {resp.status_code}"
    data = resp.json()
    
    # Check structure
    if 'items' not in data:
        return False, "Expected 'items' key in response"
    if 'total' not in data:
        return False, "Expected 'total' key in response"
    
    items = data['items']
    if not isinstance(items, list):
        return False, f"Expected 'items' to be a list, got {type(items)}"
    if len(items) > 5:
        return False, f"Expected items length <= 5, got {len(items)}"
    
    if not isinstance(data['total'], int):
        return False, f"Expected 'total' to be int, got {type(data['total'])}"
    
    # Check sorting: risk_score should be descending
    if len(items) > 1:
        for i in range(len(items) - 1):
            if items[i]['risk_score'] < items[i + 1]['risk_score']:
                return False, f"Items not sorted by risk_score descending: {items[i]['risk_score']} < {items[i + 1]['risk_score']}"
    
    return True, None


def check_filter_by_risk_level():
    """e. GET /transactions?risk_level=CRITICAL&page=1&page_size=5 -> all items have risk_level == "CRITICAL"."""
    resp = requests.get(f'{BASE_URL}/transactions?risk_level=CRITICAL&page=1&page_size=5', timeout=5)
    if resp.status_code != 200:
        return False, f"Expected status 200, got {resp.status_code}"
    data = resp.json()
    
    items = data.get('items', [])
    for item in items:
        if item.get('risk_level') != 'CRITICAL':
            return False, f"Expected all items to have risk_level='CRITICAL', got {item.get('risk_level')}"
    
    return True, None


def check_analyze_valid_transaction():
    """f. POST /risk/analyze with {"transaction_id": "txn_03795"} -> expect 200, has "risk", "transaction", "investigation" keys."""
    resp = requests.post(
        f'{BASE_URL}/risk/analyze',
        json={'transaction_id': 'txn_03795'},
        timeout=5
    )
    if resp.status_code != 200:
        return False, f"Expected status 200, got {resp.status_code}"
    data = resp.json()
    
    required_keys = ['risk', 'transaction', 'investigation']
    for key in required_keys:
        if key not in data:
            return False, f"Expected '{key}' key in response, got keys: {list(data.keys())}"
    
    return True, None


def check_analyze_invalid_transaction():
    """g. POST /risk/analyze with {"transaction_id": "this_id_does_not_exist"} -> expect 404."""
    resp = requests.post(
        f'{BASE_URL}/risk/analyze',
        json={'transaction_id': 'this_id_does_not_exist'},
        timeout=5
    )
    if resp.status_code != 404:
        return False, f"Expected status 404, got {resp.status_code}"
    
    return True, None


def check_get_risk():
    """h. GET /risk/txn_03795 -> expect 200, JSON has "risk" key."""
    resp = requests.get(f'{BASE_URL}/risk/txn_03795', timeout=5)
    if resp.status_code != 200:
        return False, f"Expected status 200, got {resp.status_code}"
    data = resp.json()
    
    if 'risk' not in data:
        return False, f"Expected 'risk' key in response, got keys: {list(data.keys())}"
    
    return True, None


def main():
    """Run all checks and report results."""
    print("Starting API health checks...\n")
    
    run_check("GET /health", check_health)
    run_check("GET /docs", check_docs)
    run_check("GET /metrics", check_metrics)
    run_check("GET /transactions?page=1&page_size=5", check_list_transactions)
    run_check("GET /transactions?risk_level=CRITICAL&page=1&page_size=5", check_filter_by_risk_level)
    run_check("POST /risk/analyze (valid transaction)", check_analyze_valid_transaction)
    run_check("POST /risk/analyze (invalid transaction)", check_analyze_invalid_transaction)
    run_check("GET /risk/txn_03795", check_get_risk)
    
    print()
    passed_count = sum(CHECKS)
    total_count = len(CHECKS)
    print(f"{passed_count}/{total_count} checks passed.")
    
    if passed_count == total_count:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
