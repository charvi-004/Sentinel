from __future__ import annotations

import json
import os
import unittest

from fastapi.testclient import TestClient

from backend.main import app


class SentinelApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')
        self.assertEqual(response.json()['service'], 'sentinel-risk-engine')

    def test_post_risk_analyze_valid_transaction(self):
        response = self.client.post('/risk/analyze', json={'transaction_id': 'txn_00001'})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn('transaction_id', payload)
        self.assertIn('risk', payload)
        self.assertIn('investigation', payload)
        self.assertIn('level', payload['risk'])

    def test_get_risk_by_transaction_id(self):
        response = self.client.get('/risk/txn_00001')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['transaction_id'], 'txn_00001')

    def test_get_metrics(self):
        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200)
        metrics = response.json()
        self.assertIn('dataset', metrics)
        self.assertIn('model', metrics)
        self.assertIn('detection_metrics', metrics)

    def test_unknown_transaction_404(self):
        response = self.client.get('/risk/txn_unknown')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()['error'], 'transaction_not_found')

    def test_malformed_request_422(self):
        response = self.client.post('/risk/analyze', json={'wrong': 'value'})
        self.assertEqual(response.status_code, 422)

    def test_response_schema_validation(self):
        response = self.client.post('/risk/analyze', json={'transaction_id': 'txn_00001'})
        payload = response.json()
        self.assertIn('transaction', payload)
        self.assertIn('amount', payload['transaction'])
        self.assertIn('currency', payload['transaction'])
        self.assertIn('timestamp', payload['transaction'])
        self.assertIn('score', payload['risk'])
        self.assertIn('level', payload['risk'])
        self.assertIn('recommended_action', payload['risk'])
        self.assertIn('executive_summary', payload['investigation'])

    def test_deterministic_risk_decision_preservation(self):
        response = self.client.post('/risk/analyze', json={'transaction_id': 'txn_00001'})
        payload = response.json()
        self.assertEqual(payload['risk']['level'], payload['investigation']['risk_assessment']['risk_level'])
        self.assertEqual(payload['risk']['recommended_action'], payload['investigation']['risk_assessment']['recommended_action'])

    def test_no_ground_truth_leakage(self):
        response = self.client.post('/risk/analyze', json={'transaction_id': 'txn_00001'})
        body = json.dumps(response.json())
        for forbidden in ['is_abuse', 'abuse_type', 'abuse_cluster_id', 'ground_truth']:
            self.assertNotIn(forbidden, body)

    def test_ai_fallback_works_without_api_key(self):
        original = os.environ.get('SENTINEL_LLM_API_KEY')
        os.environ.pop('SENTINEL_LLM_API_KEY', None)
        try:
            response = self.client.post('/risk/analyze', json={'transaction_id': 'txn_00001'})
            self.assertEqual(response.status_code, 200)
            self.assertIn('executive_summary', response.json()['investigation'])
        finally:
            if original is not None:
                os.environ['SENTINEL_LLM_API_KEY'] = original

    def test_ai_fallback_works_with_dummy_api_key(self):
        original = os.environ.get('SENTINEL_LLM_API_KEY')
        os.environ['SENTINEL_LLM_API_KEY'] = 'dummy_configured_api_key'
        try:
            response = self.client.post('/risk/analyze', json={'transaction_id': 'txn_00001'})
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIn('transaction_id', payload)
            self.assertIn('transaction', payload)
            self.assertIn('risk', payload)
            self.assertIn('reasons', payload)
            self.assertIn('network_context', payload)
            self.assertIn('investigation', payload)
            self.assertEqual(payload['transaction_id'], 'txn_00001')
            self.assertIn('score', payload['risk'])
            self.assertIn('level', payload['risk'])
            self.assertIn('recommended_action', payload['risk'])
            self.assertIn('executive_summary', payload['investigation'])
            self.assertIn('risk_assessment', payload['investigation'])
            self.assertEqual(payload['risk']['level'], payload['investigation']['risk_assessment']['risk_level'])
            self.assertEqual(payload['risk']['recommended_action'], payload['investigation']['risk_assessment']['recommended_action'])
        finally:
            if original is not None:
                os.environ['SENTINEL_LLM_API_KEY'] = original
            else:
                os.environ.pop('SENTINEL_LLM_API_KEY', None)


if __name__ == '__main__':
    unittest.main()
