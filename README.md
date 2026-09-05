# SENTINEL

SENTINEL is an explainable AI risk-management layer designed for merchants using payment platforms such as Razorpay. It helps identify suspicious transaction behavior, surface coordinated activity across customers, devices, and payment instruments, prioritize operational review, and balance fraud or abuse risk against customer friction and business cost.

## Risk Workflow

SENTINEL is designed around a merchant-facing payment-risk workflow:

Payment transaction
-> behavioral feature extraction
-> ML risk score
-> network and graph evidence
-> explainable risk assessment
-> merchant policy
-> operational action

The system is intended to help merchants investigate:

- velocity bursts
- entity reuse
- coordinated activity
- cross-merchant patterns
- excessive false positives

These statements describe SENTINEL's intended product use. They do not claim Razorpay deployment, proprietary data access, internal API usage, or production performance.

## Evaluation Scope and Deployment Readiness

SENTINEL is evaluated on a controlled synthetic dataset designed to reproduce observable behavioral and network patterns. The dataset contains 10,000 transactions across 100 merchants, 2,000 customers, 500 devices, and 1,000 payment instruments over 120 days. It includes 355 modeled abuse transactions covering shared devices, shared payment instruments, velocity bursts, and cross-merchant rings.

The data uses a chronological 70/15/15 split:

- 7,000 training transactions
- 1,500 validation transactions
- 1,500 test transactions

The benchmark was designed to demonstrate behavioral velocity features, entity reuse, graph relationships, explainability, and business-cost trade-offs. No target or temporal leakage was found in the feature audit.

Reported precision, recall, F1, ROC-AUC, PR-AUC, and simulated costs describe performance on this controlled synthetic benchmark. They must not be interpreted as real-world or production fraud performance.

Before deployment, SENTINEL would require validation on historical merchant and payment data, testing against real-world distributions, threshold recalibration, drift monitoring, and merchant-specific policy and cost calibration.

## Model Evaluation

The primary production risk scorer is a behavioral Logistic Regression model using 32 behavioral features.

On the held-out synthetic test set:

- Precision: 51.14%
- Recall: 100%
- F1: 67.67%
- ROC-AUC: 99.45%
- PR-AUC: 80.86%

The production operating threshold is 0.50.

The threshold was selected using the validation set only, under the current simulated business-cost assumptions:

- False positive cost: $5
- False negative cost: $100

At threshold 0.50 on the held-out test set:

- False positives: 43
- False negatives: 0
- Simulated cost: $215

These are synthetic benchmark/test-set evaluation results, not production fraud-detection performance.

## Graph Feature Ablation

A controlled ablation compared the behavioral model with and without graph-derived features.

After removing nine duplicated graph representations, the graph-enhanced experiment used 55 features and achieved:

- Precision: 74.55%
- Recall: 91.11%
- F1: 82.00%
- FPR: 0.96%
- PR-AUC: 85.17%

compared with the behavioral-only model:

- Precision: 51.14%
- Recall: 100%
- F1: 67.67%
- FPR: 2.96%
- PR-AUC: 80.86%

The experiment indicates that graph-derived features provide incremental predictive signal beyond the behavioral features. The graph experiment is an evaluation/ablation result and does not replace the current production model.

## Baseline Comparison

A reproducible, point-in-time-safe naive rule baseline was evaluated on the held-out test split (1,500 transactions, chronological 70/15/15 split) to benchmark against the behavioral Logistic Regression.

### Frozen Heuristic Rule
The rule uses four fixed heuristic thresholds derived strictly from training-distribution observations (not optimized on validation or test labels):

FLAG transaction if ANY of:
- `customer_txn_count_30m >= 2`
- `customer_txn_count_1h >= 3`
- `device_account_count_before >= 10`
- `instrument_account_count_before >= 6`

### Synthetic Test-Set Performance Comparison
Simulated business costs assume **$5 per false positive (FP)** (operational review / customer friction) and **$100 per false negative (FN)** (unmitigated abuse loss). These represent synthetic benchmark evaluation assumptions, not real merchant loss reductions or Razorpay production metrics.

| Metric | Frozen Naive Rule | Behavioral Logistic Regression |
|---|---|---|
| **Precision** | 4.03% (0.0403) | 51.14% (0.5114) |
| **Recall** | 64.44% (0.6444) | 100.00% (1.0000) |
| **F1 Score** | 7.58% (0.0758) | 67.67% (0.6767) |
| **False Positive Rate (FPR)** | 47.49% | 2.96% |
| **False Positives (FP)** | 691 | 43 |
| **False Negatives (FN)** | 16 | 0 |
| **Transactions Flagged** | 720 / 1,500 (48.0%) | 88 / 1,500 (5.87%) |
| **Simulated Cost** | **$5,055** | **$215** |

The naive heuristic flags 48.0% of all test transactions while still missing 16 abuse cases (35.56% false negative rate), resulting in $5,055 in simulated test-set cost ($3,455 in review friction + $1,600 in missed fraud). In contrast, the behavioral Logistic Regression achieves 100% recall with only 43 false positives (2.96% FPR) and $215 in simulated cost.

This demonstrates the value of learned probabilistic risk scoring over an uncalibrated heuristic rule on the controlled benchmark. This heuristic is a simple strawman baseline, not an optimized industry rule system.

### Reproduction Command
To reproduce the naive baseline evaluation and regenerate the comparison artifacts:

```bash
python experiments/rule_baseline.py
```

Generated artifacts:
- `ml/models/artifacts/rule_baseline_comparison.csv`
- `ml/models/artifacts/rule_baseline_summary.json`

Again, these are controlled synthetic test-set evaluation results, not production fraud-detection performance.

## Running SENTINEL Locally

### Architecture Note: Risk Engine Implementations
- `ml/risk/risk_engine.py`: The **active production inference engine** utilized by the FastAPI backend (`backend/main.py`). It evaluates transaction risk using the calibrated 32-feature behavioral Logistic Regression model (`baseline_logistic_regression.joblib`) at the validated 0.50 operating threshold.
- `ml/models/risk_engine.py`: A **historical / non-production HGB experiment script** from an exploratory 64-feature duplicate-inclusive evaluation. It is an offline research artifact and must not be confused with the active production risk scorer.

### Backend Setup
Install dependencies and run the FastAPI server:

```bash
python -m pip install -r ml/requirements.txt fastapi uvicorn httpx
uvicorn backend.main:app --reload
```
The backend API serves on `http://127.0.0.1:8000`.

### Frontend Setup
The Next.js frontend application is located directly at the project root (`app/` directory with `package.json` at the root; no `cd app` required):

```bash
npm install
npm run dev
```
The user interface serves on `http://localhost:3000`.

### Running Tests
Execute the lightweight automated regression suite:

```bash
python -m unittest backend/test_api.py ml/investigator/test_investigator.py ml/risk/test_risk_engine.py
```

### Experiment Reproduction
- **Naive Rule Baseline**:
  ```bash
  python experiments/rule_baseline.py
  ```
  *(Reproduces naive heuristic evaluation; generates `rule_baseline_comparison.csv` and `rule_baseline_summary.json`)*

- **Fair Graph Feature Ablation**:
  ```bash
  python experiments/graph_ablation.py
  ```
  *(Reproduces fair 32 vs 55 feature ablation; generates `graph_ablation_comparison.csv` and `graph_ablation_summary.json`)*

## Deployment Considerations

The current benchmark demonstrates the architecture, feature engineering, explainability, graph context, thresholding methodology, and business-cost framework.

It does not establish production fraud-detection performance.

A real deployment would require:

- historical merchant/payment data
- validation against real transaction distributions
- merchant-specific threshold and policy calibration
- monitoring for distribution and behavior drift
- ongoing precision/recall and business-cost monitoring
- operational review and feedback loops

Do not claim the system is production-ready solely because of the benchmark metrics.