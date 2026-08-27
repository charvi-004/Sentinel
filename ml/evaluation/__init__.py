"""Evaluation and end-to-end orchestration for Sentinel."""

from .end_to_end import evaluate_transaction, generate_demo_cases, run_evaluation

__all__ = ["evaluate_transaction", "generate_demo_cases", "run_evaluation"]
