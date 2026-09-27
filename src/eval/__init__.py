"""Evaluation metrics, Argo validation, and benchmarking for OceanEmbed."""

from src.eval.argo_matcher import ArgoMatcher
from src.eval.metrics import calculate_depth_metrics, evaluate_predictions_against_argo
from src.eval.benchmark import run_full_benchmark

__all__ = [
    "ArgoMatcher",
    "calculate_depth_metrics",
    "evaluate_predictions_against_argo",
    "run_full_benchmark"
]
