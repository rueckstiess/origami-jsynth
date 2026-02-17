"""Evaluation metrics for synthetic data."""

from .detection import DetectionResult, compute_detection
from .evaluate import EvaluationResult, evaluate_synthetic_data
from .fidelity import FidelityResult, compute_fidelity
from .privacy import PrivacyResult, compute_privacy
from .utility import UtilityResult, compute_utility

__all__ = [
    "EvaluationResult",
    "evaluate_synthetic_data",
    "FidelityResult",
    "compute_fidelity",
    "UtilityResult",
    "compute_utility",
    "PrivacyResult",
    "compute_privacy",
    "DetectionResult",
    "compute_detection",
]
