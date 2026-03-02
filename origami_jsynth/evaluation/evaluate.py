"""Unified evaluation API for synthetic data.

Provides a single entry point for evaluating synthetic data quality against
real training and evaluation data. Runs all evaluation metrics:
fidelity, utility, privacy, and detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .detection import DetectionResult, compute_detection
from .fidelity import FidelityResult, compute_fidelity
from .privacy import PrivacyResult, compute_privacy
from .utility import UtilityResult, compute_utility


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results for synthetic data.

    Attributes:
        metrics: Flat dictionary of primary metrics for logging/comparison.
        details: Detailed result dictionaries for each evaluation type.
        config: Configuration used for evaluation (for reproducibility).
        fidelity_result: Detailed fidelity results (None if not computed).
        utility_result: Detailed utility results (None if not computed).
        privacy_result: Detailed privacy results (None if not computed).
        detection_result: Detailed detection results (None if not computed).
    """

    metrics: dict[str, float | int]
    details: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    fidelity_result: FidelityResult | None = None
    utility_result: UtilityResult | None = None
    privacy_result: PrivacyResult | None = None
    detection_result: DetectionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": self.metrics,
            "details": self.details,
            "config": self.config,
        }

    def __repr__(self) -> str:
        """Pretty-print primary metrics."""
        lines = ["EvaluationResult:"]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def evaluate_synthetic_data(
    train_records: list[dict],
    eval_records: list[dict],
    synthetic_records: list[dict],
    *,
    target_column: str | None = None,
    task_type: Literal["classification", "regression"] = "classification",
    fidelity: bool = True,
    utility: bool = True,
    privacy: bool = True,
    detection: bool = True,
    fidelity_kwargs: dict | None = None,
    utility_kwargs: dict | None = None,
    privacy_kwargs: dict | None = None,
    detection_kwargs: dict | None = None,
    verbose: bool = True,
) -> EvaluationResult:
    """Evaluate synthetic data quality against real train/eval data.

    All inputs are JSON records (list of dicts). Each metric uses type
    separation internally to handle heterogeneous and nested data.

    Args:
        train_records: Training data records (reference for fidelity/privacy).
        eval_records: Evaluation/test data records (held out, for utility).
        synthetic_records: Generated synthetic data records to evaluate.
        target_column: Field name for ML utility (e.g., "income"). If None,
            utility is skipped.
        task_type: "classification" or "regression" for ML utility.
        fidelity: Whether to compute fidelity metrics.
        utility: Whether to compute ML utility metrics.
        privacy: Whether to compute privacy (DCR) metrics.
        detection: Whether to compute detection (C2ST) metrics.
        verbose: Whether to print progress messages.

    Returns:
        EvaluationResult with metrics dict and detailed results.
    """
    metrics: dict[str, float | int] = {}
    details: dict[str, Any] = {}

    # Track sample counts
    metrics["num_train"] = len(train_records)
    metrics["num_eval"] = len(eval_records)
    metrics["num_synth"] = len(synthetic_records)

    # Fidelity
    fidelity_result = None
    if fidelity:
        if verbose:
            print("Computing fidelity...")
        fidelity_result = compute_fidelity(
            train_records, synthetic_records, **(fidelity_kwargs or {})
        )
        metrics.update(
            {
                "fidelity": fidelity_result.overall_score,
                "fidelity_shapes": fidelity_result.column_shapes_score,
                "fidelity_trends": fidelity_result.column_pair_trends_score,
            }
        )
        details["fidelity"] = fidelity_result.to_dict()

    # ML Utility
    utility_result = None
    if utility and target_column:
        if verbose:
            print(f"Computing ML utility (target: {target_column})...")
        utility_result = compute_utility(
            train_records=train_records,
            test_records=eval_records,
            synthetic_records=synthetic_records,
            target_field=target_column,
            task_type=task_type,
            **(utility_kwargs or {}),
        )
        primary_metric = "f1_weighted" if utility_result.task_type == "classification" else "r2"
        trtr_primary = utility_result.trtr_metrics[primary_metric]
        tstr_primary = utility_result.tstr_metrics[primary_metric]
        metrics.update(
            {
                "utility": utility_result.utility_score,
                f"utility_trtr_{primary_metric}": trtr_primary,
                f"utility_tstr_{primary_metric}": tstr_primary,
            }
        )
        details["utility"] = utility_result.to_dict()
    elif utility and verbose:
        print("Skipping ML utility (no target column specified)")

    # Privacy (DCR)
    privacy_result = None
    if privacy:
        if verbose:
            print("Computing privacy (DCR)...")
        privacy_result = compute_privacy(
            train_records=train_records,
            test_records=eval_records,
            synthetic_records=synthetic_records,
            verbose=verbose,
            **(privacy_kwargs or {}),
        )
        metrics.update(
            {
                "privacy": privacy_result.privacy_score,
                "privacy_dcr_score": privacy_result.dcr_score,
                "privacy_exact_matches_train": privacy_result.exact_matches_train,
                "privacy_exact_matches_test": privacy_result.exact_matches_test,
                "privacy_exact_matches_train_only": privacy_result.exact_matches_train_only,
            }
        )
        details["privacy"] = privacy_result.to_dict()

    # Detection (C2ST)
    detection_result = None
    if detection:
        if verbose:
            print("Computing detection (C2ST)...")
        detection_result = compute_detection(
            real_records=train_records,
            synthetic_records=synthetic_records,
            **(detection_kwargs or {}),
        )
        metrics.update(
            {
                "detection": detection_result.detection_score,
                "detection_roc_auc": detection_result.roc_auc,
            }
        )
        details["detection"] = detection_result.to_dict()

    return EvaluationResult(
        metrics=metrics,
        details=details,
        config={
            "fidelity": fidelity,
            "utility": utility,
            "privacy": privacy,
            "detection": detection,
            "target_column": target_column,
        },
        fidelity_result=fidelity_result,
        utility_result=utility_result,
        privacy_result=privacy_result,
        detection_result=detection_result,
    )
