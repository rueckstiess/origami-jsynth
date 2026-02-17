"""Detection metric using Classifier Two-Sample Test (C2ST).

Measures how easily a classifier can distinguish synthetic data from real data.
Uses cross-validation with either XGBoost (default) or logistic regression.

A good synthetic dataset will be difficult to distinguish from real data,
resulting in classifier accuracy near 50% (random guessing).
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class DetectionResult:
    """Results from detection evaluation.

    Attributes:
        detection_score: Primary score (0-1, higher = better).
            Score of 1.0 = classifier at random chance (ideal synthetic data).
            Score of 0.0 = classifier perfectly distinguishes (poor synthetic data).
        roc_auc: Estimated ROC AUC of the classifier.
            0.5 = random guessing (good), 1.0 = perfect separation (bad).
    """

    detection_score: float  # 0-1, 1.0 = best (can't distinguish)
    roc_auc: float  # ROC AUC, 0.5 = random guessing
    method: str = "xgboost"
    feature_importances: dict[str, float] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "detection_score": self.detection_score,
            "roc_auc": self.roc_auc,
            "method": self.method,
            "feature_importances": self.feature_importances,
        }

    def format_breakdown(self, top_n: int = 10) -> str:
        """Format a human-readable breakdown of detection results.

        Args:
            top_n: Number of top features to show

        Returns:
            Formatted string showing score and feature importances
        """
        lines = [
            f"Detection Score: {self.detection_score:.4f}",
            f"  ROC AUC: {self.roc_auc:.4f}",
        ]

        if self.feature_importances:
            total_gain = sum(self.feature_importances.values())
            items = list(self.feature_importances.items())[:top_n]
            lines.append("")
            lines.append(f"Top {len(items)} discriminative features (gain):")
            for name, gain in items:
                pct = gain / total_gain * 100 if total_gain > 0 else 0
                lines.append(f"  {name}: {gain:.2f} ({pct:.1f}%)")
            if len(self.feature_importances) > top_n:
                rest = sum(v for v in list(self.feature_importances.values())[top_n:])
                pct = rest / total_gain * 100 if total_gain > 0 else 0
                lines.append(f"  ... {len(self.feature_importances) - top_n} more ({pct:.1f}%)")

        return "\n".join(lines)


def compute_detection(
    real_records: list[dict],
    synthetic_records: list[dict],
    *,
    method: Literal["xgboost", "logistic_regression"] = "xgboost",
    n_splits: int = 3,
    random_state: int = 42,
) -> DetectionResult:
    """Compute detection metric for JSON data.

    Uses type separation to handle heterogeneous JSON, then trains a classifier
    to distinguish real from synthetic records via cross-validation.

    Args:
        real_records: List of dictionaries (real data)
        synthetic_records: List of dictionaries (synthetic data)
        method: Classifier to use. "xgboost" (default) uses XGBoost with native
            categorical support. "logistic_regression" uses sklearn LogisticRegression
            with imputation and scaling.
        n_splits: Number of cross-validation folds (default 3)
        random_state: Random seed for reproducibility

    Returns:
        DetectionResult with:
        - detection_score: 0-1 (1.0 = can't distinguish = best)
        - roc_auc: Estimated ROC AUC (0.5 = random guessing)
    """
    from .shared import encode_features, prepare_union_table

    # Prepare type-separated data
    df, _ = prepare_union_table(real=real_records, synth=synthetic_records)
    y = (df["__split__"] == "real").astype(int).to_numpy()

    # Encode features based on method
    if method == "xgboost":
        X = encode_features(df, exclude_columns=["__split__"], categorical_encoding="native")
    else:
        X = encode_features(df, exclude_columns=["__split__"]).to_numpy(dtype=np.float64)

    # Adjust n_splits if dataset is too small
    min_class_size = min(y.sum(), len(y) - y.sum())
    actual_splits = min(n_splits, min_class_size)
    if actual_splits < 2:
        return DetectionResult(detection_score=1.0, roc_auc=0.5, method=method)

    # Cross-validated detection
    kf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    scores = []
    raw_aucs = []
    fold_importances = []

    for train_idx, test_idx in kf.split(X, y):
        if method == "xgboost":
            y_pred_proba, importances = _fit_predict_xgboost(
                X.iloc[train_idx],
                y[train_idx],
                X.iloc[test_idx],
                random_state=random_state,
            )
            fold_importances.append(importances)
        elif method == "logistic_regression":
            y_pred_proba = _fit_predict_logistic(
                X[train_idx],
                y[train_idx],
                X[test_idx],
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        roc_auc = roc_auc_score(y[test_idx], y_pred_proba)
        raw_aucs.append(roc_auc)
        # SDMetrics formula: clamp to [0.5, 1.0], transform to [0, 1]
        scores.append(max(0.5, roc_auc) * 2 - 1)

    detection_score = 1 - np.mean(scores)
    mean_roc_auc = float(np.mean(raw_aucs))

    # Average feature importances across folds
    feature_importances: dict[str, float] = {}
    if fold_importances:
        all_features = fold_importances[0].keys()
        feature_importances = {
            f: float(np.mean([fold[f] for fold in fold_importances])) for f in all_features
        }
        # Sort by importance descending
        feature_importances = dict(
            sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        )

    return DetectionResult(
        detection_score=detection_score,
        roc_auc=mean_roc_auc,
        method=method,
        feature_importances=feature_importances,
    )


def _fit_predict_xgboost(
    X_train, y_train, X_test, *, random_state: int = 42
) -> tuple[np.ndarray, dict[str, float]]:
    from xgboost import XGBClassifier  # Lazy import to avoid PyTorch conflict

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        enable_categorical=True,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    # gain importance: average information gain when the feature is used for splitting
    gain_scores = model.get_booster().get_score(importance_type="gain")
    # Include features with zero importance (never used for splitting)
    importances = {name: gain_scores.get(name, 0.0) for name in X_train.columns}
    return proba, importances


def _fit_predict_logistic(X_train, y_train, X_test) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=100)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline.predict_proba(X_test)[:, 1]
