"""ML utility evaluation using Train-Synthetic-Test-Real (TSTR) approach.

Utility measures how useful synthetic data is for training ML models.
We compare:
- TRTR: Train on Real, Test on Real (baseline)
- TSTR: Train on Synthetic, Test on Real

A good synthetic dataset will have TSTR performance close to TRTR.

Uses XGBoost with native categorical support for ML evaluation.
"""

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)


@dataclass
class UtilityResult:
    """Results from ML utility evaluation.

    Attributes:
        task_type: "classification" or "regression"
        target_column: Name of the target column
        trtr_metrics: Metrics from Train-Real-Test-Real
        tstr_metrics: Metrics from Train-Synthetic-Test-Real
        utility_score: Ratio of TSTR/TRTR performance (higher is better, max 1.0)
    """

    task_type: Literal["classification", "regression"]
    target_column: str
    trtr_metrics: dict[str, float]
    tstr_metrics: dict[str, float]
    utility_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "trtr_metrics": self.trtr_metrics,
            "tstr_metrics": self.tstr_metrics,
            "utility_score": self.utility_score,
        }


def compute_utility(
    train_records: list[dict],
    test_records: list[dict],
    synthetic_records: list[dict],
    target_field: str,
    task_type: Literal["classification", "regression"],
    random_state: int = 42,
) -> UtilityResult:
    """Compute ML utility for JSON data.

    Uses type separation for feature engineering and XGBoost with
    enable_categorical=True for native categorical support (no one-hot
    encoding). Evaluates using the TSTR/TRTR protocol.

    Args:
        train_records: Training data as list of JSON records
        test_records: Test data as list of JSON records (held out)
        synthetic_records: Synthetic training data as list of JSON records
        target_field: JSON field name to predict (e.g., "income")
        task_type: "classification" or "regression"
        random_state: Random seed for XGBoost

    Returns:
        UtilityResult with TRTR, TSTR metrics and utility score
    """
    from .shared import encode_features, prepare_union_table

    # Flatten and type-separate all records together for consistent columns
    df, masks = prepare_union_table(train=train_records, test=test_records, synth=synthetic_records)
    train_mask = masks["train"]
    test_mask = masks["test"]
    synth_mask = masks["synth"]

    # Identify target columns to exclude from features
    target_prefix = f"{target_field}."
    target_cols = [c for c in df.columns if c.startswith(target_prefix)]
    if not target_cols:
        available = sorted(
            {c.rsplit(".", 1)[0] for c in df.columns if "." in c and c != "__split__"}
        )
        raise ValueError(
            f"Target field '{target_field}' not found in type-separated data. "
            f"Available fields: {available}"
        )

    # Extract targets for each split
    y_train = _extract_target(df, train_mask, target_field, task_type)
    y_test = _extract_target(df, test_mask, target_field, task_type)
    y_synth = _extract_target(df, synth_mask, target_field, task_type)

    # Prepare features (CategoricalDtype for XGBoost native categorical support)
    features_df = encode_features(
        df, exclude_columns=["__split__"] + target_cols, categorical_encoding="native"
    )

    # Split features by mask, reset index for clean alignment
    X_train = features_df.loc[train_mask].reset_index(drop=True)
    X_test = features_df.loc[test_mask].reset_index(drop=True)
    X_synth = features_df.loc[synth_mask].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_synth = y_synth.reset_index(drop=True)

    # Run TSTR/TRTR evaluation
    trtr_metrics, tstr_metrics, utility_score = _run_tstr(
        X_train,
        y_train,
        X_test,
        y_test,
        X_synth,
        y_synth,
        task_type,
        random_state=random_state,
    )

    return UtilityResult(
        task_type=task_type,
        target_column=target_field,
        trtr_metrics=trtr_metrics,
        tstr_metrics=tstr_metrics,
        utility_score=utility_score,
    )


def _extract_target(
    df: pd.DataFrame,
    mask: pd.Series,
    target_field: str,
    task_type: Literal["classification", "regression"],
) -> pd.Series:
    """Extract target values from type-separated columns for a given split.

    Classification: uses .cat column (fallback .bool), NaN -> "__missing__".
    Regression: uses .num column, NaN preserved.
    """
    if task_type == "classification":
        cat_col = f"{target_field}.cat"
        bool_col = f"{target_field}.bool"
        num_col = f"{target_field}.num"

        if cat_col in df.columns and df.loc[mask, cat_col].notna().any():
            return df.loc[mask, cat_col].fillna("__missing__").astype(str)
        elif bool_col in df.columns and df.loc[mask, bool_col].notna().any():
            return df.loc[mask, bool_col].map({True: "True", False: "False"}).fillna("__missing__")
        elif num_col in df.columns and df.loc[mask, num_col].notna().any():
            # Numeric class labels (e.g., 0/1) — convert to string
            return df.loc[mask, num_col].fillna("__missing__").astype(str)
        else:
            return pd.Series(["__missing__"] * mask.sum(), index=df.index[mask])
    else:
        num_col = f"{target_field}.num"
        if num_col in df.columns:
            return df.loc[mask, num_col].astype(float)
        else:
            import numpy as np

            return pd.Series([np.nan] * mask.sum(), index=df.index[mask])


def _encode_target(
    train1: pd.Series, test: pd.Series, train2: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    """Encode categorical target for sklearn.

    Converts all values to strings to handle mixed types, then maps to
    contiguous integer labels across all three series consistently.
    """
    if train1.dtype not in ["object", "category"]:
        return train1, test, train2, {}

    str_series = [s.astype(str) for s in (train1, test, train2)]
    all_values = pd.concat(str_series).unique()
    label_map = {val: i for i, val in enumerate(sorted(all_values))}
    encoded = [s.map(label_map).astype(int) for s in str_series]
    return encoded[0], encoded[1], encoded[2], label_map


def _run_tstr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_synth: pd.DataFrame,
    y_synth: pd.Series,
    task_type: Literal["classification", "regression"],
    **model_params: Any,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Run TRTR + TSTR evaluation and compute utility score.

    For classification, y values must be string Series (label encoding
    is handled internally via _encode_target). For regression, y values
    must be float Series.
    """
    if task_type == "classification":
        y_train, y_test, y_synth, _ = _encode_target(y_train, y_test, y_synth)

        trtr_metrics = _evaluate_classification(X_train, y_train, X_test, y_test, **model_params)
        tstr_metrics = _evaluate_classification(X_synth, y_synth, X_test, y_test, **model_params)

        utility_score = min(tstr_metrics["f1_weighted"] / max(trtr_metrics["f1_weighted"], 1e-10), 1.0)
    else:
        trtr_metrics = _evaluate_regression(X_train, y_train, X_test, y_test, **model_params)
        tstr_metrics = _evaluate_regression(X_synth, y_synth, X_test, y_test, **model_params)

        trtr_r2 = max(trtr_metrics["r2"], 0)
        tstr_r2 = max(tstr_metrics["r2"], 0)
        utility_score = min(tstr_r2 / max(trtr_r2, 1e-10), 1.0) if trtr_r2 > 0 else 0.0

    return trtr_metrics, tstr_metrics, utility_score


def _evaluate_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **model_params: Any,
) -> dict[str, float]:
    """Train and evaluate XGBoost classifier."""
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from xgboost import XGBClassifier  # Lazy import to avoid PyTorch conflict

    # XGBoost requires class labels to be contiguous starting from 0
    # Re-map training labels to ensure this (handles cases where encode_series
    # created non-contiguous labels due to class mismatch between datasets)
    train_classes_list = sorted(y_train.unique())

    # Handle edge case: training data has only one class
    # This can happen when synthetic data has all NaN targets (becomes single "nan" class)
    # Use DummyClassifier which handles single-class gracefully
    if len(train_classes_list) == 1:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # DummyClassifier with single class returns 1D probabilities
        # Expand to 2D for consistent handling below
        y_pred_proba_raw = model.predict_proba(X_test)
        # Create probability matrix with zeros for unseen classes
        test_classes = set(y_test.unique())
        all_classes = sorted(set(train_classes_list) | test_classes)
        n_samples = len(y_test)
        y_pred_proba = np.zeros((n_samples, len(all_classes)))
        # Fill in the probability for the single trained class
        trained_class_idx = all_classes.index(train_classes_list[0])
        y_pred_proba[:, trained_class_idx] = y_pred_proba_raw[:, 0]
    else:
        model_params.setdefault("n_estimators", 100)
        model_params.setdefault("max_depth", 6)
        model_params.setdefault("learning_rate", 0.3)
        model_params.setdefault("random_state", 42)
        model_params.setdefault("n_jobs", -1)
        model_params.setdefault("enable_categorical", True)

        train_label_map = {cls: i for i, cls in enumerate(train_classes_list)}
        y_train_remapped = y_train.map(train_label_map)

        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train_remapped)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Remap predictions back to original labels
        reverse_label_map = {i: cls for cls, i in train_label_map.items()}
        y_pred = pd.Series(y_pred).map(reverse_label_map).values

    # Determine classes from test data (real data defines the task).
    # Synthetic train may have extra classes (e.g., "nan" from NaN values)
    # which are artifacts, not real classes to evaluate against.
    train_classes = set(train_classes_list)
    test_classes = sorted(y_test.unique())
    n_test_classes = len(test_classes)

    # Build all_classes as union (needed for probability matrix alignment)
    all_classes = sorted(train_classes | set(test_classes))
    n_all_classes = len(all_classes)

    # For XGBoost path, remap probability matrix from remapped labels to original labels
    # (DummyClassifier path already has properly aligned probability matrix)
    if len(train_classes_list) > 1:
        # XGBoost used remapped labels [0, 1, 2, ...] corresponding to train_classes_list
        # Rebuild probability matrix aligned to all_classes
        n_samples = len(y_test)
        full_proba = np.zeros((n_samples, n_all_classes))
        for i, orig_cls in enumerate(train_classes_list):
            cls_idx = all_classes.index(orig_cls)
            full_proba[:, cls_idx] = y_pred_proba[:, i]
        y_pred_proba = full_proba

    # Compute ROC AUC — binary vs multi-class determined by test classes
    # (real data defines the task; synthetic artifacts don't change it)
    if n_test_classes < 2:
        # Test set has only one class — can't compute ROC AUC
        roc_auc = 0.5
    elif n_test_classes == 2:
        # Binary classification: use probability of positive class
        if len(train_classes) == 1:
            # Model only saw one class, can't compute meaningful ROC AUC
            roc_auc = 0.5
        else:
            # Positive class is the second in sorted test classes
            pos_class = test_classes[1]
            pos_idx = all_classes.index(pos_class)
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, pos_idx])
    else:
        # Multi-class: use one-vs-rest with weighted average (only test classes)
        # Slice probability columns to match test classes only
        test_class_indices = [all_classes.index(c) for c in test_classes]
        test_proba = y_pred_proba[:, test_class_indices]
        # Renormalize so rows sum to 1.0 (required by sklearn for multiclass)
        # Probability mass on training-only classes is redistributed proportionally
        row_sums = test_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid division by zero
        test_proba = test_proba / row_sums
        roc_auc = roc_auc_score(
            y_test,
            test_proba,
            multi_class="ovr",
            average="weighted",
            labels=test_classes,
        )

    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }


def _evaluate_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **model_params: Any,
) -> dict[str, float]:
    """Train and evaluate XGBoost regressor."""
    import numpy as np
    from xgboost import XGBRegressor  # Lazy import to avoid PyTorch conflict

    model_params.setdefault("n_estimators", 100)
    model_params.setdefault("max_depth", 6)
    model_params.setdefault("learning_rate", 0.3)
    model_params.setdefault("random_state", 42)
    model_params.setdefault("n_jobs", -1)
    model_params.setdefault("verbosity", 0)  # Suppress XGBoost warnings
    model_params.setdefault("enable_categorical", True)

    # Handle NaN in training targets by imputing with random values from distribution
    # This penalizes models trained on synthetic data with type errors
    # without incentivizing prediction of NaN
    y_train_clean = y_train.copy()
    nan_mask = y_train_clean.isna()
    if nan_mask.any():
        non_nan_values = y_train_clean[~nan_mask].values
        if len(non_nan_values) > 0:
            # Sample random values from the non-NaN distribution
            rng = np.random.default_rng(model_params.get("random_state", 42))
            imputed = rng.choice(non_nan_values, size=nan_mask.sum(), replace=True)
            y_train_clean.loc[nan_mask] = imputed

    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train_clean)
    y_pred = model.predict(X_test)

    return {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
    }
