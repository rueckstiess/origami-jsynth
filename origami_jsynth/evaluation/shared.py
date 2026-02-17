"""Shared utilities for JSON evaluation metrics.

Provides core functions used by all evaluation metrics:

1. prepare_union_table() — flattens and type-separates multiple groups of
   JSON records into a single DataFrame with split masks.

2. encode_features() — encodes a type-separated DataFrame into numeric
   features suitable for ML or distance computation.
"""

import numpy as np
import pandas as pd

from .flatten import flatten_records
from .type_separation import separate_types


def prepare_union_table(
    **record_groups: list[dict],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Flatten and type-separate multiple groups of JSON records.

    Tags each group's records, flattens nested JSON, and applies type
    separation. All groups are processed together to ensure consistent
    columns across groups.

    Args:
        **record_groups: Named groups of records.
            E.g.: real=real_records, synth=synth_records
            Or: train=train_records, test=test_records, synth=synth_records

    Returns:
        (df, masks) where df has type-separated columns and masks is a dict
        mapping group names to boolean Series.

    Example:
        >>> df, masks = prepare_union_table(real=real_recs, synth=synth_recs)
        >>> real_data = df[masks["real"]]
    """
    # Build split tags without copying every record dict
    all_records = []
    split_tags = []
    for tag, records in record_groups.items():
        all_records.extend(records)
        split_tags.extend([tag] * len(records))

    df = flatten_records(all_records, include_non_leaf=True)
    separated = separate_types(df, columns=list(df.columns), force=True)
    df = separated.df
    df["__split__"] = split_tags

    masks = {tag: df["__split__"] == tag for tag in record_groups}
    return df, masks


def encode_features(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
    categorical_encoding: str = "onehot",
) -> pd.DataFrame:
    """Encode a type-separated DataFrame into features.

    Processes columns based on their suffix:
    - .num, .alen: Float (NaN preserved)
    - .date: Converted to nanoseconds, then float (NaT → NaN)
    - .cat, .dtype: Encoding depends on categorical_encoding parameter
    - .bool: Numeric (True→1, False→0, NaN preserved)

    Uninformative columns (all-NaN or single-value) are dropped.
    inf values are replaced with NaN in numeric columns.
    NaN is preserved — callers decide how to handle it.

    Args:
        df: Type-separated DataFrame from prepare_union_table().
        exclude_columns: Columns to skip (e.g., ["__split__"]).
        categorical_encoding: How to encode .cat/.dtype columns:
            "onehot" (default) — one-hot via pd.get_dummies (for sklearn/distances)
            "native" — pd.CategoricalDtype (for XGBoost enable_categorical=True)

    Returns:
        DataFrame with encoded columns. NaN preserved.
    """
    exclude = set(exclude_columns or [])
    result_data = {}
    categorical_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        series = df[col]

        # Skip uninformative columns
        non_nan = series.dropna()
        if len(non_nan) == 0:
            continue
        if non_nan.nunique() <= 1:
            continue

        # Classify by suffix
        if col.endswith(".num") or col.endswith(".alen"):
            result_data[col] = series.astype(float)

        elif col.endswith(".date"):
            dt_vals = pd.to_datetime(series, errors="coerce")
            result_data[col] = pd.to_numeric(dt_vals, errors="coerce").astype("float64")

        elif col.endswith(".dtype") or col.endswith(".cat"):
            str_vals = series.fillna("NaN").astype(str)
            if categorical_encoding == "native":
                cat_type = pd.CategoricalDtype(categories=sorted(str_vals.unique()))
                result_data[col] = str_vals.astype(cat_type)
            else:
                categorical_cols.append(col)
                result_data[col] = str_vals

        elif col.endswith(".bool"):
            result_data[col] = series.map({True: 1.0, False: 0.0})

    if not result_data:
        raise ValueError("No valid features found for encoding")

    result_df = pd.DataFrame(result_data, index=df.index)

    # Replace inf in numeric columns
    numeric_cols = result_df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        result_df[numeric_cols] = result_df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # One-hot encode categoricals (only for "onehot" mode)
    if categorical_cols:
        result_df = pd.get_dummies(result_df, columns=categorical_cols, drop_first=False)

    return result_df
