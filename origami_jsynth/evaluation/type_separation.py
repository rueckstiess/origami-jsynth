"""Type-separation preprocessing for semi-structured data.

Converts mixed-type columns into dtype indicator + typed value columns.
Enables proper evaluation of semi-structured synthetic data.

Example:
    Original column `foo`: [42, "text", None, NaN, 3.14, True]

    After separation:
        foo.dtype: [num, cat, null, missing, num, bool]
        foo.num:   [42, NaN, NaN, NaN, 3.14, NaN]
        foo.cat:   [NaN, "text", NaN, NaN, NaN, NaN]
        foo.bool:  [NaN, NaN, NaN, NaN, NaN, True]

    Note: "missing" in dtype column indicates field was absent (not present in record).
    This is distinct from "null" which means field was present with null value.
    Using a string value instead of NaN ensures missingness survives CSV round-trips.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

DType = Literal["num", "cat", "date", "bool", "null", "missing", "array", "object"]

_ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?")


@dataclass
class TypeSeparationResult:
    """Result of type separation.

    Attributes:
        df: Type-separated DataFrame with dtype and typed columns
        column_map: Mapping from original column names to their typed columns
                   e.g., {"foo": ["foo.dtype", "foo.num", "foo.cat"]}
    """

    df: pd.DataFrame
    column_map: dict[str, list[str]]


def _is_iso_datetime(s: str) -> bool:
    """Check if string is ISO format datetime (conservative).

    Only matches:
    - YYYY-MM-DD
    - YYYY-MM-DDTHH:MM:SS (with optional timezone/fractional seconds)
    """
    return _ISO_DATETIME_RE.match(s) is not None


def infer_cell_dtype(value: Any) -> DType:
    """Infer the dtype of a single cell value.

    Args:
        value: Any cell value

    Returns:
        DType literal. Returns "missing" for NaN/absent values (field not present).
        Valid types: "num", "cat", "date", "bool", "null", "missing", "array", "object"
    """
    # Check None first (explicit null - field present with null value)
    if value is None:
        return "null"

    # Check array (list) - before pd.isna which fails on lists
    if isinstance(value, list):
        return "array"

    # Check object (dict) - before pd.isna which fails on dicts
    if isinstance(value, dict):
        return "object"

    # Check NaN (field absent after flattening)
    if pd.isna(value):
        return "missing"

    # Boolean before numeric (bool is subclass of int in Python)
    if isinstance(value, (bool, np.bool_)):
        return "bool"

    # Numeric (int or float)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "num"

    # Timestamp objects
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return "date"

    # String - check for ISO datetime
    if isinstance(value, str):
        if _is_iso_datetime(value):
            return "date"
        return "cat"

    # Default to categorical for other types
    return "cat"


def _infer_dtypes_column(series: pd.Series) -> pd.Series:
    """Fast per-cell dtype inference via direct numpy array iteration.

    Equivalent to series.apply(infer_cell_dtype) but avoids pandas .apply()
    overhead (~2-5x faster for large Series). Returns "missing" for absent fields.
    """
    arr = series.values
    n = len(arr)
    result = np.empty(n, dtype=object)

    for i in range(n):
        val = arr[i]
        if val is None:
            result[i] = "null"
        elif isinstance(val, (bool, np.bool_)):
            # Check bool before int (bool is subclass of int in Python)
            result[i] = "bool"
        elif isinstance(val, (int, np.integer)):
            result[i] = "num"
        elif isinstance(val, float):
            result[i] = "missing" if val != val else "num"  # NaN check
        elif isinstance(val, np.floating):
            result[i] = "missing" if np.isnan(val) else "num"
        elif isinstance(val, str):
            result[i] = "date" if _is_iso_datetime(val) else "cat"
        elif isinstance(val, list):
            result[i] = "array"
        elif isinstance(val, dict):
            result[i] = "object"
        elif isinstance(val, (pd.Timestamp, np.datetime64)):
            result[i] = "date"
        else:
            result[i] = "cat"

    return pd.Series(result, index=series.index)


def separate_types(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    force: bool = False,
) -> TypeSeparationResult:
    """Separate mixed-type columns into dtype + typed value columns.

    For columns with multiple value types (e.g., both numeric and categorical),
    creates:
    - foo.dtype: column with values num/cat/date/bool/null/missing/array/object
    - foo.num: numeric values (NaN for non-numeric rows)
    - foo.cat: categorical values (NaN for non-categorical rows)
    - foo.date: datetime values (NaN for non-datetime rows)
    - foo.bool: boolean values (NaN for non-boolean rows)
    - foo.alen: array length (for array dtype only, NaN otherwise)

    Array/object values are not stored directly - they are reconstructed from
    children columns during merge_types().

    For homogeneous columns (single value type, no missing values),
    the column is kept as-is without type separation, UNLESS force=True.
    Columns with missing values are always separated so that missingness
    is explicitly captured in the .dtype column.

    Args:
        df: Input DataFrame (typically flattened from JSON)
        columns: Columns to process. If None, process all columns.
        force: If True, separate ALL columns regardless of homogeneity.
               Useful for evaluation to ensure consistent column structure.

    Returns:
        TypeSeparationResult with separated DataFrame and column mapping
    """
    if columns is None:
        columns = list(df.columns)

    columns_set = set(columns)
    result_cols: dict[str, pd.Series] = {}
    column_map: dict[str, list[str]] = {}

    for col in df.columns:
        if col not in columns_set:
            # Keep column as-is (not processed)
            result_cols[col] = df[col]
            column_map[col] = [col]
            continue

        # Fast dtype inference (avoids .apply() overhead)
        dtypes = _infer_dtypes_column(df[col])
        unique_dtypes = set(dtypes.unique())

        # Check if column is homogeneous (only one value type, no missing values)
        value_types = unique_dtypes - {"missing", "null"}
        has_missing = "missing" in unique_dtypes
        if len(value_types) <= 1 and not has_missing and not force:
            # Truly homogeneous with no missing values — keep as-is
            result_cols[col] = df[col]
            column_map[col] = [col]
            continue

        # Multiple value types OR force=True - create type-separated columns
        dtype_col = f"{col}.dtype"
        result_cols[dtype_col] = dtypes
        typed_cols = [dtype_col]

        # Create typed columns using vectorized boolean masking
        if "num" in unique_dtypes:
            num_col = f"{col}.num"
            num_mask = dtypes == "num"
            result_cols[num_col] = pd.to_numeric(df[col], errors="coerce").where(num_mask)
            typed_cols.append(num_col)

        if "cat" in unique_dtypes:
            cat_col = f"{col}.cat"
            cat_mask = dtypes == "cat"
            result_cols[cat_col] = df[col].where(cat_mask)
            typed_cols.append(cat_col)

        if "date" in unique_dtypes:
            date_col = f"{col}.date"
            date_mask = dtypes == "date"
            result_cols[date_col] = pd.to_datetime(df[col].where(date_mask), errors="coerce")
            typed_cols.append(date_col)

        if "bool" in unique_dtypes:
            bool_col = f"{col}.bool"
            bool_mask = dtypes == "bool"
            result_cols[bool_col] = df[col].where(bool_mask)
            typed_cols.append(bool_col)

        if "array" in unique_dtypes:
            alen_col = f"{col}.alen"
            array_mask = dtypes == "array"
            alen_series = pd.Series(np.nan, index=df.index)
            if array_mask.any():
                alen_series.loc[array_mask] = df.loc[array_mask, col].apply(len)
            result_cols[alen_col] = alen_series
            typed_cols.append(alen_col)
            # Note: actual array values are reconstructed from children during merge

        # Note: "object" dtype doesn't need extra columns - values reconstructed from children

        column_map[col] = typed_cols

    return TypeSeparationResult(
        df=pd.DataFrame(result_cols),
        column_map=column_map,
    )


def _infer_column_map(df: pd.DataFrame) -> dict[str, list[str]]:
    """Infer column map from DataFrame by detecting .dtype columns.

    Looks for columns ending in '.dtype' and finds their corresponding
    typed columns (.num, .cat, .date, .bool, .alen). Columns without a .dtype
    counterpart are treated as unseparated.

    Args:
        df: DataFrame potentially containing type-separated columns

    Returns:
        Mapping from original column names to their typed columns
    """
    column_map: dict[str, list[str]] = {}
    processed_cols: set[str] = set()

    # Find all .dtype columns and their corresponding typed columns
    for col in df.columns:
        if col.endswith(".dtype"):
            orig_col = col[:-6]  # Remove ".dtype" suffix
            typed_cols = [col]
            processed_cols.add(col)

            # Find corresponding typed columns
            for suffix in [".num", ".cat", ".date", ".bool", ".alen"]:
                typed_col = f"{orig_col}{suffix}"
                if typed_col in df.columns:
                    typed_cols.append(typed_col)
                    processed_cols.add(typed_col)

            column_map[orig_col] = typed_cols

    # Add columns that weren't type-separated (no .dtype indicator)
    for col in df.columns:
        if col not in processed_cols:
            column_map[col] = [col]

    return column_map


def merge_types(
    df: pd.DataFrame,
    column_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Inverse of separate_types: merge typed columns back to original structure.

    Reconstructs the original column values based on the dtype indicator column.
    Array/object values are reconstructed from their child columns.

    Args:
        df: Type-separated DataFrame
        column_map: Mapping from original columns to typed columns.
                   If None, infers from column names by detecting .dtype columns.

    Returns:
        DataFrame with original column structure
    """
    if column_map is None:
        column_map = _infer_column_map(df)

    result_cols: dict[str, pd.Series] = {}

    # Sort columns by depth (deeper first) so children are processed before parents
    # This ensures array/object reconstruction can find already-merged children
    sorted_cols = sorted(column_map.keys(), key=lambda k: k.count("."), reverse=True)

    for orig_col in sorted_cols:
        typed_cols = column_map[orig_col]

        # Check if column was not separated
        if len(typed_cols) == 1 and typed_cols[0] == orig_col:
            if orig_col not in df.columns:
                continue
            result_cols[orig_col] = df[orig_col]
            continue

        dtype_col = typed_cols[0]  # e.g., "foo.dtype"

        if dtype_col not in df.columns:
            # Column doesn't exist in this DataFrame, skip
            continue

        dtypes = df[dtype_col]

        # Initialize with object dtype to hold mixed values
        values = pd.Series([np.nan] * len(df), index=df.index, dtype=object)

        # Fill from typed columns based on dtype
        for typed_col in typed_cols[1:]:
            if typed_col not in df.columns:
                continue

            if typed_col.endswith(".num"):
                mask = dtypes == "num"
                values.loc[mask] = df.loc[mask, typed_col]
            elif typed_col.endswith(".cat"):
                mask = dtypes == "cat"
                values.loc[mask] = df.loc[mask, typed_col]
            elif typed_col.endswith(".date"):
                mask = dtypes == "date"
                # Convert datetime to ISO format string for JSON serialization
                date_vals = df.loc[mask, typed_col]
                iso_strings = date_vals.apply(lambda x: x.isoformat() if pd.notna(x) else np.nan)
                values.loc[mask] = iso_strings
            elif typed_col.endswith(".bool"):
                mask = dtypes == "bool"
                bool_vals = df.loc[mask, typed_col]
                # Convert string "True"/"False" back to actual booleans.
                # Synthesizers may return object dtype (TabDiff) or
                # StringDtype (MostlyAI Engine) — handle both.
                if bool_vals.dtype == object or isinstance(
                    bool_vals.dtype, pd.StringDtype
                ):
                    bool_vals = bool_vals.map({"True": True, "False": False}).fillna(bool_vals)
                values.loc[mask] = bool_vals
            # Note: .alen is metadata used below, not for direct value reconstruction

        # Handle array reconstruction from children
        array_mask = dtypes == "array"
        if array_mask.any():
            # Get array length from .alen column
            alen_col = f"{orig_col}.alen"
            if alen_col in df.columns:
                for idx in df.index[array_mask]:
                    alen = df.at[idx, alen_col]
                    if pd.notna(alen):
                        alen = int(alen)
                        # Reconstruct array from child columns
                        arr = []
                        for i in range(alen):
                            child_col = f"{orig_col}.{i}"
                            if child_col in result_cols:
                                val = result_cols[child_col].at[idx]
                                # Extract scalar value if it's a numpy type
                                if hasattr(val, "item"):
                                    val = val.item()
                                arr.append(val)
                            else:
                                arr.append(None)
                        # Use .at for scalar assignment to avoid dict/list expansion
                        values.at[idx] = arr

        # Handle object reconstruction from children
        object_mask = dtypes == "object"
        if object_mask.any():
            # Find all child columns that match pattern {orig_col}.{key}
            child_prefix = f"{orig_col}."
            child_cols = [
                c
                for c in result_cols
                if c.startswith(child_prefix) and "." not in c[len(child_prefix) :]
            ]
            for idx in df.index[object_mask]:
                obj = {}
                for child_col in child_cols:
                    key = child_col[len(child_prefix) :]
                    val = result_cols[child_col].at[idx]
                    # Extract scalar value if it's a numpy type
                    if hasattr(val, "item"):
                        val = val.item()
                    if pd.notna(val) or val is None:  # Include None but not NaN
                        obj[key] = val
                # Use .at for scalar assignment to avoid dict expansion
                values.at[idx] = obj if obj else {}

        # Handle null (explicit None) - set to None
        null_mask = dtypes == "null"
        values.loc[null_mask] = None

        # Absent fields ("missing" in dtype column) stay as NaN (already initialized)

        result_cols[orig_col] = values

    return pd.DataFrame(result_cols)
