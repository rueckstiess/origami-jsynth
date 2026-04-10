"""Shared records <-> DataFrame conversion for baseline synthesizers."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..evaluation.flatten import flatten_records, unflatten_dataframe
from ..evaluation.type_separation import merge_types, separate_types


def build_metadata(df: pd.DataFrame) -> Any:
    """Build SDV Metadata using only basic sdtypes inferred from pandas dtypes.

    Unlike ``Metadata.detect_from_dataframe()``, this does **not** infer
    semantic types (city, email, …) which SDV marks as PII and replaces
    with Faker-generated values during sampling.  We want the synthesizer
    to learn the actual training distribution for every column.

    Mapping:
        int / float          → numerical
        bool                 → boolean
        datetime64           → datetime
        everything else      → categorical
    """
    from sdv.metadata import Metadata

    metadata = Metadata()
    metadata.add_table("table")

    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            sdtype = "boolean"
        elif pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            sdtype = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sdtype = "datetime"
        else:
            sdtype = "categorical"
        metadata.add_column(col, sdtype=sdtype, table_name="table")

    return metadata


@dataclass
class PreprocessingState:
    """Captures the transformations applied so they can be inverted."""

    is_nested: bool
    column_map: dict[str, list[str]]
    # Original pandas dtypes for passthrough (unseparated) columns.
    # Synthesizers may change dtypes (e.g., bool → StringDtype "True"/"False").
    passthrough_dtypes: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> PreprocessingState:
        with open(path, "rb") as f:
            state = pickle.load(f)
        # Backward compat: older pickles lack passthrough_dtypes
        if not hasattr(state, "passthrough_dtypes"):
            state.passthrough_dtypes = {}
        return state


def records_to_dataframe(
    records: list[dict[str, Any]],
    tabular: bool,
    exclude_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, PreprocessingState]:
    """Convert JSON records to a DataFrame suitable for tabular baselines.

    Flattens nested JSON (no-op for already-flat tabular data) and runs
    separate_types(force=False) to split sparse and mixed-type columns into 
    multiple typed columns.

    Args:
        exclude_columns: Columns to exclude from type separation (e.g. target
            columns that a model needs to reference by their original name).

    Returns the DataFrame and the state needed to invert the transformation.
    """
    df = flatten_records(records, include_non_leaf=False)
    cols = [c for c in df.columns if c not in (exclude_columns or [])]
    result = separate_types(df, columns=cols, force=False)

    # Record original dtypes of passthrough columns so we can restore them
    # after sampling (synthesizers may change dtypes, e.g. bool → string).
    passthrough_dtypes: dict[str, str] = {}
    for col, typed_cols in result.column_map.items():
        if len(typed_cols) == 1 and typed_cols[0] == col:
            passthrough_dtypes[col] = str(result.df[col].dtype)

    return result.df, PreprocessingState(
        is_nested=not tabular,
        column_map=result.column_map,
        passthrough_dtypes=passthrough_dtypes,
    )


def dataframe_to_records(
    df: pd.DataFrame,
    state: PreprocessingState,
) -> list[dict[str, Any]]:
    """Invert the preprocessing: DataFrame -> list[dict].

    Always runs merge_types to reconstruct mixed-type columns.
    For semi-structured data, also unflattens nested structure.
    """
    merged = merge_types(df, column_map=state.column_map)

    # Restore dtypes that synthesizers may have changed (e.g., bool → string).
    # Use stored passthrough_dtypes when available; for old checkpoints that
    # lack it, detect string-encoded booleans in passthrough columns heuristically.
    for col, typed_cols in state.column_map.items():
        if col not in merged.columns:
            continue
        if len(typed_cols) != 1 or typed_cols[0] != col:
            continue  # type-separated column, handled by merge_types
        if pd.api.types.is_bool_dtype(merged[col]):
            continue  # already boolean

        orig_dtype = state.passthrough_dtypes.get(col)
        if orig_dtype == "bool" or (
            orig_dtype is None
            and (merged[col].dtype == object or isinstance(merged[col].dtype, pd.StringDtype))
            and merged[col].dropna().isin(["True", "False", True, False]).all()
        ):
            merged[col] = merged[col].map({"True": True, "False": False, True: True, False: False})

    if not state.is_nested:
        return merged.to_dict(orient="records")
    return unflatten_dataframe(merged)
