"""Shared records <-> DataFrame conversion for baseline synthesizers."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
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

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> PreprocessingState:
        with open(path, "rb") as f:
            return pickle.load(f)


def records_to_dataframe(
    records: list[dict[str, Any]],
    tabular: bool,
) -> tuple[pd.DataFrame, PreprocessingState]:
    """Convert JSON records to a DataFrame suitable for tabular baselines.

    Flattens nested JSON (no-op for already-flat tabular data) and runs
    separate_types(force=False) to handle mixed-type columns.

    Returns the DataFrame and the state needed to invert the transformation.
    """
    df = flatten_records(records, include_non_leaf=False)
    result = separate_types(df, force=False)
    return result.df, PreprocessingState(is_nested=not tabular, column_map=result.column_map)


def dataframe_to_records(
    df: pd.DataFrame,
    state: PreprocessingState,
) -> list[dict[str, Any]]:
    """Invert the preprocessing: DataFrame -> list[dict].

    Always runs merge_types to reconstruct mixed-type columns.
    For semi-structured data, also unflattens nested structure.
    """
    merged = merge_types(df, column_map=state.column_map)
    if not state.is_nested:
        return merged.to_dict(orient="records")
    return unflatten_dataframe(merged)
