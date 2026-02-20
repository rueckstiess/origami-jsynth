"""JSON flatten/unflatten utilities for semi-structured data."""

from typing import Any

import pandas as pd


def is_nested(records: list[dict]) -> bool:
    """Check if any record contains nested structures (dicts or lists).

    Args:
        records: List of dictionaries to check

    Returns:
        True if any record contains nested dicts or lists as values
    """
    if not records:
        return False

    for record in records:
        for value in record.values():
            if isinstance(value, (dict, list)):
                return True
    return False


def has_nested_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame has columns with nested structures.

    Args:
        df: DataFrame to check

    Returns:
        True if any column contains dicts or lists
    """
    if df.empty:
        return False

    for col in df.columns:
        for val in df[col]:
            if isinstance(val, (dict, list)):
                return True
    return False


def has_flattened_columns(df: pd.DataFrame, sep: str = ".") -> bool:
    """Check if DataFrame has flattened column names (contains separator).

    Args:
        df: DataFrame to check
        sep: Separator used in flattened column names

    Returns:
        True if any column name contains the separator
    """
    return any(sep in str(col) for col in df.columns)


def flatten_json(
    obj: dict,
    sep: str = ".",
    include_non_leaf: bool = False,
    _prefix: str = "",
) -> dict:
    """Flatten nested JSON to flat dict with dot-separated keys.

    Args:
        obj: Nested dictionary to flatten
        sep: Separator for nested keys (default ".")
        include_non_leaf: If True, include non-leaf values (arrays/objects)
                         alongside their flattened children
        _prefix: Internal prefix for recursion

    Returns:
        Flat dictionary with dot-separated keys

    Example:
        >>> flatten_json({"user": {"name": "Alice", "age": 30}, "scores": [1, 2]})
        {"user.name": "Alice", "user.age": 30, "scores.0": 1, "scores.1": 2}

        >>> flatten_json({"scores": [1, 2]}, include_non_leaf=True)
        {"scores": [1, 2], "scores.0": 1, "scores.1": 2}
    """
    result = {}

    for key, value in obj.items():
        full_key = f"{_prefix}{sep}{key}" if _prefix else key

        if isinstance(value, dict):
            # Include the dict itself if include_non_leaf=True
            if include_non_leaf:
                result[full_key] = value
            # Recurse into nested dict
            result.update(
                flatten_json(value, sep=sep, include_non_leaf=include_non_leaf, _prefix=full_key)
            )
        elif isinstance(value, list):
            # Include the list itself if include_non_leaf=True
            if include_non_leaf:
                result[full_key] = value
            # Flatten list with numeric indices
            for i, item in enumerate(value):
                list_key = f"{full_key}{sep}{i}"
                if isinstance(item, dict):
                    if include_non_leaf:
                        result[list_key] = item
                    result.update(
                        flatten_json(
                            item, sep=sep, include_non_leaf=include_non_leaf, _prefix=list_key
                        )
                    )
                elif isinstance(item, list):
                    # Nested list - recurse
                    if include_non_leaf:
                        result[list_key] = item
                    nested = {str(j): v for j, v in enumerate(item)}
                    result.update(
                        flatten_json(
                            nested, sep=sep, include_non_leaf=include_non_leaf, _prefix=list_key
                        )
                    )
                else:
                    result[list_key] = item
        else:
            result[full_key] = value

    return result


def _has_non_leaf_values(flat: dict) -> bool:
    """Check if any values in flat dict are lists or dicts (non-leaf values).

    This is used to detect if flattening was done with include_non_leaf=True.
    """
    return any(isinstance(v, (list, dict)) for v in flat.values())


def unflatten_json(flat: dict, sep: str = ".") -> dict:
    """Reconstruct nested JSON from flat dict.

    Automatically detects if flattening was done with include_non_leaf=True
    by checking if any values are lists or dicts. When non-leaf values exist,
    they take priority over reconstructed children (which are redundant).

    Args:
        flat: Flat dictionary with dot-separated keys
        sep: Separator used in keys (default ".")

    Returns:
        Nested dictionary

    Example:
        >>> unflatten_json({"user.name": "Alice", "user.age": 30, "scores.0": 1, "scores.1": 2})
        {"user": {"name": "Alice", "age": 30}, "scores": [1, 2]}

    Note:
        NaN values (from pandas) are skipped to handle variable-length lists
        that get padded with NaN when converted to DataFrame.
    """
    if _has_non_leaf_values(flat):
        return _unflatten_with_non_leaf(flat, sep)
    else:
        return _unflatten_leaf_only(flat, sep)


def _unflatten_leaf_only(flat: dict, sep: str = ".") -> dict:
    """Original unflatten logic for leaf-only flattened data."""
    result: dict = {}

    for key, value in flat.items():
        # Skip NaN values (common from pandas DataFrames with variable-length lists)
        if _is_nan(value):
            continue

        parts = key.split(sep)
        current = result

        # Navigate/create path, always using dicts
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value

    # Post-process: convert dict with numeric keys to lists where appropriate
    return _convert_numeric_dicts_to_lists(result)


def _unflatten_with_non_leaf(flat: dict, sep: str = ".") -> dict:
    """Unflatten when non-leaf values (arrays/objects) are present.

    Non-leaf values are authoritative and their children are skipped (redundant).
    """
    result: dict = {}

    # Sort keys by depth (shortest first) to process parents before children
    sorted_keys = sorted(flat.keys(), key=lambda k: k.count(sep))

    # Track which prefixes have non-leaf values (their children should be skipped)
    non_leaf_prefixes: set[str] = set()

    for key in sorted_keys:
        value = flat[key]

        if _is_nan(value):
            continue

        # Check if this key is a child of a non-leaf prefix
        is_child_of_non_leaf = any(key.startswith(prefix + sep) for prefix in non_leaf_prefixes)
        if is_child_of_non_leaf:
            continue  # Skip - parent non-leaf value is authoritative

        parts = key.split(sep)
        current = result

        # Navigate to parent, creating dicts as needed
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value
        final_key = parts[-1]

        if isinstance(value, (list, dict)):
            # This is a non-leaf value - mark prefix for skipping children
            non_leaf_prefixes.add(key)
            current[final_key] = value
        else:
            current[final_key] = value

    # Post-process: convert dict with numeric keys to lists where appropriate
    return _convert_numeric_dicts_to_lists(result)


def _is_nan(value: Any) -> bool:
    """Check if value is NaN (works for float NaN and pandas NA)."""
    if value is None:
        return False
    try:
        # Check for float NaN
        import math

        if isinstance(value, float) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    try:
        # Check for pandas NA
        import pandas as pd

        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _convert_numeric_dicts_to_lists(obj: Any) -> Any:
    """Convert dicts with all-numeric keys to lists, recursively.

    Values are packed contiguously in sorted key order — gaps from skipped
    NaN entries do not produce None holes in the resulting list.
    """
    if isinstance(obj, dict):
        # Check if all keys are numeric strings
        if obj and all(k.isdigit() for k in obj):
            # Convert to list, packing values contiguously in key order
            result = [
                _convert_numeric_dicts_to_lists(v)
                for _, v in sorted(obj.items(), key=lambda kv: int(kv[0]))
            ]
            return result
        else:
            return {k: _convert_numeric_dicts_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numeric_dicts_to_lists(item) for item in obj]
    else:
        return obj


def flatten_records(
    records: list[dict],
    sep: str = ".",
    include_non_leaf: bool = False,
) -> pd.DataFrame:
    """Flatten list of JSON objects to DataFrame.

    Args:
        records: List of nested dictionaries
        sep: Separator for nested keys
        include_non_leaf: If True, include non-leaf values (arrays/objects)
                         alongside their flattened children

    Returns:
        DataFrame with flattened columns
    """
    flattened = [
        flatten_json(record, sep=sep, include_non_leaf=include_non_leaf) for record in records
    ]
    df = pd.DataFrame(flattened)
    return df[sorted(df.columns)]


def unflatten_dataframe(df: pd.DataFrame, sep: str = ".") -> list[dict]:
    """Convert DataFrame back to list of nested JSON objects.

    Args:
        df: DataFrame with dot-separated column names
        sep: Separator used in column names

    Returns:
        List of nested dictionaries
    """
    records = df.to_dict(orient="records")
    return [unflatten_json(record, sep=sep) for record in records]
