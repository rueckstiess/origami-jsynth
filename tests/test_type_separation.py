"""Tests for origami_jsynth.evaluation.type_separation."""

import numpy as np
import pandas as pd
import pytest

from origami_jsynth.evaluation.type_separation import (
    infer_cell_dtype,
    merge_types,
    separate_types,
)


# ---------------------------------------------------------------------------
# TestInferCellDtype
# ---------------------------------------------------------------------------


class TestInferCellDtype:
    def test_none_is_null(self):
        assert infer_cell_dtype(None) == "null"

    def test_float_nan_is_missing(self):
        assert infer_cell_dtype(float("nan")) == "missing"

    def test_pandas_na_is_missing(self):
        assert infer_cell_dtype(pd.NA) == "missing"

    def test_true_is_bool(self):
        assert infer_cell_dtype(True) == "bool"

    def test_false_is_bool(self):
        assert infer_cell_dtype(False) == "bool"

    def test_numpy_bool_is_bool(self):
        assert infer_cell_dtype(np.bool_(True)) == "bool"

    def test_int_is_num(self):
        assert infer_cell_dtype(42) == "num"

    def test_float_is_num(self):
        assert infer_cell_dtype(3.14) == "num"

    def test_numpy_int_is_num(self):
        assert infer_cell_dtype(np.int64(7)) == "num"

    def test_numpy_float_is_num(self):
        assert infer_cell_dtype(np.float64(2.5)) == "num"

    def test_string_is_cat(self):
        assert infer_cell_dtype("hello") == "cat"

    def test_iso_date_string_is_date(self):
        assert infer_cell_dtype("2024-01-15") == "date"

    def test_iso_datetime_string_is_date(self):
        assert infer_cell_dtype("2024-01-15T10:30:00") == "date"

    def test_non_iso_string_is_cat(self):
        assert infer_cell_dtype("not-a-date") == "cat"

    def test_list_is_array(self):
        assert infer_cell_dtype([1, 2, 3]) == "array"

    def test_dict_is_object(self):
        assert infer_cell_dtype({"key": "val"}) == "object"

    def test_pandas_timestamp_is_date(self):
        assert infer_cell_dtype(pd.Timestamp("2024-01-15")) == "date"


# ---------------------------------------------------------------------------
# TestSeparateTypes
# ---------------------------------------------------------------------------


class TestSeparateTypes:
    def test_homogeneous_numeric_column_kept_as_is(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = separate_types(df, force=False)
        # No type-separated columns — kept as-is.
        assert "x" in result.df.columns
        assert "x.dtype" not in result.df.columns

    def test_homogeneous_string_column_kept_as_is(self):
        df = pd.DataFrame({"label": ["a", "b", "c"]})
        result = separate_types(df, force=False)
        assert "label" in result.df.columns
        assert "label.dtype" not in result.df.columns

    def test_column_with_missing_values_always_separated(self):
        # Even a single NaN triggers separation so missingness is captured.
        df = pd.DataFrame({"x": [1, float("nan"), 3]})
        result = separate_types(df, force=False)
        assert "x.dtype" in result.df.columns

    def test_mixed_num_and_cat_creates_separate_columns(self):
        df = pd.DataFrame({"v": [42, "text", 7]})
        result = separate_types(df)
        assert "v.dtype" in result.df.columns
        assert "v.num" in result.df.columns
        assert "v.cat" in result.df.columns

    def test_mixed_with_bool_creates_bool_column(self):
        df = pd.DataFrame({"v": [True, 42, "cat"]})
        result = separate_types(df)
        assert "v.bool" in result.df.columns

    def test_array_values_create_alen_column(self):
        # force=True is needed here because an all-array column is homogeneous and would
        # otherwise be kept as-is. In practice, mixed-type columns (where some rows have
        # arrays and others don't) are what triggers separation automatically.
        df = pd.DataFrame({"v": [[1, 2], [3], [1, 2, 3]]})
        result = separate_types(df, force=True)
        assert "v.alen" in result.df.columns

    def test_alen_values_match_array_lengths(self):
        df = pd.DataFrame({"v": [[1, 2], [3], [1, 2, 3]]})
        result = separate_types(df, force=True)
        alens = result.df["v.alen"].tolist()
        assert alens == [2.0, 1.0, 3.0]

    def test_force_true_separates_homogeneous_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = separate_types(df, force=True)
        assert "x.dtype" in result.df.columns

    def test_force_false_skips_homogeneous_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = separate_types(df, force=False)
        assert "x.dtype" not in result.df.columns

    def test_column_map_output_structure(self):
        df = pd.DataFrame({"x": [1, "text"]})
        result = separate_types(df)
        assert "x" in result.column_map
        assert "x.dtype" in result.column_map["x"]

    def test_array_values_not_stored_directly(self):
        # Array values themselves should not appear in the df (only .alen).
        # Use force=True to ensure the column is separated.
        df = pd.DataFrame({"v": [[1, 2], [3]]})
        result = separate_types(df, force=True)
        # There should be no column named exactly "v"
        assert "v" not in result.df.columns


# ---------------------------------------------------------------------------
# TestMergeTypes
# ---------------------------------------------------------------------------


class TestMergeTypes:
    def test_passthrough_columns_preserved(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = separate_types(df, force=False)
        merged = merge_types(result.df, result.column_map)
        assert list(merged["a"]) == [1, 2]

    def test_numeric_values_restored(self):
        df = pd.DataFrame({"v": [1, "text"]})
        result = separate_types(df)
        merged = merge_types(result.df, result.column_map)
        assert merged["v"][0] == 1

    def test_categorical_values_restored(self):
        df = pd.DataFrame({"v": [1, "text"]})
        result = separate_types(df)
        merged = merge_types(result.df, result.column_map)
        assert merged["v"][1] == "text"

    def test_null_values_are_none(self):
        # Build a DataFrame where "null" is injected explicitly into the .dtype column,
        # bypassing the pandas None→NaN coercion that occurs during DataFrame construction.
        # This tests that merge_types correctly converts "null" dtype rows back to None.
        df = pd.DataFrame({"v": [1, 2]})
        result = separate_types(df, force=True)
        # Manually set the first row's dtype to "null" (as a synthesizer might produce).
        result.df = result.df.copy()
        result.df.loc[0, "v.dtype"] = "null"
        merged = merge_types(result.df, result.column_map)
        assert merged["v"][0] is None

    def test_missing_values_are_nan(self):
        # "missing" dtype means the field was absent (NaN in the .dtype column).
        # Note: pandas stores Python None as NaN, so None in a Series becomes "missing".
        df = pd.DataFrame({"v": [float("nan"), 1]})
        result = separate_types(df, force=True)
        merged = merge_types(result.df, result.column_map)
        assert pd.isna(merged["v"][0])

    def test_bool_values_restored_as_bool(self):
        df = pd.DataFrame({"v": [True, False]})
        result = separate_types(df, force=True)
        merged = merge_types(result.df, result.column_map)
        assert merged["v"][0] is True or merged["v"][0] == True  # noqa: E712
        assert merged["v"][1] is False or merged["v"][1] == False  # noqa: E712

    def test_string_true_false_restored_as_bool(self):
        # Synthesizers may produce "True"/"False" strings for bool columns.
        # merge_types must convert these back to actual booleans.
        df = pd.DataFrame({"v": [True, False, True]})
        result = separate_types(df, force=True)
        # Simulate what a synthesizer might return (string booleans).
        result.df["v.bool"] = result.df["v.bool"].astype(object)
        result.df.loc[result.df["v.bool"].notna(), "v.bool"] = result.df.loc[
            result.df["v.bool"].notna(), "v.bool"
        ].apply(lambda x: str(x) if x is not None else x)
        merged = merge_types(result.df, result.column_map)
        for val in merged["v"].dropna():
            assert isinstance(val, bool), f"Expected bool, got {type(val)}: {val!r}"

    def test_roundtrip_separate_merge(self):
        df = pd.DataFrame({"a": [1, "text", None], "b": [True, False, True]})
        result = separate_types(df, force=True)
        merged = merge_types(result.df, result.column_map)
        # Original numeric value.
        assert merged["a"][0] == 1
        # Original string value.
        assert merged["a"][1] == "text"
        # Original None.
        assert merged["a"][2] is None


# ---------------------------------------------------------------------------
# TestSeparateMergeRoundtrip — parametrized
# ---------------------------------------------------------------------------


class TestSeparateMergeRoundtrip:
    @pytest.mark.parametrize(
        "records",
        [
            pytest.param([{"x": 1}, {"x": 2}, {"x": 3}], id="flat_numeric"),
            pytest.param([{"s": "a"}, {"s": "b"}, {"s": "c"}], id="flat_categorical"),
            pytest.param([{"v": None}, {"v": 1}, {"v": "x"}], id="with_none"),
            pytest.param([{"v": 42}, {"v": "text"}, {"v": True}], id="mixed_types"),
            pytest.param([{"b": True}, {"b": False}, {"b": True}], id="booleans"),
        ],
    )
    def test_roundtrip(self, records):
        df = pd.DataFrame(records)
        result = separate_types(df, force=True)
        merged = merge_types(result.df, result.column_map)
        # Verify every non-NaN value matches original.
        for col in df.columns:
            if col not in merged.columns:
                continue
            for i, (orig, restored) in enumerate(zip(df[col], merged[col])):
                if orig is None:
                    assert restored is None, f"Row {i} col {col}: expected None, got {restored!r}"
                elif isinstance(orig, float) and (orig != orig):  # NaN check
                    assert pd.isna(restored), f"Row {i} col {col}: expected NaN, got {restored!r}"
                else:
                    assert orig == restored, (
                        f"Row {i} col {col}: expected {orig!r}, got {restored!r}"
                    )
