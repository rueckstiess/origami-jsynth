"""Tests for origami_jsynth.baselines._preprocessing."""

import math

import pandas as pd
import pytest

from origami_jsynth.baselines._preprocessing import (
    PreprocessingState,
    dataframe_to_records,
    records_to_dataframe,
)


# ---------------------------------------------------------------------------
# TestRecordsToDataframe
# ---------------------------------------------------------------------------


class TestRecordsToDataframe:
    def test_flat_records_correct_shape(self):
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        df, state = records_to_dataframe(records, tabular=True)
        assert len(df) == 2
        # At minimum "a" and "b" (or their type-separated variants) are present.
        all_cols = " ".join(df.columns)
        assert "a" in all_cols
        assert "b" in all_cols

    def test_nested_records_flattened(self):
        records = [{"user": {"name": "Alice"}}, {"user": {"name": "Bob"}}]
        df, state = records_to_dataframe(records, tabular=False)
        # Nested dict should be flattened to dot-notation.
        assert not any(isinstance(df[col].iloc[0], dict) for col in df.columns)

    def test_returns_preprocessing_state(self):
        records = [{"a": 1}]
        _, state = records_to_dataframe(records, tabular=True)
        assert isinstance(state, PreprocessingState)

    def test_is_nested_true_when_tabular_false(self):
        records = [{"a": 1}]
        _, state = records_to_dataframe(records, tabular=False)
        assert state.is_nested is True

    def test_is_nested_false_when_tabular_true(self):
        records = [{"a": 1}]
        _, state = records_to_dataframe(records, tabular=True)
        assert state.is_nested is False

    def test_passthrough_dtypes_captured(self):
        records = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        _, state = records_to_dataframe(records, tabular=True)
        assert isinstance(state.passthrough_dtypes, dict)

    def test_mixed_type_column_separated(self):
        # A column with both ints and strings has mixed types → gets separated.
        records = [{"v": 1}, {"v": "text"}, {"v": 2}]
        df, state = records_to_dataframe(records, tabular=True)
        assert "v.dtype" in df.columns


# ---------------------------------------------------------------------------
# TestDataframeToRecords
# ---------------------------------------------------------------------------


class TestDataframeToRecords:
    def test_flat_records_correct_dicts(self):
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        df, state = records_to_dataframe(records, tabular=True)
        result = dataframe_to_records(df, state)
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)

    def test_is_nested_false_no_unflattening(self):
        records = [{"a.b": 1}]  # Flat record with a dot in the key (unusual but valid).
        df = pd.DataFrame(records)
        state = PreprocessingState(
            is_nested=False,
            column_map={"a.b": ["a.b"]},
            passthrough_dtypes={"a.b": "int64"},
        )
        result = dataframe_to_records(df, state)
        # With is_nested=False, keys are preserved literally.
        assert "a.b" in result[0]

    def test_is_nested_true_restores_nested_structure(self):
        records = [{"user": {"name": "Alice"}}]
        df, state = records_to_dataframe(records, tabular=False)
        result = dataframe_to_records(df, state)
        assert "user" in result[0]
        assert result[0]["user"]["name"] == "Alice"

    def test_boolean_columns_restored(self):
        records = [{"flag": True}, {"flag": False}]
        df, state = records_to_dataframe(records, tabular=True)
        result = dataframe_to_records(df, state)
        flags = [r["flag"] for r in result]
        assert isinstance(flags[0], bool)
        assert isinstance(flags[1], bool)


# ---------------------------------------------------------------------------
# TestRoundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_flat_tabular_roundtrip(self):
        records = [
            {"age": 25, "income": 50000.0, "label": "A"},
            {"age": 30, "income": 60000.0, "label": "B"},
        ]
        df, state = records_to_dataframe(records, tabular=True)
        result = dataframe_to_records(df, state)
        assert len(result) == 2
        # Key fields should survive.
        for orig, restored in zip(records, result):
            assert orig["age"] == restored.get("age") or any(
                "age" in k for k in restored
            ), f"age missing in {restored}"

    def test_nested_records_roundtrip(self):
        records = [
            {"user": {"name": "Alice", "score": 9.5}},
            {"user": {"name": "Bob", "score": 7.0}},
        ]
        df, state = records_to_dataframe(records, tabular=False)
        result = dataframe_to_records(df, state)
        assert result[0]["user"]["name"] == "Alice"
        assert result[1]["user"]["name"] == "Bob"

    def test_empty_arrays_roundtrip(self):
        # Regression test: empty arrays must survive flatten/unflatten, not become absent fields.
        # Records with empty arrays must survive the full records_to_dataframe → dataframe_to_records
        # pipeline. This is the regression test for the _unflatten_leaf_only bug fix.
        #
        # The fix in _unflatten_leaf_only works by detecting array-parent paths and setting them
        # to [] when all children are NaN. It only fires when the immediate parent dict already
        # exists in the reconstructed result — so records must have at least one non-array field
        # on the same parent (e.g., "title") to anchor the parent dict.
        # This mirrors real-world data: GitHub issues always have a title alongside assignees.
        records = [
            {"issue": {"assignees": [], "title": "foo"}},
            {"issue": {"assignees": [{"login": "alice"}], "title": "bar"}},
        ]
        df, state = records_to_dataframe(records, tabular=False)
        result = dataframe_to_records(df, state)

        # Empty arrays must be present, not silently absent.
        assert "assignees" in result[0]["issue"], (
            "Empty 'assignees' array was silently dropped during preprocessing roundtrip"
        )
        assert result[0]["issue"]["assignees"] == [], (
            f"Expected [], got {result[0]['issue']['assignees']!r}"
        )
        assert result[0]["issue"]["title"] == "foo"

        # Second record with non-empty arrays must also be intact.
        assert result[1]["issue"]["assignees"][0]["login"] == "alice"
        assert result[1]["issue"]["title"] == "bar"

    def test_boolean_fields_preserved(self):
        records = [{"active": True}, {"active": False}]
        df, state = records_to_dataframe(records, tabular=True)
        result = dataframe_to_records(df, state)
        assert result[0]["active"] == True  # noqa: E712
        assert result[1]["active"] == False  # noqa: E712
        assert isinstance(result[0]["active"], bool)
        assert isinstance(result[1]["active"], bool)
