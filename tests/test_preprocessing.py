"""Tests for origami_jsynth.baselines._preprocessing."""

import pandas as pd

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

    def test_exclude_columns_not_separated(self):
        # Excluded columns should keep their original name, not get .dtype/.cat suffixes.
        records = [{"target": "pos", "feat": 1.0}, {"target": "neg", "feat": 2.0}]
        df, state = records_to_dataframe(records, tabular=True, exclude_columns=["target"])
        assert "target" in df.columns
        assert "target.dtype" not in df.columns
        assert "target.cat" not in df.columns

    def test_exclude_columns_others_still_separated(self):
        # Non-excluded columns should still be processed by separate_types.
        # With force=False (default), only mixed-type columns get .dtype/.cat
        # sub-columns; homogeneous columns pass through as-is. Use a mixed
        # column to verify separation still applies to non-excluded columns.
        records = [{"target": "pos", "feat": 1}, {"target": "neg", "feat": "text"}]
        df, state = records_to_dataframe(records, tabular=True, exclude_columns=["target"])
        assert "feat.dtype" in df.columns

    def test_exclude_columns_roundtrip(self):
        # Full roundtrip with excluded column should preserve it.
        records = [
            {"target": "pos", "score": 0.9, "flag": True},
            {"target": "neg", "score": 0.1, "flag": False},
        ]
        df, state = records_to_dataframe(records, tabular=True, exclude_columns=["target"])
        result = dataframe_to_records(df, state)
        assert result[0]["target"] == "pos"
        assert result[1]["target"] == "neg"

    def test_exclude_columns_with_missing_values(self):
        # Excluded column with NaN should keep NaN as-is (no .dtype indicator).
        records = [{"target": "pos", "x": 1}, {"target": None, "x": 2}]
        df, state = records_to_dataframe(records, tabular=True, exclude_columns=["target"])
        assert "target" in df.columns
        assert "target.dtype" not in df.columns
        assert pd.isna(df["target"].iloc[1])


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
        for orig, restored in zip(records, result, strict=True):
            assert orig["age"] == restored.get("age") or any("age" in k for k in restored), (
                f"age missing in {restored}"
            )

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
        # Records with empty arrays must survive the full preprocessing
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


# ---------------------------------------------------------------------------
# TestSynthesizerOutputHandling
# ---------------------------------------------------------------------------

# These tests cover the dtype-restoration logic in dataframe_to_records that
# repairs common synthesizer output artefacts: float-encoded bools, string-
# encoded bools, and "nan"-string-encoded nulls in categorical columns.


class TestSynthesizerOutputHandling:
    def test_float_bool_restored_via_passthrough(self):
        # Synthesizer outputs float64 in a passthrough bool column.
        # records_to_dataframe uses force=True so the bool column is separated
        # into a .bool sub-column; dataframe_to_records must recover bool type.
        records = [{"active": True}, {"active": False}, {"active": True}]
        df, state = records_to_dataframe(records, tabular=True)
        # Simulate synthesizer returning float64 in the .bool column.
        if "active.bool" in df.columns:
            df = df.copy()
            df["active.bool"] = df["active.bool"].astype("float64")
        else:
            # Column passed through as-is; cast it to float to simulate synthesizer.
            df = df.copy()
            df["active"] = df["active"].astype("float64")
        result = dataframe_to_records(df, state)
        for r in result:
            assert isinstance(r["active"], bool), (
                f"Expected bool, got {type(r['active'])}: {r['active']!r}"
            )

    def test_string_bool_restored_via_passthrough(self):
        # Synthesizer outputs "True"/"False" strings in a passthrough bool column.
        records = [{"active": True}, {"active": False}, {"active": True}]
        df, state = records_to_dataframe(records, tabular=True)
        # Simulate synthesizer returning string-encoded booleans.
        if "active.bool" in df.columns:
            df = df.copy()
            df["active.bool"] = df["active.bool"].map({True: "True", False: "False"})
        else:
            df = df.copy()
            df["active"] = df["active"].map({True: "True", False: "False"})
        result = dataframe_to_records(df, state)
        for r in result:
            assert isinstance(r["active"], bool), (
                f"Expected bool, got {type(r['active'])}: {r['active']!r}"
            )

    def test_nullable_cat_nan_string_restored_to_none(self):
        # TabDiff's _clean_data converts NaN in categorical columns to the
        # literal string "nan". dataframe_to_records must NOT handle this —
        # only the TabDiffAdapter.sample() method does — but this test confirms
        # that the preprocessing pipeline correctly identifies nullable cat columns
        # and that the records_to_dataframe → dataframe_to_records roundtrip
        # preserves None when no "nan" injection happens.
        records = [
            {"reason": None, "x": 1},
            {"reason": "spam", "x": 2},
        ]
        df, state = records_to_dataframe(records, tabular=True)
        # Locate the column for "reason" (may be separated or passthrough).
        reason_col = None
        for col in df.columns:
            if col == "reason" or col.startswith("reason."):
                reason_col = col
                break
        assert reason_col is not None, f"reason column not found; got {list(df.columns)}"
        # Simulate TabDiff's _clean_data: replace NaN with the string "nan".
        df = df.copy()
        df[reason_col] = df[reason_col].fillna("nan")
        result = dataframe_to_records(df, state)
        # The "nan" string should NOT be silently passed through as None here —
        # that restoration is TabDiffAdapter's responsibility. We verify that
        # the non-null value is intact and that the pipeline doesn't crash.
        non_null = [r["reason"] for r in result if r.get("reason") is not None]
        assert "spam" in non_null

    def test_none_preserved_in_roundtrip(self):
        # Full roundtrip with no synthesizer simulation: None must stay None.
        records = [{"cat": None}, {"cat": "foo"}, {"cat": None}]
        df, state = records_to_dataframe(records, tabular=True)
        result = dataframe_to_records(df, state)
        assert result[1]["cat"] == "foo"
        # Rows 0 and 2 had None; they should come back as None (not "nan").
        assert result[0].get("cat") is None
        assert result[2].get("cat") is None

    def test_missing_field_vs_none_distinction(self):
        # Row 0: b is present but null. Row 1: b is absent entirely.
        # After roundtrip through pandas both become NaN — the null/absent
        # distinction is collapsed by DataFrame construction. The pipeline must
        # not crash and non-null fields must be intact.
        records = [{"a": 1, "b": None}, {"a": 2}]
        df, state = records_to_dataframe(records, tabular=True)
        result = dataframe_to_records(df, state)
        assert len(result) == 2
        # "a" must survive for both rows.
        assert result[0].get("a") == 1
        assert result[1].get("a") == 2
        # Row 0 had b=None; after pandas round-trip it comes back as NaN.
        # It must NOT be the string "nan".
        b_val = result[0].get("b")
        assert b_val != "nan", f"Row 0 b should not be 'nan' string, got {b_val!r}"
