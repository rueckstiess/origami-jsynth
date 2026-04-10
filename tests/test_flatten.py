"""Tests for origami_jsynth.evaluation.flatten."""

import math

import pandas as pd
import pytest

from origami_jsynth.evaluation.flatten import (
    flatten_json,
    flatten_records,
    unflatten_dataframe,
    unflatten_json,
)

# ---------------------------------------------------------------------------
# TestFlattenJson
# ---------------------------------------------------------------------------


class TestFlattenJson:
    def test_flat_dict_unchanged(self):
        obj = {"a": 1, "b": "hello", "c": 3.14}
        assert flatten_json(obj) == obj

    def test_nested_dict_dot_notation(self):
        obj = {"user": {"name": "Alice", "age": 30}}
        result = flatten_json(obj)
        assert result == {"user.name": "Alice", "user.age": 30}

    def test_deeply_nested_dict(self):
        obj = {"a": {"b": {"c": {"d": 42}}}}
        result = flatten_json(obj)
        assert result == {"a.b.c.d": 42}

    def test_list_numeric_indices(self):
        obj = {"scores": [10, 20, 30]}
        result = flatten_json(obj)
        assert result == {"scores.0": 10, "scores.1": 20, "scores.2": 30}

    def test_list_of_dicts(self):
        obj = {"items": [{"x": 1}, {"x": 2}]}
        result = flatten_json(obj)
        assert result == {"items.0.x": 1, "items.1.x": 2}

    def test_empty_dict(self):
        obj = {}
        result = flatten_json(obj)
        assert result == {}

    def test_empty_list_produces_no_keys_leaf_only(self):
        # With include_non_leaf=False (default), an empty list produces no keys.
        obj = {"items": []}
        result = flatten_json(obj)
        assert result == {}

    def test_none_value_preserved(self):
        obj = {"a": None, "b": 1}
        result = flatten_json(obj)
        assert result["a"] is None
        assert result["b"] == 1

    def test_custom_separator(self):
        obj = {"a": {"b": 1}}
        result = flatten_json(obj, sep="/")
        assert result == {"a/b": 1}

    def test_include_non_leaf_stores_list_alongside_children(self):
        obj = {"scores": [1, 2]}
        result = flatten_json(obj, include_non_leaf=True)
        assert result["scores"] == [1, 2]
        assert result["scores.0"] == 1
        assert result["scores.1"] == 2

    def test_include_non_leaf_empty_array_stored(self):
        # Empty array must be stored as the list value itself when include_non_leaf=True.
        obj = {"items": []}
        result = flatten_json(obj, include_non_leaf=True)
        assert result == {"items": []}

    def test_include_non_leaf_empty_dict_stored(self):
        obj = {"meta": {}}
        result = flatten_json(obj, include_non_leaf=True)
        assert result == {"meta": {}}

    def test_include_non_leaf_nested_object_in_array(self):
        obj = {"users": [{"name": "Alice"}]}
        result = flatten_json(obj, include_non_leaf=True)
        # The list, the nested dict, and the leaf are all present.
        assert result["users"] == [{"name": "Alice"}]
        assert result["users.0"] == {"name": "Alice"}
        assert result["users.0.name"] == "Alice"


# ---------------------------------------------------------------------------
# TestUnflattenJson — leaf-only mode
# ---------------------------------------------------------------------------


class TestUnflattenJson:
    def test_flat_dict_unchanged(self):
        flat = {"a": 1, "b": "hello"}
        assert unflatten_json(flat) == flat

    def test_nested_dict_reconstruction(self):
        flat = {"user.name": "Alice", "user.age": 30}
        result = unflatten_json(flat)
        assert result == {"user": {"name": "Alice", "age": 30}}

    def test_list_reconstruction_from_numeric_indices(self):
        flat = {"scores.0": 10, "scores.1": 20, "scores.2": 30}
        result = unflatten_json(flat)
        assert result == {"scores": [10, 20, 30]}

    def test_list_of_dicts_reconstruction(self):
        flat = {"items.0.x": 1, "items.1.x": 2}
        result = unflatten_json(flat)
        assert result == {"items": [{"x": 1}, {"x": 2}]}

    def test_custom_separator(self):
        flat = {"a/b": 1}
        result = unflatten_json(flat, sep="/")
        assert result == {"a": {"b": 1}}

    def test_nan_gaps_compacted(self):
        # NaN at index 1 — the remaining values are packed contiguously.
        flat = {"arr.0": "a", "arr.1": float("nan"), "arr.2": "c"}
        result = unflatten_json(flat)
        assert result == {"arr": ["a", "c"]}

    def test_trailing_nans_dropped(self):
        flat = {"arr.0": "x", "arr.1": float("nan")}
        result = unflatten_json(flat)
        assert result == {"arr": ["x"]}

    def test_empty_array_not_absent(self):
        # Regression test: empty arrays must survive flatten/unflatten, not become absent fields.
        # A single NaN slot ("arr.0": NaN) encodes an empty array — must produce {"arr": []}
        # not an absent key.
        flat = {"arr.0": float("nan")}
        result = unflatten_json(flat)
        assert "arr" in result, "empty array must be present, not silently dropped"
        assert result["arr"] == []

    def test_all_nan_array_with_known_parent_produces_empty_list(self):
        # Regression test: empty arrays must survive flatten/unflatten, not become absent fields.
        # Multiple NaN slots should also yield an empty list.
        flat = {"arr.0": float("nan"), "arr.1": float("nan")}
        result = unflatten_json(flat)
        assert "arr" in result
        assert result["arr"] == []

    def test_sibling_fields_present_when_array_is_empty(self):
        # Regression test: empty arrays must survive flatten/unflatten, not become absent fields.
        flat = {"name": "Bob", "tags.0": float("nan")}
        result = unflatten_json(flat)
        assert result["name"] == "Bob"
        assert "tags" in result
        assert result["tags"] == []


# ---------------------------------------------------------------------------
# TestUnflattenNonLeaf — non-leaf mode auto-detected
# ---------------------------------------------------------------------------


class TestUnflattenNonLeaf:
    def test_non_leaf_values_take_priority(self):
        # Non-leaf value at "scores" should take priority over child keys.
        flat = {"scores": [1, 2], "scores.0": 1, "scores.1": 2}
        result = unflatten_json(flat)
        assert result == {"scores": [1, 2]}

    def test_empty_array_roundtrip_non_leaf(self):
        # Regression test: empty arrays must survive flatten/unflatten, not become absent fields.
        # flatten(include_non_leaf=True) stores {"items": []} and unflatten must reproduce it.
        obj = {"items": []}
        flat = flatten_json(obj, include_non_leaf=True)
        result = unflatten_json(flat)
        assert result == {"items": []}


# ---------------------------------------------------------------------------
# TestRoundtrip — parametrized
# ---------------------------------------------------------------------------


class TestRoundtrip:
    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param({"a": 1, "b": "x"}, id="flat_dict"),
            pytest.param({"user": {"name": "Alice", "age": 30}}, id="nested_dict"),
            pytest.param({"scores": [1, 2, 3]}, id="list"),
            pytest.param({"items": [{"x": 1}, {"x": 2}]}, id="list_of_dicts"),
        ],
    )
    def test_roundtrip_leaf_only(self, obj):
        flat = flatten_json(obj, include_non_leaf=False)
        result = unflatten_json(flat)
        assert result == obj, f"Roundtrip failed for {obj!r}: got {result!r}"

    def test_empty_list_leaf_only_is_lossy(self):
        # In leaf-only mode, flatten_json({"items": []}) produces {} because there
        # are no leaf values to emit. This is a known limitation of leaf-only flattening:
        # an empty list cannot survive a pure in-memory flatten/unflatten cycle.
        # The real pipeline goes through a DataFrame, which adds NaN-padded index columns
        # (e.g., "items.0": NaN) that allow unflatten to reconstruct the empty list.
        # See TestDataFrameRoundtrip for the regression test that actually covers the bug fix.
        flat = flatten_json({"items": []}, include_non_leaf=False)
        assert flat == {}, "leaf-only flatten of empty list produces no keys (by design)"

    def test_variable_length_arrays_roundtrip(self):
        # Two records with arrays of different lengths; shorter gets NaN-padded in DataFrame.
        objs = [
            {"tags": ["a", "b", "c"]},
            {"tags": ["x"]},
        ]
        flat_records = [flatten_json(o) for o in objs]
        df = pd.DataFrame(flat_records)
        records_back = [unflatten_json(r) for r in df.to_dict(orient="records")]
        assert records_back[0] == {"tags": ["a", "b", "c"]}
        assert records_back[1] == {"tags": ["x"]}


# ---------------------------------------------------------------------------
# TestFlattenRecords / TestUnflattenDataframe
# ---------------------------------------------------------------------------


class TestFlattenRecords:
    def test_basic_usage(self):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        df = flatten_records(records)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_empty_records_empty_dataframe(self):
        df = flatten_records([])
        assert df.empty

    def test_columns_sorted_alphabetically(self):
        records = [{"z": 1, "a": 2, "m": 3}]
        df = flatten_records(records)
        assert list(df.columns) == sorted(df.columns)

    def test_variable_length_arrays_nan_padding(self):
        records = [
            {"tags": ["a", "b", "c"]},
            {"tags": ["x"]},
        ]
        df = flatten_records(records)
        # Row 1 should have NaN for tags.1 and tags.2.
        assert math.isnan(df.loc[1, "tags.1"])
        assert math.isnan(df.loc[1, "tags.2"])
        # Row 0 should have all values.
        assert df.loc[0, "tags.0"] == "a"
        assert df.loc[0, "tags.1"] == "b"
        assert df.loc[0, "tags.2"] == "c"


class TestUnflattenDataframe:
    def test_basic_usage(self):
        df = pd.DataFrame([{"user.name": "Alice", "user.age": 30}])
        result = unflatten_dataframe(df)
        assert result == [{"user": {"name": "Alice", "age": 30}}]

    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame(columns=["a", "b"])
        result = unflatten_dataframe(df)
        assert result == []


# ---------------------------------------------------------------------------
# TestDataFrameRoundtrip
# ---------------------------------------------------------------------------


class TestDataFrameRoundtrip:
    def test_nested_records_with_empty_arrays(self):
        # Regression test: empty arrays must survive flatten/unflatten, not become absent fields.
        # Records where some rows have empty arrays and others have non-empty arrays.
        records = [
            {"issue": {"assignees": [], "title": "foo"}},
            {"issue": {"assignees": [{"login": "bob"}], "title": "bar"}},
        ]
        df = flatten_records(records)
        restored = unflatten_dataframe(df)

        # First record: assignees must be [] not absent.
        assert restored[0]["issue"]["assignees"] == [], (
            "Empty assignees array was silently dropped during roundtrip"
        )
        assert restored[0]["issue"]["title"] == "foo"

        # Second record intact.
        assert restored[1]["issue"]["assignees"] == [{"login": "bob"}]
        assert restored[1]["issue"]["title"] == "bar"

    def test_flat_roundtrip(self):
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        df = flatten_records(records)
        restored = unflatten_dataframe(df)
        assert restored == records
