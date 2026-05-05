"""Tests for origami_jsynth.evaluation.shared."""

import pandas as pd
import pytest

from origami_jsynth.evaluation.shared import encode_features, prepare_union_table

# ---------------------------------------------------------------------------
# TestPrepareUnionTable
# ---------------------------------------------------------------------------


class TestPrepareUnionTable:
    def test_two_groups_produce_correct_masks(self):
        real = [{"a": 1}, {"a": 2}]
        synth = [{"a": 3}]
        df, masks = prepare_union_table(real=real, synth=synth)
        assert masks["real"].sum() == 2
        assert masks["synth"].sum() == 1

    def test_masks_are_boolean_series(self):
        real = [{"a": 1}]
        synth = [{"a": 2}]
        df, masks = prepare_union_table(real=real, synth=synth)
        assert masks["real"].dtype == bool
        assert masks["synth"].dtype == bool

    def test_masks_are_disjoint(self):
        real = [{"a": 1}, {"a": 2}]
        synth = [{"a": 3}]
        df, masks = prepare_union_table(real=real, synth=synth)
        overlap = masks["real"] & masks["synth"]
        assert not overlap.any()

    def test_masks_cover_all_rows(self):
        real = [{"a": 1}, {"a": 2}]
        synth = [{"a": 3}]
        df, masks = prepare_union_table(real=real, synth=synth)
        combined = masks["real"] | masks["synth"]
        assert combined.all()

    def test_columns_are_type_separated(self):
        # force=True in prepare_union_table — all columns get .dtype suffix.
        real = [{"x": 1}]
        synth = [{"x": 2}]
        df, _ = prepare_union_table(real=real, synth=synth)
        assert "x.dtype" in df.columns

    def test_split_column_present(self):
        real = [{"a": 1}]
        synth = [{"a": 2}]
        df, _ = prepare_union_table(real=real, synth=synth)
        assert "__split__" in df.columns

    def test_split_column_correct_values(self):
        real = [{"a": 1}, {"a": 2}]
        synth = [{"a": 3}]
        df, masks = prepare_union_table(real=real, synth=synth)
        assert (df.loc[masks["real"], "__split__"] == "real").all()
        assert (df.loc[masks["synth"], "__split__"] == "synth").all()

    def test_three_groups(self):
        train = [{"v": 1}]
        test = [{"v": 2}]
        synth = [{"v": 3}]
        df, masks = prepare_union_table(train=train, test=test, synth=synth)
        assert set(masks.keys()) == {"train", "test", "synth"}
        assert len(df) == 3

    def test_nested_records_handled(self):
        real = [{"user": {"name": "Alice"}}]
        synth = [{"user": {"name": "Bob"}}]
        df, masks = prepare_union_table(real=real, synth=synth)
        # Nested fields should be flattened.
        flat_cols = [c for c in df.columns if "user" in c and c != "__split__"]
        assert len(flat_cols) > 0

    def test_same_fields_same_column_structure(self):
        real = [{"a": 1, "b": "x"}]
        synth = [{"a": 2, "b": "y"}]
        df, masks = prepare_union_table(real=real, synth=synth)
        # Both groups have same fields — columns should include both.
        assert any("a" in c for c in df.columns)
        assert any("b" in c for c in df.columns)


# ---------------------------------------------------------------------------
# TestEncodeFeatures
# ---------------------------------------------------------------------------


class TestEncodeFeatures:
    def _make_df(self):
        """Build a small type-separated DataFrame for testing encode_features."""
        real = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        synth = [{"x": 3, "y": "a"}]
        df, _ = prepare_union_table(real=real, synth=synth)
        return df

    def test_num_columns_are_float(self):
        df = self._make_df()
        encoded = encode_features(df, exclude_columns=["__split__"])
        num_cols = [c for c in encoded.columns if c.endswith(".num")]
        for col in num_cols:
            assert encoded[col].dtype == float

    def test_cat_columns_onehot(self):
        df = self._make_df()
        encoded = encode_features(df, exclude_columns=["__split__"], categorical_encoding="onehot")
        # One-hot columns are boolean / uint8 dtype; at least one column per category value.
        cat_orig_cols = [c for c in df.columns if c.endswith(".cat")]
        for orig_col in cat_orig_cols:
            # After get_dummies, there should be columns like "{orig_col}_a", "{orig_col}_b".
            dummies = [c for c in encoded.columns if c.startswith(orig_col + "_")]
            assert len(dummies) > 0, f"No one-hot columns found for {orig_col}"

    def test_cat_columns_native(self):
        df = self._make_df()
        encoded = encode_features(df, exclude_columns=["__split__"], categorical_encoding="native")
        cat_cols = [c for c in encoded.columns if c.endswith(".cat")]
        for col in cat_cols:
            assert isinstance(encoded[col].dtype, pd.CategoricalDtype)

    def test_bool_columns_numeric(self):
        records_a = [{"flag": True, "x": 1}]
        records_b = [{"flag": False, "x": 2}]
        df, _ = prepare_union_table(a=records_a, b=records_b)
        encoded = encode_features(df, exclude_columns=["__split__"])
        bool_cols = [c for c in encoded.columns if c.endswith(".bool")]
        for col in bool_cols:
            vals = encoded[col].dropna()
            assert set(vals.unique()).issubset({0.0, 1.0})

    def test_alen_columns_are_float(self):
        records_a = [{"tags": ["a", "b"]}]
        records_b = [{"tags": ["x"]}]
        df, _ = prepare_union_table(a=records_a, b=records_b)
        encoded = encode_features(df, exclude_columns=["__split__"])
        alen_cols = [c for c in encoded.columns if c.endswith(".alen")]
        for col in alen_cols:
            assert encoded[col].dtype == float

    def test_all_nan_columns_dropped(self):
        # If a column is all-NaN it should be dropped.
        real = [{"a": 1}]
        synth = [{"a": 2}]
        df, _ = prepare_union_table(real=real, synth=synth)
        # Inject an all-NaN column.
        df = df.copy()
        df["ghost.num"] = float("nan")
        encoded = encode_features(df, exclude_columns=["__split__"])
        assert "ghost.num" not in encoded.columns

    def test_single_value_columns_dropped(self):
        # Columns with only a single unique value carry no information.
        # When ALL columns are constant and therefore uninformative, encode_features raises
        # ValueError because there are no valid features left.
        real = [{"a": 1}, {"a": 1}]
        synth = [{"a": 1}]
        df, _ = prepare_union_table(real=real, synth=synth)
        with pytest.raises(ValueError, match="No valid features found"):
            encode_features(df, exclude_columns=["__split__"])

    def test_exclude_columns_respected(self):
        df = self._make_df()
        encoded = encode_features(df, exclude_columns=["__split__"])
        assert "__split__" not in encoded.columns

    def test_inf_values_replaced_with_nan(self):
        real = [{"x": 1.0}]
        synth = [{"x": 2.0}]
        df, _ = prepare_union_table(real=real, synth=synth)
        df = df.copy()
        num_cols = [c for c in df.columns if c.endswith(".num")]
        if num_cols:
            # Cast to float first to avoid dtype incompatibility warning.
            df[num_cols[0]] = df[num_cols[0]].astype(float)
            df.loc[0, num_cols[0]] = float("inf")
        encoded = encode_features(df, exclude_columns=["__split__"])
        num_enc_cols = [c for c in encoded.columns if c.endswith(".num")]
        for col in num_enc_cols:
            import numpy as np

            assert not np.isinf(encoded[col].dropna()).any()
