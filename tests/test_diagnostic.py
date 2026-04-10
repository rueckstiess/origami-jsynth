"""Tests for the diagnostic benchmark dataset generator."""

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostic import generate_records, audit_synthetic


class TestGenerateRecords:
    @pytest.fixture(scope="class")
    def records(self):
        return generate_records(1000, seed=42)

    @pytest.fixture(scope="class")
    def df(self, records):
        return pd.DataFrame(records)

    def test_correct_row_count(self, records):
        assert len(records) == 1000

    def test_all_columns_present(self, df):
        expected = {
            "cont_a", "cont_b", "cont_c", "int_count", "int_rating",
            "cat_group", "cat_level", "cat_region", "bool_flag", "bool_active",
            "bool_nullable", "cont_conditional", "cat_nullable", "int_sparse",
            "cont_interaction", "mixed_value", "target",
        }
        assert expected == set(df.columns)

    def test_dense_columns_no_missing(self, df):
        dense = ["cont_a", "cont_b", "int_count", "cat_group", "cat_level",
                 "bool_flag", "bool_active", "cont_interaction", "target"]
        for col in dense:
            assert df[col].isna().sum() == 0, f"{col} has unexpected NaN"

    def test_sparse_columns_have_missing(self, df):
        sparse = ["cont_c", "int_rating", "cat_region", "int_sparse",
                  "bool_nullable", "cont_conditional", "cat_nullable"]
        for col in sparse:
            assert df[col].isna().sum() > 0, f"{col} has no NaN"

    def test_conditional_missingness_cont_conditional(self, df):
        # NaN whenever cat_group == "alpha", never otherwise
        alpha = df[df["cat_group"] == "alpha"]
        non_alpha = df[df["cat_group"] != "alpha"]
        assert alpha["cont_conditional"].isna().all()
        assert non_alpha["cont_conditional"].notna().all()

    def test_conditional_missingness_cat_nullable(self, df):
        # NaN whenever bool_active == False, never otherwise
        inactive = df[df["bool_active"] == False]  # noqa: E712
        active = df[df["bool_active"] == True]  # noqa: E712
        assert inactive["cat_nullable"].isna().all()
        assert active["cat_nullable"].notna().all()

    def test_types_in_json_records(self, records):
        row = records[0]
        assert isinstance(row["cont_a"], float)
        assert isinstance(row["int_count"], int)
        assert isinstance(row["cat_group"], str)
        assert isinstance(row["bool_flag"], bool)
        assert isinstance(row["target"], str)

    def test_target_binary(self, df):
        assert set(df["target"].unique()) == {"positive", "negative"}

    def test_cat_group_values(self, df):
        assert set(df["cat_group"].unique()) == {"alpha", "beta", "gamma"}

    def test_bool_columns_are_bool(self, df):
        assert df["bool_flag"].dtype == bool
        assert df["bool_active"].dtype == bool

    def test_mixed_value_has_both_types(self, records):
        nums = [r for r in records if isinstance(r["mixed_value"], (int, float))]
        strs = [r for r in records if isinstance(r["mixed_value"], str)]
        assert len(nums) > 0 and len(strs) > 0
        # ~60% numeric, ~40% string
        num_pct = len(nums) / len(records)
        assert 0.45 < num_pct < 0.75

    def test_reproducible(self):
        r1 = generate_records(100, seed=123)
        r2 = generate_records(100, seed=123)
        assert r1 == r2

    def test_different_seeds_differ(self):
        r1 = generate_records(100, seed=1)
        r2 = generate_records(100, seed=2)
        assert r1 != r2


class TestAuditSynthetic:
    def test_real_data_all_pass(self):
        records = generate_records(2000, seed=42)
        results = audit_synthetic(records, records)
        failures = [k for k, v in results.items() if v["status"] == "FAIL"]
        assert failures == [], f"Unexpected failures on real data: {failures}"

    def test_empty_dataframe_detected(self):
        results = audit_synthetic([], generate_records(100))
        assert results["all_columns_present"]["status"] == "FAIL"

    def test_missing_column_detected(self):
        records = generate_records(500, seed=42)
        # Remove a column
        for r in records:
            r.pop("bool_flag", None)
        results = audit_synthetic(records, generate_records(500))
        assert results["all_columns_present"]["status"] == "FAIL"
