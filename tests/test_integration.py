"""Preprocessing pipeline tests using known synthesizer output formats.

Each tabular synthesizer transforms the type-separated DataFrame in a
characteristic way before returning it from sample(). These tests simulate
those known transformations directly — no training required — and verify that
dataframe_to_records correctly restores the original Python types.

Known output formats (discovered empirically):
  - TabDiff:        bool columns → float32 (0.0/1.0)
                    null categoricals → string "nan" in .cat column
  - CTGAN/TVAE:     bool columns → native bool (no conversion needed)
  - MostlyAI:       bool columns → StringDtype "True"/"False"
                    null categoricals → StringDtype, .dtype column correct
"""

import random

import numpy as np
import pandas as pd
import pytest

from origami_jsynth.baselines._preprocessing import dataframe_to_records, records_to_dataframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def type_check_records():
    """Records covering the type cases that caused real bugs."""
    rng = random.Random(42)
    records = []
    for _ in range(80):
        records.append(
            {
                "age": rng.randint(18, 80),
                "income": round(rng.uniform(20000, 100000), 2),
                "category": rng.choice(["A", "B", "C"]),
                "active": rng.choice([True, False]),  # bool
                "has_feature": rng.choice([True, False]),  # bool
                "reason": rng.choice([None, None, None, "spam", "resolved"]),  # nullable cat
                "score": rng.choice([None, round(rng.uniform(0, 1), 4)]),  # nullable num
            }
        )
    return records


def _is_null(v):
    """Check if a value is null (None or float NaN)."""
    return v is None or (isinstance(v, float) and np.isnan(v))


def check_types(records_out):
    """Assert that bool and nullable-cat fields have the correct Python types."""
    for r in records_out:
        for field in ("active", "has_feature"):
            if field in r and not _is_null(r[field]):
                assert isinstance(r[field], bool), (
                    f"{field} should be bool, got {type(r[field])}: {r[field]!r}"
                )
        if "reason" in r:
            assert r["reason"] != "nan", (
                f"reason should not be literal 'nan' string, got {r['reason']!r}"
            )
            assert _is_null(r["reason"]) or isinstance(r["reason"], str), (
                f"reason should be None/NaN or str, got {type(r['reason'])}: {r['reason']!r}"
            )


# ---------------------------------------------------------------------------
# TestTabDiffOutputFormat
#
# TabDiff converts bool columns to float64 (0.0/1.0) and fills null
# categoricals with the string "nan". Simulate both transformations and verify
# dataframe_to_records restores the correct types.
# ---------------------------------------------------------------------------


class TestTabDiffOutputFormat:
    def _simulate_tabdiff_output(self, df):
        """Apply the transformations TabDiff is known to make to the DataFrame.

        TabDiff converts bools to float64 and fills NaN in cat columns with
        "nan" string. The TabDiffAdapter.sample() cleans up the "nan" strings
        before calling dataframe_to_records, so we simulate both the corruption
        and the cleanup here.
        """
        df = df.copy()
        for col in df.columns:
            # Bool columns: TabDiff outputs float64
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype("float64")
            # Object (nullable cat) columns: TabDiff fills NaN with "nan" string,
            # then TabDiffAdapter.sample() restores them to NaN before handoff.
            elif df[col].dtype == object and df[col].isna().any():
                df[col] = df[col].fillna("nan")
                df[col] = df[col].replace("nan", np.nan)
        return df

    def test_bool_columns_restored_from_float(self, type_check_records):
        df, state = records_to_dataframe(type_check_records, tabular=True)
        df_synth = self._simulate_tabdiff_output(df)
        result = dataframe_to_records(df_synth, state)
        check_types(result)

    def test_nullable_cat_restored(self, type_check_records):
        df, state = records_to_dataframe(type_check_records, tabular=True)
        df_synth = self._simulate_tabdiff_output(df)
        result = dataframe_to_records(df_synth, state)
        check_types(result)

    def test_both_transformations_together(self, type_check_records):
        # Both bugs at once: float bools AND "nan" strings.
        df, state = records_to_dataframe(type_check_records, tabular=True)
        df_synth = self._simulate_tabdiff_output(df)
        # Confirm the simulation actually applied the bool transformation.
        bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
        assert bool_cols, "Expected at least one bool column"
        for col in bool_cols:
            assert pd.api.types.is_float_dtype(df_synth[col]), (
                f"Simulation should have converted {col} to float"
            )
        result = dataframe_to_records(df_synth, state)
        assert len(result) == len(type_check_records)
        check_types(result)

    def test_exclude_columns_preserves_target(self, type_check_records):
        # TabDiff excludes target from type separation.
        df, state = records_to_dataframe(
            type_check_records,
            tabular=True,
            exclude_columns=["category"],
        )
        assert "category" in df.columns
        assert "category.dtype" not in df.columns


# ---------------------------------------------------------------------------
# TestSDVOutputFormat
#
# SDV-based synthesizers (CTGAN, TVAE, MostlyAI) encode bool columns as
# "True"/"False" object strings or StringDtype. Simulate that and verify
# dataframe_to_records restores booleans correctly.
# ---------------------------------------------------------------------------


class TestSDVOutputFormat:
    def _simulate_sdv_output(self, df):
        """Apply the transformation SDV-based synthesizers make to bool columns."""
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(object).map(str)  # True → "True"
        return df

    def test_bool_columns_restored_from_string(self, type_check_records):
        df, state = records_to_dataframe(type_check_records, tabular=True)
        df_synth = self._simulate_sdv_output(df)
        result = dataframe_to_records(df_synth, state)
        check_types(result)

    def test_bool_columns_restored_from_stringdtype(self, type_check_records):
        # MostlyAI Engine returns StringDtype specifically.
        df, state = records_to_dataframe(type_check_records, tabular=True)
        df_synth = df.copy()
        for col in df.columns:
            if pd.api.types.is_bool_dtype(df[col]):
                df_synth[col] = df[col].astype(object).map(str).astype(pd.StringDtype())
        result = dataframe_to_records(df_synth, state)
        check_types(result)

    def test_both_bool_formats_produce_correct_types(self, type_check_records):
        df, state = records_to_dataframe(type_check_records, tabular=True)
        for simulate in [
            lambda d: d.assign(
                **{
                    c: d[c].astype(object).map(str)
                    for c in d.columns
                    if pd.api.types.is_bool_dtype(d[c])
                }
            ),
            lambda d: d.assign(
                **{c: d[c].astype("float64") for c in d.columns if pd.api.types.is_bool_dtype(d[c])}
            ),
        ]:
            df_synth = simulate(df.copy())
            result = dataframe_to_records(df_synth, state)
            check_types(result)
