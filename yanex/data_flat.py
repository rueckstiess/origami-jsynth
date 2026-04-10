"""Yanex stage 1b: prepare a dataset with flat (type-separated) preprocessing.

Loads train/test JSONL, applies records_to_dataframe (flatten + type separation)
excluding the target column, converts back to records (dropping NaN = absent keys),
and writes train.jsonl + test.jsonl as artifacts.

Fit is done on combined train+test so both splits share an identical column schema.

Usage:
    yanex run yanex/data_flat.py -p dataset=yelp -n "yelp-1-flat"
"""

from __future__ import annotations

import math
from pathlib import Path

try:
    import yanex
except ImportError as e:
    raise SystemExit(
        "yanex is not installed. Install it with: pip install yanex\n"
        "This script must be run via: yanex run yanex/data_flat.py"
    ) from e

import tempfile

from origami_jsynth.baselines._preprocessing import records_to_dataframe
from origami_jsynth.data import load_jsonl, prepare_dataset, save_jsonl
from origami_jsynth.registry import get_dataset

# =============================================================================
# Parameters
# =============================================================================

dataset: str = yanex.get_param("dataset")
dcr: bool = yanex.get_param("dcr", False)
seed: int = yanex.get_param("seed", 42)

# =============================================================================
# Prepare data
# =============================================================================

print(f"Dataset: {dataset}")
print(f"DCR mode: {dcr}")
print(f"Seed: {seed}")

info = get_dataset(dataset)
exclude = [info.target_column] if info.target_column else []


def df_to_records(df) -> list[dict]:
    """Convert DataFrame to records, dropping NaN keys (= missing values)."""
    records = df.to_dict(orient="records")
    return [
        {k: v for k, v in r.items() if not (isinstance(v, float) and math.isnan(v))}
        for r in records
    ]


with tempfile.TemporaryDirectory() as tmp:
    data_dir = Path(tmp)
    prepare_dataset(dataset, data_dir, dcr=dcr, seed=seed)

    train_records = load_jsonl(data_dir / "train.jsonl")
    test_records = load_jsonl(data_dir / "test.jsonl")

n_train = len(train_records)
n_test = len(test_records)
print(f"Loaded: {n_train} train / {n_test} test")

# Fit on combined so train and test share identical column schema.
print("Applying flat preprocessing (flatten + type separation)...")
combined = train_records + test_records
df, _ = records_to_dataframe(combined, tabular=info.tabular, exclude_columns=exclude)
train_df = df.iloc[:n_train]
test_df = df.iloc[n_train:]

train_flat = df_to_records(train_df)
test_flat = df_to_records(test_df)

print(f"Flat columns: {len(df.columns)}")
print(f"Sample keys (first 10): {sorted(train_flat[0].keys())[:10]}...")

with tempfile.TemporaryDirectory() as tmp:
    train_path = Path(tmp) / "train.jsonl"
    test_path = Path(tmp) / "test.jsonl"
    save_jsonl(train_flat, train_path)
    save_jsonl(test_flat, test_path)
    yanex.copy_artifact(train_path)
    yanex.copy_artifact(test_path)

if yanex.has_context():
    yanex.log_metrics(
        {
            "train_records": n_train,
            "test_records": n_test,
            "flat_columns": len(df.columns),
            "dcr": int(dcr),
        }
    )

print(f"\nDone. train={n_train}, test={n_test}, columns={len(df.columns)}")
