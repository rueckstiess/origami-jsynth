"""Yanex stage 1: download and prepare a dataset.

Writes train.jsonl and test.jsonl to a temp directory, copies them into the
yanex artifacts directory, and logs record counts.

Usage:
    yanex run yanex/data.py -p dataset=adult -n "adult-1-data"
"""

from __future__ import annotations

import tempfile
from pathlib import Path

try:
    import yanex
except ImportError as e:
    raise SystemExit(
        "yanex is not installed. Install it with: pip install yanex\n"
        "This script must be run via: yanex run yanex/data.py"
    ) from e

from origami_jsynth.data import load_jsonl, prepare_dataset

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

with tempfile.TemporaryDirectory() as tmp:
    data_dir = Path(tmp)
    prepare_dataset(dataset, data_dir, dcr=dcr, seed=seed)

    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"

    yanex.copy_artifact(train_path)
    yanex.copy_artifact(test_path)

    # Count records for metrics (load from temp before dir is cleaned up)
    train_records = load_jsonl(train_path)
    test_records = load_jsonl(test_path)

if yanex.has_context():
    yanex.log_metrics(
        {
            "train_records": len(train_records),
            "test_records": len(test_records),
            "dcr": int(dcr),
        }
    )

print(f"\nDone. train={len(train_records)}, test={len(test_records)}")
