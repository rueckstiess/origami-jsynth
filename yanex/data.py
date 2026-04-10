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

from origami_jsynth.data import load_jsonl, prepare_dataset, save_jsonl
from origami_jsynth.registry import get_dataset

# =============================================================================
# Parameters
# =============================================================================

dataset: str = yanex.get_param("dataset")
dcr: bool = yanex.get_param("dcr", False)
seed: int = yanex.get_param("seed", 42)
subset_fraction: float | None = yanex.get_param("subset_fraction", None)

# =============================================================================
# Prepare data
# =============================================================================

print(f"Dataset: {dataset}")
print(f"DCR mode: {dcr}")
print(f"Seed: {seed}")
if subset_fraction is not None:
    print(f"Subset fraction: {subset_fraction}")


def stratified_subsample(
    records: list[dict],
    fraction: float,
    target_column: str | None,
    seed: int,
) -> list[dict]:
    """Return a stratified random subsample of records."""
    import random as _random

    if not 0 < fraction < 1:
        raise ValueError(f"subset_fraction must be in (0, 1), got {fraction}")

    rng = _random.Random(seed)

    if target_column is None or not any(target_column in r for r in records):
        records = list(records)
        rng.shuffle(records)
        return records[: max(1, int(len(records) * fraction))]

    # Group by target value, sample proportionally from each group.
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = str(r.get(target_column, "__missing__"))
        groups.setdefault(key, []).append(r)

    sampled: list[dict] = []
    for group in groups.values():
        rng.shuffle(group)
        sampled.extend(group[: max(1, round(len(group) * fraction))])

    rng.shuffle(sampled)
    return sampled


with tempfile.TemporaryDirectory() as tmp:
    data_dir = Path(tmp)
    prepare_dataset(dataset, data_dir, dcr=dcr, seed=seed)

    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"

    if subset_fraction is not None:
        info = get_dataset(dataset)
        train_records = load_jsonl(train_path)
        n_before = len(train_records)
        train_records = stratified_subsample(
            train_records, subset_fraction, info.target_column, seed=seed
        )
        save_jsonl(train_records, train_path)
        print(f"Subsampled train: {n_before} -> {len(train_records)} records")

    yanex.copy_artifact(train_path)
    yanex.copy_artifact(test_path)

    # Count records for metrics (load from temp before dir is cleaned up)
    train_records = load_jsonl(train_path)
    test_records = load_jsonl(test_path)

if yanex.has_context():
    metrics: dict = {
        "train_records": len(train_records),
        "test_records": len(test_records),
        "dcr": int(dcr),
    }
    if subset_fraction is not None:
        metrics["subset_fraction"] = subset_fraction
    yanex.log_metrics(metrics)

print(f"\nDone. train={len(train_records)}, test={len(test_records)}")
