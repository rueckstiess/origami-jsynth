"""Yanex stage 3: generate synthetic replicates from a trained model.

Loads the model checkpoint from the train dependency, generates N synthetic
replicates, and registers each as a yanex artifact.

Usage:
    yanex run yanex/sample.py \\
        -p replicates=10 \\
        -D model=<train-experiment-id> \\
        -n "adult-1-sample"
"""

from __future__ import annotations

from pathlib import Path

try:
    import yanex
except ImportError as e:
    raise SystemExit(
        "yanex is not installed. Install it with: pip install yanex\n"
        "This script must be run via: yanex run yanex/sample.py"
    ) from e

from origami_jsynth.data import load_jsonl
from origami_jsynth.registry import get_dataset
from origami_jsynth.sample import sample_dataset

# =============================================================================
# Parameters
# =============================================================================

replicates: int = yanex.get_param("replicates", 1)
num_workers: int = yanex.get_param("num_workers", 4)
batch_size: int = yanex.get_param("batch_size", 100)
seed_base: int = yanex.get_param("seed_base", 42)

# =============================================================================
# Dependencies: resolve data dir transitively, model dir directly
# =============================================================================

yanex.assert_dependency("train.py", "model")

dataset: str = yanex.get_graph().get_param("dataset")
all_deps = {dep.script_path.stem: dep for dep in yanex.get_dependencies(transitive=True)}
data_dir: Path = all_deps["data"].artifacts_dir
checkpoint_dir: Path = yanex.get_dependency("model").artifacts_dir
samples_dir: Path = yanex.get_artifacts_dir()

print(f"Dataset: {dataset}")
print(f"Replicates: {replicates}")
print(f"Checkpoint dir: {checkpoint_dir}")
print(f"Output dir: {samples_dir}")

# =============================================================================
# Sample
# =============================================================================

info = get_dataset(dataset)

sample_dataset(
    dataset,
    checkpoint_dir,
    samples_dir,
    num_workers=num_workers,
    batch_size=batch_size,
    tabular=info.tabular,
    data_dir=data_dir,
    replicates=replicates,
    seed_base=seed_base,
)

total_records = sum(
    len(load_jsonl(samples_dir / f"synthetic_{i}.jsonl")) for i in range(1, replicates + 1)
)
yanex.log_metrics({"replicates": replicates, "total_synthetic_records": total_records})

print(f"\nDone. {replicates} replicate(s), {total_records} total records.")
