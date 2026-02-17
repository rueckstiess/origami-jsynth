"""Evaluation orchestration for synthetic data."""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

from .data import load_jsonl
from .evaluation import evaluate_synthetic_data
from .registry import get_dataset


def _discover_replicate_files(samples_dir: Path) -> list[tuple[int, Path]]:
    """Find synthetic_*.jsonl files and return sorted (index, path) pairs."""
    pairs: list[tuple[int, Path]] = []
    for p in samples_dir.glob("synthetic_*.jsonl"):
        m = re.match(r"synthetic_(\d+)\.jsonl$", p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort()
    return pairs


def _evaluate_single(
    dataset: str,
    train_records: list[dict],
    test_records: list[dict],
    synthetic_records: list[dict],
    *,
    dcr: bool = False,
):
    """Run evaluation for a single set of synthetic records."""
    info = get_dataset(dataset)

    if dcr:
        return evaluate_synthetic_data(
            train_records,
            test_records,
            synthetic_records,
            target_column=info.target_column,
            task_type=_map_task_type(info.task_type),
            fidelity=False,
            utility=False,
            privacy=True,
            detection=False,
        )
    else:
        return evaluate_synthetic_data(
            train_records,
            test_records,
            synthetic_records,
            target_column=info.target_column,
            task_type=_map_task_type(info.task_type),
            fidelity=True,
            utility=True,
            privacy=False,
            detection=True,
        )


def _aggregate_metrics(
    all_metrics: list[dict[str, float | int]],
) -> dict[str, dict]:
    """Compute mean and stddev for each float metric across replicates."""
    agg: dict[str, dict] = {}
    if not all_metrics:
        return agg

    keys = all_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        if all(isinstance(v, (int, float)) for v in values):
            float_values = [float(v) for v in values]
            mean = statistics.mean(float_values)
            std = statistics.stdev(float_values) if len(float_values) > 1 else 0.0
            agg[key] = {"mean": mean, "std": std, "values": float_values}
    return agg


def evaluate_dataset(
    dataset: str,
    data_dir: Path,
    samples_dir: Path,
    report_dir: Path,
    *,
    dcr: bool = False,
) -> Path:
    """Evaluate synthetic data against real data.

    Discovers all synthetic_*.jsonl files in samples_dir, evaluates each
    independently, and produces per-replicate results plus an aggregate
    summary with mean/stddev.

    Args:
        dataset: Dataset name from registry.
        data_dir: Directory containing train.jsonl and test.jsonl.
        samples_dir: Directory containing synthetic_{i}.jsonl files.
        report_dir: Directory to save results_{i}.json and agg_results.json.
        dcr: If True, evaluate privacy only (DCR mode).

    Returns:
        Path to the agg_results.json file.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    replicate_files = _discover_replicate_files(samples_dir)
    if not replicate_files:
        print(
            f"Error: No synthetic_*.jsonl files found in {samples_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    train_records = load_jsonl(data_dir / "train.jsonl")
    test_records = load_jsonl(data_dir / "test.jsonl")

    mode_str = "DCR (privacy only)" if dcr else "Standard (fidelity + utility + detection)"
    print(f"Evaluating {dataset} ({len(replicate_files)} replicate(s), mode: {mode_str}):")
    print(f"  Train: {len(train_records)}, Test: {len(test_records)}")

    all_metrics: list[dict[str, float | int]] = []
    config: dict = {}

    for idx, synth_path in replicate_files:
        print(f"\n--- Replicate {idx} ---")
        synthetic_records = load_jsonl(synth_path)
        print(f"  Synthetic: {len(synthetic_records)} records")

        result = _evaluate_single(
            dataset,
            train_records,
            test_records,
            synthetic_records,
            dcr=dcr,
        )

        all_metrics.append(result.metrics)
        config = result.to_dict()["config"]

        # Save individual results
        result_path = report_dir / f"results_{idx}.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"  Saved to {result_path}")

    # Aggregate
    agg_metrics = _aggregate_metrics(all_metrics)
    agg_result = {
        "num_replicates": len(replicate_files),
        "metrics": agg_metrics,
        "config": config,
    }

    agg_path = report_dir / "agg_results.json"
    with open(agg_path, "w") as f:
        json.dump(agg_result, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Aggregate results for {dataset}" + (" (DCR)" if dcr else ""))
    print(f"  Replicates: {len(replicate_files)}")
    print("=" * 60)
    for key, stats in agg_metrics.items():
        if key.startswith("num_"):
            continue
        print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print("=" * 60)
    print(f"\nAggregate results saved to {agg_path}")

    return agg_path


def _map_task_type(task_type: str) -> str:
    """Map registry task_type to evaluation task_type."""
    if task_type in ("binclass", "multiclass"):
        return "classification"
    return task_type
