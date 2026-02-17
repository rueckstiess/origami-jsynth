"""Evaluation orchestration for synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

from .data import load_jsonl
from .evaluation import evaluate_synthetic_data
from .registry import get_dataset


def evaluate_dataset(
    dataset: str,
    data_dir: Path,
    samples_dir: Path,
    report_dir: Path,
    *,
    dcr: bool = False,
) -> Path:
    """Evaluate synthetic data against real data.

    Args:
        dataset: Dataset name from registry.
        data_dir: Directory containing train.jsonl and test.jsonl.
        samples_dir: Directory containing synthetic.jsonl.
        report_dir: Directory to save results.json.
        dcr: If True, evaluate privacy only (DCR mode).

    Returns:
        Path to the results.json file.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "results.json"

    info = get_dataset(dataset)

    train_records = load_jsonl(data_dir / "train.jsonl")
    test_records = load_jsonl(data_dir / "test.jsonl")
    synthetic_records = load_jsonl(samples_dir / "synthetic.jsonl")

    print(f"Evaluating {dataset}:")
    print(
        f"  Train: {len(train_records)}, Test: {len(test_records)}, "
        f"Synthetic: {len(synthetic_records)}"
    )

    if dcr:
        print("  Mode: DCR (privacy only)")
        result = evaluate_synthetic_data(
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
        print("  Mode: Standard (fidelity + utility + detection)")
        result = evaluate_synthetic_data(
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

    # Print summary
    print("\n" + "=" * 60)
    print(f"Results for {dataset}" + (" (DCR)" if dcr else ""))
    print("=" * 60)
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)

    # Save full results
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nFull results saved to {output_path}")

    return output_path


def _map_task_type(task_type: str) -> str:
    """Map registry task_type to evaluation task_type."""
    if task_type in ("binclass", "multiclass"):
        return "classification"
    return task_type
