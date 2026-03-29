"""Evaluate a specific Origami checkpoint without running the full CLI pipeline."""

from pathlib import Path

from origami_jsynth.data import load_jsonl
from origami_jsynth.eval import _evaluate_single

CHECKPOINT = Path("results/electric_vehicles/origami/checkpoints/epoch_150.pt")
DATA_DIR = Path("results/electric_vehicles/data")
DATASET = "electric_vehicles"
NUM_WORKERS = 12


def main() -> None:
    from origami_jsynth.sample import sample_parallel

    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    test_records = load_jsonl(DATA_DIR / "test.jsonl")
    n = len(train_records)
    print(f"Train: {n}, Test: {len(test_records)}")

    print(f"\nSampling {n} records from {CHECKPOINT} ({NUM_WORKERS} workers)...")
    synthetic_records = sample_parallel(
        CHECKPOINT,
        n=n,
        num_workers=NUM_WORKERS,
        max_length=2048,
    )
    print(f"Generated {len(synthetic_records)} records")

    print("\nEvaluating...")
    result = _evaluate_single(
        DATASET,
        train_records,
        test_records,
        synthetic_records,
    )

    print("\n" + "=" * 60)
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
