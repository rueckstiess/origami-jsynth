"""Parallel sampling from trained Origami models."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .data import load_jsonl, save_jsonl


def sample_parallel(
    model_path: str | Path,
    n: int,
    num_workers: int = 4,
    batch_size: int = 100,
    verbose: bool = True,
    seed: int | None = None,
    **sample_kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate synthetic samples in parallel using multiple processes.

    Each worker loads the model independently and generates its share of
    samples in a separate process, avoiding GIL and CUDA contention.

    Args:
        model_path: Path to a saved Origami model checkpoint.
        n: Total number of samples to generate.
        num_workers: Number of parallel worker processes.
        batch_size: Number of samples per generate() call within each worker.
        verbose: Print progress updates (~5 per worker).
        seed: Random seed for reproducibility. Each worker uses seed + worker_index.
        **sample_kwargs: Additional keyword arguments passed to generate().

    Returns:
        List of generated records.

    Raises:
        RuntimeError: If all workers fail.
    """
    model_path = str(Path(model_path).resolve())
    samples_per_worker = n // num_workers
    remainder = n % num_workers

    with tempfile.TemporaryDirectory() as tmpdir:
        procs: list[tuple[subprocess.Popen, Path]] = []
        for i in range(num_workers):
            worker_n = samples_per_worker + (1 if i < remainder else 0)
            if worker_n == 0:
                continue
            out_file = Path(tmpdir) / f"part_{i}.jsonl"
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in sample_kwargs.items())
            sample_args_suffix = ", " + kwargs_str if kwargs_str else ""
            seed_lines = ""
            if seed is not None:
                worker_seed = seed + i
                seed_lines = (
                    "import torch\n"
                    f"torch.manual_seed({worker_seed})\n"
                )
            script = (
                "import json\n"
                f"{seed_lines}"
                "from origami import OrigamiPipeline\n"
                f"pipeline = OrigamiPipeline.load({model_path!r})\n"
                f"total = {worker_n}\n"
                f"batch_size = min({batch_size}, total)\n"
                "n_batches = (total + batch_size - 1) // batch_size\n"
                "report_every = max(1, n_batches // 5)\n"
                "records = []\n"
                "for batch_i in range(n_batches):\n"
                "    n = min(batch_size, total - batch_i * batch_size)\n"
                f"    records.extend(pipeline.generate(n{sample_args_suffix}))\n"
                + (
                    "    if batch_i % report_every == 0 or batch_i == n_batches - 1:\n"
                    f"        print(f'  Worker {i}: "
                    f"{{len(records)}}/{{total}} samples', flush=True)\n"
                    if verbose
                    else ""
                )
                + f"with open({str(out_file)!r}, 'w') as f:\n"
                "    f.writelines(json.dumps(r) + '\\n' for r in records)\n"
            )
            p = subprocess.Popen([sys.executable, "-c", script])
            procs.append((p, out_file))

        failed = 0
        for p, _ in procs:
            p.wait()
            if p.returncode != 0:
                failed += 1

        if failed == len(procs):
            raise RuntimeError(f"All {len(procs)} sampling workers failed")
        if failed > 0:
            print(f"Warning: {failed}/{len(procs)} workers failed")

        records: list[dict[str, Any]] = []
        for _, out_file in procs:
            if out_file.exists():
                with open(out_file) as f:
                    records.extend(json.loads(line) for line in f if line.strip())

    return records


def sample_dataset(
    dataset: str,
    checkpoint_dir: Path,
    samples_dir: Path,
    *,
    num_workers: int = 4,
    batch_size: int = 100,
    tabular: bool = True,
    n_train: int | None = None,
    data_dir: Path | None = None,
    replicates: int = 1,
    seed_base: int = 42,
) -> list[Path]:
    """Sample from a trained model and save results.

    Args:
        dataset: Dataset name (for logging).
        checkpoint_dir: Directory containing the trained model.
        samples_dir: Directory to save synthetic_{i}.jsonl files.
        num_workers: Number of parallel sampling workers.
        batch_size: Batch size per worker.
        tabular: Whether dataset is tabular (affects allow_complex_values).
        n_train: Number of training records (determines sample count).
            If None, reads from data_dir/train.jsonl.
        data_dir: Directory containing train.jsonl (used if n_train is None).
        replicates: Number of independent sampling rounds.
        seed_base: Base seed for reproducibility. Replicate i uses seed_base + i - 1.

    Returns:
        List of paths to the synthetic_{i}.jsonl files.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Find the model
    model_path = checkpoint_dir / "final.pt"
    if not model_path.exists():
        model_path = checkpoint_dir / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {checkpoint_dir}")

    # Determine sample count
    if n_train is None:
        if data_dir is None:
            raise ValueError("Either n_train or data_dir must be provided")
        train_records = load_jsonl(data_dir / "train.jsonl")
        n_train = len(train_records)

    sample_kwargs = {}
    if not tabular:
        sample_kwargs["allow_complex_values"] = True

    output_paths: list[Path] = []
    for i in range(1, replicates + 1):
        output_path = samples_dir / f"synthetic_{i}.jsonl"
        output_paths.append(output_path)

        if output_path.exists():
            existing = load_jsonl(output_path)
            if len(existing) == n_train:
                print(f"Replicate {i}/{replicates}: already exists ({len(existing)} records)")
                continue
            print(
                f"Replicate {i}/{replicates}: incomplete ({len(existing)}/{n_train} records), "
                f"regenerating..."
            )
            output_path.unlink()

        seed = seed_base + i - 1
        print(
            f"Replicate {i}/{replicates}: sampling {n_train} records "
            f"(seed={seed}, {num_workers} workers)..."
        )

        records = sample_parallel(
            model_path,
            n=n_train,
            num_workers=num_workers,
            batch_size=batch_size,
            seed=seed,
            **sample_kwargs,
        )

        save_jsonl(records, output_path)
        print(f"Replicate {i}/{replicates}: saved {len(records)} records to {output_path}")

    return output_paths
