"""Dataset downloading, splitting, and JSONL I/O."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any

from .registry import HF_REPO, YELP_INSTRUCTIONS, DatasetInfo, get_dataset


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Save records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _download_hf_file(hf_path: str, filename: str, cache_dir: str | None = None) -> Path:
    """Download a single file from HuggingFace."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{hf_path}/{filename}",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )


def _split_records(
    records: list[dict], ratio: float, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split records into train/test using a fixed seed."""
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    split_idx = int(len(records) * ratio)
    return records[:split_idx], records[split_idx:]


def prepare_dataset(
    dataset: str,
    data_dir: Path,
    *,
    dcr: bool = False,
    seed: int = 42,
) -> Path:
    """Download and prepare a dataset, writing train.jsonl and test.jsonl.

    Args:
        dataset: Dataset name from registry.
        data_dir: Directory where train.jsonl and test.jsonl will be written.
        dcr: If True, use 50/50 split for DCR privacy evaluation.
        seed: Random seed for splitting.

    Returns:
        Path to the data directory containing train.jsonl and test.jsonl.
    """
    info = get_dataset(dataset)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"

    if train_path.exists() and test_path.exists():
        print(f"Data already prepared at {data_dir}")
        return data_dir

    if info.hf_path is None:
        # Yelp: manual download required
        return _prepare_yelp(info, data_dir, dcr=dcr, seed=seed)

    # Download from HuggingFace
    print(f"Downloading {dataset} from HuggingFace ({HF_REPO})...")
    hf_train = _download_hf_file(info.hf_path, "train.jsonl")
    hf_test = _download_hf_file(info.hf_path, "test.jsonl")

    train_records = load_jsonl(hf_train)
    test_records = load_jsonl(hf_test)

    if dcr:
        # Merge and re-split 50/50
        all_records = train_records + test_records
        train_records, test_records = _split_records(all_records, 0.5, seed=seed)
        print(f"DCR split: {len(train_records)} train / {len(test_records)} test")
    else:
        print(f"Loaded: {len(train_records)} train / {len(test_records)} test")

    save_jsonl(train_records, train_path)
    save_jsonl(test_records, test_path)
    print(f"Saved to {data_dir}")
    return data_dir


def _prepare_yelp(
    info: DatasetInfo,
    data_dir: Path,
    *,
    dcr: bool = False,
    seed: int = 42,
) -> Path:
    """Prepare Yelp dataset from manual download."""
    raw_path = data_dir.parent.parent / "yelp_academic_dataset_business.json"

    if not raw_path.exists():
        print(YELP_INSTRUCTIONS.format(raw_path=raw_path))
        raise FileNotFoundError(
            f"Yelp raw data not found at {raw_path}. See instructions above for downloading."
        )

    print("Loading Yelp data from manual download...")
    records = load_jsonl(raw_path)

    # Strip fields that are unique identifiers / PII and not useful for synthesis.
    drop_fields = {"name", "business_id", "address"}
    records = [{k: v for k, v in r.items() if k not in drop_fields} for r in records]

    # Parse comma-separated categories string into a list of strings.
    for r in records:
        if isinstance(r.get("categories"), str):
            r["categories"] = [c.strip() for c in r["categories"].split(",")]

    # The raw Yelp dump stores attribute values as Python repr strings
    # (e.g. "True", "u'casual'", "{'garage': False, ...}").
    # Parse them into proper JSON types.
    for r in records:
        attrs = r.get("attributes")
        if not isinstance(attrs, dict):
            continue
        for k, v in attrs.items():
            if isinstance(v, str):
                try:
                    attrs[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass  # keep as string if not a valid Python literal

    ratio = 0.5 if dcr else info.split_ratio
    train_records, test_records = _split_records(records, ratio, seed=seed)
    print(f"Split: {len(train_records)} train / {len(test_records)} test")

    save_jsonl(train_records, data_dir / "train.jsonl")
    save_jsonl(test_records, data_dir / "test.jsonl")
    print(f"Saved to {data_dir}")
    return data_dir
