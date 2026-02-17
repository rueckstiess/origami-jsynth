"""Dataset registry for origami-jsynth experiments."""

from __future__ import annotations

from dataclasses import dataclass

HF_REPO = "origami-ml/jsynth-data"

YELP_INSTRUCTIONS = """\
The Yelp dataset cannot be redistributed due to the Yelp Dataset Terms of Use.
To use this dataset:

1. Go to https://www.yelp.com/dataset and accept the terms
2. Download the dataset (you need yelp_academic_dataset_business.json)
3. Place it at: {raw_path}
4. Re-run: origami-jsynth data --dataset yelp
"""


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    target_column: str | None
    task_type: str  # "binclass", "multiclass", or "none"
    tabular: bool
    has_canonical_split: bool
    split_ratio: float = 0.9
    hf_path: str | None = None  # subdirectory in HF repo (None for yelp)


DATASETS: dict[str, DatasetInfo] = {
    "adult": DatasetInfo(
        name="adult",
        target_column="income",
        task_type="binclass",
        tabular=True,
        has_canonical_split=True,
        hf_path="adult",
    ),
    "diabetes": DatasetInfo(
        name="diabetes",
        target_column="readmitted",
        task_type="binclass",
        tabular=True,
        has_canonical_split=True,
        hf_path="diabetes",
    ),
    "electric_vehicles": DatasetInfo(
        name="electric_vehicles",
        target_column="Body Type",
        task_type="multiclass",
        tabular=True,
        has_canonical_split=False,
        hf_path="electric_vehicles",
    ),
    "ddxplus": DatasetInfo(
        name="ddxplus",
        target_column="PATHOLOGY",
        task_type="multiclass",
        tabular=False,
        has_canonical_split=True,
        hf_path="ddxplus",
    ),
    "mtg": DatasetInfo(
        name="mtg",
        target_column="rarity",
        task_type="multiclass",
        tabular=False,
        has_canonical_split=False,
        hf_path="mtg",
    ),
    "yelp": DatasetInfo(
        name="yelp",
        target_column="stars",
        task_type="multiclass",
        tabular=False,
        has_canonical_split=False,
        hf_path=None,  # manual download required
    ),
}

DATASET_NAMES = list(DATASETS.keys())


def get_dataset(name: str) -> DatasetInfo:
    if name not in DATASETS:
        available = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASETS[name]
