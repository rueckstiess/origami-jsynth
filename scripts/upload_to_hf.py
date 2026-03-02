"""Upload prepared dataset splits to HuggingFace."""

from pathlib import Path

from huggingface_hub import HfApi

HF_REPO = "origami-ml/jsynth-data"

# Yanex experiment IDs → dataset names
DATASETS = {
    "adult": "699e374a",
    "diabetes": "8fb645ce",
    "electric_vehicles": "0618fea3",
    "ddxplus": "411be808",
}

YANEX_DIR = Path.home() / ".yanex" / "experiments"


def main():
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=HF_REPO,
        repo_type="dataset",
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/datasets/{HF_REPO}")

    for dataset, experiment_id in DATASETS.items():
        artifacts_dir = YANEX_DIR / experiment_id / "artifacts"
        train_path = artifacts_dir / "train.jsonl"
        test_path = artifacts_dir / "test.jsonl"

        assert train_path.exists(), f"Missing: {train_path}"
        assert test_path.exists(), f"Missing: {test_path}"

        print(f"\nUploading {dataset}...")
        for local_path, remote_name in [(train_path, "train.jsonl"), (test_path, "test.jsonl")]:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{dataset}/{remote_name}",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            print(f"  {remote_name} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\nDone! https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
