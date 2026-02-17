"""Training orchestration for Origami models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from origami import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    OrigamiConfig,
    OrigamiPipeline,
    TrainingConfig,
)
from origami.training import TrainResult
from origami.training.callbacks import TableLogCallback, TrainerCallback

from .data import load_jsonl


class CheckpointCallback(TrainerCallback):
    """Save checkpoints to a local directory."""

    def __init__(
        self,
        pipeline: OrigamiPipeline,
        checkpoint_dir: Path,
        save_every_epoch: int = 0,
        save_best: bool = True,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.checkpoint_dir = checkpoint_dir
        self.save_every_epoch = save_every_epoch
        self.save_best = save_best

    def on_best(self, _trainer: Any, _state: Any, _payload: Any) -> None:
        if self.save_best:
            path = self.checkpoint_dir / "best.pt"
            self.pipeline.save(path, include_training_state=True)

    def on_epoch_end(self, _trainer: Any, state: TrainResult, _payload: Any) -> None:
        if self.save_every_epoch > 0 and (state.epoch + 1) % self.save_every_epoch == 0:
            path = self.checkpoint_dir / f"epoch_{state.epoch + 1}.pt"
            self.pipeline.save(path, include_training_state=True)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the latest epoch checkpoint, falling back to best.pt."""
    epoch_checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if epoch_checkpoints:
        return max(epoch_checkpoints, key=lambda p: int(p.stem.split("_")[1]))
    best = checkpoint_dir / "best.pt"
    if best.exists():
        return best
    return None


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply dot-separated key=value overrides to a nested config dict.

    Example: "training.num_epochs=2" sets config["training"]["num_epochs"] = 2.
    Values are auto-cast: ints, floats, bools, then strings.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid param format: {override!r} (expected key=value)")
        key, value = override.split("=", 1)
        parts = key.split(".")

        # Auto-cast value
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # keep as string

        # Navigate to parent dict, creating intermediate dicts if needed
        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
        print(f"  Override: {key} = {value!r}")

    return config


def train_dataset(
    dataset: str,
    data_dir: Path,
    checkpoint_dir: Path,
    config_path: Path,
    *,
    overrides: list[str] | None = None,
    seed: int = 42,
) -> Path:
    """Train an Origami model on a prepared dataset.

    Args:
        dataset: Dataset name (for logging).
        data_dir: Directory with train.jsonl and test.jsonl.
        checkpoint_dir: Directory to save checkpoints.
        config_path: Path to YAML config file.
        overrides: List of "key=value" strings to override config values.
        seed: Random seed.

    Returns:
        Path to the final checkpoint.
    """
    import torch

    torch.manual_seed(seed)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_records = load_jsonl(data_dir / "train.jsonl")
    eval_records = load_jsonl(data_dir / "test.jsonl")
    print(f"Dataset: {dataset}")
    print(f"Train: {len(train_records)} records, Eval: {len(eval_records)} records")

    # Load config
    with open(config_path) as f:
        origami_config = yaml.safe_load(f)

    if overrides:
        origami_config = _apply_overrides(origami_config, overrides)

    # Check for existing checkpoint to resume from
    checkpoint = find_latest_checkpoint(checkpoint_dir)
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
        pipeline = OrigamiPipeline.load(checkpoint)
    else:
        print("Starting training from scratch.")
        config = OrigamiConfig(
            data=DataConfig(**origami_config.get("data", {})),
            model=ModelConfig(**origami_config.get("model", {})),
            training=TrainingConfig(**origami_config.get("training", {})),
            inference=InferenceConfig(**origami_config.get("inference", {})),
            device=origami_config.get("device", "auto"),
        )
        pipeline = OrigamiPipeline(config)

    num_epochs = origami_config.get("training", {}).get("num_epochs", 100)
    save_every = max(1, num_epochs // 10)

    callbacks = [
        TableLogCallback(print_every=100),
        CheckpointCallback(
            pipeline=pipeline,
            checkpoint_dir=checkpoint_dir,
            save_every_epoch=save_every,
            save_best=True,
        ),
    ]

    pipeline.fit(train_records, eval_data=eval_records, callbacks=callbacks, verbose=True)

    final_path = checkpoint_dir / "final.pt"
    pipeline.save(final_path, include_training_state=True)
    print(f"\nTraining complete. Final model saved to {final_path}")
    return final_path
