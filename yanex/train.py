"""Yanex stage 2: train an Origami model.

Loads a dataset config YAML, merges any per-section yanex param overrides,
trains the model, and saves checkpoints as yanex artifacts. Supports resume
from the latest existing checkpoint.

Usage:
    yanex run yanex/train.py \\
        -p seed=42 training.num_epochs=500 \\
        -D data=<data-experiment-id> \\
        -n "adult-1-train"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import yanex
except ImportError as e:
    raise SystemExit(
        "yanex is not installed. Install it with: pip install yanex\n"
        "This script must be run via: yanex run yanex/train.py"
    ) from e

import torch
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

from origami_jsynth.train import (
    CheckpointCallback,
    TimeoutCallback,
    WandbCallback,
    find_latest_checkpoint,
)

# =============================================================================
# YanexCallback (inline — no changes to origami_jsynth package)
# =============================================================================


class YanexCallback(TrainerCallback):
    """Log training metrics to yanex at a fixed step cadence."""

    def __init__(self, log_every: int = 100) -> None:
        super().__init__()
        self.log_every = log_every

    def on_batch_end(self, _trainer: Any, state: TrainResult, _payload: Any) -> None:
        if state.global_step % self.log_every != 0:
            return
        if yanex.has_context():
            yanex.log_metrics(
                {
                    "loss": state.current_batch_loss,
                    "lr": state.current_lr,
                    "batch_dt": state.current_batch_dt,
                    "epoch": state.epoch,
                },
                step=state.global_step,
            )

    def on_evaluate(self, trainer: Any, state: TrainResult, payload: dict[str, float]) -> None:
        if payload and yanex.has_context():
            yanex.log_metrics(payload, step=state.global_step)

    def on_best(self, _trainer: Any, _state: TrainResult, payload: dict[str, float]) -> None:
        if payload and yanex.has_context():
            yanex.log_metrics({f"best_{k}": v for k, v in payload.items()})


# =============================================================================
# Parameters
# =============================================================================

seed: int = yanex.get_param("seed", 42)
use_wandb: bool = yanex.get_param("wandb", False)
max_minutes: float | None = yanex.get_param("max_minutes", None)
verbose: bool = "--show-config" in sys.argv

# Per-section overrides: each is a partial dict merged on top of the YAML config
param_data: dict = yanex.get_param("data", {})
param_model: dict = yanex.get_param("model", {})
param_training: dict = yanex.get_param("training", {})
param_inference: dict = yanex.get_param("inference", {})

# =============================================================================
# Dependency: data stage
# =============================================================================

yanex.assert_dependency("data.py", "data")

dataset: str = yanex.get_graph().get_param("dataset")
data_dir: Path = yanex.get_dependency("data").artifacts_dir
train_records = yanex.load_artifact("train.jsonl")
test_records = yanex.load_artifact("test.jsonl")

print(f"Dataset: {dataset}")
print(f"Train: {len(train_records)} records, Eval: {len(test_records)} records")

# =============================================================================
# Config: load YAML, merge yanex param overrides per section
# =============================================================================

config_path = Path(__file__).parent.parent / "configs" / f"{dataset}.yaml"
with open(config_path) as f:
    base_config: dict = yaml.safe_load(f)

merged_data = {**base_config.get("data", {}), **param_data}
merged_model = {**base_config.get("model", {}), **param_model}
merged_training = {**base_config.get("training", {}), **param_training}
merged_inference = {**base_config.get("inference", {}), **param_inference}
merged_device: str = base_config.get("device", "auto")

# Flattened view for logging and W&B config
merged_config = {
    "data": merged_data,
    "model": merged_model,
    "training": merged_training,
    "inference": merged_inference,
    "device": merged_device,
}

num_epochs: int = merged_training.get("num_epochs", 100)

if verbose:
    print("Merged config:")
    print(yaml.dump(merged_config, default_flow_style=False))

# =============================================================================
# Build pipeline (resume if checkpoint exists)
# =============================================================================

torch.manual_seed(seed)

checkpoint_dir: Path = yanex.get_artifacts_dir()
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint = find_latest_checkpoint(checkpoint_dir)
if checkpoint:
    print(f"Resuming from checkpoint: {checkpoint}")
    pipeline = OrigamiPipeline.load(checkpoint)
else:
    print("Starting training from scratch.")
    config = OrigamiConfig(
        data=DataConfig(**merged_data),
        model=ModelConfig(**merged_model),
        training=TrainingConfig(**merged_training),
        inference=InferenceConfig(**merged_inference),
        device=merged_device,
    )
    pipeline = OrigamiPipeline(config)

# =============================================================================
# Callbacks
# =============================================================================

save_every = max(1, num_epochs // 10)

callbacks: list[TrainerCallback] = [
    TableLogCallback(print_every=100),
    YanexCallback(log_every=100),
    CheckpointCallback(
        pipeline=pipeline,
        checkpoint_dir=checkpoint_dir,
        save_every_epoch=save_every,
        save_best=True,
    ),
]

if max_minutes is not None:
    callbacks.append(TimeoutCallback(max_minutes * 60))

if use_wandb:
    callbacks.append(
        WandbCallback(
            project="origami-jsynth",
            name=f"origami-{dataset}",
            config=merged_config,
            group=dataset,
        )
    )

# =============================================================================
# Train
# =============================================================================

pipeline.fit(train_records, eval_data=test_records, callbacks=callbacks, verbose=True)

final_path = checkpoint_dir / "final.pt"
pipeline.save(final_path, include_training_state=True)
print(f"\nTraining complete. Final model saved to {final_path}")

