"""TabDiff training and sampling orchestration.

Extracted from the original tabdiff/main.py, adapted for use as a library
rather than a CLI-only entry point. Evaluation code removed (handled by
origami-jsynth).
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tabdiff import src
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.modules.main_modules import Model, UniModMLP
from tabdiff.trainer import Trainer, split_num_cat_target, recover_data
from tabdiff.utils_train import TabDiffDataset

warnings.filterwarnings("ignore")


class _NoOpLogger:
    """Minimal wandb-like logger that does nothing."""

    def log(self, data: dict) -> None:
        pass

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        pass


@dataclass
class TrainResult:
    model_save_path: str
    config: dict


@dataclass
class SampleResult:
    df: pd.DataFrame
    num_samples: int


def train(
    data_dir: str,
    info: dict,
    save_dir: str,
    *,
    device: str = "cpu",
    learnable_schedule: bool = True,
    config_overrides: dict | None = None,
    ckpt_path: str | None = None,
    max_seconds: float | None = None,
    logger: Any | None = None,
) -> TrainResult:
    """Train a TabDiff model.

    Args:
        data_dir: Path to directory containing info.json, .npy files, CSVs.
        info: The info.json dict (already loaded).
        save_dir: Base directory for checkpoints and results.
        device: Torch device string.
        learnable_schedule: Whether to use per-column learnable noise schedules.
        config_overrides: Dict of overrides for raw_config (e.g. batch_size, lr).
        ckpt_path: Optional path to resume training from.

    Returns:
        TrainResult with model_save_path and the raw_config used.
    """
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    dataname = info["name"]

    # Load default config
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f"{curr_dir}/configs/tabdiff_configs.toml"
    raw_config = src.load_config(config_path)

    # Apply overrides
    if config_overrides:
        if "batch_size" in config_overrides:
            raw_config["train"]["main"]["batch_size"] = config_overrides["batch_size"]
        if "lr" in config_overrides:
            raw_config["train"]["main"]["lr"] = config_overrides["lr"]
        if "sample_batch_size" in config_overrides:
            raw_config["sample"]["batch_size"] = config_overrides["sample_batch_size"]
        if "check_val_every" in config_overrides:
            raw_config["train"]["main"]["check_val_every"] = config_overrides[
                "check_val_every"
            ]
        if "steps" in config_overrides:
            raw_config["train"]["main"]["steps"] = config_overrides["steps"]
        if "clip_gradients" in config_overrides:
            raw_config["train"]["main"]["clip_gradients"] = config_overrides["clip_gradients"]

    # Paths
    exp_name = "learnable_schedule" if learnable_schedule else "non_learnable_schedule"
    model_save_path = os.path.join(save_dir, "ckpt", dataname, exp_name)
    result_save_path = model_save_path.replace("ckpt", "result")
    raw_config["model_save_path"] = model_save_path
    raw_config["result_save_path"] = result_save_path
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(result_save_path, exist_ok=True)

    # Load data
    batch_size = raw_config["train"]["main"]["batch_size"]
    train_data = TabDiffDataset(
        dataname,
        data_dir,
        info,
        isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    d_numerical, categories = train_data.d_numerical, train_data.categories

    val_data = TabDiffDataset(
        dataname,
        data_dir,
        info,
        isTrain=False,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )

    # Build model
    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist()

    if learnable_schedule:
        raw_config["diffusion_params"]["scheduler"] = "power_mean_per_column"
        raw_config["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=device,
    )
    num_params = sum(p.numel() for p in diffusion.parameters())
    print(f"TabDiff model parameters: {num_params:,}")
    diffusion.to(device)
    diffusion.train()

    # Save config
    with open(os.path.join(model_save_path, "config.pkl"), "wb") as f:
        pickle.dump(raw_config, f)

    # Create trainer
    if logger is None:
        logger = _NoOpLogger()
    sample_batch_size = raw_config["sample"]["batch_size"]

    # Build a minimal metrics-like object that only holds info and real_data_size
    class _MinimalMetrics:
        def __init__(self, info_dict, data_size):
            self.info = info_dict
            self.real_data_size = data_size

    metrics_stub = _MinimalMetrics(info, len(train_data))

    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        metrics_stub,
        logger,
        **raw_config["train"]["main"],
        sample_batch_size=sample_batch_size,
        num_samples_to_generate=None,
        model_save_path=model_save_path,
        result_save_path=result_save_path,
        device=device,
        ckpt_path=ckpt_path,
        max_seconds=max_seconds,
    )

    trainer.run_loop()

    return TrainResult(model_save_path=model_save_path, config=raw_config)


def _find_checkpoint(ckpt_parent: str) -> str:
    """Return the path of the best checkpoint in *ckpt_parent*.

    Selection rules:
    1. Merge ``best_ema_model*`` and ``best_model_*`` into one pool and pick
       the file with the lowest loss encoded in its name (second-to-last
       underscore-separated segment, e.g. ``best_ema_model_1.6117_7856.pt``
       → 1.6117).  EMA is not unconditionally better — with few training
       steps the regular model may have a lower loss, so we let the loss
       value decide.
    2. If no ``best_*`` checkpoint exists fall back to ``final_ema_model*``.
    3. If that is also absent fall back to ``final_model*``.
    4. Raise ``FileNotFoundError`` if nothing is found.
    """

    def _by_loss(p: str) -> float:
        """Extract loss from filenames like best_ema_model_1.6117_7856.pt."""
        parts = Path(p).stem.split("_")
        try:
            return float(parts[-2])
        except (ValueError, IndexError):
            return float("inf")

    best_ckpts = glob.glob(
        os.path.join(ckpt_parent, "best_ema_model*")
    ) + glob.glob(os.path.join(ckpt_parent, "best_model_*"))
    ckpt_files = sorted(best_ckpts, key=_by_loss)

    if not ckpt_files:
        ckpt_files = glob.glob(os.path.join(ckpt_parent, "final_ema_model*"))
    if not ckpt_files:
        ckpt_files = glob.glob(os.path.join(ckpt_parent, "final_model*"))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No TabDiff checkpoints found in {ckpt_parent}. "
            "Train a model first."
        )
    return ckpt_files[0]


def sample(
    data_dir: str,
    info: dict,
    save_dir: str,
    num_samples: int,
    *,
    device: str = "cpu",
    ckpt_path: str | None = None,
    learnable_schedule: bool = True,
    sample_batch_size: int = 10000,
) -> pd.DataFrame:
    """Generate synthetic data from a trained TabDiff model.

    Args:
        data_dir: Path to directory containing info.json, .npy files.
        info: The info.json dict.
        save_dir: Base directory where checkpoints were saved during training.
        num_samples: Number of synthetic rows to generate.
        device: Torch device string.
        ckpt_path: Explicit checkpoint path. If None, auto-discovers best EMA model.
        learnable_schedule: Must match what was used during training.
        sample_batch_size: Batch size for generation.

    Returns:
        DataFrame with synthetic data, columns named as original column names.
    """
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    dataname = info["name"]
    exp_name = "learnable_schedule" if learnable_schedule else "non_learnable_schedule"

    # Find checkpoint
    if ckpt_path is None:
        ckpt_parent = os.path.join(save_dir, "ckpt", dataname, exp_name)
        ckpt_path = _find_checkpoint(ckpt_parent)

    print(f"Sampling from checkpoint: {ckpt_path}")

    # Load config from checkpoint directory
    config_pkl = os.path.join(os.path.dirname(ckpt_path), "config.pkl")
    if os.path.exists(config_pkl):
        with open(config_pkl, "rb") as f:
            raw_config = pickle.load(f)
    else:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        raw_config = src.load_config(f"{curr_dir}/configs/tabdiff_configs.toml")

    # Load data (needed for inverse transforms)
    train_data = TabDiffDataset(
        dataname,
        data_dir,
        info,
        isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    d_numerical, categories = train_data.d_numerical, train_data.categories

    # Rebuild model
    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist()

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=device,
    )
    diffusion.to(device)

    # Load weights
    state_dicts = torch.load(ckpt_path, map_location=device, weights_only=False)
    diffusion._denoise_fn.load_state_dict(state_dicts["denoise_fn"])
    diffusion.num_schedule.load_state_dict(state_dicts["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state_dicts["cat_schedule"])
    print(f"Loaded TabDiff checkpoint from {ckpt_path}")

    diffusion.eval()

    # Generate
    print(f"Generating {num_samples} synthetic samples...")
    syn_data = diffusion.sample_all(
        num_samples, sample_batch_size, keep_nan_samples=True
    )
    print(f"Generated tensor shape: {syn_data.shape}")

    # Inverse transforms
    num_inverse = train_data.num_inverse
    int_inverse = train_data.int_inverse
    cat_inverse = train_data.cat_inverse

    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, info, num_inverse, int_inverse, cat_inverse
    )
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    # Rename columns to original names
    idx_name_mapping = info["idx_name_mapping"]
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)

    return syn_df
