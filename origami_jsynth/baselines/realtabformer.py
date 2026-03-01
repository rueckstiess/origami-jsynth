"""REaLTabFormer baseline adapter."""

from __future__ import annotations

import time
from pathlib import Path, PurePath
from typing import Any

from ._preprocessing import PreprocessingState, dataframe_to_records, records_to_dataframe

_DEFAULTS = {
    "model_type": "tabular",
    "epochs": 200,
    "gradient_accumulation_steps": 4,
    "logging_steps": 100,
}

# Parameters accepted by REaLTabFormer.fit() (not the constructor).
# These are extracted from kwargs and forwarded to .fit() separately.
_FIT_PARAMS = {
    "resume_from_checkpoint",
    "device",
    "num_bootstrap",
    "frac",
    "frac_max_data",
    "qt_max",
    "qt_max_default",
    "qt_interval",
    "qt_interval_unique",
    "quantile",
    "n_critic",
    "n_critic_stop",
    "gen_rounds",
    "sensitivity_max_col_nums",
    "use_ks",
    "full_sensitivity",
    "sensitivity_orig_frac_multiple",
    "orig_samples_rounds",
    "load_from_best_mean_sensitivity",
    "target_col",
    "save_full_every_epoch",
}


def _patch_save(model: Any) -> None:
    """Monkey-patch model.save() to fix upstream serialization bugs.

    REaLTabFormer.save() dumps self.__dict__ to JSON but:
    - Only converts checkpoints_dir and samples_save_dir to strings (misses
      full_save_dir and any other Path attributes).
    - Doesn't exclude non-serializable objects like HuggingFace Dataset
      (which is attached during training and deleted after).

    This patch wraps save() to temporarily sanitize __dict__ before calling
    the original, then restores everything. This makes save() safe to call
    both internally (during _train_with_sensitivity) and externally.
    """
    original_save = model.save

    def patched_save(*args: Any, **kwargs: Any) -> None:
        # Remove non-serializable attributes temporarily
        stashed: dict[str, Any] = {}
        for key in ("dataset", "save"):
            if key in model.__dict__:
                stashed[key] = model.__dict__.pop(key)

        # Convert Path objects that upstream doesn't handle
        _upstream_handles = {"checkpoints_dir", "samples_save_dir"}
        path_backup: dict[str, PurePath] = {}
        for k, v in model.__dict__.items():
            if isinstance(v, PurePath) and k not in _upstream_handles:
                path_backup[k] = v
                model.__dict__[k] = str(v)

        try:
            original_save(*args, **kwargs)
        finally:
            # Restore everything
            model.__dict__.update(stashed)
            model.__dict__.update(path_backup)

    model.save = patched_save


class REaLTabFormerAdapter:
    name = "realtabformer"

    def __init__(self, tabular: bool = True, **kwargs: Any) -> None:
        self.tabular = tabular
        merged = {**_DEFAULTS, **kwargs}
        self.fit_kwargs: dict[str, Any] = {}
        for key in list(merged):
            if key in _FIT_PARAMS:
                self.fit_kwargs[key] = merged.pop(key)
        self.kwargs = merged
        self._model: Any = None
        self._state: PreprocessingState | None = None

    def fit(
        self,
        records: list[dict[str, Any]],
        checkpoint_dir: Path | None = None,
    ) -> None:
        try:
            from realtabformer import REaLTabFormer
        except ImportError:
            raise ImportError(
                "REaLTabFormer requires the 'realtabformer' package. "
                "Install with: pip install origami-jsynth[baselines]"
            ) from None

        df, self._state = records_to_dataframe(records, self.tabular)

        # Redirect RTF's internal working directories into the checkpoint
        # directory so they don't pollute the working directory.
        kwargs = dict(self.kwargs)
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            kwargs.setdefault("checkpoints_dir", str(checkpoint_dir / "rtf_checkpoints"))
            kwargs.setdefault("samples_save_dir", str(checkpoint_dir / "rtf_samples"))
            kwargs.setdefault("full_save_dir", str(checkpoint_dir / "rtf_full_save"))
            # Save preprocessing state early so checkpoints are usable
            # even if training is interrupted.
            self._state.save(checkpoint_dir / "preprocess_state.pkl")

        self._model = REaLTabFormer(**kwargs)
        # Upstream bug: fit() sets experiment_id *after* _train_with_sensitivity()
        # returns, but that method calls self.save() internally which asserts
        # experiment_id is not None. Pre-set it here.
        self._model.experiment_id = f"id{int(time.time() * 10**10):024}"
        # Patch save() to handle serialization bugs (see docstring).
        _patch_save(self._model)
        # gen_kwargs defaults to None upstream, causing **None TypeError during
        # early-stopping sampling. Pass empty dict to work around the bug.
        fit_kwargs = {**self.fit_kwargs, "gen_kwargs": {}}
        self._model.fit(df, **fit_kwargs)

    def sample(self, n: int) -> list[dict[str, Any]]:
        assert self._model is not None and self._state is not None
        df = self._model.sample(n_samples=n, gen_batch=min(512, n))
        return dataframe_to_records(df, self._state)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        # REaLTabFormer creates a subdirectory with an auto-generated ID.
        # Save into a known subdirectory so we can find it on load.
        model_dir = path / "model"
        self._model.save(str(model_dir))
        # Discover the auto-generated id subdirectory
        id_dirs = sorted(model_dir.glob("id*"))
        if id_dirs:
            (path / "_rtf_model_path.txt").write_text(str(id_dirs[-1]))
        self._state.save(path / "preprocess_state.pkl")

    @classmethod
    def load(cls, path: Path, tabular: bool = True) -> REaLTabFormerAdapter:
        try:
            from realtabformer import REaLTabFormer
        except ImportError:
            raise ImportError(
                "REaLTabFormer requires the 'realtabformer' package. "
                "Install with: pip install origami-jsynth[baselines]"
            ) from None

        adapter = cls(tabular=tabular)
        # Read the saved model path
        path_file = path / "_rtf_model_path.txt"
        if path_file.exists():
            model_path = path_file.read_text().strip()
        else:
            # Fallback: find the id subdirectory
            id_dirs = sorted((path / "model").glob("id*"))
            if not id_dirs:
                raise FileNotFoundError(f"No REaLTabFormer model found in {path / 'model'}")
            model_path = str(id_dirs[-1])
        adapter._model = REaLTabFormer.load_from_dir(model_path)
        adapter._state = PreprocessingState.load(path / "preprocess_state.pkl")
        return adapter
