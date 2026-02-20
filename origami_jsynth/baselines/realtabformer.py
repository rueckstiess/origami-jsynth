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
        self.kwargs = {**_DEFAULTS, **kwargs}
        self._model: Any = None
        self._state: PreprocessingState | None = None

    def fit(self, records: list[dict[str, Any]]) -> None:
        try:
            from realtabformer import REaLTabFormer
        except ImportError:
            raise ImportError(
                "REaLTabFormer requires the 'realtabformer' package. "
                "Install with: pip install origami-jsynth[baselines]"
            ) from None

        df, self._state = records_to_dataframe(records, self.tabular)
        self._model = REaLTabFormer(**self.kwargs)
        # Upstream bug: fit() sets experiment_id *after* _train_with_sensitivity()
        # returns, but that method calls self.save() internally which asserts
        # experiment_id is not None. Pre-set it here.
        self._model.experiment_id = f"id{int(time.time() * 10**10):024}"
        # Patch save() to handle serialization bugs (see docstring).
        _patch_save(self._model)
        # gen_kwargs defaults to None upstream, causing **None TypeError during
        # early-stopping sampling. Pass empty dict to work around the bug.
        self._model.fit(df, gen_kwargs={})

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
