"""GReaT baseline adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ._preprocessing import PreprocessingState, dataframe_to_records, records_to_dataframe

_DEFAULTS = {
    "llm": "distilgpt2",
    "epochs": 100,
    "batch_size": 32,
}


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GReaTAdapter:
    name = "great"

    def __init__(self, tabular: bool = True, **kwargs: Any) -> None:
        self.tabular = tabular
        self.kwargs = {**_DEFAULTS, **kwargs}
        self._model: Any = None
        self._state: PreprocessingState | None = None

    def fit(self, records: list[dict[str, Any]], **kwargs: Any) -> None:
        try:
            from be_great import GReaT
        except ImportError:
            raise ImportError(
                "GReaT requires the 'be-great' package. "
                "Install with: pip install origami-jsynth[baselines]"
            ) from None

        df, self._state = records_to_dataframe(records, self.tabular)
        self._model = GReaT(**self.kwargs)
        self._model.fit(df)

    def sample(self, n: int) -> list[dict[str, Any]]:
        assert self._model is not None and self._state is not None
        df = self._model.sample(n_samples=n, device=_resolve_device())
        return dataframe_to_records(df, self._state)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "model"))
        self._state.save(path / "preprocess_state.pkl")

    @classmethod
    def load(cls, path: Path, tabular: bool = True) -> GReaTAdapter:
        try:
            from be_great import GReaT
        except ImportError:
            raise ImportError(
                "GReaT requires the 'be-great' package. "
                "Install with: pip install origami-jsynth[baselines]"
            ) from None

        adapter = cls(tabular=tabular)
        adapter._model = GReaT.load_from_dir(str(path / "model"))
        adapter._state = PreprocessingState.load(path / "preprocess_state.pkl")
        return adapter
