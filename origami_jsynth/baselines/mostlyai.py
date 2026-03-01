"""MOSTLY AI Engine baseline adapter."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from ._preprocessing import (
    PreprocessingState,
    dataframe_to_records,
    records_to_dataframe,
)

_DEFAULTS: dict[str, Any] = {
    "max_training_time": 14400,
    "verbose": 1,
}


class MostlyAIAdapter:
    name = "mostlyai"

    def __init__(self, tabular: bool = True, **kwargs: Any) -> None:
        self.tabular = tabular
        self.kwargs = {**_DEFAULTS, **kwargs}
        self._model: Any = None
        self._state: PreprocessingState | None = None

    def fit(self, records: list[dict[str, Any]], **kwargs: Any) -> None:
        try:
            from mostlyai.engine import TabularARGN
        except ImportError:
            raise ImportError(
                "MOSTLY AI Engine requires the 'mostlyai-engine' package. "
                "Install with: pip install mostlyai-engine"
            ) from None

        df, self._state = records_to_dataframe(records, self.tabular)

        # Persist model artifacts under the checkpoint directory when provided,
        # so they survive beyond a TemporaryDirectory lifetime.
        model_kwargs = dict(self.kwargs)
        checkpoint_dir = kwargs.get("checkpoint_dir")
        if checkpoint_dir is not None:
            workspace = Path(checkpoint_dir) / "mostlyai_workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            model_kwargs["workspace_dir"] = str(workspace)

        self._model = TabularARGN(**model_kwargs)
        self._model.fit(df)

    def sample(self, n: int) -> list[dict[str, Any]]:
        assert self._model is not None and self._state is not None
        df = self._model.sample(n_samples=n)
        return dataframe_to_records(df, self._state)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self._model, f)
        self._state.save(path / "preprocess_state.pkl")

    @classmethod
    def load(cls, path: Path, tabular: bool = True) -> MostlyAIAdapter:
        try:
            from mostlyai.engine import TabularARGN  # noqa: F401
        except ImportError:
            raise ImportError(
                "MOSTLY AI Engine requires the 'mostlyai-engine' package. "
                "Install with: pip install mostlyai-engine"
            ) from None

        adapter = cls(tabular=tabular)
        with open(path / "model.pkl", "rb") as f:
            adapter._model = pickle.load(f)
        adapter._state = PreprocessingState.load(path / "preprocess_state.pkl")
        return adapter
