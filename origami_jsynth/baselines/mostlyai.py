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

    def __init__(self, tabular: bool = True, dataset_info: Any = None, **kwargs: Any) -> None:
        self.tabular = tabular
        self.kwargs = {**_DEFAULTS, **kwargs}
        self._model: Any = None
        self._state: PreprocessingState | None = None

    def fit(
        self, records: list[dict[str, Any]], max_seconds: float | None = None, **kwargs: Any
    ) -> None:
        try:
            from mostlyai.engine import TabularARGN
        except ImportError:
            raise ImportError(
                "MOSTLY AI Engine requires the 'mostlyai-engine' package. "
                "Install with: pip install mostlyai-engine"
            ) from None

        from ._timeout import TrainingTimeout

        df, self._state = records_to_dataframe(records, self.tabular)

        # Persist model artifacts under the checkpoint directory when provided,
        # so they survive beyond a TemporaryDirectory lifetime.
        model_kwargs = dict(self.kwargs)
        checkpoint_dir = kwargs.get("checkpoint_dir")
        if checkpoint_dir is not None:
            workspace = Path(checkpoint_dir) / "mostlyai_workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            model_kwargs["workspace_dir"] = str(workspace)

        # Also cap via the native max_training_time if --max-minutes is tighter
        if max_seconds is not None:
            native = model_kwargs.get("max_training_time", float("inf"))
            model_kwargs["max_training_time"] = int(min(native, max_seconds))

        self._model = TabularARGN(**model_kwargs)
        with TrainingTimeout(max_seconds):
            self._model.fit(df)
        # If training was interrupted by our timeout, the engine never sets its
        # internal _fitted flag even though checkpoints were saved.  Force it so
        # that save/sample work after a timeout-interrupted run.
        if hasattr(self._model, "_fitted") and not self._model._fitted:
            self._model._fitted = True

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
        # A saved model was trained; ensure the fitted flag is set even if
        # training was interrupted before the engine could set it.
        if hasattr(adapter._model, "_fitted") and not adapter._model._fitted:
            adapter._model._fitted = True
        adapter._state = PreprocessingState.load(path / "preprocess_state.pkl")
        return adapter
