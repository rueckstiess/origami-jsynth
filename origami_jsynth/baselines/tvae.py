"""TVAE baseline adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._preprocessing import (
    PreprocessingState,
    build_metadata,
    dataframe_to_records,
    records_to_dataframe,
)

_DEFAULTS = {
    "epochs": 300,
    "verbose": True,
}


class TVAEAdapter:
    name = "tvae"

    def __init__(self, tabular: bool = True, **kwargs: Any) -> None:
        self.tabular = tabular
        self.kwargs = {**_DEFAULTS, **kwargs}
        self._model: Any = None
        self._state: PreprocessingState | None = None

    def fit(self, records: list[dict[str, Any]], **kwargs: Any) -> None:
        try:
            from sdv.single_table import TVAESynthesizer
        except ImportError:
            raise ImportError(
                "TVAE requires the 'sdv' package. "
                "Install with: pip install origami-jsynth[baselines]"
            ) from None

        df, self._state = records_to_dataframe(records, self.tabular)
        metadata = build_metadata(df)
        self._model = TVAESynthesizer(metadata, **self.kwargs)
        self._model.fit(df)

    def sample(self, n: int) -> list[dict[str, Any]]:
        assert self._model is not None and self._state is not None
        df = self._model.sample(n)
        return dataframe_to_records(df, self._state)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "model.pkl"))
        self._state.save(path / "preprocess_state.pkl")

    @classmethod
    def load(cls, path: Path, tabular: bool = True) -> TVAEAdapter:
        try:
            from sdv.single_table import TVAESynthesizer
        except ImportError:
            raise ImportError(
                "TVAE requires the 'sdv' package. "
                "Install with: pip install origami-jsynth[baselines]"
            ) from None

        adapter = cls(tabular=tabular)
        adapter._model = TVAESynthesizer.load(str(path / "model.pkl"))
        adapter._state = PreprocessingState.load(path / "preprocess_state.pkl")
        return adapter
