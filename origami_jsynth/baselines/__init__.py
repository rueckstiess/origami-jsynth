"""Baseline synthesizer registry."""

from __future__ import annotations

import importlib
from typing import Any

# Model name -> (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "ctgan": (".ctgan", "CTGANAdapter"),
    "tvae": (".tvae", "TVAEAdapter"),
    "great": (".great", "GReaTAdapter"),
    "realtabformer": (".realtabformer", "REaLTabFormerAdapter"),
    "mostlyai": (".mostlyai", "MostlyAIAdapter"),
    "tabdiff": (".tabdiff", "TabDiffAdapter"),
}

BASELINE_NAMES = list(_REGISTRY.keys())
MODEL_NAMES = ["origami"] + BASELINE_NAMES


def get_synthesizer(name: str, tabular: bool = True, dataset_info: Any = None, **kwargs: Any) -> Any:
    """Instantiate a baseline synthesizer by name.

    Args:
        name: Baseline model name.
        tabular: Whether the dataset is tabular.
        dataset_info: A ``DatasetInfo`` from the registry. Adapters that need
            dataset metadata (e.g. ``target_column``, ``task_type``) should
            read it from here rather than receiving it via kwargs.
        kwargs: Model-specific hyperparameter overrides.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline: {name!r}. Available: {BASELINE_NAMES}")

    module_path, class_name = _REGISTRY[name]
    module = importlib.import_module(module_path, package=__name__)
    cls = getattr(module, class_name)
    return cls(tabular=tabular, dataset_info=dataset_info, **kwargs)
