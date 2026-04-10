"""TabDiff baseline adapter.

Integrates the vendored TabDiff package (vendor/tabdiff/) into the
origami-jsynth experiment framework. Handles conversion between JSONL
records and TabDiff's internal format (info.json + .npy arrays + CSVs).

Data conversion logic is adapted from TabDiff's convert_to_tabdiff.py.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..evaluation.type_separation import merge_types
from ._preprocessing import PreprocessingState, dataframe_to_records, records_to_dataframe

_DEFAULTS: dict[str, Any] = {
    "steps": 8000,
    "batch_size": 4096,
    "lr": 0.001,
    "check_val_every": 2000,
}


# ---------------------------------------------------------------------------
# Data-conversion helpers (from convert_to_tabdiff.py)
# ---------------------------------------------------------------------------


def _classify_columns(
    df: pd.DataFrame, target_column: str
) -> tuple[list[int], list[int], list[int]]:
    columns = list(df.columns)
    if target_column not in columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available: {columns}"
        )
    target_idx = [columns.index(target_column)]
    num_col_idx = []
    cat_col_idx = []
    for i, col in enumerate(columns):
        if i in target_idx:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_col_idx.append(i)
        else:
            cat_col_idx.append(i)
    return num_col_idx, cat_col_idx, target_idx


def _clean_data(
    df: pd.DataFrame,
    cat_cols: list[str],
    target_cols: list[str],
    task_type: str,
) -> pd.DataFrame:
    for col in cat_cols:
        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)
        df[col] = df[col].fillna("nan").replace("", "nan").replace("NA", "nan")
    for col in target_cols:
        if task_type == "regression":
            df = df.dropna(subset=[col])
        else:
            df[col] = df[col].fillna("nan").replace("", "nan").replace("NA", "nan")
    return df


def _get_tabdiff_task_type(task_type: str, target_series: pd.Series) -> str:
    if task_type == "regression":
        return "regression"
    n_classes = target_series.nunique()
    return "binclass" if n_classes == 2 else "multiclass"


def _get_column_name_mapping(
    num_col_idx: list[int],
    cat_col_idx: list[int],
    target_col_idx: list[int],
    column_names: list[str],
) -> tuple[dict, dict, dict]:
    idx_mapping: dict[int, int] = {}
    curr_num = 0
    curr_cat = len(num_col_idx)
    curr_target = curr_cat + len(cat_col_idx)
    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[idx] = curr_num
            curr_num += 1
        elif idx in cat_col_idx:
            idx_mapping[idx] = curr_cat
            curr_cat += 1
        else:
            idx_mapping[idx] = curr_target
            curr_target += 1
    inverse_idx_mapping = {v: k for k, v in idx_mapping.items()}
    idx_name_mapping = {i: column_names[i] for i in range(len(column_names))}
    return (
        {str(k): v for k, v in idx_mapping.items()},
        {str(k): v for k, v in inverse_idx_mapping.items()},
        {str(k): v for k, v in idx_name_mapping.items()},
    )


def _compute_int_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_col_idx: list[int],
    column_names: list[str],
) -> tuple[list[int], list[str], list[int]]:
    complete_df = pd.concat([train_df, test_df], axis=0)
    int_col_idx = []
    int_columns = []
    int_col_idx_wrt_num = []
    for i, col_idx in enumerate(num_col_idx):
        col = column_names[col_idx]
        col_data = complete_df[col].dropna()
        if len(col_data) > 0 and (col_data % 1 == 0).all():
            int_columns.append(col)
            int_col_idx.append(col_idx)
            int_col_idx_wrt_num.append(i)
    return int_col_idx, int_columns, int_col_idx_wrt_num


def _build_info(
    name: str,
    tabdiff_task_type: str,
    column_names: list[str],
    num_col_idx: list[int],
    cat_col_idx: list[int],
    target_col_idx: list[int],
    int_col_idx: list[int],
    int_columns: list[str],
    int_col_idx_wrt_num: list[int],
    idx_mapping: dict,
    inverse_idx_mapping: dict,
    idx_name_mapping: dict,
    train_num: int,
    test_num: int,
) -> dict:
    metadata: dict[str, Any] = {"columns": {}}
    for i in num_col_idx:
        metadata["columns"][str(i)] = {
            "sdtype": "numerical",
            "computer_representation": "Float",
        }
    for i in cat_col_idx:
        metadata["columns"][str(i)] = {"sdtype": "categorical"}
    for i in target_col_idx:
        if tabdiff_task_type == "regression":
            metadata["columns"][str(i)] = {
                "sdtype": "numerical",
                "computer_representation": "Float",
            }
        else:
            metadata["columns"][str(i)] = {"sdtype": "categorical"}
    return {
        "name": name,
        "task_type": tabdiff_task_type,
        "header": "infer",
        "column_names": column_names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": f"data/{name}/{name}_train.csv",
        "test_path": f"data/{name}/{name}_test.csv",
        "val_path": None,
        "int_col_idx": int_col_idx,
        "int_columns": int_columns,
        "int_col_idx_wrt_num": int_col_idx_wrt_num,
        "train_num": train_num,
        "test_num": test_num,
        "val_num": 0,
        "idx_mapping": idx_mapping,
        "inverse_idx_mapping": inverse_idx_mapping,
        "idx_name_mapping": idx_name_mapping,
        "metadata": metadata,
    }


def _save_npy_files(
    save_dir: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_columns: list[str],
    cat_columns: list[str],
    target_columns: list[str],
) -> None:
    for split_name, df in [("train", train_df), ("test", test_df)]:
        X_num = df[num_columns].to_numpy().astype(np.float32) if num_columns else np.empty((len(df), 0), dtype=np.float32)
        X_cat = df[cat_columns].to_numpy() if cat_columns else np.empty((len(df), 0), dtype=object)
        y = df[target_columns].to_numpy()
        np.save(os.path.join(save_dir, f"X_num_{split_name}.npy"), X_num)
        np.save(os.path.join(save_dir, f"X_cat_{split_name}.npy"), X_cat)
        np.save(os.path.join(save_dir, f"y_{split_name}.npy"), y)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TabDiffAdapter:
    name = "tabdiff"

    def __init__(self, tabular: bool = True, dataset_info: Any = None, **kwargs: Any) -> None:
        self.tabular = tabular
        # Read dataset metadata from the registry.
        self.target_column: str | None = dataset_info.target_column if dataset_info else None
        self.task_type: str = dataset_info.task_type if dataset_info else "binclass"
        kwargs.pop("tabdiff_dir", None)
        self.kwargs = {**_DEFAULTS, **kwargs}

        # State set during fit()
        self._state: PreprocessingState | None = None
        self._info: dict | None = None
        self._data_dir: str | None = None
        self._save_dir: str | None = None
        self._dataset_name: str = "dataset"

    def fit(
        self,
        records: list[dict[str, Any]],
        checkpoint_dir: Path | None = None,
        max_seconds: float | None = None,
        wandb: bool = False,
        dataset: str = "",
        **kwargs: Any,
    ) -> None:
        try:
            from tabdiff.train import train
        except ImportError:
            raise ImportError(
                "TabDiff requires the 'tabdiff' vendored package. "
                "Install with: pip install -e './vendor/tabdiff'"
            ) from None

        if self.target_column is None:
            raise ValueError(
                "TabDiff requires a target column. Pass --param target_column=<col> "
                "or ensure the dataset has a target_column in the registry."
            )

        if checkpoint_dir is None:
            checkpoint_dir = Path("tabdiff_work")
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Convert records to DataFrame with type separation
        exclude = [self.target_column]
        df, self._state = records_to_dataframe(records, self.tabular, exclude_columns=exclude)

        # Step 2: Split into train/test (TabDiff requires both)
        n_train = int(len(df) * 0.9)
        train_df = df.iloc[:n_train].reset_index(drop=True)
        test_df = df.iloc[n_train:].reset_index(drop=True)

        # Step 3: Classify columns
        columns = list(df.columns)
        num_col_idx, cat_col_idx, target_col_idx = _classify_columns(
            df, self.target_column
        )
        num_columns = [columns[i] for i in num_col_idx]
        cat_columns = [columns[i] for i in cat_col_idx]
        target_columns = [columns[i] for i in target_col_idx]
        print(
            f"TabDiff columns: {len(num_col_idx)} numeric, "
            f"{len(cat_col_idx)} categorical, {len(target_col_idx)} target"
        )

        # Step 4: Clean data
        train_df = _clean_data(train_df, cat_columns, target_columns, self.task_type)
        test_df = _clean_data(test_df, cat_columns, target_columns, self.task_type)

        # Step 5: Compute metadata
        tabdiff_task_type = _get_tabdiff_task_type(
            self.task_type, train_df[self.target_column]
        )
        int_col_idx, int_columns, int_col_idx_wrt_num = _compute_int_columns(
            train_df, test_df, num_col_idx, columns
        )
        idx_mapping, inverse_idx_mapping, idx_name_mapping = _get_column_name_mapping(
            num_col_idx, cat_col_idx, target_col_idx, columns
        )

        # Step 6: Write TabDiff data directory
        self._dataset_name = "dataset"
        work_dir = str(checkpoint_dir / "tabdiff_work")
        data_dir = os.path.join(work_dir, "data", self._dataset_name)
        os.makedirs(data_dir, exist_ok=True)

        # CSVs
        train_df.to_csv(
            os.path.join(data_dir, f"{self._dataset_name}_train.csv"), index=False
        )
        test_df.to_csv(
            os.path.join(data_dir, f"{self._dataset_name}_test.csv"), index=False
        )

        # .npy files
        _save_npy_files(data_dir, train_df, test_df, num_columns, cat_columns, target_columns)

        # info.json
        info = _build_info(
            name=self._dataset_name,
            tabdiff_task_type=tabdiff_task_type,
            column_names=columns,
            num_col_idx=num_col_idx,
            cat_col_idx=cat_col_idx,
            target_col_idx=target_col_idx,
            int_col_idx=int_col_idx,
            int_columns=int_columns,
            int_col_idx_wrt_num=int_col_idx_wrt_num,
            idx_mapping=idx_mapping,
            inverse_idx_mapping=inverse_idx_mapping,
            idx_name_mapping=idx_name_mapping,
            train_num=len(train_df),
            test_num=len(test_df),
        )
        with open(os.path.join(data_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)

        self._info = info
        self._data_dir = data_dir
        self._save_dir = work_dir

        # Step 7: Detect device
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Step 8: Build config overrides from kwargs
        config_overrides = {}
        for key in ("steps", "batch_size", "lr", "check_val_every", "sample_batch_size", "clip_gradients"):
            if key in self.kwargs:
                config_overrides[key] = self.kwargs[key]

        # Step 9: Save adapter state before training so it survives interruptions
        self.save(checkpoint_dir)

        # Step 10: Train
        logger = None
        if wandb:
            import wandb as _wandb

            _wandb.init(
                project="origami-jsynth",
                name=f"tabdiff-{dataset}" if dataset else "tabdiff",
                config=config_overrides or self.kwargs,
                group=dataset or None,
                job_type="train",
            )
            logger = _wandb.run

        try:
            train(
                data_dir=data_dir,
                info=info,
                save_dir=work_dir,
                device=device,
                learnable_schedule=True,
                config_overrides=config_overrides if config_overrides else None,
                max_seconds=max_seconds,
                logger=logger,
            )
        finally:
            if wandb and logger is not None:
                import wandb as _wandb

                _wandb.finish()

    def sample(self, n: int) -> list[dict[str, Any]]:
        try:
            from tabdiff.train import sample as tabdiff_sample
        except ImportError:
            raise ImportError(
                "TabDiff requires the 'tabdiff' vendored package. "
                "Install with: pip install -e './vendor/tabdiff'"
            ) from None

        assert self._info is not None and self._save_dir is not None

        # Detect device
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        sample_batch_size = self.kwargs.get("sample_batch_size", 10000)

        syn_df = tabdiff_sample(
            data_dir=self._data_dir,
            info=self._info,
            save_dir=self._save_dir,
            num_samples=n,
            device=device,
            learnable_schedule=True,
            sample_batch_size=sample_batch_size,
        )

        # TabDiff's _clean_data replaces NaN with the literal string "nan" in
        # categorical columns. Restore actual NaN before dataframe_to_records,
        # since with force=False nullable cat columns may not be type-separated.
        for col in syn_df.select_dtypes(include="object").columns:
            syn_df[col] = syn_df[col].replace("nan", np.nan)

        return dataframe_to_records(syn_df, self._state)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "state": self._state,
            "info": self._info,
            "data_dir": self._data_dir,
            "save_dir": self._save_dir,
            "dataset_name": self._dataset_name,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "kwargs": self.kwargs,
            "tabular": self.tabular,
        }
        with open(path / "tabdiff_state.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: Path, tabular: bool = True) -> TabDiffAdapter:
        with open(path / "tabdiff_state.pkl", "rb") as f:
            meta = pickle.load(f)

        adapter = cls(tabular=meta.get("tabular", tabular))
        adapter._state = meta.get("state") or PreprocessingState(
            is_nested=meta.get("is_nested", not tabular),
            column_map=meta.get("column_map", {}),
        )
        adapter._info = meta["info"]
        adapter._dataset_name = meta["dataset_name"]
        adapter.target_column = meta["target_column"]
        adapter.task_type = meta["task_type"]
        adapter.kwargs = meta["kwargs"]

        # Reconstruct paths relative to the current checkpoint directory
        # instead of using the absolute paths from the original machine.
        work_dir = str(path / "tabdiff_work")
        adapter._save_dir = work_dir
        adapter._data_dir = os.path.join(work_dir, "data", adapter._dataset_name)
        return adapter
