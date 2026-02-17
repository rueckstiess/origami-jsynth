---
configs:
  - config_name: adult
    data_files:
      - split: train
        path: adult/train.jsonl
      - split: test
        path: adult/test.jsonl
  - config_name: diabetes
    data_files:
      - split: train
        path: diabetes/train.jsonl
      - split: test
        path: diabetes/test.jsonl
  - config_name: electric_vehicles
    data_files:
      - split: train
        path: electric_vehicles/train.jsonl
      - split: test
        path: electric_vehicles/test.jsonl
  - config_name: ddxplus
    data_files:
      - split: train
        path: ddxplus/train.jsonl
      - split: test
        path: ddxplus/test.jsonl
---

# jsynth-data

Preprocessed datasets for Origami tabular/JSON synthesis experiments.

Each configuration is an independent dataset with its own schema and train/test split.

## Datasets

| Dataset | Train | Test | Type |
|---------|-------|------|------|
| adult | 32,561 | 16,281 | Tabular |
| diabetes | 61,059 | 20,354 | Tabular |
| electric_vehicles | 189,010 | 21,001 | Tabular |
| ddxplus | 1,025,602 | 134,529 | Semi-structured |

## Usage

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="origami-ml/jsynth-data",
    filename="adult/train.jsonl",
    repo_type="dataset",
)
```

Or with the `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("origami-ml/jsynth-data", "adult")
```
