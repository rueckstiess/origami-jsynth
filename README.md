# origami-jsynth

Minimal reproduction package for Origami tabular/JSON synthesis experiments.

## Setup

Requires Python 3.12+. MPS (MacOS) or CUDA-capable GPU recommended) for training, but CPU-only is possible for small datasets.

```bash
pip install -e .
```

## Quick Start

Run the full pipeline for a dataset:

```bash
origami-jsynth all --dataset adult
```

Or run each step individually:

```text
origami-jsynth data --dataset adult
origami-jsynth train --dataset adult
origami-jsynth sample --dataset adult
origami-jsynth eval --dataset adult
```

Results are saved to `./results/<dataset>/`.

## DCR Privacy Evaluation

To evaluate privacy using Distance to Closest Record (DCR), use the `--dcr` flag. This creates a 50/50 train/test split and evaluates privacy only:

```bash
origami-jsynth all --dataset adult --dcr
```

DCR results are saved to `./results/<dataset>/dcr/`.

## Datasets

| Dataset | Type | Task | Records | Source |
|---------|------|------|---------|--------|
| adult | Tabular | Binary classification | 48,842 | HuggingFace |
| diabetes | Tabular | Binary classification | ~81K | HuggingFace |
| electric_vehicles | Tabular | Multiclass | ~210K | HuggingFace |
| ddxplus | Semi-structured | Multiclass | ~1.16M | HuggingFace |
| mtg | Semi-structured | Multiclass | ~74K | HuggingFace |
| yelp | Semi-structured | Multiclass | ~150K | Manual download |

### Yelp Dataset

The Yelp dataset cannot be redistributed due to the [Yelp Dataset Terms of Use](https://www.yelp.com/dataset/download). To use it:

1. Go to https://www.yelp.com/dataset and accept the terms
2. Download the dataset (you need `yelp_academic_dataset_business.json`)
3. Place it at `./results/yelp/yelp_academic_dataset_business.json`
4. Run `origami-jsynth data --dataset yelp`

## CLI Reference

```
origami-jsynth <command> --dataset <name> [--output-dir ./results] [--dcr]

Commands:
  data     Download and prepare dataset splits
  train    Train an Origami model
  sample   Generate synthetic data from a trained model
  eval     Evaluate synthetic data quality
  all      Run the full pipeline (data -> train -> sample -> eval)

Options:
  --dataset       Dataset name (adult, diabetes, electric_vehicles, ddxplus, mtg, yelp)
  --output-dir    Base output directory (default: ./results)
  --dcr           DCR mode: 50/50 split, privacy evaluation only
  --num-workers   Number of parallel sampling workers (default: 4, sample/all commands only)
```

## Output Structure

```
results/
└── adult/
    ├── data/
    │   ├── train.jsonl
    │   └── test.jsonl
    ├── checkpoints/
    │   └── final.pt
    ├── samples/
    │   └── synthetic.jsonl
    └── report/
        └── results.json
```
