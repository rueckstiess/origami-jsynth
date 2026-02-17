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

## Replicates

To measure variance in evaluation metrics, use `--replicates` (`-R`) to run multiple independent sampling rounds with different seeds:

```bash
origami-jsynth sample --dataset adult --replicates 10
origami-jsynth eval --dataset adult
```

This produces `synthetic_1.jsonl` through `synthetic_10.jsonl` in the samples directory. The `eval` command automatically discovers all replicate files, evaluates each independently, and reports aggregate statistics (mean and standard deviation). Results are saved as individual `results_{i}.json` files plus an `agg_results.json` with the aggregate summary.

Sampling is resumable: if interrupted, re-running the same command will skip replicates that already have output files.

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
  --num-workers   Number of parallel sampling workers (default: 4, sample/all only)
  -R, --replicates  Number of independent sampling rounds (default: 1, sample/all only)
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
    │   ├── synthetic_1.jsonl
    │   └── ...                  # synthetic_{N}.jsonl when using --replicates N
    └── report/
        ├── results_1.json
        ├── ...                  # results_{N}.json when using --replicates N
        └── agg_results.json
```
