# Autoregresive Synthesis of Sparse and Semi-Structured Mixed-Type Data

Reproduction package for Origami tabular/JSON synthesis experiments and baselines for the publication
"Autoregressive Synthesis of Sparse and Semi-Structured Mixed-Type Data".


> The Origami model architecture is published as a standalone package `origami-ml` for flexible use in various contexts.
> If you want to train an Origami model on your own datasets or for use cases beyond synthetic data generation, we recommend 
> you use the [Origami repository](https://github.com/rueckstiess/origami/) directly.


## Setup

Requires Python 3.11+. MPS (Apple Silicon) or CUDA-capable GPU recommended for training, but CPU-only is possible for small datasets.

### Optional: Create a virtual environment

It's recommended to create a virtual environment to manage dependencies. You can use the built-in `venv` module:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install the package

```bash
pip install -e .
```

To run baseline synthesizers (CTGAN, TVAE, GReaT, REaLTabFormer, Mostly AI), instead install with the `baselines` extra:

```bash
pip install -e ".[baselines]"
```

## Quick Start

Run the full pipeline for a dataset:

```bash
origami-jsynth all --dataset adult --model origami
```

Or run each step individually:

```text
origami-jsynth data --dataset adult --model origami
origami-jsynth train --dataset adult --model origami
origami-jsynth sample --dataset adult --model origami
origami-jsynth eval --dataset adult --model origami
```

Results are saved to `./results/<dataset>/<model>/`.

## Quick Sanity Check

To verify the full pipeline works end-to-end (splitting, training, sampling, evaluation):

```bash
origami-jsynth all --dataset adult --model origami -R 2 --param training.num_epochs=5
```

This completes in approximately 5 minutes with an MPS or CUDA device. The evaluation results from this quick run are not representative of the final results reported in the paper, which require significantly longer training (~400 epochs on the adult dataset).

## Baselines

The `--model` flag selects which synthesizer to run. The default is `origami`.

| Model | Package | Description |
|-------|---------|-------------|
| `origami` | `origami-ml` | Origami (ours) |
| `ctgan` | `sdv` | CTGAN |
| `tvae` | `sdv` | TVAE |
| `great` | `be-great` | GReaT (distilgpt2) |
| `realtabformer` | `realtabformer` | REaLTabFormer |
| `mostlyai` | `mostlyai-engine` | Mostly AI (TabularARGN) |

```bash
origami-jsynth all --dataset adult --model ctgan
```

All models share the same `data`, `sample`, and `eval` pipeline. Each model's artifacts (checkpoints, samples, reports) are stored under `results/<dataset>/<model>/`.

Tabby and TabDiff are not included in this package because they do not offer a Python SDK and instead operate on `.csv` files directly. We ran these baselines in their respective repositories using the flattened `.csv` splits produced by the `data` command.

## Replicates

To measure variance in evaluation metrics, use `--replicates` (`-R`) to run multiple independent sampling rounds with different seeds:

```bash
origami-jsynth sample --dataset adult --replicates 10
origami-jsynth eval --dataset adult
```

The default in this package is `--replicates 1` for a quick run, but the paper results use `--replicates 10` for all datasets, except DDXPlus for DCR, which uses `--replicates 3` due to the much longer privacy metric evaluation time.

This produces `synthetic_1.jsonl` through `synthetic_10.jsonl` in the samples directory. The `eval` command automatically discovers all replicate files, evaluates each independently, and reports aggregate statistics (mean and standard deviation). Results are saved as individual `results_{i}.json` files plus an `agg_results.json` with the aggregate summary.

Sampling is resumable: if interrupted, re-running the same command will skip replicates that already have completed.

## DCR Privacy Evaluation

To evaluate privacy using Distance to Closest Record (DCR), use the `--dcr` flag. This creates a 50/50 train/test split and evaluates privacy only:

```bash
origami-jsynth all --dataset adult --dcr
```

DCR results are saved to `./results/<dataset>_dcr/<model>/`.

## Datasets

| Dataset | Type | Task | Records | Source |
|---------|------|------|---------|--------|
| adult | Tabular | Binary classification | 48,842 | HuggingFace |
| diabetes | Tabular | Binary classification | ~81K | HuggingFace |
| electric_vehicles | Tabular | Multiclass | ~210K | HuggingFace |
| ddxplus | Semi-structured | Multiclass | ~1.16M | HuggingFace |
| yelp | Semi-structured | Multiclass | ~150K | Manual download |

### Yelp Dataset

The Yelp dataset cannot be redistributed due to the [Yelp Dataset Terms of Use](https://www.yelp.com/dataset/download). To use it:

1. Go to https://www.yelp.com/dataset and accept the terms
2. Download the dataset (you need `yelp_academic_dataset_business.json`)
3. Place it at `./results/yelp_academic_dataset_business.json`
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
  --dataset       Dataset name (adult, diabetes, electric_vehicles, ddxplus, yelp)
  --model         Synthesizer to use (default: origami; see Baselines section)
  --output-dir    Base output directory (default: ./results)
  --dcr           DCR mode: 50/50 split, privacy evaluation only
  --num-workers   Number of parallel sampling workers (default: 4, sample/all only)
  -R, --replicates  Number of independent sampling rounds (default: 1, sample/all only)
```

## Output Structure

```
results/
├── adult/                       # regular run
│   ├── data/
│   │   ├── train.jsonl
│   │   ├── test.jsonl
│   │   ├── train.csv            # flattened tabular view
│   │   └── test.csv
│   └── origami/                 # one subdirectory per --model
│       ├── checkpoints/
│       ├── samples/
│       │   ├── synthetic_1.jsonl
│       │   └── ...              # synthetic_{N}.jsonl when using --replicates N
│       └── report/
│           ├── results_1.json
│           ├── ...              # results_{N}.json when using --replicates N
│           └── agg_results.json
└── adult_dcr/                   # DCR privacy run (--dcr flag)
    └── ...                      # same layout as above
```
