# Model Parameters

All models accept parameters via the `--param` flag:

```bash
origami-jsynth train --model <model> --dataset <dataset> --param key=value --param key2=value2
```

Parameters are auto-cast: `true`/`false` become booleans, numeric strings become int/float.

## Training Time Limit

All models support the `--max-minutes` flag to cap training wall-clock time:

```bash
origami-jsynth train --model <model> --dataset <dataset> --max-minutes 1440  # 24 hours
```

When the time limit is reached, training stops gracefully and saves a checkpoint so that sampling can proceed. Default: no limit.

---

## Origami

The autoregressive JSON synthesizer. Uses dot-notation for nested config sections.

### Training

| Parameter | Default | Description |
|---|---|---|
| `training.num_epochs` | 10 | Number of training epochs |
| `training.batch_size` | 32 | Training batch size |
| `training.learning_rate` | 0.001 | Learning rate |
| `training.warmup_steps` | 1000 | LR warmup steps |
| `training.weight_decay` | 0.01 | L2 regularization |
| `training.mixed_precision` | `no` | Mixed precision: `no`, `fp16`, `bf16` |
| `training.shuffle_keys` | true | Shuffle JSON keys during training (data augmentation) |
| `training.eval_strategy` | `epoch` | When to evaluate: `no`, `steps`, `epoch` |
| `training.eval_steps` | 100 | Eval every N steps (when `eval_strategy=steps`) |
| `training.target_key` | *(none)* | Key to upweight in loss |
| `training.target_loss_weight` | 1.0 | Multiplier for target key loss |
| `training.constrain_grammar` | true | Enforce JSON grammar during generation |
| `training.constrain_schema` | false | Enforce learned schema during generation |

### Model Architecture

| Parameter | Default | Description |
|---|---|---|
| `model.d_model` | 128 | Embedding dimension |
| `model.n_heads` | 4 | Attention heads |
| `model.n_layers` | 4 | Transformer layers |
| `model.d_ff` | 512 | Feed-forward dimension |
| `model.dropout` | 0.0 | Dropout rate |
| `model.backbone` | `transformer` | Architecture: `transformer`, `lstm`, `mamba` |
| `model.max_seq_length` | 2048 | Maximum sequence length |
| `model.use_continuous_head` | false | Use mixture density head for numeric values |
| `model.num_mixture_components` | 5 | Components in mixture head |

### Data Processing

| Parameter | Default | Description |
|---|---|---|
| `data.numeric_mode` | `disabled` | Numeric handling: `disabled`, `discretize`, `scale` |
| `data.n_bins` | 20 | Bins for discretization |
| `data.bin_strategy` | `quantile` | Binning: `quantile`, `uniform`, `kmeans` |
| `data.cat_threshold` | 100 | Unique-value threshold for categorical treatment |
| `data.infer_schema` | false | Infer schema from training data |

**Example:**

```bash
origami-jsynth train --model origami --dataset adult \
  --param training.num_epochs=50 \
  --param model.d_model=256 \
  --param model.n_layers=6
```

---

## TabDiff

Continuous-time diffusion model for tabular data. Requires a target column (auto-injected from dataset registry).

| Parameter | Default | Description |
|---|---|---|
| `steps` | 8000 | Number of training epochs |
| `batch_size` | 4096 | Training batch size |
| `lr` | 0.001 | Learning rate |
| `check_val_every` | 2000 | Checkpoint interval (in epochs) |
| `sample_batch_size` | 10000 | Batch size during sample generation |

**Example:**

```bash
origami-jsynth train --model tabdiff --dataset adult \
  --param steps=1000 --param batch_size=2048
```

---

## CTGAN

Conditional GAN with mode-specific normalization (via SDV).

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 300 | Number of training epochs |
| `verbose` | true | Print training progress |

Additional SDV `CTGANSynthesizer` parameters (e.g. `embedding_dim`, `generator_dim`, `discriminator_dim`) are passed through.

**Example:**

```bash
origami-jsynth train --model ctgan --dataset adult --param epochs=500
```

---

## TVAE

Variational autoencoder for tabular data (via SDV).

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 300 | Number of training epochs |
| `verbose` | true | Print training progress |

Additional SDV `TVAESynthesizer` parameters (e.g. `embedding_dim`, `compress_dims`, `decompress_dims`) are passed through.

**Example:**

```bash
origami-jsynth train --model tvae --dataset adult --param epochs=500
```

---

## GReaT

LLM-based tabular synthesis using GPT-2 fine-tuning.

| Parameter | Default | Description |
|---|---|---|
| `llm` | `distilgpt2` | HuggingFace model identifier |
| `epochs` | 100 | Number of training epochs |
| `batch_size` | 32 | Training batch size |

**Example:**

```bash
origami-jsynth train --model great --dataset adult \
  --param epochs=50 --param llm=gpt2
```

---

## REaLTabFormer

Autoregressive transformer for tabular data with sensitivity-based early stopping.

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 200 | Number of training epochs |
| `model_type` | `tabular` | Model type |
| `gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `logging_steps` | 100 | Logging frequency |

Additional `REaLTabFormer` constructor and fit parameters are passed through.

**Example:**

```bash
origami-jsynth train --model realtabformer --dataset adult --param epochs=100
```

---

## MostlyAI

Neural network-based autoregressive synthesizer (MostlyAI Engine).

| Parameter | Default | Description |
|---|---|---|
| `max_training_time` | 14400 | Maximum training time in seconds (4 hours) |
| `verbose` | 1 | Verbosity level |

Additional `TabularARGN` parameters are passed through.

**Example:**

```bash
origami-jsynth train --model mostlyai --dataset adult \
  --param max_training_time=7200
```

---

## Installation

| Model | Install command |
|---|---|
| origami | `pip install -e .` |
| ctgan, tvae, great, realtabformer, mostlyai, tabdiff | `pip install -e ".[baselines]"` |
