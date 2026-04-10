# Model Details: Data Encoding and Preprocessing

Detailed comparison of how each baseline model natively encodes categorical and numeric values, which data types it supports, and how it handles missing values. This describes the model's own behavior, not any additional preprocessing applied by our wrapper.

---

## CTGAN

**Implementation**: SDV `CTGANSynthesizer` wrapping the `ctgan` library  
**Paper**: Xu et al., "Modeling Tabular data using Conditional GAN" (NeurIPS 2019)  
**Code**: [sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN)

CTGAN operates on flat DataFrames with two column types: **discrete** (categorical) and **continuous** (numeric). SDV adds a preprocessing layer (Reversible Data Transforms / RDT) that handles type conversion and missing value imputation before CTGAN's own `DataTransformer` applies the model-specific encoding.

### Categorical Encoding

**Method: One-hot encoding** with Gumbel softmax for generation.

CTGAN uses RDT's `OneHotEncoder` internally. Each discrete column with $N$ categories becomes $N$ binary columns (or $N+1$ if NaN is present during training — see Missing Values below). The generator outputs a softmax distribution over categories, and during training the discriminator sees one-hot vectors.

SDV sets the categorical RDT transformer to `None` for CTGAN, meaning categorical columns pass through the SDV preprocessing layer untouched and arrive at CTGAN's `DataTransformer` as raw values, which then applies the one-hot encoding.

### Numeric Encoding

**Method: Mode-specific normalization** using a Bayesian Gaussian Mixture Model (BGM).

This is CTGAN's main contribution. The paper argues that simple min-max normalization to $[-1, 1]$ fails for multimodal continuous distributions. Instead, each continuous column is processed in three steps:

1. **Fit a BGM**: A `BayesianGaussianMixture` with up to 10 components and a Dirichlet process prior (`weight_concentration_prior=0.001`) is fit. Components with weight below 0.005 are pruned — typically 1–10 modes survive.

2. **Assign each value to a mode**: For each value, the BGM predicts the probability of belonging to each component. A component is *sampled* from these probabilities (not argmax — this adds noise that helps training).

3. **Normalize within the mode**: The value is normalized using the selected component's mean $\mu_k$ and std $\sigma_k$:
   $$\alpha = \frac{x - \mu_k}{4 \sigma_k}$$
   The multiplier of 4 ensures ~99.99% of values fall in $[-1, 1]$. The result is clipped to $[-0.99, 0.99]$.

**Output representation**: Each continuous column becomes $1 + K$ dimensions:
- 1 scalar (the normalized value $\alpha$), activated with `tanh`
- $K$ binary columns (one-hot component indicator $\beta$), activated with `softmax`

where $K$ is the number of valid BGM components (typically 1–10).

Before CTGAN's own encoding, SDV applies `FloatFormatter` which enforces rounding to observed decimal precision and optionally clamps values to observed min/max (both enabled by default).

### Supported Types

CTGAN natively recognizes only two types: **discrete** and **continuous**. SDV maps column sdtypes to this binary:

| SDV sdtype | CTGAN type | Preprocessing |
|---|---|---|
| `numerical` | continuous | `FloatFormatter` → float64 |
| `datetime` | continuous | `UnixTimestampEncoder` → float64 (nanoseconds since epoch) |
| `categorical` | discrete | None (pass-through to CTGAN's one-hot) |
| `boolean` | discrete | None (pass-through to CTGAN's one-hot) |

- **Boolean**: Passed through as raw `True`/`False` values. CTGAN's `OneHotEncoder` treats them as a 2-category (or 3-category with NaN) discrete column.
- **Datetime**: Converted to Unix timestamps (nanosecond-precision float64) by `UnixTimestampEncoder`, then treated as a continuous column with mode-specific normalization.

### Missing Values

CTGAN has a **two-layer** missing value strategy due to the SDV wrapper:

**Layer 1 — SDV preprocessing (before CTGAN sees the data):**

| Column type | Strategy | Details |
|---|---|---|
| Numerical | Random imputation | NaN replaced with a random value uniformly sampled from `[min, max]` of the column. No indicator column is created (`missing_value_generation='random'`). |
| Datetime | Random imputation | Same as numerical — NaN replaced with random timestamp in observed range. |
| Categorical | Pass-through | Transformer is `None`, so NaN stays as-is. |
| Boolean | Pass-through | Transformer is `None`, so NaN stays as-is. |

**Layer 2 — CTGAN's own handling (after SDV preprocessing):**

| Column type | Strategy | Details |
|---|---|---|
| Continuous | **Strict rejection** | `_validate_null_data()` raises `InvalidDataError` if any continuous column still has NaN. In practice this never triggers because SDV already imputed them. |
| Discrete | NaN as extra category | `OneHotEncoder` adds a dedicated one-hot dimension for NaN if present during fit. |

**Net effect for each type:**

- **Numeric NaN**: Imputed with random value in observed range. The model never learns that the value was missing — it sees a plausible-looking number. During generation, missing values are reintroduced by randomly setting values to NaN at the observed marginal null rate. This is **unconditional** — missingness in the output is independent of all other columns. Empirically verified: a perfect correlation between a categorical column and numeric missingness (100% null when cat="x", 0% otherwise) is completely destroyed; output shows ~34% nulls uniformly across all category values.
- **Categorical NaN**: Encoded as a dedicated one-hot category. The model explicitly learns to generate "missing" as a category, and this *can* capture conditional missingness patterns since it is part of the learned distribution.
- **Boolean NaN**: Same as categorical NaN — treated as a third category alongside True/False.
- **Datetime NaN**: Same as numeric — imputed with random timestamp, unconditional random reinsertion on output.

### Type Roundtrip Behavior

Empirically tested: all types survive the SDV `fit()` → `sample()` roundtrip. SDV's reverse transform restores original dtypes.

| Input dtype | Output dtype | Survives roundtrip? | Notes |
|---|---|---|---|
| `float64` | `float64` | Yes | |
| `int64` | `int64` | Yes | Rounded back to int by `FloatFormatter` |
| `Int64` (nullable) | `Int64` | Yes | |
| `bool` | `bool` | Yes | |
| `datetime64[ns]` | `datetime64[ns]` | Yes | Reverse-transformed from Unix timestamp |
| `object` (string) | `object` | Yes | |

Missing values are also reintroduced during generation: SDV's `NullTransformer` tracks the observed null rate during `fit()` and probabilistically inserts NaN/NaT in generated samples to match.

### Summary

| Aspect | CTGAN |
|---|---|
| Categorical encoding | One-hot |
| Categorical generation | Gumbel softmax → one-hot |
| Numeric encoding | Mode-specific: BGM (up to 10 modes) + per-mode normalization |
| Numeric formula | $\alpha = (x - \mu_k) / (4\sigma_k)$, clipped to $[-0.99, 0.99]$ |
| Numeric output dims | $1 + K$ per column ($K$ = number of modes) |
| Types recognized | Discrete vs. continuous (binary) |
| Boolean handling | Discrete (2–3 category one-hot); roundtrips as `bool` |
| Datetime handling | Unix timestamp → continuous; roundtrips as `datetime64` |
| Integer handling | Continuous (float internally); roundtrips as `int64` via rounding |
| Missing numeric | Random imputation (SDV layer), no indicator; nulls reintroduced probabilistically on output |
| Missing categorical | Extra one-hot category (CTGAN layer); nulls generated as learned category |
| Missing boolean | Extra one-hot category; same as categorical |
| Missing datetime | Random imputation (SDV layer), no indicator; nulls reintroduced probabilistically on output |

---

## TVAE

**Implementation**: SDV `TVAESynthesizer` wrapping the `ctgan` library (same package as CTGAN)  
**Paper**: Xu et al., "Modeling Tabular data using Conditional GAN" (NeurIPS 2019) — same paper as CTGAN  
**Code**: [sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN) (`ctgan/synthesizers/tvae.py`)

TVAE shares the same `DataTransformer`, the same SDV preprocessing layer (RDT), and the same type/missing-value pipeline as CTGAN. The differences are in the model architecture (VAE vs. GAN) and how reconstruction works.

### Categorical Encoding

**Same as CTGAN.** One-hot encoding via RDT's `OneHotEncoder` inside the shared `DataTransformer`. SDV sets the categorical and boolean RDT transformers to `None` (`_model_sdtype_transformers = {'categorical': None, 'boolean': None}`), so raw category values pass through to CTGAN's `DataTransformer`.

The difference is in generation: where CTGAN uses Gumbel softmax sampling, TVAE reconstructs softmax logits and optimizes them with cross-entropy loss against the one-hot target. During sampling, the decoder produces logits that are passed through argmax in the inverse transform.

### Numeric Encoding

**Same as CTGAN.** Mode-specific normalization via Bayesian Gaussian Mixture (same `ClusterBasedNormalizer`), same formula $\alpha = (x - \mu_k) / (4\sigma_k)$, same $1 + K$ output dimensions per continuous column.

One difference in reconstruction: TVAE's decoder includes a **learnable per-element sigma** parameter (initialized to 0.1). The loss for continuous columns is a Gaussian log-likelihood:

$$\mathcal{L}_\text{cont} = \frac{(x - \tanh(\hat{x}))^2}{2\sigma^2} + \log \sigma$$

During `inverse_transform`, these learned sigmas are passed through and used to add calibrated noise to the normalized values before denormalization. CTGAN passes `None` for sigmas, so no such noise is added.

### Supported Types

**Same as CTGAN.** The SDV wrapper configures identical type mappings:

| SDV sdtype | TVAE type | Preprocessing |
|---|---|---|
| `numerical` | continuous | `FloatFormatter` → float64 |
| `datetime` | continuous | `UnixTimestampEncoder` → float64 (nanoseconds since epoch) |
| `categorical` | discrete | None (pass-through to one-hot) |
| `boolean` | discrete | None (pass-through to one-hot) |

### Missing Values

**Same as CTGAN.** Identical two-layer strategy:

- SDV layer: numeric/datetime NaN imputed with random value in `[min, max]`, no indicator column. Categorical/boolean NaN passes through.
- TVAE layer: continuous columns reject NaN (same `_validate_null_data()` — though note the TVAE source omits this check, SDV's preprocessing already removes all numeric nulls). Discrete NaN encoded as extra one-hot category.

Empirically confirmed: conditional missingness is **not preserved** — same behavior as CTGAN. A perfect correlation between a categorical column value and numeric missingness is destroyed; output shows uniform ~34% null rate across all categories regardless of the input pattern.

### Type Roundtrip Behavior

Empirically tested: identical to CTGAN — all types survive the `fit()` → `sample()` roundtrip.

| Input dtype | Output dtype | Survives? |
|---|---|---|
| `float64` | `float64` | Yes |
| `int64` | `int64` | Yes |
| `Int64` (nullable) | `Int64` | Yes |
| `bool` | `bool` | Yes |
| `datetime64[ns]` | `datetime64[ns]` | Yes |
| `object` (string) | `object` | Yes |

### Differences from CTGAN

| Aspect | CTGAN | TVAE |
|---|---|---|
| Architecture | Generator + Discriminator (GAN) | Encoder + Decoder (VAE) |
| Training signal | Adversarial (WGAN-GP) + conditional | Reconstruction + KL divergence |
| Categorical generation | Gumbel softmax | Cross-entropy on softmax logits |
| Continuous reconstruction | Implicit (adversarial) | Gaussian likelihood with learnable variance |
| Sigmas in inverse transform | `None` (no noise added) | Learned per-element sigma (noise added) |
| Loss factor | N/A | `loss_factor=2` scales reconstruction vs. KLD |
| Conditional training | Training-by-sampling + conditional vector | None — no conditional mechanism |

### Summary

| Aspect | TVAE |
|---|---|
| Categorical encoding | One-hot (same as CTGAN) |
| Numeric encoding | Mode-specific BGM (same as CTGAN) + learnable reconstruction variance |
| Types recognized | Discrete vs. continuous (same as CTGAN) |
| Boolean handling | Discrete one-hot; roundtrips as `bool` |
| Datetime handling | Unix timestamp → continuous; roundtrips as `datetime64` |
| Integer handling | Continuous (float internally); roundtrips as `int64` |
| Missing numeric | Random imputation, unconditional reinsertion (same as CTGAN) |
| Missing categorical | Extra one-hot category (same as CTGAN) |
| Conditional missingness | **Not preserved** (empirically verified) |

---

## REaLTabFormer

**Implementation**: `realtabformer` library  
**Paper**: Solatorio & Dupriez, "REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers" (2023)  
**Code**: [worldbank/REaLTabFormer](https://github.com/worldbank/REaLTabFormer)

REaLTabFormer is fundamentally different from CTGAN/TVAE. It uses a GPT-2 transformer and treats tabular data generation as **sequence generation** — each row is serialized into a sequence of tokens from a fixed vocabulary built from the training data. Only the single-table variant is discussed here.

### Categorical Encoding

**Method: Vocabulary token per unique value.**

Each unique value in a categorical column gets its own token in a fixed vocabulary. The token format encodes column metadata:

```
02___CATEGORICAL___cat___a     → token ID 42
02___CATEGORICAL___cat___b     → token ID 43
```

There is no one-hot encoding or embedding layer in the traditional sense — the GPT-2 model learns embeddings for each token ID. Out-of-vocabulary values map to `[UNK]`. The vocabulary is built from all unique values observed during training.

### Numeric Encoding

**Method: Character-level digit tokenization** (no binning, no normalization).

Numeric values are converted to aligned strings and then split character-by-character, with each digit position becoming a separate token:

1. **Align**: Zero-pad integers to uniform width (e.g., `5` → `05` if max is 99). For floats, align decimal points and pad.
2. **Partition**: Split into single-character substrings (default `numeric_nparts=1`). Each character becomes a separate token.

Example: the value `1.5` in column `num_float` becomes 4 tokens:
```
00___NUMERIC___num_float_00___0     (tens digit)
00___NUMERIC___num_float_01___1     (units digit)
00___NUMERIC___num_float_02___.     (decimal point)
00___NUMERIC___num_float_03___5     (tenths digit)
```

Each digit position has its own set of possible tokens (0–9, `.`, `-`), giving the model a structured vocabulary over digit positions rather than treating numbers as opaque strings. No scaling or normalization is applied — values retain their original magnitude.

### Supported Types

REaLTabFormer detects types from pandas dtypes:

| pandas dtype | REaLTabFormer type | Handling |
|---|---|---|
| `int64`, `float64` | `NUMERIC` | Character-level digit tokenization |
| `bool` | `NUMERIC` | Encoded as digit `0` or `1` |
| `datetime64` | `DATETIME` | Mean-centered Unix timestamp → digit tokenization |
| `object` (string) | `CATEGORICAL` | One token per unique value |
| Nullable `bool` (`object` with `True`/`None`) | `CATEGORICAL` | Tokens: `"True"`, `"False"`, `"<NA>"` |

Notable: **Boolean handling depends on nullability.** Non-nullable `bool` columns are treated as NUMERIC (0/1 digit tokens). Nullable booleans (which pandas stores as `object` dtype) are treated as CATEGORICAL with string tokens `"True"`/`"False"`/`"<NA>"`.

**Datetime handling**: Datetimes are converted to Unix timestamps, centered by subtracting the column mean, then digit-tokenized. This means output datetimes may include time components even if inputs were date-only.

### Missing Values

**Method: Explicit missing tokens** — missingness is part of the learned sequence.

| Column type | Missing representation | Details |
|---|---|---|
| Numeric | `@` per digit position | Each digit position gets its own `@` token (e.g., `@@@@` for 4-digit column) |
| Categorical | `<NA>` string | Single token `"07___CATEGORICAL___cat_na___<NA>"` |
| Datetime | `@` per digit position | Same as numeric |

**Key design**: The model explicitly learns when to generate missing tokens as part of its autoregressive sequence. No imputation, no indicator columns — missingness is a first-class value in the vocabulary.

**Validation during sampling**: Columns that had no missing values during training are enforced non-null in generation — any sampled row where such a column is NaN is rejected (the `_validate_missing` mechanism). This can significantly reduce sampling efficiency.

**Conditional missingness**: In principle, the autoregressive architecture can learn conditional missingness patterns because later tokens are generated conditioned on earlier tokens. However, empirical testing with a strong correlation (A="x" → B always null) showed the pattern was **not well preserved**: the model generated ~9% nulls for A=x vs ~12% for A=y/z — roughly uniform and well below the input rate of 34%. The model also had 48% invalid sample rate, suggesting difficulty with the missingness pattern. This may improve with larger datasets and longer training, but the character-level digit tokenization means generating a missing numeric value requires all digit positions to independently produce `@`, which may be harder to learn than a single missing indicator.

### Type Roundtrip Behavior

Empirically tested: non-nullable types survive; nullable types degrade.

| Input dtype | Output dtype | Survives? | Notes |
|---|---|---|---|
| `float64` | `float64` | Yes | |
| `int64` | `int64` | Yes | |
| `Int64` (nullable) | `object` | **No** | Comes back as string |
| `bool` | `bool` | Yes | (non-nullable) |
| `object` (string) | `object` | Yes | |
| `datetime64[ns]` | `datetime64[ns]` | Yes | But gains time components |
| Nullable bool (`object`) | `object` | Partial | Values are strings `"True"`/`"False"`, not Python `bool` |

### Summary

| Aspect | REaLTabFormer |
|---|---|
| Architecture | GPT-2 (autoregressive transformer) |
| Categorical encoding | One token per unique value in fixed vocabulary |
| Numeric encoding | Character-level digit tokenization (no binning, no normalization) |
| Numeric output dims | 1 token per digit position per column |
| Types recognized | NUMERIC / CATEGORICAL / DATETIME |
| Boolean handling | Non-nullable → NUMERIC (0/1); nullable → CATEGORICAL ("True"/"False"/"<NA>") |
| Datetime handling | Mean-centered Unix timestamp → digit tokens; roundtrips with added time component |
| Integer handling | Digit tokens; roundtrips as `int64` |
| Missing numeric | `@` token per digit position; model learns to generate |
| Missing categorical | `<NA>` token; model learns to generate |
| Conditional missingness | Architecturally possible but **not well captured empirically** |
| Sampling validation | Rejects rows violating non-null constraints from training |

---

## TabDiff

**Implementation**: Vendored into `vendor/tabdiff/`  
**Paper**: Xu et al., "TabDiff: a Multi-Modal Diffusion Model for Tabular Data Generation" (2024)  
**Code**: [MinkaiXu/TabDiff](https://github.com/MinkaiXu/TabDiff)

TabDiff is a continuous-time diffusion model that uses **separate noise processes** for numerical and categorical columns: Gaussian diffusion for continuous features, and a masking (absorbing) process for discrete features. It requires a designated target column.

### Categorical Encoding

**Method: Ordinal integer encoding** for storage, **one-hot with [MASK] token** during diffusion.

1. **Preprocessing**: Each categorical column is mapped to integers 0..C-1 via `OrdinalEncoder`.
2. **During diffusion**: Categories are represented as one-hot vectors with an extra [MASK] dimension (C+1 total), where [MASK] represents the fully-noised state.
3. **Forward process**: A masking/absorbing process probabilistically flips categories to [MASK] based on a log-linear noise schedule. At $t=0$ the category is intact; at $t=1$ everything is [MASK].
4. **Reverse process**: The score network predicts logits over all C+1 states, and sampling progressively unmasks categories.

### Numeric Encoding

**Method: Quantile transformation** to standard normal.

Continuous columns are normalized using `QuantileTransformer(output_distribution='normal')` from scikit-learn, which maps each feature's marginal distribution to $\mathcal{N}(0, 1)$. The number of quantiles is adaptive: $\max(\min(n/30, 1000), 10)$.

The forward diffusion process adds Gaussian noise with a learnable power-mean schedule:
$$x_t = x_0 + \sigma(t) \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Integer columns are detected (by checking if all values are whole numbers) and can optionally be dequantized before diffusion, then rounded back after sampling. The inverse quantile transform recovers original-scale values.

### Supported Types

TabDiff natively supports only **numerical** and **categorical** columns. Column types must be explicitly specified via index lists (`num_col_idx`, `cat_col_idx`, `target_col_idx`) in a JSON metadata file.

| pandas dtype | TabDiff type | Handling |
|---|---|---|
| `float64` | numerical | Quantile transform → Gaussian diffusion |
| `int64` | numerical | Same, with optional dequantization; detected as integer for rounding |
| `object` (string) | categorical | Ordinal encoding → masking diffusion |
| `bool` | — | Not natively supported |
| `datetime64` | — | Not natively supported |

### Missing Values

**Numeric**: Imputed with **column mean** before any transformation. No indicator column. The model never sees that a value was missing — it is replaced with a plausible-looking number. The paper states: "numerical missing values being replaced by the column average."

**Categorical**: NaN is replaced with the string `"nan"`, which is treated as **an ordinary category** — it gets its own ordinal index and participates in the masking diffusion process. The paper states: "categorical missing values being treated as a new category."

**Important caveat**: The `"nan"` string survives in the output as a literal categorical value. It is not converted back to actual `None`/`NaN` — the model generates the string `"nan"` as just another category.

**Conditional missingness**: For numeric columns, missingness information is destroyed by mean imputation — the model cannot learn conditional missingness patterns. For categorical columns, since `"nan"` is just another category, the model can in principle learn when to generate it conditioned on other features.

### Type Roundtrip Behavior

| Input dtype | Output dtype | Survives? | Notes |
|---|---|---|---|
| `float64` | `float64` | Yes | Quantile-transformed and back |
| `int64` | `float64` | **No** | Comes back as float (continuous diffusion output) |
| `object` (string) | `object` | Yes | |
| `bool` | — | **No** | Not natively supported |
| `datetime64` | — | **No** | Not natively supported |
| Nullable numeric | `float64` | Partial | NaN → mean imputation, no indicator; NaN not restored |
| Nullable categorical | `object` | Partial | NaN becomes literal string `"nan"` in output |

### Summary

| Aspect | TabDiff |
|---|---|
| Architecture | Continuous-time diffusion (joint numerical + categorical) |
| Categorical encoding | Ordinal integers → one-hot + [MASK] token during diffusion |
| Categorical diffusion | Masking/absorbing process with log-linear schedule |
| Numeric encoding | QuantileTransformer → standard normal |
| Numeric diffusion | Gaussian noise with learnable power-mean schedule |
| Types recognized | Numerical vs. categorical (binary) |
| Boolean handling | Not natively supported |
| Datetime handling | Not natively supported |
| Integer handling | Treated as continuous; not rounded back by default |
| Missing numeric | Mean imputation (no indicator) |
| Missing categorical | `"nan"` string as ordinary category |
| Conditional missingness | **Not natively preserved** for numeric (mean imputation destroys signal); categorical `"nan"` can capture conditional patterns |
| Requires target column | Yes |

---

## MostlyAI (TabularARGN)

**Implementation**: `mostlyai-engine` library (`TabularARGN` class)  
**Paper**: Mostly AI, "Generative Tabular AI for Enterprise Data" (2025) — model called TabularARGN in the paper  
**Code**: [mostly-ai/mostlyai-engine](https://github.com/mostly-ai/mostlyai-engine)

TabularARGN is an any-order autoregressive neural network that reduces **every column to one or more categorical sub-columns**, then learns conditional probabilities $\hat{p}(x_i \mid x_{<i})$ via learned embeddings. The feature ordering is shuffled per batch (any-order training), so the model can condition on any subset of features at generation time.

### Categorical Encoding

**Method: Integer token codes** with learned embeddings.

Each categorical column is mapped to integer codes via a value-to-code dictionary. Special tokens:
- `"<<NULL>>"` — explicit null token for missing values
- `"_RARE_"` — privacy-safe bucket for values appearing in fewer than ~5 training entities

The embedding dimension for each sub-column is adaptive: $3 \cdot C^{0.25}$ where $C$ is the cardinality. During generation, the model outputs a probability distribution over all codes, and a value is sampled.

### Numeric Encoding

TabularARGN maps every numeric column to one or more **categorical sub-columns** via one of three strategies, automatically selected:

**1. Numeric-Discrete** (when <100 distinct values and >99.9% non-rare):
- Each unique value becomes a category token. Reuses the categorical encoder.

**2. Numeric-Digit** (when ≤3 significant decimal digits):
- Splits the number into individual digit positions (E18 down to E-8).
- Each digit position is a separate sub-column with cardinality 0–9.
- Adds binary `nan` and `neg` flag sub-columns.
- Example: `42.5` → digit sub-columns for tens=4, units=2, decimal=5, plus nan=0, neg=0.

**3. Numeric-Binned** (default for many distinct values):
- Up to 100 **quantile-based** bins.
- The bin index becomes a categorical sub-column.
- During generation, a uniform random value is sampled within the selected bin interval, preserving within-bin variation.
- Precision is maintained by truncating to the observed minimum decimal precision.

The auto-selection heuristic chooses discrete → digit → binned in order of preference.

### Supported Types

TabularARGN has the broadest native type support of all models reviewed:

| Type | Encoding | Sub-columns |
|---|---|---|
| Integer | numeric (auto: discrete/digit/binned) | 1 (discrete/binned) or many (digit) |
| Float | numeric (auto: discrete/digit/binned) | 1 (discrete/binned) or many (digit) |
| String/object | categorical | 1 |
| Boolean | categorical (True/False as 2 codes) | 1 |
| Datetime | decomposed into year/month/day/hour/min/sec/ms | up to 9 sub-columns |
| Date | decomposed (without time components) | up to 3 sub-columns |

**Boolean handling**: Treated as categorical with 2 categories (`True`, `False`). No special boolean type — output dtype is `string[python]`, not `bool`.

**Datetime handling**: Decomposed into calendar/clock components (year, month, day, hour, minute, second, ms digits). Each component is encoded as a numeric-digit sub-column with its own cardinality. This preserves the hierarchical structure of dates without the precision loss of Unix timestamp conversion.

### Missing Values

**Method: Explicit missingness modeling** — the most complete of all reviewed models.

The paper states: "missing or empty values are treated as separate categories and are faithfully reproduced in the synthetic data."

| Column type | Missing representation | Details |
|---|---|---|
| Categorical | `"<<NULL>>"` token | A dedicated code in the vocabulary; model learns when to generate it |
| Numeric (discrete) | `"<<NULL>>"` token | Same as categorical — null is its own category |
| Numeric (binned) | `"<<NULL>>"` token | Same — null bin separate from value bins |
| Numeric (digit) | Binary `nan` flag sub-column | NaN values are imputed from the empirical value distribution for digit encoding, but a separate nan indicator sub-column tells the model the value was missing |
| Datetime | Binary `nan` flag sub-column | Same as numeric digit — impute + indicator |

**Key design**: For digit and datetime encodings, the nan indicator is a **separate sub-column that the model predicts autoregressively**, so it can learn conditional missingness patterns based on previously generated features.

**Conditional missingness**: Empirically tested with A=x → B always null (34% overall null rate). Results:
- A=x: **82.24%** null rate (input: 100%)
- A=y/z: **17.82%** null rate (input: 0%)

This is dramatically better than CTGAN/TVAE (~34% uniform) and demonstrates that the explicit nan indicator allows the model to learn conditional missingness patterns. The correlation is not perfectly captured (82% vs 100%, 18% vs 0%), likely due to the small training set and limited epochs, but the directional signal is strong.

### Type Roundtrip Behavior

| Input dtype | Output dtype | Survives? | Notes |
|---|---|---|---|
| `float64` | `Float64` | Partial | Upcasted to nullable float variant |
| `int64` | `Int64` | Partial | Upcasted to nullable int variant |
| `Int64` (nullable) | `Int64` | Yes | |
| `object` (string) | `string[python]` | Partial | Upcasted to pandas StringDtype |
| `bool` | `string[python]` | **No** | Output is string `"True"`/`"False"`, not `bool` |
| `boolean` (nullable) | `string[python]` | **No** | Same — string output |
| `datetime64[ns]` | `datetime64[ns]` | Yes | Clean roundtrip, no spurious time components |

Notable: MostlyAI outputs nullable pandas dtypes (`Int64`, `Float64`, `string[python]`) rather than standard numpy dtypes. Booleans are consistently returned as strings. Float columns with few distinct values may be auto-selected as discrete encoding and come back as `Int64` (rounded).

### Summary

| Aspect | MostlyAI (TabularARGN) |
|---|---|
| Architecture | Any-order autoregressive neural network |
| Categorical encoding | Integer token codes with learned embeddings |
| Numeric encoding | Auto-selected: discrete (tokens) / digit (per-digit sub-columns) / binned (quantile bins + uniform residual) |
| Types recognized | Categorical, numeric, boolean, datetime, date, character, lat/long |
| Boolean handling | Categorical (2 codes); output as `string[python]` |
| Datetime handling | Decomposed into year/month/day/hour/min/sec/ms sub-columns |
| Integer handling | Auto-selected encoding; roundtrips as `Int64` |
| Missing categorical | Explicit `"<<NULL>>"` token |
| Missing numeric | Explicit `"<<NULL>>"` token (discrete/binned) or nan indicator sub-column (digit) |
| Missing datetime | nan indicator sub-column |
| Conditional missingness | **Well preserved** (82% vs 18% on test, input 100% vs 0%) |
| Rare value protection | Values below privacy threshold bucketed as `"_RARE_"` |

---

## GReaT

**Implementation**: `be_great` library  
**Paper**: Borisov et al., "Language Models are Realistic Tabular Data Generators" (ICLR 2023)  
**Code**: [tabularis-ai/be_great](https://github.com/tabularis-ai/be_great)

GReaT fine-tunes a pretrained GPT-2 language model on natural-language serializations of tabular rows. It is the simplest approach of all reviewed models: zero data preprocessing, no custom vocabulary, no normalization. The paper emphasizes: "zero data preprocessing, e.g., feature names and values are used as they are presented in original data sets."

### Categorical Encoding

**Method: Literal text strings** — no encoding at all.

Each categorical value is converted to a string via `str(value)` and inserted directly into the text. The pretrained GPT-2 BPE tokenizer handles all subword decomposition. There are no special tokens, no vocabulary construction, no ordinal mapping. High-cardinality categoricals receive no special treatment.

The serialization format for each row is:

```
column_name is value, column_name is value, ...
```

Columns are **randomly shuffled** per row during training, so the model learns to generate features in any order (similar in spirit to MostlyAI's any-order approach).

### Numeric Encoding

**Method: Text representation** — numbers become character sequences.

Numeric values are converted to strings via `str(value)`, optionally with configurable `float_precision` to control decimal places. No normalization, scaling, binning, or discretization. The paper states: "continuous and discrete numerical values are represented as character sequences" and "transformer-based models are capable of understanding and processing numerical data encoded in such a way."

During post-processing, a regex extracts numeric values from generated text (`re.search(r'-?\d+\.?\d*', value)`), and `pd.to_numeric(errors='coerce')` validates them. Rows with unparseable numbers are rejected.

### Supported Types

GReaT has a **binary type system**: numeric (detected via `pd.select_dtypes(include=np.number)`) vs. everything else (treated as categorical text).

| Type | GReaT handling |
|---|---|
| Integer | Numeric — `str(value)` text |
| Float | Numeric — `str(value)` text, optional precision |
| String/object | Categorical — `str(value)` text |
| Boolean | **Not recognized** — `str()` → `"True"`/`"False"` text, treated as categorical |
| Datetime | **Not recognized** — `str()` → timestamp string text, treated as categorical |

Booleans come back as strings `"True"`/`"False"`. Datetimes come back as strings with potentially inconsistent formatting (the LLM generates date-like text that may not parse cleanly, e.g., `"2020"` instead of `"2020-01-01 00:00:00"`).

### Missing Values

GReaT **cannot generate missing values natively** in practice, despite a claimed fix ([tabularis-ai/be_great#39](https://github.com/tabularis-ai/be_great/issues/39)).

There are two code paths for serializing NaN during training, depending on the underlying Python type:

| Null type | Stored as | `str()` produces | During training |
|---|---|---|---|
| `np.nan` (numeric) | `float('nan')` | `"nan"` | Included as `"B is nan"` |
| `None` (categorical) | `None` | `"None"` | Included as `"A is None"` |

Both are included in the training text (via `GReaTDataset._format_value` which calls `str(value)` on all values without NaN-checking). The model learns both patterns.

However, during **sampling**, the parsing pipeline blocks both paths:

1. **Omitted columns**: If the model generates a short sequence where a column is absent, that column stays as `"placeholder"` and **the entire row is rejected**.

2. **Numeric `"nan"`**: If the model generates `"B is nan"`, the string `"nan"` passes the placeholder filter but fails the numeric validation. `pd.to_numeric("nan", errors="coerce")` produces actual NaN, but the guard `df[col].isna()` checks the *pre-coercion string* `"nan"`, which is not null → **row rejected**. The fix from issue #39 (`coerced.notnull() | df[col].isna()`) has a bug: it should check `.isna()` on the coerced series, not the original string column.

3. **Categorical `"None"`**: If the model generates `"A is None"`, `df.replace("None", None)` converts it to actual Python `None` → `isna()` returns True → **this path works**. However, this was not observed in our tests, likely because categorical NaN in pandas is stored as `np.nan`/`float('nan')` (not Python `None`), so training text uses `"nan"` rather than `"None"`.

**Net result**: Empirically, 0% null rate in all output columns, even with 30% input nulls.

**Additional consequence** — **categories associated with missingness are suppressed**: in the conditional missingness test (A=x always paired with B=NaN), the model generated **zero rows with A=x** out of 200 samples. The model likely generates `"A is x, B is nan"` or short `"A is x"` sequences, but both are rejected — eliminating the category from the output entirely.

The paper shows `"Gender is Missing"` in an example figure, suggesting the original design contemplated explicit missing tokens, but the code uses `str(NaN)` → `"nan"` instead, and the parsing pipeline rejects it.

### Type Roundtrip Behavior

| Input dtype | Output dtype | Survives? | Notes |
|---|---|---|---|
| `float64` | `float64` | Yes | Parsed via `pd.to_numeric` |
| `int64` | `float64` | **No** | All numerics come back as float |
| `object` (string) | `object` | Yes | |
| `bool` | `object` | **No** | Comes back as string `"True"`/`"False"` |
| `datetime64[ns]` | `object` | **No** | Comes back as string with inconsistent formatting |
| Nullable numeric | `float64` | Partial | NaN never generated (always filled) |
| Nullable categorical | `object` | **No** | NaN never generated |

### Summary

| Aspect | GReaT |
|---|---|
| Architecture | Fine-tuned GPT-2 (autoregressive LM) |
| Categorical encoding | Literal text strings, GPT-2 BPE tokenization |
| Numeric encoding | Text representation (`str(value)`), no normalization |
| Types recognized | Numeric vs. categorical (binary) |
| Boolean handling | Not recognized; becomes string `"True"`/`"False"` |
| Datetime handling | Not recognized; becomes string with potentially inconsistent format |
| Integer handling | Text; roundtrips as `float64` |
| Missing values | **Cannot generate natively** — NaN columns included in training text but rejected during parsing |
| Conditional missingness | **Not preserved** — 0% nulls in output; associated categories suppressed |
| Tokenizer | Pretrained GPT-2 BPE (no custom vocabulary) |
| Invalid sample handling | Rejection sampling — unparseable rows discarded |

---

## Tabby

**Implementation**: Standalone scripts (not a Python package)  
**Paper**: Ceritli et al., "Tabby: Tabular Data Synthesis with the Plain Method" (2025)  
**Code**: [soCromp/tabby](https://github.com/soCromp/tabby)

Tabby extends GReaT's approach of fine-tuning a pretrained LLM on text-serialized tabular data, but introduces the "Plain" method — a simpler serialization using `<EOC>` (end-of-column) delimiter tokens instead of commas. It can optionally use GReaT under the hood (via a `-g` flag). Like GReaT, columns are randomly shuffled per row during training.

Note: Tabby is not integrated into origami-jsynth as a Python package (it operates on CSV files via scripts), so no empirical roundtrip tests were performed. Analysis is based on code review.

### Categorical Encoding

**Method: Literal text strings** (same as GReaT).

Values are serialized via `str(value)`. Optional ordinal encoding (via scikit-learn `LabelEncoder`, enabled with `-ec` flag) converts categories to integer indices, but the default is literal strings.

### Numeric Encoding

**Method: Text representation** (same as GReaT).

Numbers are converted via `str(value)` with no rounding, normalization, or discretization in the Plain method. When using GReaT mode (`-g`), numeric start tokens use 5 decimal places.

### Supported Types

Same binary type system as GReaT: numeric (int/float detected via pandas dtype) vs. everything else (categorical). No native boolean or datetime support.

### Missing Values

Tabby improves slightly on GReaT's missing value handling, but still **cannot generate missing values** in the output.

**Preprocessing**:
- **Categorical NaN**: Replaced with the literal string `"?"` before serialization. The model sees and can learn to generate `"?"`.
- **Numeric NaN**: Left as-is. `str(np.nan)` → `"nan"`, so the model sees `"col is nan<EOC>"`.

This is better than GReaT's training-time serialization (which in some code paths omits NaN columns entirely), because the model does see explicit missing-value tokens during training.

**Sampling**: Generated text is parsed by splitting on `<EOC>` and `" is "`. However, **any row where not all columns are present is rejected** (`if set(d.keys()) == cols`). This means:
- If the model generates a row where a column is absent → rejected.
- If the model generates `"col is nan"` → the string `"nan"` is included as a literal value, not converted to actual NaN. For categorical columns, it would similarly produce the string `"?"`.

So Tabby can in principle produce `"nan"` or `"?"` as string values in the output, but these are treated as literal category values rather than proper null/NaN. There is no mechanism to convert them back to actual missing values.

### Differences from GReaT

| Aspect | GReaT | Tabby (Plain) |
|---|---|---|
| Delimiter | Comma (`,`) | `<EOC>` special token |
| Numeric formatting | `str()` or 5-decimal precision | `str()` only |
| Categorical NaN | Omitted or `"nan"` via `str()` | Replaced with `"?"` |
| Numeric NaN | Omitted or `"nan"` via `str()` | `"nan"` via `str()` (left as-is) |
| Multi-head generation | No | Optional (multi-head experts, one per column) |
| GReaT compatibility | N/A | Optional via `-g` flag |

### Summary

| Aspect | Tabby |
|---|---|
| Architecture | Fine-tuned LLM (Plain method or GReaT-compatible) |
| Categorical encoding | Literal text strings (optional ordinal encoding) |
| Numeric encoding | Text representation (`str(value)`), no normalization |
| Types recognized | Numeric vs. categorical (binary) |
| Boolean handling | Not recognized; becomes string |
| Datetime handling | Not recognized; becomes string |
| Missing categorical | Replaced with `"?"` string — model can learn the pattern |
| Missing numeric | Serialized as `"nan"` string — model can learn the pattern |
| Output missing values | **Not restored** — `"nan"` and `"?"` survive as literal strings, not actual NaN |
| Conditional missingness | Not tested; architecturally possible (model sees missing tokens) but output lacks proper null conversion |
| Invalid sample handling | Rows missing any column rejected during parsing |

---

## Cross-Model Comparison

### Categorical Encoding

| Model | Method |
|---|---|
| CTGAN | One-hot encoding (via RDT `OneHotEncoder`) |
| TVAE | One-hot encoding (same as CTGAN) |
| REaLTabFormer | One token per unique value in custom vocabulary |
| TabDiff | Ordinal integers → one-hot + [MASK] during diffusion |
| MostlyAI | Integer token codes with learned embeddings |
| GReaT | Literal text strings, GPT-2 BPE tokenization |
| Tabby | Literal text strings (optional ordinal encoding) |

### Numeric Encoding

| Model | Method | Output dims per column |
|---|---|---|
| CTGAN | Mode-specific normalization (BGM, up to 10 modes), $(x - \mu_k) / (4\sigma_k)$ | $1 + K$ |
| TVAE | Same as CTGAN + learnable reconstruction variance | $1 + K$ |
| REaLTabFormer | Character-level digit tokenization (no normalization) | 1 token per digit position |
| TabDiff | QuantileTransformer → standard normal, Gaussian diffusion | 1 |
| MostlyAI | Auto-selected: discrete tokens / per-digit sub-columns / quantile bins (up to 100) | varies |
| GReaT | `str(value)` text, GPT-2 BPE tokenization | N/A (text) |
| Tabby | `str(value)` text | N/A (text) |

### Supported Types

| Model | Numeric | Categorical | Boolean | Datetime |
|---|---|---|---|---|
| CTGAN | Yes (continuous) | Yes (discrete) | As 2-category discrete | Unix timestamp → continuous |
| TVAE | Yes (continuous) | Yes (discrete) | As 2-category discrete | Unix timestamp → continuous |
| REaLTabFormer | Yes (digit tokens) | Yes (string tokens) | Non-nullable: numeric 0/1; nullable: categorical | Unix timestamp → digit tokens |
| TabDiff | Yes (Gaussian diffusion) | Yes (masking diffusion) | Not supported | Not supported |
| MostlyAI | Yes (auto-selected) | Yes (token codes) | As 2-category categorical | Decomposed into year/month/day/hour/min/sec/ms |
| GReaT | Yes (text) | Yes (text) | As text `"True"`/`"False"` | As text (inconsistent format) |
| Tabby | Yes (text) | Yes (text) | As text | As text |

### Missing Value Handling

| Model | Numeric NaN | Categorical NaN | Model sees missingness? | Generates NaN? | Conditional missingness |
|---|---|---|---|---|---|
| CTGAN | Random imputation (no indicator) | Extra one-hot category | Categorical only | Categorical: yes; numeric: random reinsertion at marginal rate | **Not preserved** (34% uniform) |
| TVAE | Random imputation (no indicator) | Extra one-hot category | Categorical only | Same as CTGAN | **Not preserved** (34% uniform) |
| REaLTabFormer | `@` token per digit | `<NA>` token | Yes (both types) | Yes (via learned tokens) | Architecturally possible; not captured empirically |
| TabDiff | Mean imputation (no indicator) | `"nan"` string as category | Categorical only | Categorical: `"nan"` string; numeric: no | **Not natively preserved** |
| MostlyAI | `<<NULL>>` token or nan indicator sub-column | `<<NULL>>` token | Yes (both types) | Yes (explicit tokens + indicator) | **Well preserved** (82% vs 18%) |
| GReaT | Included as `"nan"` in training text | Included as `"nan"`/`"None"` in training text | Yes, but parsing rejects it | No — rows with missing columns rejected | **Not preserved** (0% nulls; associated categories suppressed) |
| Tabby | Included as `"nan"` in training text | Replaced with `"?"` | Yes (both types) | As literal strings only (`"nan"`, `"?"`) — not proper NaN | Not tested |

### Type Roundtrip

Whether input dtypes survive the `fit()` → `sample()` cycle (native model behavior, not our wrapper).

| Model | `float64` | `int64` | `bool` | `datetime64` | Nullable types |
|---|---|---|---|---|---|
| CTGAN | ✓ | ✓ (via rounding) | ✓ | ✓ | NaN reintroduced probabilistically |
| TVAE | ✓ | ✓ (via rounding) | ✓ | ✓ | NaN reintroduced probabilistically |
| REaLTabFormer | ✓ | ✓ | ✓ (non-nullable only) | ✓ (gains time component) | Nullable int/bool degrade to string |
| TabDiff | ✓ | ✗ (→ float) | Not supported | Not supported | Numeric NaN not restored; categorical NaN becomes `"nan"` string |
| MostlyAI | ✓ (→ `Float64`) | ✓ (→ `Int64`) | ✗ (→ string) | ✓ | Clean nullable dtype output |
| GReaT | ✓ | ✗ (→ float) | ✗ (→ string) | ✗ (→ string) | NaN never generated |
| Tabby | N/A (CSV) | N/A (CSV) | N/A (CSV) | N/A (CSV) | Missing values as literal strings |
