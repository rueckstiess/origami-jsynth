"""Fidelity evaluation for semi-structured JSON data.

Column Shapes (per-field) uses a factorized model:
  score(field) = presence × type × value

Where:
  - presence: 1 - |rate_real - rate_synth|
  - type: TVComplement on dtype distribution (conditioned on presence)
  - value: KS/TV per type (conditioned on type), or length for arrays

Column Pair Trends operates on the type-separated union table, pairing all
columns and weighting by co-occurrence rate.

Overall score = (column_shapes + column_pair_trends) / 2.

Both column shapes and pair trends computations are embarrassingly parallel
and can be distributed across multiple processes via ``max_workers``.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Shared state for forked worker processes.
# Set by the parent before forking; read-only in workers (copy-on-write).
# ---------------------------------------------------------------------------
_SHARED: dict[str, Any] = {}


@dataclass
class FidelityResult:
    """Results from fidelity evaluation."""

    overall_score: float
    column_shapes_score: float
    column_pair_trends_score: float
    field_scores: dict[str, float]
    field_details: dict[str, dict[str, Any]]
    field_weights: dict[str, float] = dataclass_field(default_factory=dict)
    pair_scores: dict[str, float] = dataclass_field(default_factory=dict)
    pair_details: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)
    pair_weights: dict[str, float] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "column_shapes_score": self.column_shapes_score,
            "column_pair_trends_score": self.column_pair_trends_score,
            "field_scores": self.field_scores,
            "field_details": self.field_details,
            "field_weights": self.field_weights,
            "num_fields": len(self.field_weights),
            "pair_scores": self.pair_scores,
            "pair_details": self.pair_details,
            "pair_weights": self.pair_weights,
            "num_pairs": len(self.pair_weights),
        }

    @property
    def num_fields(self) -> int:
        """Number of fields in Column Shapes aggregation."""
        return len(self.field_weights)

    @property
    def num_pairs(self) -> int:
        """Number of column pairs in Column Pair Trends aggregation."""
        return len(self.pair_weights)

    def format_breakdown(self, top_n: int = 10) -> str:
        """Format a human-readable breakdown of worst-scoring fields and pairs.

        Args:
            top_n: Number of worst entries to show per section

        Returns:
            Formatted string showing score breakdown
        """
        lines = [
            f"Overall Score: {self.overall_score:.4f}",
            f"  Column Shapes:      {self.column_shapes_score:.4f} ({self.num_fields} fields)",
            f"  Column Pair Trends: {self.column_pair_trends_score:.4f} ({self.num_pairs} pairs)",
        ]

        # Worst fields by cost
        field_breakdown = []
        for field, score in self.field_scores.items():
            weight = self.field_weights.get(field, 0.0)
            cost = (1.0 - score) * weight
            field_breakdown.append((field, score, weight, cost))
        field_breakdown.sort(key=lambda x: x[3], reverse=True)

        lines.append("")
        lines.append(f"Worst {min(top_n, len(field_breakdown))} fields (Column Shapes):")
        for field, score, weight, cost in field_breakdown[:top_n]:
            details = self.field_details.get(field, {})
            parts = []
            if "presence_fidelity" in details:
                parts.append(f"pres={details['presence_fidelity']:.2f}")
            if "type_fidelity" in details:
                parts.append(f"type={details['type_fidelity']:.2f}")
            if "value_fidelity" in details:
                parts.append(f"val={details['value_fidelity']:.2f}")
            if "length_fidelity" in details:
                parts.append(f"len={details['length_fidelity']:.2f}")
            detail_str = f" [{', '.join(parts)}]" if parts else ""
            lines.append(
                f"  {field}: {score:.4f} (weight={weight:.1%}, cost={cost:.4f}){detail_str}"
            )

        # Worst pairs by cost
        pair_breakdown = []
        for pair_key, score in self.pair_scores.items():
            weight = self.pair_weights.get(pair_key, 0.0)
            cost = (1.0 - score) * weight
            pair_breakdown.append((pair_key, score, weight, cost))
        pair_breakdown.sort(key=lambda x: x[3], reverse=True)

        lines.append("")
        lines.append(f"Worst {min(top_n, len(pair_breakdown))} pairs (Column Pair Trends):")
        for pair_key, score, weight, cost in pair_breakdown[:top_n]:
            details = self.pair_details.get(pair_key, {})
            metric = details.get("metric", "")
            lines.append(
                f"  {pair_key}: {score:.4f} (weight={weight:.1%}, cost={cost:.4f}) [{metric}]"
            )

        return "\n".join(lines)


def _tv_complement(real: pd.Series, synth: pd.Series) -> float:
    """Total Variation Complement for categorical distributions.

    Returns score in [0, 1] where 1 means identical distributions.
    """
    real_counts = real.value_counts(normalize=True)
    synth_counts = synth.value_counts(normalize=True)
    real_aligned, synth_aligned = real_counts.align(synth_counts, fill_value=0.0)
    tv_distance = (real_aligned - synth_aligned).abs().sum()
    return 1.0 - 0.5 * tv_distance


def _ks_complement(real: pd.Series, synth: pd.Series) -> float:
    """Kolmogorov-Smirnov Complement for continuous distributions.

    Returns score in [0, 1] where 1 means identical distributions.
    """
    real_arr = pd.to_numeric(real, errors="coerce").dropna().values
    synth_arr = pd.to_numeric(synth, errors="coerce").dropna().values

    if len(real_arr) == 0 or len(synth_arr) == 0:
        return 0.0

    ks_stat, _ = stats.ks_2samp(real_arr, synth_arr)
    return 1.0 - ks_stat


def _compare_values(real: pd.Series, synth: pd.Series, dtype: str) -> float:
    """Compare value distributions for a given type.

    Args:
        real: Values from real data
        synth: Values from synthetic data
        dtype: Type of values ("num", "cat", "bool", "date")

    Returns:
        Score in [0, 1] where 1 means identical distributions.
    """
    if len(real) == 0 or len(synth) == 0:
        return 0.0

    if dtype in ("cat", "bool"):
        return _tv_complement(real, synth)

    # For dates, convert to numeric timestamps
    if dtype == "date":
        real = pd.to_datetime(real).astype(np.int64)
        synth = pd.to_datetime(synth).astype(np.int64)

    return _ks_complement(real, synth)


def _compute_field_fidelity(
    df: pd.DataFrame,
    field: str,
    real_mask: pd.Series,
    synth_mask: pd.Series,
) -> dict[str, Any]:
    """Compute fidelity for a single field.

    Returns dict with presence_fidelity, type_fidelity, value_fidelity (or
    length_fidelity for arrays), and combined score.
    """
    dtype_col = f"{field}.dtype"
    if dtype_col not in df.columns:
        return {"score": 1.0, "error": "no dtype column"}

    n_real, n_synth = real_mask.sum(), synth_mask.sum()
    if n_real == 0 or n_synth == 0:
        return {"score": 0.0, "error": "empty split"}

    real_dtypes = df.loc[real_mask, dtype_col]
    synth_dtypes = df.loc[synth_mask, dtype_col]

    # Presence fidelity ("missing" means absent)
    real_rate = (real_dtypes != "missing").mean()
    synth_rate = (synth_dtypes != "missing").mean()
    presence_fidelity = 1.0 - abs(real_rate - synth_rate)

    # Type fidelity (conditioned on presence — exclude "missing")
    real_present = real_dtypes[real_dtypes != "missing"]
    synth_present = synth_dtypes[synth_dtypes != "missing"]

    if len(real_present) == 0 or len(synth_present) == 0:
        type_fidelity = 0.0
    else:
        type_fidelity = _tv_complement(real_present, synth_present)

    result: dict[str, Any] = {
        "presence_fidelity": presence_fidelity,
        "type_fidelity": type_fidelity,
        "real_presence_rate": float(real_rate),
        "synth_presence_rate": float(synth_rate),
    }

    # Determine field type
    all_types = set(real_present) | set(synth_present)
    is_array = "array" in all_types
    is_object = "object" in all_types and not is_array
    value_types = {"num", "cat", "bool", "date"}
    has_values = bool(all_types & value_types)

    # Value fidelity (for leaf fields with value-bearing types)
    if has_values and not is_array:
        value_fidelity, value_details = _compute_value_fidelity(
            df, field, dtype_col, real_mask, synth_mask
        )
        result["value_fidelity"] = value_fidelity
        result["value_details"] = value_details

    # Length fidelity (for array fields)
    if is_array:
        result["length_fidelity"] = _compute_length_fidelity(
            df, field, dtype_col, real_mask, synth_mask
        )

    # Combined score via chain rule
    if is_array:
        score = presence_fidelity * type_fidelity * result.get("length_fidelity", 1.0)
    elif is_object:
        score = presence_fidelity * type_fidelity
    elif has_values:
        score = presence_fidelity * type_fidelity * result.get("value_fidelity", 1.0)
    else:
        score = presence_fidelity * type_fidelity

    result["score"] = score
    return result


def _compute_value_fidelity(
    df: pd.DataFrame,
    field: str,
    dtype_col: str,
    real_mask: pd.Series,
    synth_mask: pd.Series,
) -> tuple[float, dict[str, Any]]:
    """Compute value fidelity across all value-bearing types.

    Returns (weighted_score, details_dict).
    """
    value_types = ["num", "cat", "bool", "date"]

    # Count values by type (pooled for symmetric weighting)
    type_counts = {}
    for t in value_types:
        real_count = ((df[dtype_col] == t) & real_mask).sum()
        synth_count = ((df[dtype_col] == t) & synth_mask).sum()
        type_counts[t] = real_count + synth_count

    total = sum(type_counts.values())
    if total == 0:
        return 1.0, {}

    value_fidelity = 0.0
    details: dict[str, Any] = {}

    for t in value_types:
        if type_counts[t] == 0:
            continue

        weight = type_counts[t] / total
        value_col = f"{field}.{t}"

        if value_col not in df.columns:
            value_fidelity += weight * 0.0
            details[t] = {"weight": weight, "score": 0.0, "error": "no column"}
            continue

        t_real_mask = real_mask & (df[dtype_col] == t)
        t_synth_mask = synth_mask & (df[dtype_col] == t)

        real_vals = df.loc[t_real_mask, value_col].dropna()
        synth_vals = df.loc[t_synth_mask, value_col].dropna()

        t_score = _compare_values(real_vals, synth_vals, t)
        value_fidelity += weight * t_score
        details[t] = {"weight": weight, "score": t_score}

    return value_fidelity, details


def _compute_length_fidelity(
    df: pd.DataFrame,
    field: str,
    dtype_col: str,
    real_mask: pd.Series,
    synth_mask: pd.Series,
) -> float:
    """Compute length distribution fidelity for array fields."""
    alen_col = f"{field}.alen"
    if alen_col not in df.columns:
        return 1.0

    array_real = real_mask & (df[dtype_col] == "array")
    array_synth = synth_mask & (df[dtype_col] == "array")

    real_lengths = df.loc[array_real, alen_col].dropna()
    synth_lengths = df.loc[array_synth, alen_col].dropna()

    if len(real_lengths) == 0 and len(synth_lengths) == 0:
        return 1.0
    if len(real_lengths) == 0 or len(synth_lengths) == 0:
        return 0.0

    return _ks_complement(real_lengths, synth_lengths)


# ---------------------------------------------------------------------------
# Worker functions for parallel execution (used with fork context)
# ---------------------------------------------------------------------------


def _field_fidelity_task(field: str) -> tuple[str, dict[str, Any]]:
    """Worker: compute fidelity for one field using inherited shared state."""
    return field, _compute_field_fidelity(
        _SHARED["df"], field, _SHARED["real_mask"], _SHARED["synth_mask"]
    )


def _pair_batch_task(pairs: list[tuple[int, int]]) -> list[tuple]:
    """Worker: compute scores for a batch of column pairs.

    Returns list of (pair_key, score, metric, details_dict, weight).
    """
    columns = _SHARED["columns"]
    notna_masks = _SHARED["notna_masks"]
    col_arrays = _SHARED["col_arrays"]
    str_arrays = _SHARED["str_arrays"]
    real_arr = _SHARED["real_arr"]
    synth_arr = _SHARED["synth_arr"]
    n_real = _SHARED["n_real"]
    n_synth = _SHARED["n_synth"]
    sep = "\x00"

    results = []
    for i, j in pairs:
        col_a, col_b = columns[i], columns[j]
        a_continuous = _is_continuous(col_a)
        b_continuous = _is_continuous(col_b)
        pair_key = f"{col_a}|{col_b}"

        both_present = notna_masks[col_a] & notna_masks[col_b]
        real_co = real_arr & both_present
        synth_co = synth_arr & both_present

        n_real_co = real_co.sum()
        n_synth_co = synth_co.sum()
        co_rate_real = n_real_co / n_real if n_real > 0 else 0.0
        co_rate_synth = n_synth_co / n_synth if n_synth > 0 else 0.0
        weight = max(co_rate_real, co_rate_synth)

        if n_real_co == 0 or n_synth_co == 0:
            results.append(
                (
                    pair_key,
                    0.0,
                    "N/A",
                    {
                        "score": 0.0,
                        "metric": "N/A",
                        "co_rate_real": co_rate_real,
                        "co_rate_synth": co_rate_synth,
                        "error": "no co-present rows on one side",
                    },
                    weight,
                )
            )
            continue

        if not a_continuous and not b_continuous:
            real_keys = str_arrays[col_a][real_co] + sep + str_arrays[col_b][real_co]
            synth_keys = str_arrays[col_a][synth_co] + sep + str_arrays[col_b][synth_co]
            score = _tv_complement_from_arrays(real_keys, synth_keys)
            metric = "ContingencySimilarity"
        else:
            real_a_s = pd.Series(col_arrays[col_a][real_co])
            real_b_s = pd.Series(col_arrays[col_b][real_co])
            synth_a_s = pd.Series(col_arrays[col_a][synth_co])
            synth_b_s = pd.Series(col_arrays[col_b][synth_co])
            score, metric = _compute_pair_score(
                real_a_s,
                real_b_s,
                synth_a_s,
                synth_b_s,
                a_continuous,
                b_continuous,
            )

        results.append(
            (
                pair_key,
                score,
                metric,
                {
                    "score": score,
                    "metric": metric,
                    "co_rate_real": co_rate_real,
                    "co_rate_synth": co_rate_synth,
                    "n_real": int(n_real_co),
                    "n_synth": int(n_synth_co),
                },
                weight,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Column Pair Trends
# ---------------------------------------------------------------------------

_CONTINUOUS_SUFFIXES = (".num", ".date", ".alen")
_DISCRETE_SUFFIXES = (".dtype", ".cat", ".bool")


def _is_continuous(col: str) -> bool:
    return any(col.endswith(s) for s in _CONTINUOUS_SUFFIXES)


def _is_discrete(col: str) -> bool:
    return any(col.endswith(s) for s in _DISCRETE_SUFFIXES)


def _has_variance(series: pd.Series) -> bool:
    """Check if a series has more than one unique non-NaN value."""
    return series.dropna().nunique() > 1


def _correlation_similarity(
    real_a: pd.Series,
    real_b: pd.Series,
    synth_a: pd.Series,
    synth_b: pd.Series,
) -> float:
    """Compare Pearson correlation between real and synthetic column pairs.

    Returns score in [0, 1] where 1 means identical correlations.
    """
    if len(real_a) < 2 or len(synth_a) < 2:
        return 0.0

    # Check for constant columns to avoid numpy RuntimeWarning from corrcoef
    if (
        real_a.nunique() < 2
        or real_b.nunique() < 2
        or synth_a.nunique() < 2
        or synth_b.nunique() < 2
    ):
        return 0.0

    real_corr = np.corrcoef(real_a.values, real_b.values)[0, 1]
    synth_corr = np.corrcoef(synth_a.values, synth_b.values)[0, 1]

    if np.isnan(real_corr) or np.isnan(synth_corr):
        return 0.0

    return 1.0 - abs(real_corr - synth_corr) / 2.0


def _contingency_similarity(
    real_a: pd.Series,
    real_b: pd.Series,
    synth_a: pd.Series,
    synth_b: pd.Series,
) -> float:
    """Compare joint categorical distributions (2D TV complement).

    Returns score in [0, 1] where 1 means identical joint distributions.
    """
    if len(real_a) == 0 or len(synth_a) == 0:
        return 0.0

    # Use string concatenation with null byte separator to create composite keys
    # (avoids materializing N Python tuples + pd.Series overhead)
    sep = "\x00"
    real_keys = real_a.astype(str).values + sep + real_b.astype(str).values
    synth_keys = synth_a.astype(str).values + sep + synth_b.astype(str).values

    return _tv_complement_from_arrays(real_keys, synth_keys)


def _tv_complement_from_arrays(real_keys: np.ndarray, synth_keys: np.ndarray) -> float:
    """TV complement from pre-built numpy string arrays."""
    real_counts = pd.Series(real_keys).value_counts(normalize=True)
    synth_counts = pd.Series(synth_keys).value_counts(normalize=True)
    real_aligned, synth_aligned = real_counts.align(synth_counts, fill_value=0.0)
    tv_distance = (real_aligned - synth_aligned).abs().sum()
    return 1.0 - 0.5 * tv_distance


def _discretize(
    real_vals: pd.Series, synth_vals: pd.Series, n_bins: int = 10
) -> tuple[pd.Series, pd.Series]:
    """Discretize continuous values into bins using shared edges."""
    # Convert datetimes to numeric (epoch nanos) so histogram_bin_edges works
    if pd.api.types.is_datetime64_any_dtype(real_vals):
        real_vals = pd.to_numeric(real_vals, errors="coerce")
        synth_vals = pd.to_numeric(synth_vals, errors="coerce")

    pooled = pd.concat([real_vals, synth_vals]).dropna()
    if len(pooled) == 0:
        return real_vals, synth_vals

    n_unique = pooled.nunique()
    if n_unique <= 1:
        return real_vals, synth_vals
    n_bins = min(n_bins, n_unique)

    bin_edges = np.histogram_bin_edges(pooled, bins=n_bins)
    real_binned = pd.cut(real_vals, bins=bin_edges, labels=False, include_lowest=True)
    synth_binned = pd.cut(synth_vals, bins=bin_edges, labels=False, include_lowest=True)
    return real_binned, synth_binned


def _compute_pair_score(
    real_a: pd.Series,
    real_b: pd.Series,
    synth_a: pd.Series,
    synth_b: pd.Series,
    a_continuous: bool,
    b_continuous: bool,
) -> tuple[float, str]:
    """Compute trend similarity for a single pair.

    Returns (score, metric_name).
    """
    if a_continuous and b_continuous:
        # Convert dates to numeric timestamps for correlation
        real_a = pd.to_numeric(real_a, errors="coerce")
        real_b = pd.to_numeric(real_b, errors="coerce")
        synth_a = pd.to_numeric(synth_a, errors="coerce")
        synth_b = pd.to_numeric(synth_b, errors="coerce")
        return (
            _correlation_similarity(real_a, real_b, synth_a, synth_b),
            "CorrelationSimilarity",
        )

    if not a_continuous and not b_continuous:
        return (
            _contingency_similarity(real_a, real_b, synth_a, synth_b),
            "ContingencySimilarity",
        )

    # Mixed: discretize the continuous column, then contingency
    if a_continuous:
        real_a, synth_a = _discretize(real_a, synth_a)
    else:
        real_b, synth_b = _discretize(real_b, synth_b)

    return (
        _contingency_similarity(real_a, real_b, synth_a, synth_b),
        "ContingencySimilarity",
    )


def _precompute_pair_arrays(
    df: pd.DataFrame,
    real_mask: pd.Series,
    synth_mask: pd.Series,
) -> tuple[list[str], dict, dict, dict, np.ndarray, np.ndarray, int, int]:
    """Precompute arrays needed for pair trends (shared across workers).

    Returns (columns, notna_masks, col_arrays, str_arrays,
             real_arr, synth_arr, n_real, n_synth).
    """
    n_real = int(real_mask.sum())
    n_synth = int(synth_mask.sum())

    candidates = [
        c for c in df.columns if c != "__split__" and (_is_continuous(c) or _is_discrete(c))
    ]
    columns = [c for c in candidates if _has_variance(df[c])]

    def _presence_mask(series: pd.Series, col_name: str) -> np.ndarray:
        if col_name.endswith(".dtype"):
            return (series != "missing").values
        return series.notna().values

    notna_masks = {col: _presence_mask(df[col], col) for col in columns}
    col_arrays = {col: df[col].values for col in columns}

    str_arrays: dict[str, np.ndarray] = {}
    for col in columns:
        if _is_discrete(col):
            str_arrays[col] = df[col].astype(str).values

    return (
        columns,
        notna_masks,
        col_arrays,
        str_arrays,
        real_mask.values,
        synth_mask.values,
        n_real,
        n_synth,
    )


def _compute_pair_trends(
    df: pd.DataFrame,
    real_mask: pd.Series,
    synth_mask: pd.Series,
    max_workers: int | None = None,
) -> tuple[float, dict[str, float], dict[str, dict[str, Any]], dict[str, float]]:
    """Compute column pair trends for the type-separated DataFrame.

    Returns (overall_score, pair_scores, pair_details, pair_weights).
    """
    (columns, notna_masks, col_arrays, str_arrays, real_arr, synth_arr, n_real, n_synth) = (
        _precompute_pair_arrays(
            df,
            real_mask,
            synth_mask,
        )
    )

    # Generate all (i, j) pairs
    pairs = [(i, j) for i in range(len(columns)) for j in range(i + 1, len(columns))]
    n_pairs = len(pairs)

    if n_pairs == 0:
        return 1.0, {}, {}, {}

    pair_scores: dict[str, float] = {}
    pair_details: dict[str, dict[str, Any]] = {}
    pair_weights: dict[str, float] = {}

    _MIN_PAIRS_FOR_PARALLEL = 20
    use_parallel = max_workers != 1 and n_pairs >= _MIN_PAIRS_FOR_PARALLEL

    if use_parallel:
        # Store precomputed arrays in shared state for forked workers
        _SHARED.update(
            columns=columns,
            notna_masks=notna_masks,
            col_arrays=col_arrays,
            str_arrays=str_arrays,
            real_arr=real_arr,
            synth_arr=synth_arr,
            n_real=n_real,
            n_synth=n_synth,
        )

        batch_size = max(1, min(200, n_pairs // (max_workers or mp.cpu_count()) + 1))
        batches = [pairs[k : k + batch_size] for k in range(0, n_pairs, batch_size)]

        print(
            f"  Column pair trends: {n_pairs} pairs across "
            f"{len(batches)} batches ({max_workers or mp.cpu_count()} workers)..."
        )

        try:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
                done = 0
                for batch_results in pool.map(_pair_batch_task, batches):
                    for pair_key, score, _metric, details, weight in batch_results:
                        pair_scores[pair_key] = score
                        pair_details[pair_key] = details
                        pair_weights[pair_key] = weight
                    done += len(batch_results)
                    print(f"  Column pair trends: {done}/{n_pairs} pairs done")
        finally:
            for key in (
                "columns",
                "notna_masks",
                "col_arrays",
                "str_arrays",
                "real_arr",
                "synth_arr",
                "n_real",
                "n_synth",
            ):
                _SHARED.pop(key, None)
    else:
        # Sequential fallback
        sep = "\x00"
        for idx, (i, j) in enumerate(pairs):
            col_a, col_b = columns[i], columns[j]
            a_continuous = _is_continuous(col_a)
            b_continuous = _is_continuous(col_b)
            pair_key = f"{col_a}|{col_b}"

            both_present = notna_masks[col_a] & notna_masks[col_b]
            real_co = real_arr & both_present
            synth_co = synth_arr & both_present

            n_real_co = real_co.sum()
            n_synth_co = synth_co.sum()
            co_rate_real = n_real_co / n_real if n_real > 0 else 0.0
            co_rate_synth = n_synth_co / n_synth if n_synth > 0 else 0.0
            weight = max(co_rate_real, co_rate_synth)
            pair_weights[pair_key] = weight

            if n_real_co == 0 or n_synth_co == 0:
                pair_scores[pair_key] = 0.0
                pair_details[pair_key] = {
                    "score": 0.0,
                    "metric": "N/A",
                    "co_rate_real": co_rate_real,
                    "co_rate_synth": co_rate_synth,
                    "error": "no co-present rows on one side",
                }
                continue

            if not a_continuous and not b_continuous:
                real_keys = str_arrays[col_a][real_co] + sep + str_arrays[col_b][real_co]
                synth_keys = str_arrays[col_a][synth_co] + sep + str_arrays[col_b][synth_co]
                score = _tv_complement_from_arrays(real_keys, synth_keys)
                metric = "ContingencySimilarity"
            else:
                real_a = pd.Series(col_arrays[col_a][real_co])
                real_b = pd.Series(col_arrays[col_b][real_co])
                synth_a = pd.Series(col_arrays[col_a][synth_co])
                synth_b = pd.Series(col_arrays[col_b][synth_co])
                score, metric = _compute_pair_score(
                    real_a,
                    real_b,
                    synth_a,
                    synth_b,
                    a_continuous,
                    b_continuous,
                )

            pair_scores[pair_key] = score
            pair_details[pair_key] = {
                "score": score,
                "metric": metric,
                "co_rate_real": co_rate_real,
                "co_rate_synth": co_rate_synth,
                "n_real": int(n_real_co),
                "n_synth": int(n_synth_co),
            }

            if (idx + 1) % 500 == 0:
                print(f"  Column pair trends: {idx + 1}/{n_pairs} pairs done")

    # Weighted average
    total_weight = sum(pair_weights.values())
    if total_weight > 0:
        pair_weights_norm = {k: w / total_weight for k, w in pair_weights.items()}
        overall = sum(pair_weights_norm[k] * pair_scores[k] for k in pair_scores)
    else:
        pair_weights_norm = pair_weights
        overall = 1.0

    return overall, pair_scores, pair_details, pair_weights_norm


def compute_fidelity(
    real_records: list[dict],
    synthetic_records: list[dict],
    *,
    max_workers: int | None = None,
) -> FidelityResult:
    """Compute fidelity for semi-structured JSON data.

    Column Shapes: factorized model per field (presence × type × value).
    Column Pair Trends: pairwise trend similarity on type-separated columns.
    Overall score = (column_shapes + column_pair_trends) / 2.

    Args:
        real_records: List of dictionaries (real data)
        synthetic_records: List of dictionaries (synthetic data)
        max_workers: Number of parallel workers. ``None`` uses all CPUs,
            ``1`` disables multiprocessing (sequential execution).

    Returns:
        FidelityResult with overall, per-field, and per-pair scores
    """
    from .shared import prepare_union_table

    df, masks = prepare_union_table(real=real_records, synth=synthetic_records)
    real_mask, synth_mask = masks["real"], masks["synth"]

    # --- Column Shapes ---
    fields = sorted(c[:-6] for c in df.columns if c.endswith(".dtype"))

    field_scores: dict[str, float] = {}
    field_details: dict[str, dict[str, Any]] = {}

    # Only parallelize when there are enough fields to justify the overhead
    _MIN_FIELDS_FOR_PARALLEL = 10
    use_parallel = max_workers != 1 and len(fields) >= _MIN_FIELDS_FOR_PARALLEL

    if use_parallel:
        _SHARED.update(df=df, real_mask=real_mask, synth_mask=synth_mask)
        print(f"  Column shapes: {len(fields)} fields ({max_workers or mp.cpu_count()} workers)...")
        try:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
                for done, (field, details) in enumerate(pool.map(_field_fidelity_task, fields), 1):
                    field_scores[field] = details["score"]
                    field_details[field] = details
                    if done % 50 == 0 or done == len(fields):
                        print(f"  Column shapes: {done}/{len(fields)} fields done")
        finally:
            for key in ("df", "real_mask", "synth_mask"):
                _SHARED.pop(key, None)
    else:
        for i, field in enumerate(fields):
            details = _compute_field_fidelity(df, field, real_mask, synth_mask)
            field_scores[field] = details["score"]
            field_details[field] = details
            if (i + 1) % 50 == 0 or (i + 1) == len(fields):
                print(f"  Column shapes: {i + 1}/{len(fields)} fields done")

    field_weights: dict[str, float] = {}
    for field in fields:
        real_rate = field_details[field].get("real_presence_rate", 0.0)
        synth_rate = field_details[field].get("synth_presence_rate", 0.0)
        field_weights[field] = max(real_rate, synth_rate)

    total = sum(field_weights.values())
    if total > 0:
        field_weights = {f: w / total for f, w in field_weights.items()}
    else:
        field_weights = {f: 1.0 / len(fields) for f in fields} if fields else {}

    shapes_score = sum(field_weights[f] * field_scores[f] for f in fields) if fields else 1.0

    # --- Column Pair Trends ---
    trends_score, pair_scores, pair_details, pair_weights = _compute_pair_trends(
        df,
        real_mask,
        synth_mask,
        max_workers=max_workers,
    )

    # --- Combined ---
    overall = (shapes_score + trends_score) / 2.0

    return FidelityResult(
        overall_score=overall,
        column_shapes_score=shapes_score,
        column_pair_trends_score=trends_score,
        field_scores=field_scores,
        field_details=field_details,
        field_weights=field_weights,
        pair_scores=pair_scores,
        pair_details=pair_details,
        pair_weights=pair_weights,
    )
