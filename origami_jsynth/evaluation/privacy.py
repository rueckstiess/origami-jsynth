"""Privacy evaluation using Distance to Closest Record (DCR).

Uses approximate L2 DCR with Johnson-Lindenstrauss random projections on
type-separated JSON data for efficient privacy measurement.

A score close to 50% indicates good privacy - the synthetic data is equally
likely to be near training or holdout data, suggesting no memorization.

Reference: https://github.com/amazon-science/tabsyn/blob/main/eval/eval_dcr.py
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PrivacyResult:
    """Results from privacy evaluation.

    Attributes:
        privacy_score: Primary score (0-1, 1.0 = best).
            Computed as: 1 - 2 * max(dcr_score/100 - 0.5, 0).
            Only penalizes dcr_score > 50 (memorization). dcr_score <= 50 gives 1.0.
        dcr_score: DCR (% of synthetic closer to train than test).
            Computed on equal-sized train/test reference sets. 50% = ideal.
        exact_matches_train: Synthetic records that are exact copies of a training record.
        exact_matches_test: Synthetic records that are exact copies of a test record.
        exact_matches_train_only: Synthetic records matching train but NOT test.
            This is the true memorization signal — the model reproduced a training
            record that doesn't also appear in the test set.
        ref_size: Number of samples in each reference set (train subsampled to
            match test size for unbiased comparison).
    """

    privacy_score: float  # 0-1, 1.0 = best (primary)
    dcr_score: float  # % closer to train (50% = ideal)
    exact_matches_train: int
    exact_matches_test: int
    exact_matches_train_only: int
    ref_size: int  # size of each reference set used

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "privacy_score": self.privacy_score,
            "dcr_score": self.dcr_score,
            "exact_matches_train": self.exact_matches_train,
            "exact_matches_test": self.exact_matches_test,
            "exact_matches_train_only": self.exact_matches_train_only,
            "ref_size": self.ref_size,
        }


def _compute_dcr_l2_batched(
    query: np.ndarray,
    reference: np.ndarray,
    batch_size: int = 100,
    desc: str | None = None,
) -> np.ndarray:
    """Compute distance to closest record using L2 (Euclidean) distance.

    Uses ||a-b||² = ||a||² + ||b||² - 2a·b to avoid O(B×M×D) intermediate
    arrays, reducing memory from O(B×M×D) to O(B×M) per batch and leveraging
    BLAS-optimized matrix multiplication.

    Args:
        query: Array of query points (N x D)
        reference: Array of reference points (M x D)
        batch_size: Number of query points to process at once (default 100)
        desc: Description for progress logging (optional)

    Returns:
        Array of distances (N,), one per query point
    """
    n_query = len(query)
    dcr = np.zeros(n_query)

    # Precompute ||r||² once for all batches
    ref_sq = np.sum(reference**2, axis=1)  # (M,)

    n_batches = (n_query + batch_size - 1) // batch_size

    for idx, i in enumerate(range(0, n_query, batch_size)):
        batch = query[i : i + batch_size]
        batch_sq = np.sum(batch**2, axis=1, keepdims=True)  # (B, 1)
        sq_distances = batch_sq + ref_sq - 2.0 * batch @ reference.T  # (B, M)
        np.maximum(sq_distances, 0.0, out=sq_distances)
        dcr[i : i + batch_size] = np.sqrt(sq_distances.min(axis=1))

        if desc and ((idx + 1) % max(1, n_batches // 5) == 0 or idx + 1 == n_batches):
            print(f"  {desc}: {idx + 1}/{n_batches} batches done")

    return dcr


def compute_privacy(
    train_records: list[dict],
    test_records: list[dict],
    synthetic_records: list[dict],
    *,
    n_components: int = 30000,
    batch_size: int = 100,
    max_synth_samples: int | None = None,
    max_workers: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> PrivacyResult:
    """Compute privacy metrics using random projections and L2 DCR.

    Uses type separation to handle heterogeneous JSON, then applies
    Johnson-Lindenstrauss random projections before computing distances.

    For each synthetic sample, computes distance to nearest training sample
    and nearest test sample. DCR score is the % of synthetic samples that
    are closer to training than to test. A score of 50% means the synthetic
    data is equidistant from train and test — no memorization detected.

    Args:
        train_records: Training data records (sensitive)
        test_records: Held-out test data records
        synthetic_records: Generated synthetic data records
        n_components: Number of dimensions after projection. Default 500
            gives ~10% distance distortion for typical dataset sizes.
        batch_size: Batch size for distance computation
        max_synth_samples: Maximum synthetic samples to evaluate. None = use all.
        max_workers: Number of parallel workers. ``None`` runs the two DCR
            computations concurrently, ``1`` runs them sequentially.
        random_state: Random seed for sampling and projection reproducibility
        verbose: Print subsampling and projection details

    Returns:
        PrivacyResult with privacy_score, dcr_score, exact match counts, ref_size
    """
    from sklearn.random_projection import SparseRandomProjection

    from .shared import encode_features, prepare_union_table

    rng = np.random.default_rng(random_state)

    # Subsample synthetic records if too large
    n_synth_orig = len(synthetic_records)
    if max_synth_samples is not None and n_synth_orig > max_synth_samples:
        synth_idx = rng.choice(n_synth_orig, max_synth_samples, replace=False)
        synthetic_records = [synthetic_records[i] for i in synth_idx]
    if verbose and len(synthetic_records) < n_synth_orig:
        print(f"  Synthetic: {n_synth_orig} → subsampled to {len(synthetic_records)}")

    # Hash-based exact match detection on original records (before any
    # encoding/projection). Uses full training set — not the DCR subsample —
    # so we don't miss memorized records that fall outside the subsample.
    def _record_hash(record: dict) -> str:
        return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()

    train_hashes = {_record_hash(r) for r in train_records}
    test_hashes = {_record_hash(r) for r in test_records}
    synth_hash_list = [_record_hash(r) for r in synthetic_records]

    exact_matches_train = sum(1 for h in synth_hash_list if h in train_hashes)
    exact_matches_test = sum(1 for h in synth_hash_list if h in test_hashes)
    exact_matches_train_only = sum(
        1 for h in synth_hash_list if h in train_hashes and h not in test_hashes
    )

    if verbose and (exact_matches_train > 0 or exact_matches_test > 0):
        print(
            f"  Exact matches: {exact_matches_train} train, "
            f"{exact_matches_test} test, "
            f"{exact_matches_train_only} train-only"
        )

    # Subsample train to match test size for unbiased comparison
    n_train_orig = len(train_records)
    ref_size = len(test_records)
    if n_train_orig > ref_size:
        train_idx = rng.choice(n_train_orig, ref_size, replace=False)
        train_records = [train_records[i] for i in train_idx]
    if verbose and len(train_records) < n_train_orig:
        print(f"  Train: {n_train_orig} → subsampled to {len(train_records)}")

    # Flatten and type-separate all three groups together
    df, masks = prepare_union_table(train=train_records, test=test_records, synth=synthetic_records)
    train_mask, test_mask, synth_mask = masks["train"], masks["test"], masks["synth"]

    # Encode type-separated columns into numeric features
    encoded = encode_features(df, exclude_columns=["__split__"])

    # Normalize continuous columns by training range.
    # One-hot (.cat/.dtype) and boolean (.bool) columns are already 0/1.
    num_suffixes = (".num", ".alen", ".date")
    for col in encoded.columns:
        if any(col.endswith(s) for s in num_suffixes):
            train_vals = encoded.loc[train_mask, col].astype(float)
            col_range = train_vals.max() - train_vals.min()
            if col_range > 0:
                encoded[col] = encoded[col].astype(float) / col_range

    # Fill NaN with 0 (absent fields contribute nothing to distance)
    encoded = encoded.fillna(0.0)

    # Split into arrays by group
    train_arr = encoded.loc[train_mask].to_numpy(dtype=np.float64)
    test_arr = encoded.loc[test_mask].to_numpy(dtype=np.float64)
    synth_arr = encoded.loc[synth_mask].to_numpy(dtype=np.float64)

    # Apply random projection (JL dimensionality reduction)
    # Skip projection if n_components >= input dimensionality (no reduction needed)
    n_features = train_arr.shape[1]
    if verbose:
        if n_components < n_features:
            print(f"  Features: {n_features} → projected to {n_components}")
        else:
            print(
                f"  Features: {n_features}"
                f" (no projection, n_components={n_components} >= n_features)"
            )
    if n_components < n_features:
        projector = SparseRandomProjection(n_components=n_components, random_state=random_state)
        train_proj = projector.fit_transform(train_arr)
        test_proj = projector.transform(test_arr)
        synth_proj = projector.transform(synth_arr)
    else:
        train_proj, test_proj, synth_proj = train_arr, test_arr, synth_arr

    # Compute L2 DCR in projected space.
    # The two DCR calls (synth→train, synth→test) are independent, so we run
    # them concurrently using threads. numpy matmul releases the GIL, giving
    # ~1.4x speedup. (ProcessPoolExecutor with fork crashes here because
    # BLAS thread state is not fork-safe on macOS.)
    if max_workers != 1:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_train = pool.submit(
                _compute_dcr_l2_batched,
                synth_proj,
                train_proj,
                batch_size,
                desc="DCR synth→train",
            )
            future_test = pool.submit(
                _compute_dcr_l2_batched,
                synth_proj,
                test_proj,
                batch_size,
                desc="DCR synth→test",
            )
            dcr_to_train = future_train.result()
            dcr_to_test = future_test.result()
    else:
        dcr_to_train = _compute_dcr_l2_batched(
            synth_proj,
            train_proj,
            batch_size,
            desc="DCR synth→train",
        )
        dcr_to_test = _compute_dcr_l2_batched(
            synth_proj,
            test_proj,
            batch_size,
            desc="DCR synth→test",
        )

    # Score computation
    closer_to_train = np.sum(dcr_to_train < dcr_to_test)
    total = len(synth_proj)

    dcr_score = (closer_to_train / total) * 100
    # Only penalize dcr_score > 50 (closer to train = memorization risk).
    # dcr_score <= 50 means closer to test, which is not a privacy concern.
    privacy_score = 1.0 - 2.0 * max(dcr_score / 100.0 - 0.5, 0.0)

    return PrivacyResult(
        privacy_score=privacy_score,
        dcr_score=dcr_score,
        exact_matches_train=exact_matches_train,
        exact_matches_test=exact_matches_test,
        exact_matches_train_only=exact_matches_train_only,
        ref_size=ref_size,
    )
