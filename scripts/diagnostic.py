#!/usr/bin/env python3
"""Diagnostic benchmark dataset: generation and audit.

Generate a dataset with known ground-truth correlations, mixed types
(float, int, bool, categorical), and controlled missingness patterns.
Audit synthetic data against those ground-truth properties.

Usage:
    python scripts/diagnostic.py generate [--output-dir DIR] [--seed 42]
    python scripts/diagnostic.py audit results/diagnostic/ctgan/samples/synthetic_1.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_TOTAL = 10_000
SPLIT_RATIO = 0.9
SEED = 42

REGION_NAMES = [f"R{i:02d}" for i in range(50)]
LEVEL_NAMES = [f"L{i:02d}" for i in range(1, 11)]
GROUP_MAP = {"A": "alpha", "B": "beta", "C": "gamma"}
GROUP_SHIFT = {"alpha": 0.0, "beta": -0.5, "gamma": 1.5}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def generate_records(n: int, seed: int = SEED) -> list[dict]:
    """Generate n records with known ground-truth relationships."""
    rng = np.random.default_rng(seed)

    # --- Latent variables ---
    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)
    Z3 = rng.binomial(1, 0.3, size=n)
    Z4_codes = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])

    def eps():
        return rng.standard_normal(n)

    # --- Observed: continuous ---
    cont_a = Z1 + 0.1 * eps()

    cat_group = np.array([GROUP_MAP[z] for z in Z4_codes])
    group_shifts = np.array([GROUP_SHIFT[g] for g in cat_group])
    cont_b = 0.8 * Z1 + 0.3 * Z2 + group_shifts + 0.1 * eps()

    cont_c = np.sin(2 * Z1) + 0.5 * Z2 + 0.15 * eps()

    cont_conditional = 2.0 * Z2 + 0.2 * eps()

    cont_interaction = Z1 * Z2 + 0.1 * eps()

    # --- Observed: integer ---
    int_count = np.clip(np.floor(np.exp(0.5 * Z1 + 0.3) + 2 * Z3), 0, 20).astype(int)
    int_rating = np.clip(np.round(3 + Z2 + 0.5 * eps()), 1, 5).astype(int)
    int_sparse = np.round(np.abs(Z1) * 10).astype(int)

    # --- Observed: categorical ---
    # cat_level: quantile-bin Z1 into 10 levels
    z1_rank = (Z1 - Z1.min()) / (Z1.max() - Z1.min() + 1e-9)
    cat_level_idx = np.clip(np.floor(z1_rank * 10), 0, 9).astype(int)
    cat_level = np.array([LEVEL_NAMES[i] for i in cat_level_idx])

    # cat_region: Z4-dependent multinomial over 50 region codes
    region_rng = np.random.default_rng(seed + 1)
    region_probs = {code: region_rng.dirichlet(np.ones(50) * 0.5) for code in ["A", "B", "C"]}
    cat_region = np.array([rng.choice(REGION_NAMES, p=region_probs[z4]) for z4 in Z4_codes])

    # cat_nullable: simple 3-way
    cat_nullable_base = rng.choice(["yes", "no", "maybe"], size=n)

    # --- Observed: mixed type ---
    # ~60% numeric (when Z2 > -0.25), ~40% string label (otherwise)
    mixed_is_num = Z2 > -0.25
    mixed_num_vals = np.round(Z1 * 10 + 50, 1)  # numeric: score around 50
    mixed_cat_vals = rng.choice(["low", "medium", "high"], size=n)

    # --- Observed: boolean ---
    threshold = np.quantile(cont_a + cont_b, 0.65)
    bool_flag = (cont_a + cont_b) > threshold
    bool_active = Z3.astype(bool)
    bool_nullable = Z2 > 0  # ~50% True

    # --- Target ---
    logit = (
        0.8 * Z1 - 0.6 * Z2 + 1.2 * (Z4_codes == "C").astype(float) + 0.5 * Z3 + 0.3 * Z1 * Z2 - 0.3
    )
    p_target = sigmoid(logit)
    target = np.where(rng.uniform(size=n) < p_target, "positive", "negative")

    # --- Missingness masks ---
    miss_cont_c = rng.uniform(size=n) < 0.20
    miss_int_rating = rng.uniform(size=n) < 0.15
    miss_cat_region = rng.uniform(size=n) < 0.25
    miss_int_sparse = rng.uniform(size=n) < 0.30
    miss_bool_nullable = rng.uniform(size=n) < 0.25
    miss_cont_cond = cat_group == "alpha"  # conditional on cat_group
    miss_cat_null = ~bool_active  # conditional on bool_active

    # --- Assemble records (omit key for missing values) ---
    records = []
    for i in range(n):
        row: dict = {
            "cont_a": float(cont_a[i]),
            "cont_b": float(cont_b[i]),
            "int_count": int(int_count[i]),
            "cat_group": str(cat_group[i]),
            "cat_level": str(cat_level[i]),
            "bool_flag": bool(bool_flag[i]),
            "bool_active": bool(bool_active[i]),
            "cont_interaction": float(cont_interaction[i]),
            "mixed_value": float(mixed_num_vals[i]) if mixed_is_num[i] else str(mixed_cat_vals[i]),
            "target": str(target[i]),
        }
        if not miss_cont_c[i]:
            row["cont_c"] = float(cont_c[i])
        if not miss_int_rating[i]:
            row["int_rating"] = int(int_rating[i])
        if not miss_cat_region[i]:
            row["cat_region"] = str(cat_region[i])
        if not miss_int_sparse[i]:
            row["int_sparse"] = int(int_sparse[i])
        if not miss_bool_nullable[i]:
            row["bool_nullable"] = bool(bool_nullable[i])
        if not miss_cont_cond[i]:
            row["cont_conditional"] = float(cont_conditional[i])
        if not miss_cat_null[i]:
            row["cat_nullable"] = str(cat_nullable_base[i])
        records.append(row)

    return records


def verify(records: list[dict]) -> None:
    """Print verification summary of ground-truth properties."""
    df = pd.DataFrame(records)
    n = len(df)

    print(f"\n{'=' * 60}")
    print(f"DIAGNOSTIC DATASET VERIFICATION ({n} rows, {len(df.columns)} columns)")
    print(f"{'=' * 60}")

    # --- Types ---
    print("\n--- Column types ---")
    for col in sorted(df.columns):
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        print(f"  {col:25s}  dtype={str(dtype):10s}  nunique={n_unique}")

    # --- Missing rates ---
    print("\n--- Missing rates ---")
    all_cols = [
        "cont_a",
        "cont_b",
        "cont_c",
        "int_count",
        "int_rating",
        "cat_group",
        "cat_level",
        "cat_region",
        "bool_flag",
        "bool_active",
        "bool_nullable",
        "cont_conditional",
        "cat_nullable",
        "int_sparse",
        "cont_interaction",
        "mixed_value",
        "target",
    ]
    for col in all_cols:
        if col in df.columns:
            rate = df[col].isna().mean()
            print(f"  {col:25s}  {rate:6.1%}")
        else:
            print(f"  {col:25s}  100.0% (column absent)")

    # --- Correlations ---
    print("\n--- Correlation: cont_a vs cont_b ---")
    r = df["cont_a"].corr(df["cont_b"])
    print(f"  Pearson r = {r:.4f}  (expected ~0.85-0.93)")

    print("\n--- Correlation: cont_a vs cont_c (non-linear) ---")
    r_lin = df["cont_a"].corr(df["cont_c"])
    print(f"  Pearson r = {r_lin:.4f}  (expected near 0, relationship is sinusoidal)")

    # --- Cat-num dependence ---
    print("\n--- cont_b mean by cat_group ---")
    for g in ["alpha", "beta", "gamma"]:
        subset = df[df["cat_group"] == g]["cont_b"]
        print(f"  {g:8s}  mean={subset.mean():.3f}  (shift={GROUP_SHIFT[g]:+.1f})")

    # --- Conditional missingness ---
    print("\n--- Conditional missingness: cont_conditional ~ cat_group ---")
    for g in ["alpha", "beta", "gamma"]:
        subset = df[df["cat_group"] == g]
        rate = subset["cont_conditional"].isna().mean()
        print(f"  cat_group={g:8s}  cont_conditional NaN rate = {rate:.1%}")

    print("\n--- Conditional missingness: cat_nullable ~ bool_active ---")
    for val in [True, False]:
        subset = df[df["bool_active"] == val]
        rate = subset["cat_nullable"].isna().mean()
        print(f"  bool_active={str(val):6s}  cat_nullable NaN rate = {rate:.1%}")

    # --- Target balance ---
    print("\n--- Target class balance ---")
    vc = df["target"].value_counts(normalize=True)
    for cls, pct in vc.items():
        print(f"  {cls:12s}  {pct:.1%}")

    print(f"\n{'=' * 60}\n")


def audit_synthetic(
    synthetic_records: list[dict],
    real_records: list[dict] | None = None,
    tolerance: float = 0.10,
) -> dict:
    """Audit synthetic records against the known ground-truth properties.

    Returns a dict with per-check results: {name: {status, detail, ...}}.
    Prints a summary report. Designed to be run after train+sample to verify
    that the synthesizer preserved the dataset's structural properties.

    Args:
        synthetic_records: Generated synthetic data (list of dicts).
        real_records: Original training data for reference stats. If None,
            generates fresh reference data.
        tolerance: Acceptable absolute deviation for rate/correlation checks.
    """
    syn = pd.DataFrame(synthetic_records)
    if real_records is not None:
        real = pd.DataFrame(real_records)
    else:
        real = pd.DataFrame(generate_records(10_000))

    results: dict[str, dict] = {}
    passes = 0
    fails = 0
    warns = 0

    def check(name: str, passed: bool, detail: str, warn_only: bool = False):
        nonlocal passes, fails, warns
        if passed:
            status = "PASS"
            passes += 1
        elif warn_only:
            status = "WARN"
            warns += 1
        else:
            status = "FAIL"
            fails += 1
        results[name] = {"status": status, "detail": detail}
        marker = {"PASS": "+", "FAIL": "X", "WARN": "~"}[status]
        print(f"  [{marker}] {name}: {detail}")

    print(f"\n{'=' * 70}")
    print(f"SYNTHETIC DATA AUDIT ({len(syn)} rows, {len(syn.columns)} columns)")
    print(f"{'=' * 70}")

    # ---- 1. Column presence ----
    print("\n--- Column presence ---")
    expected_cols = {
        "cont_a",
        "cont_b",
        "cont_c",
        "int_count",
        "int_rating",
        "cat_group",
        "cat_level",
        "cat_region",
        "bool_flag",
        "bool_active",
        "bool_nullable",
        "cont_conditional",
        "cat_nullable",
        "int_sparse",
        "cont_interaction",
        "mixed_value",
        "target",
    }
    syn_cols = set(syn.columns)
    missing_cols = expected_cols - syn_cols
    extra_cols = syn_cols - expected_cols
    check(
        "all_columns_present",
        len(missing_cols) == 0,
        f"missing={missing_cols or 'none'}, extra={extra_cols or 'none'}",
    )

    # ---- 2. Type checks ----
    print("\n--- Type checks ---")
    # We check the JSON-level types, not pandas dtypes (which can differ)
    type_checks = {
        "cont_a": (float, int),
        "cont_b": (float, int),
        "int_count": (int,),
        "cat_group": (str,),
        "bool_flag": (bool,),
        "bool_active": (bool,),
        "target": (str,),
    }
    for col, expected_types in type_checks.items():
        if col not in syn.columns:
            check(f"type_{col}", False, "column missing")
            continue
        sample_vals = syn[col].dropna().head(20).tolist()
        bad = [v for v in sample_vals if not isinstance(v, expected_types)]
        type_names = "/".join(t.__name__ for t in expected_types)
        if bad:
            detail = f"expected {type_names}, got {[type(v).__name__ for v in bad[:3]]}"
        else:
            detail = f"all {type_names}"
        check(f"type_{col}", len(bad) == 0, detail, warn_only=True)

    # ---- 3. Missing value rates ----
    print("\n--- Missing value rates ---")
    expected_missing = {
        "cont_a": 0.0,
        "cont_b": 0.0,
        "int_count": 0.0,
        "cat_group": 0.0,
        "cat_level": 0.0,
        "bool_flag": 0.0,
        "bool_active": 0.0,
        "cont_interaction": 0.0,
        "target": 0.0,
        "cont_c": 0.20,
        "int_rating": 0.15,
        "cat_region": 0.25,
        "int_sparse": 0.30,
        "bool_nullable": 0.25,
        "mixed_value": 0.0,
        "cont_conditional": 0.50,
        "cat_nullable": 0.70,
    }
    for col, expected_rate in expected_missing.items():
        actual = 1.0 if col not in syn.columns else syn[col].isna().mean()
        close = abs(actual - expected_rate) < tolerance
        # Dense columns that should have 0% missing are stricter
        if expected_rate == 0.0:
            close = actual < 0.02  # allow up to 2% for dense columns
        check(
            f"missing_{col}",
            close,
            f"actual={actual:.1%}, expected~{expected_rate:.0%}",
            warn_only=(expected_rate > 0),  # only fail on dense columns
        )

    # ---- 4. Conditional missingness ----
    print("\n--- Conditional missingness ---")
    if "cat_group" in syn.columns and "cont_conditional" in syn.columns:
        alpha_miss = syn[syn["cat_group"] == "alpha"]["cont_conditional"].isna().mean()
        other_miss = syn[syn["cat_group"] != "alpha"]["cont_conditional"].isna().mean()
        # The key check: alpha should have much higher missing rate than others
        check(
            "cond_miss_cont_conditional",
            alpha_miss > other_miss + 0.2,
            f"alpha NaN={alpha_miss:.1%}, non-alpha NaN={other_miss:.1%} (real: 100% vs 0%)",
        )
    else:
        check("cond_miss_cont_conditional", False, "required columns missing")

    if "bool_active" in syn.columns and "cat_nullable" in syn.columns:
        inactive_miss = syn[syn["bool_active"] == False]["cat_nullable"].isna().mean()  # noqa: E712
        active_miss = syn[syn["bool_active"] == True]["cat_nullable"].isna().mean()  # noqa: E712
        check(
            "cond_miss_cat_nullable",
            inactive_miss > active_miss + 0.2,
            f"inactive NaN={inactive_miss:.1%}, active NaN={active_miss:.1%} (real: 100% vs 0%)",
        )
    else:
        check("cond_miss_cat_nullable", False, "required columns missing")

    # ---- 5. Correlations ----
    print("\n--- Correlations ---")
    if {"cont_a", "cont_b"} <= syn_cols:
        r_real = real["cont_a"].corr(real["cont_b"])
        r_syn = syn["cont_a"].corr(syn["cont_b"])
        check(
            "corr_cont_a_cont_b",
            abs(r_syn - r_real) < 0.15,
            f"r_syn={r_syn:.3f}, r_real={r_real:.3f}",
        )
    else:
        check("corr_cont_a_cont_b", False, "required columns missing")

    # ---- 6. Cat→num shift ----
    print("\n--- Categorical-numeric dependence ---")
    if {"cont_b", "cat_group"} <= syn_cols:
        group_means_real = real.groupby("cat_group")["cont_b"].mean()
        group_means_syn = syn.groupby("cat_group")["cont_b"].mean()
        # Check that the ordering is preserved (gamma > alpha > beta)
        try:
            order_real = group_means_real.sort_values().index.tolist()
            order_syn = group_means_syn.sort_values().index.tolist()
            check(
                "cat_num_shift_order",
                order_real == order_syn,
                f"real order={order_real}, syn order={order_syn}",
            )
        except Exception as e:
            check("cat_num_shift_order", False, str(e))
    else:
        check("cat_num_shift_order", False, "required columns missing")

    # ---- 7. Target balance ----
    print("\n--- Target balance ---")
    if "target" in syn.columns:
        real_rate = (real["target"] == "positive").mean()
        syn_rate = (syn["target"] == "positive").mean()
        check(
            "target_balance",
            abs(syn_rate - real_rate) < 0.15,
            f"syn={syn_rate:.1%} positive, real={real_rate:.1%}",
        )
        # Check that target has exactly 2 classes
        n_classes = syn["target"].nunique()
        check(
            "target_classes",
            n_classes == 2,
            f"{n_classes} classes (expected 2): {syn['target'].unique().tolist()[:5]}",
        )
    else:
        check("target_balance", False, "target column missing")
        check("target_classes", False, "target column missing")

    # ---- 8. Category sets ----
    print("\n--- Category sets ---")
    cat_checks = {
        "cat_group": {"alpha", "beta", "gamma"},
        "target": {"positive", "negative"},
    }
    for col, expected_vals in cat_checks.items():
        if col not in syn.columns:
            check(f"categories_{col}", False, "column missing")
            continue
        actual_vals = set(syn[col].dropna().unique())
        check(
            f"categories_{col}",
            expected_vals <= actual_vals,
            f"expected={expected_vals}, actual={actual_vals}",
            warn_only=True,
        )

    # ---- 9. Mixed-type column ----
    print("\n--- Mixed-type column ---")
    if "mixed_value" in syn.columns:
        # Check that both numeric and string values are present
        vals = syn["mixed_value"].dropna()
        n_num = sum(isinstance(v, (int, float)) for v in vals)
        n_str = sum(isinstance(v, str) for v in vals)
        total_vals = len(vals)
        num_pct = n_num / total_vals if total_vals > 0 else 0
        str_pct = n_str / total_vals if total_vals > 0 else 0

        real_vals = real["mixed_value"].dropna()
        real_num_pct = sum(isinstance(v, (int, float)) for v in real_vals) / len(real_vals)

        check(
            "mixed_has_both_types",
            n_num > 0 and n_str > 0,
            f"num={n_num} ({num_pct:.0%}), str={n_str} ({str_pct:.0%})",
        )
        check(
            "mixed_type_proportion",
            abs(num_pct - real_num_pct) < 0.15,
            f"syn num={num_pct:.0%}, real num={real_num_pct:.0%}",
            warn_only=True,
        )
    else:
        check("mixed_has_both_types", False, "column missing")
        check("mixed_type_proportion", False, "column missing")

    # ---- Summary ----
    total = passes + fails + warns
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passes}/{total} passed, {fails} failed, {warns} warnings")
    print(f"{'=' * 70}\n")

    return results


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate or audit diagnostic dataset")
    sub = parser.add_subparsers(dest="command")

    # --- generate ---
    gen = sub.add_parser("generate", help="Generate the diagnostic dataset")
    gen.add_argument("--output-dir", type=Path, default=Path("results/diagnostic/data"))
    gen.add_argument("--seed", type=int, default=SEED)
    gen.add_argument("--n", type=int, default=N_TOTAL)

    # --- audit ---
    aud = sub.add_parser("audit", help="Audit synthetic data against ground truth")
    aud.add_argument("synthetic", type=Path, help="Path to synthetic JSONL file")
    aud.add_argument(
        "--real",
        type=Path,
        default=Path("results/diagnostic/data/train.jsonl"),
        help="Path to real train.jsonl",
    )
    aud.add_argument("--tolerance", type=float, default=0.10)

    args = parser.parse_args()

    if args.command == "audit":
        if not str(args.synthetic).endswith(".jsonl"):
            parser.error(
                f"Expected a .jsonl file, got: {args.synthetic}\n"
                "Hint: use the synthetic samples file, e.g. "
                "results/diagnostic/ctgan/samples/synthetic_1.jsonl"
            )
        syn_records = load_jsonl(args.synthetic)
        real_records = load_jsonl(args.real) if args.real else None
        audit_synthetic(syn_records, real_records, tolerance=args.tolerance)

    else:
        # Default to generate
        output_dir = getattr(args, "output_dir", Path("results/diagnostic/data"))
        seed = getattr(args, "seed", SEED)
        n = getattr(args, "n", N_TOTAL)

        print(f"Generating {n} records with seed={seed}...")
        records = generate_records(n, seed=seed)

        rng = np.random.default_rng(seed + 99)
        indices = rng.permutation(len(records))
        split_idx = int(len(records) * SPLIT_RATIO)
        train_records = [records[i] for i in indices[:split_idx]]
        test_records = [records[i] for i in indices[split_idx:]]

        print(f"Split: {len(train_records)} train / {len(test_records)} test")

        save_jsonl(train_records, output_dir / "train.jsonl")
        save_jsonl(test_records, output_dir / "test.jsonl")
        print(f"Saved to {output_dir}")

        verify(records)


if __name__ == "__main__":
    main()
