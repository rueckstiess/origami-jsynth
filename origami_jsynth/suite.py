"""Full-suite orchestration: run all model+dataset combos (base + DCR)."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .cli import cmd_all
from .sync import RemoteSync

SUITE_DATASETS = ["adult", "diabetes", "electric_vehicles", "yelp", "ddxplus", "github_issues"]
SUITE_MODELS = ["tvae", "ctgan", "realtabformer", "mostlyai", "tabdiff", "origami"]

# Known OOM combos — matches the authoritative set in cli.py cmd_overview.
SKIP_OOM: set[tuple[str, str]] = {
    ("ctgan", "ddxplus"),
    ("ctgan", "yelp"),
    ("ctgan", "electric_vehicles"),
    ("ctgan", "github_issues"),
    ("tvae", "ddxplus"),
    ("tvae", "yelp"),
    ("tvae", "electric_vehicles"),
    ("tvae", "github_issues"),
    ("realtabformer", "ddxplus"),
    ("realtabformer", "github_issues"),
}

# V100-specific param overrides per (model, dataset).
# train_params / sample_params are "KEY=VALUE" strings (same format as --param).
V100_OVERRIDES: dict[tuple[str, str], dict] = {
    ("tabdiff", "yelp"): {
        "train_params": ["batch_size=512", "check_val_every=20"],
        "sample_params": ["sample_batch_size=512"],
    },
    ("tabdiff", "ddxplus"): {
        "train_params": ["batch_size=512", "check_val_every=20"],
        "sample_params": ["sample_batch_size=512"],
    },
    ("tabdiff", "github_issues"): {
        "train_params": ["batch_size=512", "check_val_every=20"],
        "sample_params": ["sample_batch_size=512"],
    },
}


def _is_combo_complete(output_dir: str, model: str, dataset: str, dcr: bool) -> bool:
    """Check whether the final evaluation artifact exists for this combo."""
    dataset_name = f"{dataset}_dcr" if dcr else dataset
    return (Path(output_dir) / dataset_name / model / "report" / "agg_results.json").exists()


def _build_args(
    model: str,
    dataset: str,
    dcr: bool,
    output_dir: str,
    replicates: int,
    num_workers: int,
    no_wandb: bool,
    max_minutes: float | None = None,
) -> argparse.Namespace:
    """Construct an argparse.Namespace for cmd_all."""
    overrides = V100_OVERRIDES.get((model, dataset), {})
    train_params = overrides.get("train_params", [])
    sample_params = overrides.get("sample_params", [])
    # Merge — unknown params are harmlessly ignored by the respective commands.
    all_params = list(dict.fromkeys(train_params + sample_params))

    # Per-combo override takes precedence over global default.
    effective_max_minutes = overrides.get("max_minutes", max_minutes)

    return argparse.Namespace(
        dataset=dataset,
        model=model,
        dcr=dcr,
        output_dir=output_dir,
        remote=None,  # outer RemoteSync handles S3
        replicates=replicates,
        num_workers=num_workers,
        param=all_params,
        max_minutes=effective_max_minutes,
        no_wandb=no_wandb,
    )


def _build_combos(dcr: bool) -> list[tuple[str, str, bool]]:
    """Return (model, dataset, dcr) triples for all non-OOM combos in *dcr* mode."""
    combos = []
    for model in SUITE_MODELS:
        for dataset in SUITE_DATASETS:
            if (model, dataset) in SKIP_OOM:
                continue
            combos.append((model, dataset, dcr))
    return combos


def run_full_suite(
    dcr: bool = False,
    output_dir: str = "./results",
    remote: str | None = None,
    replicates: int = 10,
    num_workers: int = 4,
    no_wandb: bool = False,
    max_minutes: float | None = None,
) -> dict[tuple[str, str, bool], str]:
    """Run all model+dataset combos for *dcr* mode.

    When ``dcr=False`` (default): runs base experiments and evaluates
    fidelity, utility, and detection.
    When ``dcr=True``: runs DCR experiments and evaluates privacy only.

    Returns a status dict mapping (model, dataset, dcr) to one of:
    ``"completed"``, ``"skipped_oom"``, ``"skipped_done"``, ``"failed"``.
    """
    combos = _build_combos(dcr)
    total = len(combos)

    status: dict[tuple[str, str, bool], str] = {}
    for model in SUITE_MODELS:
        for dataset in SUITE_DATASETS:
            if (model, dataset) in SKIP_OOM:
                status[(model, dataset, dcr)] = "skipped_oom"

    with RemoteSync(local_dir=output_dir, remote_url=remote):
        for i, (model, dataset, dcr) in enumerate(combos, 1):
            mode = "DCR" if dcr else "base"
            print(f"\n{'=' * 70}")
            print(f"[{i}/{total}] {model} + {dataset} ({mode})")
            print(f"{'=' * 70}")

            if _is_combo_complete(output_dir, model, dataset, dcr):
                print("  Already complete, skipping.")
                status[(model, dataset, dcr)] = "skipped_done"
                continue

            args = _build_args(
                model,
                dataset,
                dcr,
                output_dir,
                replicates,
                num_workers,
                no_wandb,
                max_minutes,
            )
            try:
                cmd_all(args)
                status[(model, dataset, dcr)] = "completed"
            except KeyboardInterrupt:
                status[(model, dataset, dcr)] = "failed"
                print("\nInterrupted by user.")
                _print_summary(status)
                raise
            except SystemExit as exc:
                # _require_* helpers in cli.py call sys.exit(1) on errors.
                status[(model, dataset, dcr)] = "failed" if exc.code else "completed"
                print(f"  Status: {'FAILED' if exc.code else 'completed'}")
            except Exception:
                traceback.print_exc()
                status[(model, dataset, dcr)] = "failed"

    _print_summary(status)
    return status


DATASET_LABELS = {
    "adult": "Adult",
    "diabetes": "Diabetes",
    "electric_vehicles": "Elec. Vehicles",
    "yelp": "Yelp",
    "ddxplus": "DDXPlus",
    "github_issues": "GitHub Issues",
}
MODEL_LABELS = {
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "realtabformer": "REaLTabFormer",
    "mostlyai": "TabularARGN",
    "tabdiff": "TabDiff",
    "origami": "Origami",
}


def _print_summary(status: dict[tuple[str, str, bool], str]) -> None:
    """Print a colored grid of results for base and DCR modes."""
    GREEN = "\033[32m"
    RED = "\033[31m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Plain-text symbols (4 chars each) — ANSI wrapping added separately.
    plain = {
        "completed": "OK",
        "skipped_done": "DONE",
        "skipped_oom": "OOM",
        "failed": "FAIL",
    }
    colors = {
        "completed": GREEN,
        "skipped_done": GREEN,
        "skipped_oom": GRAY,
        "failed": RED,
    }

    model_col = max(len(MODEL_LABELS.get(m, m)) for m in SUITE_MODELS) + 2
    col_w = max(len(DATASET_LABELS.get(d, d)) for d in SUITE_DATASETS) + 2

    print(f"\n{'=' * 70}")
    print(f"{BOLD}Full Suite Summary{RESET}")
    print(f"{'=' * 70}")

    # Determine which mode is present in the status dict.
    dcr_flags = {dcr for (_, _, dcr) in status}
    for dcr_flag in sorted(dcr_flags):
        mode_label = "DCR" if dcr_flag else "Base"
        print(f"\n{BOLD}{mode_label}:{RESET}")
        header = "".ljust(model_col) + "  ".join(
            DATASET_LABELS.get(d, d).rjust(col_w) for d in SUITE_DATASETS
        )
        print(header)
        print("-" * len(header))
        for model in SUITE_MODELS:
            row = MODEL_LABELS.get(model, model).ljust(model_col)
            cells = []
            for dataset in SUITE_DATASETS:
                s = status.get((model, dataset, dcr_flag), "???")
                txt = plain.get(s, s)
                c = colors.get(s, "")
                # Pad first, then wrap in ANSI so escape codes don't break alignment.
                cells.append(c + txt.rjust(col_w) + RESET)
            row += "  ".join(cells)
            print(row)

    counts: dict[str, int] = {}
    for v in status.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"\nTotal: {sum(counts.values())} combos")
    for label in ["completed", "skipped_done", "skipped_oom", "failed"]:
        if label in counts:
            print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    # Allow quick testing: python -m origami_jsynth.suite --help
    run_full_suite()
