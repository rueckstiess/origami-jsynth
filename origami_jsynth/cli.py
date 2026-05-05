"""CLI entry point for origami-jsynth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .baselines import MODEL_NAMES
from .registry import DATASET_NAMES

if TYPE_CHECKING:
    from .sync import RemoteSync


def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Compute standard paths from CLI arguments.

    Layout: results/{dataset}/data/ (shared) + results/{dataset}/{model}/...
    In DCR mode: results/{dataset}_dcr/data/ + results/{dataset}_dcr/{model}/...
    """
    dataset_name = f"{args.dataset}_dcr" if args.dcr else args.dataset
    dataset_dir = Path(args.output_dir) / dataset_name
    model_dir = dataset_dir / args.model
    return {
        "data_dir": dataset_dir / "data",
        "checkpoint_dir": model_dir / "checkpoints",
        "samples_dir": model_dir / "samples",
        "report_dir": model_dir / "report",
    }


def _config_path(dataset: str) -> Path:
    """Get the config YAML path for a dataset."""
    return Path(__file__).parent.parent / "configs" / f"{dataset}.yaml"


def _model_flag(args: argparse.Namespace) -> str:
    """Return ' --model X' for error messages."""
    return f" --model {args.model}"


def _dcr_flag(args: argparse.Namespace) -> str:
    """Return ' --dcr' if dcr mode, else ''."""
    return " --dcr" if args.dcr else ""


def _require_data(paths: dict[str, Path], args: argparse.Namespace) -> None:
    """Check that data has been prepared, exit with helpful message if not."""
    train_path = paths["data_dir"] / "train.jsonl"
    test_path = paths["data_dir"] / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        print(
            f"Error: Data not found at {paths['data_dir']}\n\n"
            f"Run this first:\n"
            f"  origami-jsynth data --dataset {args.dataset}{_dcr_flag(args)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_model(paths: dict[str, Path], args: argparse.Namespace) -> None:
    """Check that a trained model exists, exit with helpful message if not."""
    checkpoint_dir = paths["checkpoint_dir"]
    if args.model == "origami":
        has_model = (
            (checkpoint_dir / "final.pt").exists()
            or (checkpoint_dir / "best.pt").exists()
            or list(checkpoint_dir.glob("epoch_*.pt"))
            if checkpoint_dir.exists()
            else False
        )
    else:
        has_model = checkpoint_dir.exists() and any(checkpoint_dir.iterdir())
    if not has_model:
        print(
            f"Error: No trained model found in {checkpoint_dir}\n\n"
            f"Run this first:\n"
            f"  origami-jsynth train --dataset {args.dataset}"
            f"{_model_flag(args)}{_dcr_flag(args)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_samples(paths: dict[str, Path], args: argparse.Namespace) -> None:
    """Check that synthetic samples exist, exit with helpful message if not."""
    has_samples = paths["samples_dir"].exists() and any(
        paths["samples_dir"].glob("synthetic_*.jsonl")
    )
    if not has_samples:
        print(
            f"Error: No synthetic_*.jsonl files found in {paths['samples_dir']}\n\n"
            f"Run this first:\n"
            f"  origami-jsynth sample --dataset {args.dataset}"
            f"{_model_flag(args)}{_dcr_flag(args)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _derive_log_dir(args: argparse.Namespace) -> Path | None:
    """Return the per-combo log directory, or None if logging shouldn't apply.

    Logging is enabled for commands that have both --dataset and --model
    (i.e. all/train/sample/eval), not for data/results/full-suite.
    """
    dataset = getattr(args, "dataset", None)
    model = getattr(args, "model", None)
    output_dir = getattr(args, "output_dir", None)
    if not dataset or not model or not output_dir:
        return None
    dataset_name = f"{dataset}_dcr" if getattr(args, "dcr", False) else dataset
    return Path(output_dir) / dataset_name / model


def _parse_overrides(overrides: list[str]) -> dict:
    """Parse key=value overrides into a dict with auto-casting."""
    kwargs = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --param format: {override!r} (expected KEY=VALUE)")
        key, value = override.split("=", 1)
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        kwargs[key] = value
    return kwargs


def _sync_context(args: argparse.Namespace) -> RemoteSync:
    """Build a RemoteSync context manager from CLI args."""
    from .sync import RemoteSync

    return RemoteSync(
        local_dir=args.output_dir,
        remote_url=getattr(args, "remote", None),
    )


def cmd_data(args: argparse.Namespace) -> None:
    from .baselines._preprocessing import records_to_dataframe
    from .data import load_jsonl, prepare_dataset
    from .registry import get_dataset

    paths = _resolve_paths(args)
    prepare_dataset(
        args.dataset,
        paths["data_dir"],
        dcr=args.dcr,
    )

    # Export preprocessed (flattened + type-separated) CSVs for external baselines
    data_dir = paths["data_dir"]
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"

    if train_csv.exists() and test_csv.exists():
        print(f"Preprocessed CSVs already exist at {data_dir}")
        return

    info = get_dataset(args.dataset)
    train_records = load_jsonl(data_dir / "train.jsonl")
    test_records = load_jsonl(data_dir / "test.jsonl")

    # Process combined records so both splits get an identical column schema
    print(f"Flattening records for {args.dataset}...")
    n_train = len(train_records)
    df, _ = records_to_dataframe(train_records + test_records, tabular=info.tabular)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Saved preprocessed CSVs ({df.shape[1]} columns) to {data_dir}")


def cmd_train(args: argparse.Namespace) -> None:
    paths = _resolve_paths(args)
    _require_data(paths, args)

    max_seconds = args.max_minutes * 60 if getattr(args, "max_minutes", None) else None
    use_wandb = _wandb_available() and not getattr(args, "no_wandb", False)
    wandb_dataset = f"{args.dataset}-dcr" if args.dcr else args.dataset

    if use_wandb:
        import os

        os.environ.setdefault("WANDB_PROJECT", "origami-jsynth")

    with _sync_context(args):
        if args.model == "origami":
            from .train import train_dataset

            config_path = _config_path(args.dataset)
            if not config_path.exists():
                print(f"Error: Config file not found: {config_path}", file=sys.stderr)
                sys.exit(1)

            train_dataset(
                args.dataset,
                data_dir=paths["data_dir"],
                checkpoint_dir=paths["checkpoint_dir"],
                config_path=config_path,
                overrides=args.param,
                max_seconds=max_seconds,
                wandb=use_wandb,
                wandb_dataset=wandb_dataset,
            )
        else:
            from .baselines import get_synthesizer
            from .data import load_jsonl
            from .registry import get_dataset

            info = get_dataset(args.dataset)
            kwargs = _parse_overrides(args.param)
            synth = get_synthesizer(args.model, tabular=info.tabular, dataset_info=info, **kwargs)

            train_records = load_jsonl(paths["data_dir"] / "train.jsonl")
            print(f"Training {args.model} on {args.dataset} ({len(train_records)} records)")
            synth.fit(
                train_records,
                checkpoint_dir=paths["checkpoint_dir"],
                max_seconds=max_seconds,
                wandb=use_wandb,
                dataset=wandb_dataset,
            )
            synth.save(paths["checkpoint_dir"])
            print(f"Model saved to {paths['checkpoint_dir']}")


def cmd_sample(args: argparse.Namespace) -> None:
    from .data import load_jsonl, save_jsonl
    from .registry import get_dataset

    paths = _resolve_paths(args)
    info = get_dataset(args.dataset)

    _require_data(paths, args)
    _require_model(paths, args)

    with _sync_context(args):
        if args.model == "origami":
            from .sample import sample_dataset

            sample_overrides = _parse_overrides(args.param) if args.param else {}
            sample_kwargs = dict(
                num_workers=args.num_workers,
                replicates=args.replicates,
            )
            sample_kwargs.update(sample_overrides)
            sample_dataset(
                args.dataset,
                checkpoint_dir=paths["checkpoint_dir"],
                samples_dir=paths["samples_dir"],
                tabular=info.tabular,
                data_dir=paths["data_dir"],
                **sample_kwargs,
            )
        else:
            from .baselines import get_synthesizer

            paths["samples_dir"].mkdir(parents=True, exist_ok=True)
            n_train = len(load_jsonl(paths["data_dir"] / "train.jsonl"))
            synth = type(get_synthesizer(args.model, tabular=info.tabular)).load(
                paths["checkpoint_dir"], tabular=info.tabular
            )
            if args.param:
                overrides = _parse_overrides(args.param)
                synth.kwargs.update(overrides)

            for i in range(1, args.replicates + 1):
                output_path = paths["samples_dir"] / f"synthetic_{i}.jsonl"
                if output_path.exists():
                    existing = load_jsonl(output_path)
                    if len(existing) == n_train:
                        print(
                            f"Replicate {i}/{args.replicates}: "
                            f"already exists ({len(existing)} records)"
                        )
                        continue
                print(f"Replicate {i}/{args.replicates}: sampling {n_train} records...")
                records = synth.sample(n_train)
                save_jsonl(records, output_path)
                print(
                    f"Replicate {i}/{args.replicates}: "
                    f"saved {len(records)} records to {output_path}"
                )


def cmd_eval(args: argparse.Namespace) -> None:
    from .eval import evaluate_dataset

    paths = _resolve_paths(args)

    _require_data(paths, args)
    _require_samples(paths, args)

    with _sync_context(args):
        evaluate_dataset(
            args.dataset,
            data_dir=paths["data_dir"],
            samples_dir=paths["samples_dir"],
            report_dir=paths["report_dir"],
            dcr=args.dcr,
        )


def cmd_overview(args: argparse.Namespace) -> None:
    """Show evaluation status and results across all datasets and models."""
    import json

    results_dir = Path(args.output_dir)

    # Fixed order matching generate_latex_tables.py
    datasets = ["adult", "diabetes", "electric_vehicles", "yelp", "ddxplus", "github_issues"]
    models = ["tabby", "tvae", "ctgan", "realtabformer", "mostlyai", "tabdiff", "origami"]

    model_labels = {
        "tabby": "Tabby",
        "origami": "Origami",
        "ctgan": "CTGAN",
        "tvae": "TVAE",
        "realtabformer": "REaLTabFormer",
        "mostlyai": "TabularARGN",
        "tabdiff": "TabDiff",
    }
    dataset_labels = {
        "adult": "Adult",
        "diabetes": "Diabetes",
        "electric_vehicles": "Elec. Vehicles",
        "ddxplus": "DDXPlus",
        "github_issues": "GitHub Issues",
        "yelp": "Yelp",
    }

    def load_agg(dataset, model, dcr=False):
        suffix = f"{dataset}_dcr" if dcr else dataset
        path = results_dir / suffix / model / "report" / "agg_results.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    # Load all results into a nested dict: data[dataset][source][model] -> metrics
    data = {}
    for d in datasets:
        data[d] = {"base": {}, "dcr": {}}
        for m in models:
            agg = load_agg(d, m, dcr=False)
            if agg:
                data[d]["base"][m] = agg["metrics"]
            agg = load_agg(d, m, dcr=True)
            if agg:
                data[d]["dcr"][m] = agg["metrics"]

    # Models that ran out of memory on certain datasets
    oom = {
        ("ctgan", "ddxplus"),
        ("ctgan", "yelp"),
        ("ctgan", "electric_vehicles"),
        ("ctgan", "github_issues"),
        ("tvae", "github_issues"),
        ("tvae", "ddxplus"),
        ("tvae", "yelp"),
        ("tvae", "electric_vehicles"),
        ("realtabformer", "ddxplus"),
        ("realtabformer", "github_issues"),
        ("tabby", "diabetes"),
        ("tabby", "electric_vehicles"),
        ("tabby", "yelp"),
        ("tabby", "ddxplus"),
        ("tabby", "github_issues"),
    }

    if args.latex:
        _print_latex_tables(datasets, models, model_labels, dataset_labels, data, oom)
        return

    if args.markdown:
        _print_markdown_tables(datasets, models, model_labels, dataset_labels, data, oom)
        return

    # ANSI formatting
    GREEN = "\033[32m"
    DARK_RED = "\033[31m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    OOM_STR = f"{DARK_RED}OOM{RESET}"

    # --- Table 1: Replicate counts ---
    def count_replicates(report_dir: Path) -> int:
        if not report_dir.is_dir():
            return 0
        return len(list(report_dir.glob("results_*.json")))

    base_counts = {}  # (dataset, model) -> int
    dcr_counts = {}
    for d in datasets:
        for m in models:
            base_counts[(d, m)] = count_replicates(results_dir / d / m / "report")
            dcr_counts[(d, m)] = count_replicates(results_dir / f"{d}_dcr" / m / "report")

    ds_col_w = max(len(dataset_labels.get(d, d)) for d in datasets) + 1
    model_ws = {m: max(len(model_labels.get(m, m)), 3) for m in models}

    def print_count_table(title, counts):
        print(f"\n{title}")
        header = "".ljust(ds_col_w) + "  ".join(
            model_labels.get(m, m).rjust(model_ws[m]) for m in models
        )
        print(header)
        print("-" * len(header))
        for d in datasets:
            row = dataset_labels.get(d, d).ljust(ds_col_w)
            cells = []
            for m in models:
                c = counts[(d, m)]
                if c > 0:
                    s = str(c)
                elif (m, d) in oom:
                    pad = model_ws[m] - 3  # 3 = len("OOM")
                    cells.append(" " * max(0, pad) + OOM_STR)
                    continue
                else:
                    s = "-"
                cells.append(s.rjust(model_ws[m]))
            row += "  ".join(cells)
            print(row)

    print_count_table("Evaluation replicates (base):", base_counts)
    print_count_table("Evaluation replicates (DCR):", dcr_counts)

    # --- Table 2: Primary scores ---
    def fmt(mean, std, highlight=False):
        s = f"{mean:.3f}±{std:.3f}"
        if highlight:
            return f"{GREEN}{BOLD}{s}{RESET}"
        return s

    def print_results_table(title, metric_defs, source):
        metric_col_w = max(len(label) for _, label in metric_defs) + 1
        val_w = max(11, *(len(model_labels.get(m, m)) for m in models))

        CYAN = "\033[36m"
        print(f"\n\n{CYAN}{BOLD}{title}{RESET}")
        header = (
            "".ljust(ds_col_w)
            + "".ljust(metric_col_w)
            + "  ".join(model_labels.get(m, m).rjust(val_w) for m in models)
        )
        print(header)
        print("-" * len(header))

        for di, d in enumerate(datasets):
            if di > 0 and len(metric_defs) > 1:
                print()
            for i, (metric_key, metric_label) in enumerate(metric_defs):
                # Find best mean for this row
                means = {}
                for m in models:
                    model_data = data[d].get(source, {}).get(m)
                    if model_data and metric_key in model_data:
                        means[m] = model_data[metric_key]["mean"]
                best_val = max(means.values()) if means else None

                row = (dataset_labels.get(d, d) if i == 0 else "").ljust(ds_col_w)
                row += metric_label.ljust(metric_col_w)
                cells = []
                for m in models:
                    model_data = data[d].get(source, {}).get(m)
                    if model_data and metric_key in model_data:
                        met = model_data[metric_key]
                        is_best = met["mean"] == best_val
                        cell = fmt(met["mean"], met["std"], highlight=is_best)
                        # ANSI codes are invisible but affect rjust, so pad manually
                        pad = val_w - 11  # 11 = len("0.000±0.000")
                        cells.append((" " * max(0, pad)) + cell)
                    elif (m, d) in oom and i == 0:
                        pad = val_w - 3  # 3 = len("OOM")
                        cells.append(" " * max(0, pad) + OOM_STR)
                    else:
                        cells.append("-".rjust(val_w))
                row += "  ".join(cells)
                print(row)

    print_results_table("Fidelity:", [("fidelity", "Fidelity")], source="base")
    print_results_table("Utility:", [("utility", "Utility")], source="base")
    print_results_table("Detection:", [("detection", "Detection")], source="base")
    print_results_table("Privacy:", [("privacy", "Privacy")], source="dcr")
    print()


def _print_markdown_tables(datasets, models, model_labels, dataset_labels, data, oom):
    """Print primary metric scores as GitHub-flavored Markdown tables."""
    OOM = "❌"
    metric_tables = [
        ("Fidelity", "fidelity", "base"),
        ("Utility", "utility", "base"),
        ("Detection", "detection", "base"),
        ("Privacy", "privacy", "dcr"),
    ]

    available_models = [
        m
        for m in models
        if any(data[d].get(src, {}).get(m) for d in datasets for src in ("base", "dcr"))
    ]

    for title, metric_key, source in metric_tables:
        print(f"\n### {title}\n")
        header = "| Dataset | " + " | ".join(model_labels[m] for m in available_models) + " |"
        sep = "| --- | " + " | ".join("---" for _ in available_models) + " |"
        print(header)
        print(sep)
        for d in datasets:
            means = {}
            for m in available_models:
                model_data = data[d].get(source, {}).get(m)
                if model_data and metric_key in model_data:
                    means[m] = model_data[metric_key]["mean"]
            best_val = max(means.values()) if means else None

            cells = []
            for m in available_models:
                model_data = data[d].get(source, {}).get(m)
                if model_data and metric_key in model_data:
                    met = model_data[metric_key]
                    s = f"{met['mean']:.3f}"
                    if met["mean"] == best_val:
                        s = f"**{s}**"
                    cells.append(s)
                elif (m, d) in oom:
                    cells.append(OOM)
                else:
                    cells.append("-")
            print(f"| {dataset_labels.get(d, d)} | " + " | ".join(cells) + " |")
    print()


def _print_latex_tables(datasets, models, model_labels, dataset_labels, data, oom):
    """Print transposed LaTeX tables (models as columns, datasets as rows)."""

    # LaTeX-specific labels (override ASCII labels)
    latex_dataset_labels = {
        "adult": "Adult",
        "diabetes": "Diabetes",
        "electric_vehicles": "\\shortstack{Electric\\\\Vehicles}",
        "ddxplus": "DDXPlus",
        "github_issues": "\\shortstack{GitHub\\\\Issues}",
        "yelp": "Yelp",
    }
    latex_model_labels = {
        "tabby": "Tabby",
        "tvae": "TVAE",
        "ctgan": "CTGAN",
        "realtabformer": "REaLTabFormer",
        "mostlyai": "TabularARGN",
        "tabdiff": "TabDiff",
        "origami": "\\origami (ours)",
    }

    tables = {
        "fidelity": {
            "caption": "Fidelity metrics across datasets "
            "(mean $\\pm$ std over 3 replicates). Higher is better.",
            "label": "tab:fidelity",
            "source": "base",
            "metrics": [
                ("fidelity", "Overall score"),
                ("fidelity_shapes", "Shapes"),
                ("fidelity_trends", "Trends"),
            ],
            "higher_is_better": True,
        },
        "utility": {
            "caption": "Utility metrics across datasets (mean $\\pm$ std over 3 replicates). "
            "Higher is better. Overall utility normalizes TSTR $F_1$ by the "
            "corresponding real-data baseline.",
            "label": "tab:utility",
            "source": "base",
            "metrics": [
                ("utility", "Overall score"),
                ("utility_tstr_f1_weighted", "TSTR $F_1$"),
            ],
            "higher_is_better": True,
        },
        "detection": {
            "caption": "Detection metrics across datasets (mean $\\pm$ std over 3 replicates). "
            "Detection score: higher means harder to detect (better). "
            "XGBoost classifier ROC AUC: lower means harder to distinguish "
            "from real data (better).",
            "label": "tab:detection",
            "source": "base",
            "metrics": [
                ("detection", "Overall score $\\uparrow$"),
                ("detection_roc_auc", "ROC AUC $\\downarrow$"),
            ],
            "higher_is_better": {
                "detection": True,
                "detection_roc_auc": False,
            },
        },
        "privacy": {
            "caption": "Privacy metrics across datasets (mean $\\pm$ std over 3 replicates). "
            "Privacy score: higher is better. "
            "DCR score $\\leq$ 50 indicates no memorization. "
            "Exact-match counts are discussed in the text.",
            "label": "tab:privacy",
            "source": "dcr",
            "metrics": [
                ("privacy", "Overall score $\\uparrow$"),
                ("privacy_dcr_score", "DCR $\\downarrow$"),
            ],
            "higher_is_better": {
                "privacy": True,
                "privacy_dcr_score": None,
            },
        },
    }

    def fmt_val(mean, std, is_count=False, bold=False):
        if is_count:
            mean_str, std_str = f"{mean:.1f}", f"{std:.1f}"
        else:
            mean_str, std_str = f"{mean:.3f}", f"{std:.3f}"
        if bold:
            return (
                f"$\\underline{{\\mathbf{{{mean_str}}}}}"
                f"{{\\color{{gray}}\\scriptstyle\\,\\pm\\,{std_str}}}$"
            )
        return f"${mean_str}{{\\color{{gray}}\\scriptstyle\\,\\pm\\,{std_str}}}$"

    def find_best(table_cfg, metric_key):
        source = table_cfg["source"]
        hib = table_cfg["higher_is_better"]
        direction = hib.get(metric_key) if isinstance(hib, dict) else hib
        if direction is None:
            return {}
        best = {}
        for ds in datasets:
            best_val, best_models = None, []
            for model in models:
                model_data = data[ds].get(source, {}).get(model)
                if model_data is None or metric_key not in model_data:
                    continue
                val = model_data[metric_key]["mean"]
                if (
                    best_val is None
                    or (direction and val > best_val)
                    or (not direction and val < best_val)
                ):
                    best_val, best_models = val, [model]
                elif val == best_val:
                    best_models.append(model)
            for m in best_models:
                best[(ds, m)] = True
        return best

    print("% Auto-generated results tables (models as columns, datasets as rows)")
    print("% Requires: \\usepackage{booktabs, multirow, xcolor}")
    print()

    for table_name, table_cfg in tables.items():
        source = table_cfg["source"]
        metrics = table_cfg["metrics"]

        available_models = [
            m for m in models if any(m in data[ds].get(source, {}) for ds in datasets)
        ]
        if not available_models:
            continue

        col_spec = "ll" + "c" * len(available_models)

        best_per_metric = {mk: find_best(table_cfg, mk) for mk, _ in metrics}

        lines = []
        lines.append("\\begin{table*}[t]")
        lines.append(f"  \\caption{{{table_cfg['caption']}}}")
        lines.append(f"  \\label{{{table_cfg['label']}}}")
        lines.append("  \\small")
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append("    \\toprule")

        header_cols = ["", ""] + [latex_model_labels[m] for m in available_models]
        lines.append("    " + " & ".join(header_cols) + " \\\\")
        lines.append("    \\midrule")

        for i, ds in enumerate(datasets):
            ds_label = latex_dataset_labels[ds]
            num_metrics = len(metrics)

            for j, (metric_key, metric_label) in enumerate(metrics):
                is_primary = j == 0
                row_parts = []
                if j == 0:
                    row_parts.append(f"\\multirow{{{num_metrics}}}{{*}}{{{ds_label}}}")
                else:
                    row_parts.append("")

                row_parts.append(f"  {metric_label}" if is_primary else f"  \\quad {metric_label}")

                for model in available_models:
                    is_oom = (model, ds) in oom
                    model_data = data[ds].get(source, {}).get(model)
                    if model_data is None or metric_key not in model_data:
                        row_parts.append("OOM" if is_oom and is_primary else "--")
                    else:
                        mean_val = model_data[metric_key]["mean"]
                        std_val = model_data[metric_key]["std"]
                        is_count = metric_key.startswith("privacy_exact_matches")
                        is_best = is_primary and best_per_metric[metric_key].get((ds, model), False)
                        row_parts.append(
                            fmt_val(mean_val, std_val, is_count=is_count, bold=is_best)
                        )

                lines.append("    " + " & ".join(row_parts) + " \\\\")

            if i < len(datasets) - 1:
                lines.append("    \\addlinespace")

        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table*}")

        print(f"% === {table_name.upper()} ===")
        print("\n".join(lines))
        print()


def cmd_all(args: argparse.Namespace) -> None:
    """Run the full pipeline: data -> train -> sample -> eval."""
    with _sync_context(args):
        # Clear remote so sub-commands don't each start their own sync
        saved_remote = getattr(args, "remote", None)
        args.remote = None
        cmd_data(args)
        cmd_train(args)
        cmd_sample(args)
        cmd_eval(args)
        args.remote = saved_remote


def cmd_full_suite(args: argparse.Namespace) -> None:
    """Run all model+dataset combos in base or DCR mode with V100 overrides."""
    from .suite import run_full_suite

    def _split(val: str | None) -> list[str] | None:
        if not val:
            return None
        return [s.strip() for s in val.split(",") if s.strip()]

    status = run_full_suite(
        dcr=args.dcr,
        reverse=args.reverse,
        output_dir=args.output_dir,
        remote=getattr(args, "remote", None),
        replicates=args.replicates,
        num_workers=args.num_workers,
        no_wandb=getattr(args, "no_wandb", False),
        max_minutes=args.max_minutes,
        models=_split(getattr(args, "models", None)),
        datasets=_split(getattr(args, "datasets", None)),
    )
    if any(v == "failed" for v in status.values()):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="origami-jsynth",
        description="Origami tabular/JSON synthesis experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", required=True, choices=DATASET_NAMES, help="Dataset name")
        p.add_argument("--output-dir", default="./results", help="Base output directory")
        p.add_argument(
            "--model",
            default="origami",
            choices=MODEL_NAMES,
            help="Synthesizer model (default: origami)",
        )
        p.add_argument(
            "--dcr", action="store_true", help="DCR mode: 50/50 split, privacy eval only"
        )
        p.add_argument(
            "--remote",
            default=None,
            metavar="S3_URL",
            help="S3 URL to sync results to (e.g. s3://bucket/results). "
            "Syncs every 5 minutes during training.",
        )

    # data
    p_data = subparsers.add_parser("data", help="Download and prepare dataset")
    add_common_args(p_data)
    p_data.set_defaults(func=cmd_data)

    # train
    p_train = subparsers.add_parser("train", help="Train a synthesizer model")
    add_common_args(p_train)
    p_train.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config/model parameter (e.g. --param epochs=300)",
    )
    p_train.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Maximum training wall-clock time in minutes (default: no limit)",
    )
    p_train.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (enabled by default when wandb is installed)",
    )
    p_train.set_defaults(func=cmd_train)

    # sample
    p_sample = subparsers.add_parser("sample", help="Generate synthetic data")
    add_common_args(p_sample)
    p_sample.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    p_sample.add_argument(
        "-R",
        "--replicates",
        type=int,
        default=1,
        help="Number of independent sampling rounds (default: 1)",
    )
    p_sample.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override sampling parameter (e.g. --param sample_batch_size=512)",
    )
    p_sample.set_defaults(func=cmd_sample)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate synthetic data")
    add_common_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    # results
    p_results = subparsers.add_parser("results", help="Show evaluation status and results overview")
    p_results.add_argument("--output-dir", default="./results", help="Base output directory")
    p_results.add_argument(
        "--latex", action="store_true", help="Output LaTeX tables instead of ASCII"
    )
    p_results.add_argument(
        "--markdown", action="store_true", help="Output Markdown tables instead of ASCII"
    )
    p_results.set_defaults(func=cmd_overview)

    # all
    p_all = subparsers.add_parser("all", help="Run full pipeline: data -> train -> sample -> eval")
    add_common_args(p_all)
    p_all.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    p_all.add_argument(
        "-R",
        "--replicates",
        type=int,
        default=1,
        help="Number of independent sampling rounds (default: 1)",
    )
    p_all.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config/model parameter (e.g. --param epochs=300)",
    )
    p_all.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Maximum training wall-clock time in minutes (default: no limit)",
    )
    p_all.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (enabled by default when wandb is installed)",
    )
    p_all.set_defaults(func=cmd_all)

    # full-suite
    p_suite = subparsers.add_parser(
        "full-suite",
        help="Run all model+dataset combos in base or DCR mode with V100 overrides",
    )
    p_suite.add_argument("--output-dir", default="./results", help="Base output directory")
    p_suite.add_argument(
        "--dcr",
        action="store_true",
        help="DCR mode: run privacy-only evaluation instead of fidelity+utility+detection",
    )
    p_suite.add_argument(
        "--remote",
        default=None,
        metavar="S3_URL",
        help="S3 URL to sync results to (syncs every 5 min throughout the full suite)",
    )
    p_suite.add_argument(
        "-R",
        "--replicates",
        type=int,
        default=10,
        help="Number of independent sampling rounds (default: 10)",
    )
    p_suite.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel sampling workers",
    )
    p_suite.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Default max training time per combo in minutes "
        "(overridden by per-combo V100 settings)",
    )
    p_suite.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    p_suite.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse iteration order: github_issues→adult, origami→tvae",
    )
    p_suite.add_argument(
        "--models",
        default=None,
        metavar="M1,M2,...",
        help="Comma-separated list of models to include (default: all)",
    )
    p_suite.add_argument(
        "--datasets",
        default=None,
        metavar="D1,D2,...",
        help="Comma-separated list of datasets to include (default: all)",
    )
    p_suite.set_defaults(func=cmd_full_suite)

    args = parser.parse_args()
    log_dir = _derive_log_dir(args)
    try:
        if log_dir is not None:
            from ._logging import TeeLogger

            with TeeLogger(log_dir, cmd_name=args.command):
                args.func(args)
        else:
            args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
