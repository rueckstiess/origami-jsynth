"""Generate VLDB-formatted LaTeX results tables from agg_results.json files
and yanex experiment results."""

import csv
import io
import json
import subprocess
from pathlib import Path
from statistics import mean, stdev

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
YANEX_DIR = Path(__file__).resolve().parent.parent.parent / "origami-data-gen"
YANEX_EXPERIMENTS_DIR = Path.home() / ".yanex" / "experiments"

# Datasets and their DCR counterparts
DATASETS = ["adult", "diabetes", "electric_vehicles", "yelp", "ddxplus"]
DATASET_LABELS = {
    "adult": "Adult",
    "diabetes": "Diabetes",
    "electric_vehicles": "\\shortstack{Electric\\\\Vehicles}",
    "ddxplus": "DDXPlus",
    "yelp": "Yelp",
}

# Models in display order
MODELS = ["tabby", "tvae", "ctgan", "realtabformer", "mostlyai", "tabdiff", "origami"]
MODEL_LABELS = {
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "tabby": "Tabby",
    "realtabformer": "REaLTabFormer",
    "mostlyai": "TabularARGN",
    "origami": "\\origami (ours)",
    "tabdiff": "TabDiff",
}

# Models that ran out of memory on certain datasets.
# These get "OOM" on the primary metric row and "--" on sub-metrics.
OOM = {
    ("ctgan", "ddxplus"),
    ("ctgan", "yelp"),
    ("ctgan", "electric_vehicles"),
    ("tvae", "ddxplus"),
    ("tvae", "yelp"),
    ("tvae", "electric_vehicles"),
    ("realtabformer", "ddxplus"),
    ("tabby", "diabetes"),
    ("tabby", "electric_vehicles"),
    ("tabby", "yelp"),
    ("tabby", "ddxplus"),


}

# All metric keys we care about
METRIC_KEYS = [
    "fidelity",
    "fidelity_shapes",
    "fidelity_trends",
    "utility",
    "utility_trtr_roc_auc",
    "utility_tstr_roc_auc",
    "detection",
    "detection_roc_auc",
    "privacy",
    "privacy_dcr_score",
    "privacy_exact_matches_train",
    "privacy_exact_matches_test",
    "privacy_exact_matches_train_only",
]

# Table definitions
TABLES = {
    "fidelity": {
        "caption": "Fidelity metrics across datasets (mean $\\pm$ std over 10 replicates). Higher is better.",
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
        "caption": "Utility metrics across datasets (mean $\\pm$ std over 10 replicates). Higher is better.",
        "label": "tab:utility",
        "source": "base",
        "metrics": [
            ("utility_f1_ratio", "Overall score"),
            ("utility_trtr_f1_weighted", "TRTR $F_1$"),
            ("utility_tstr_f1_weighted", "TSTR $F_1$"),
        ],
        "higher_is_better": True,
    },
    "detection": {
        "caption": "Detection metrics across datasets (mean $\\pm$ std over 10 replicates). "
        "Detection score: higher means harder to detect (better). "
        "XGBoost classifier ROC AUC: lower means harder to distinguish from real data (better).",
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
        "caption": "Privacy metrics across datasets (mean $\\pm$ std over 10 replicates, 3 replicates for DDXPlus). "
        "Privacy score: higher is better. "
        "DCR score $\\leq$ 50 indicates no memorization.",
        "label": "tab:privacy",
        "source": "dcr",
        "metrics": [
            ("privacy", "Overall score $\\uparrow$"),
            ("privacy_dcr_score", "DCR $\\downarrow$"),
            ("privacy_exact_matches_train", "Exact match $\\downarrow$"),
        ],
        "higher_is_better": {
            "privacy": True,
            "privacy_dcr_score": None,
            "privacy_exact_matches_train": False,
        },
    },
}


def parse_yanex_name(name):
    """Parse a yanex experiment name into (dataset, model, source).

    Returns (dataset, model, source) or None if unparseable.
    Source is 'base' or 'dcr'.
    """
    # Determine model
    is_tabdiff = "tabdiff" in name
    model = "tabdiff" if is_tabdiff else "origami"

    # Determine source (base vs dcr/privacy)
    source = "dcr" if "dcr" in name else "base"

    # Determine dataset from prefix
    if name.startswith("adult-"):
        dataset = "adult"
    elif name.startswith("diabetes-"):
        dataset = "diabetes"
    elif name.startswith("electric-"):
        dataset = "electric_vehicles"
    elif name.startswith("ddxplus-"):
        dataset = "ddxplus"
    elif name.startswith("yelp-"):
        dataset = "yelp"
    else:
        return None

    return dataset, model, source


def load_yanex_results():
    """Load results from yanex experiments, returning aggregated metrics.

    Returns dict[dataset][source][model] -> dict[metric_key] -> {mean, std, values}
    """
    result = subprocess.run(
        [
            "yanex",
            "compare",
            "-t",
            "final",
            "--metrics",
            "fidelity*,utility*,detection*,privacy*",
            "--format",
            "csv",
        ],
        capture_output=True,
        text=True,
        cwd=YANEX_DIR,
    )
    if result.returncode != 0:
        print(f"Warning: yanex compare failed: {result.stderr}", flush=True)
        return {}

    reader = csv.DictReader(io.StringIO(result.stdout))

    # F1 metric keys extracted from detail files
    f1_detail_keys = [
        "utility_trtr_f1_weighted",
        "utility_tstr_f1_weighted",
        "utility_f1_ratio",
    ]
    all_keys = METRIC_KEYS + f1_detail_keys

    # Collect raw values grouped by (dataset, model, source)
    groups = {}  # (dataset, model, source) -> {metric_key: [values]}
    seen_ids = set()

    for row in reader:
        exp_id = row["meta:id"]
        # Deduplicate by experiment ID
        if exp_id in seen_ids:
            continue
        seen_ids.add(exp_id)

        name = row["meta:name"]
        parsed = parse_yanex_name(name)
        if parsed is None:
            continue

        dataset, model, source = parsed
        key = (dataset, model, source)

        if key not in groups:
            groups[key] = {m: [] for m in all_keys}

        for metric_key in METRIC_KEYS:
            csv_col = f"metric_{metric_key}"
            val = row.get(csv_col, "-")
            if val not in ("-", "", None):
                groups[key][metric_key].append(float(val))

        # Extract F1 from evaluation_details.json artifact
        details_path = (
            YANEX_EXPERIMENTS_DIR / exp_id / "artifacts" / "evaluation_details.json"
        )
        if details_path.exists():
            with open(details_path) as f:
                details = json.load(f)
            utility = details.get("utility", {})
            trtr_f1 = utility.get("trtr_metrics", {}).get("f1_weighted")
            tstr_f1 = utility.get("tstr_metrics", {}).get("f1_weighted")
            if trtr_f1 is not None:
                groups[key]["utility_trtr_f1_weighted"].append(trtr_f1)
            if tstr_f1 is not None:
                groups[key]["utility_tstr_f1_weighted"].append(tstr_f1)
            if trtr_f1 and tstr_f1:
                groups[key]["utility_f1_ratio"].append(tstr_f1 / trtr_f1)

    # Aggregate into mean/std
    data = {}
    for (dataset, model, source), metrics in groups.items():
        if dataset not in data:
            data[dataset] = {"base": {}, "dcr": {}}
        model_metrics = {}
        for metric_key, values in metrics.items():
            if values:
                m = mean(values)
                s = stdev(values) if len(values) > 1 else 0.0
                model_metrics[metric_key] = {
                    "mean": m,
                    "std": s,
                    "values": values,
                }
        if model_metrics:
            data[dataset][source][model] = model_metrics

    return data


def load_jsynth_results():
    """Load results from origami-jsynth agg_results.json and individual result files."""
    data = {}
    for dataset in DATASETS:
        if dataset not in data:
            data[dataset] = {"base": {}, "dcr": {}}
        for model in ["tvae", "ctgan", "realtabformer", "mostlyai", "tabby"]:
            # Base metrics from aggregated file
            base_path = RESULTS_DIR / dataset / model / "report" / "agg_results.json"
            if base_path.exists():
                with open(base_path) as f:
                    data[dataset]["base"][model] = json.load(f)["metrics"]

                # Extract F1 from individual result files
                f1_trtr_vals = []
                f1_tstr_vals = []
                f1_ratio_vals = []
                for i in range(1, 11):
                    rpath = (
                        RESULTS_DIR / dataset / model / "report" / f"results_{i}.json"
                    )
                    if rpath.exists():
                        with open(rpath) as f:
                            details = json.load(f).get("details", {})
                        utility = details.get("utility", {})
                        trtr_f1 = utility.get("trtr_metrics", {}).get("f1_weighted")
                        tstr_f1 = utility.get("tstr_metrics", {}).get("f1_weighted")
                        if trtr_f1 is not None:
                            f1_trtr_vals.append(trtr_f1)
                        if tstr_f1 is not None:
                            f1_tstr_vals.append(tstr_f1)
                        if trtr_f1 and tstr_f1:
                            f1_ratio_vals.append(tstr_f1 / trtr_f1)

                model_data = data[dataset]["base"][model]
                for key, vals in [
                    ("utility_trtr_f1_weighted", f1_trtr_vals),
                    ("utility_tstr_f1_weighted", f1_tstr_vals),
                    ("utility_f1_ratio", f1_ratio_vals),
                ]:
                    if vals:
                        model_data[key] = {
                            "mean": mean(vals),
                            "std": stdev(vals) if len(vals) > 1 else 0.0,
                            "values": vals,
                        }

            # DCR/privacy metrics
            dcr_path = (
                RESULTS_DIR / f"{dataset}_dcr" / model / "report" / "agg_results.json"
            )
            if dcr_path.exists():
                with open(dcr_path) as f:
                    data[dataset]["dcr"][model] = json.load(f)["metrics"]
    return data


def load_results():
    """Load and merge results from both sources."""
    data = {}
    for dataset in DATASETS:
        data[dataset] = {"base": {}, "dcr": {}}

    # Load jsynth results (ctgan, realtabformer)
    jsynth = load_jsynth_results()
    for dataset in DATASETS:
        for source in ("base", "dcr"):
            data[dataset][source].update(jsynth.get(dataset, {}).get(source, {}))

    # Load yanex results (origami, tabdiff)
    yanex = load_yanex_results()
    for dataset in DATASETS:
        for source in ("base", "dcr"):
            data[dataset][source].update(yanex.get(dataset, {}).get(source, {}))

    return data


def fmt_val(mean, std, precision=3, is_count=False):
    """Format a metric value as mean +/- std."""
    if is_count:
        return f"{mean:.1f} \\pm {std:.1f}"
    return f"{mean:.{precision}f} \\pm {std:.{precision}f}"


def find_best(data, table_cfg, metric_key):
    """Find which models are best for each dataset for bolding."""
    source = table_cfg["source"]
    hib = table_cfg["higher_is_better"]
    if isinstance(hib, dict):
        direction = hib.get(metric_key)
    else:
        direction = hib

    if direction is None:
        return {}

    best = {}
    for dataset in DATASETS:
        best_val = None
        best_models = []
        for model in MODELS:
            model_data = data[dataset].get(source, {}).get(model)
            if model_data is None or metric_key not in model_data:
                continue
            val = model_data[metric_key]["mean"]
            if best_val is None:
                best_val = val
                best_models = [model]
            elif (direction and val > best_val) or (not direction and val < best_val):
                best_val = val
                best_models = [model]
            elif val == best_val:
                best_models.append(model)
        for m in best_models:
            best[(dataset, m)] = True
    return best


def generate_table(_table_name, table_cfg, data):
    """Generate a single LaTeX table."""
    source = table_cfg["source"]
    metrics = table_cfg["metrics"]
    num_datasets = len(DATASETS)

    # Count available models for this table
    available_models = []
    for model in MODELS:
        has_data = any(model in data[ds].get(source, {}) for ds in DATASETS)
        if has_data:
            available_models.append(model)

    # Column spec: model name + sub-metric name + one col per dataset
    col_spec = "ll" + "c" * num_datasets

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append(f"  \\caption{{{table_cfg['caption']}}}")
    lines.append(f"  \\label{{{table_cfg['label']}}}")
    lines.append("  \\small")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Header row
    header_cols = ["", ""]
    for ds in DATASETS:
        header_cols.append(DATASET_LABELS[ds])
    lines.append("    " + " & ".join(header_cols) + " \\\\")
    lines.append("    \\midrule")

    # Precompute best values for bolding
    best_per_metric = {}
    for metric_key, _ in metrics:
        best_per_metric[metric_key] = find_best(data, table_cfg, metric_key)

    # Body: grouped by model
    for i, model in enumerate(available_models):
        model_label = MODEL_LABELS[model]
        num_metrics = len(metrics)

        for j, (metric_key, metric_label) in enumerate(metrics):
            is_primary = j == 0
            row_parts = []
            if j == 0:
                row_parts.append(
                    f"\\multirow{{{num_metrics}}}{{*}}{{{model_label}}}"
                )
            else:
                row_parts.append("")

            if is_primary:
                row_parts.append(f"  {metric_label}")
            else:
                row_parts.append(f"  \\quad {metric_label}")

            for ds in DATASETS:
                is_oom = (model, ds) in OOM
                model_data = data[ds].get(source, {}).get(model)
                if model_data is None or metric_key not in model_data:
                    if is_oom and is_primary:
                        row_parts.append("OOM")
                    else:
                        row_parts.append("--")
                else:
                    mean_val = model_data[metric_key]["mean"]
                    std_val = model_data[metric_key]["std"]
                    is_count = metric_key.startswith("privacy_exact_matches")
                    cell = fmt_val(mean_val, std_val, is_count=is_count)
                    is_best = is_primary and best_per_metric[metric_key].get(
                        (ds, model), False
                    )
                    if is_best:
                        row_parts.append(f"{{\\boldmath${cell}$}}")
                    else:
                        row_parts.append(f"${cell}$")

            lines.append("    " + " & ".join(row_parts) + " \\\\")

        # Add a small separator between models (but not after the last)
        if i < len(available_models) - 1:
            lines.append("    \\addlinespace")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def generate_table_transposed(_table_name, table_cfg, data):
    """Generate a LaTeX table with models as columns and datasets as rows."""
    source = table_cfg["source"]
    metrics = table_cfg["metrics"]

    # Find available models for this table
    available_models = []
    for model in MODELS:
        has_data = any(model in data[ds].get(source, {}) for ds in DATASETS)
        if has_data:
            available_models.append(model)

    num_models = len(available_models)

    # Column spec: dataset name + metric name + one col per model
    col_spec = "ll" + "c" * num_models

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append(f"  \\caption{{{table_cfg['caption']}}}")
    lines.append(f"  \\label{{{table_cfg['label']}}}")
    lines.append("  \\small")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Header row: model names
    header_cols = ["", ""]
    for model in available_models:
        header_cols.append(MODEL_LABELS[model])
    lines.append("    " + " & ".join(header_cols) + " \\\\")
    lines.append("    \\midrule")

    # Precompute best values for bolding
    best_per_metric = {}
    for metric_key, _ in metrics:
        best_per_metric[metric_key] = find_best(data, table_cfg, metric_key)

    # Body: grouped by dataset
    for i, ds in enumerate(DATASETS):
        ds_label = DATASET_LABELS[ds]
        num_metrics = len(metrics)

        for j, (metric_key, metric_label) in enumerate(metrics):
            is_primary = j == 0
            row_parts = []
            if j == 0:
                row_parts.append(
                    f"\\multirow{{{num_metrics}}}{{*}}{{{ds_label}}}"
                )
            else:
                row_parts.append("")

            if is_primary:
                row_parts.append(f"  {metric_label}")
            else:
                row_parts.append(f"  \\quad {metric_label}")

            for model in available_models:
                is_oom = (model, ds) in OOM
                model_data = data[ds].get(source, {}).get(model)
                if model_data is None or metric_key not in model_data:
                    if is_oom and is_primary:
                        row_parts.append("OOM")
                    else:
                        row_parts.append("--")
                else:
                    mean_val = model_data[metric_key]["mean"]
                    std_val = model_data[metric_key]["std"]
                    is_count = metric_key.startswith("privacy_exact_matches")
                    cell = fmt_val(mean_val, std_val, is_count=is_count)
                    is_best = is_primary and best_per_metric[metric_key].get(
                        (ds, model), False
                    )
                    if is_best:
                        row_parts.append(f"{{\\boldmath${cell}$}}")
                    else:
                        row_parts.append(f"${cell}$")

            lines.append("    " + " & ".join(row_parts) + " \\\\")

        # Separator between datasets (but not after the last)
        if i < len(DATASETS) - 1:
            lines.append("    \\addlinespace")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main():
    data = load_results()

    print("% Auto-generated results tables (models as rows, datasets as columns)")
    print("% Requires: \\usepackage{booktabs, multirow}")
    print()

    for table_name, table_cfg in TABLES.items():
        print(f"% === {table_name.upper()} ===")
        print(generate_table(table_name, table_cfg, data))
        print()

    print()
    print("% " + "=" * 60)
    print("% TRANSPOSED: models as columns, datasets as rows")
    print("% " + "=" * 60)
    print()

    for table_name, table_cfg in TABLES.items():
        # Use different labels to avoid LaTeX duplicate label warnings
        transposed_cfg = table_cfg
        print(f"% === {table_name.upper()} (transposed) ===")
        print(generate_table_transposed(table_name, transposed_cfg, data))
        print()


if __name__ == "__main__":
    main()
