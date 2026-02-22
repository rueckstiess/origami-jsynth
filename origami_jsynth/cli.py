"""CLI entry point for origami-jsynth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .baselines import MODEL_NAMES
from .registry import DATASET_NAMES


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
    has_samples = (
        paths["samples_dir"].exists()
        and any(paths["samples_dir"].glob("synthetic_*.jsonl"))
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
    n_train = len(train_records)
    df, _ = records_to_dataframe(train_records + test_records, tabular=info.tabular)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(
        f"Saved preprocessed CSVs ({df.shape[1]} columns) to {data_dir}"
    )


def cmd_train(args: argparse.Namespace) -> None:
    paths = _resolve_paths(args)
    _require_data(paths, args)

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
        )
    else:
        from .baselines import get_synthesizer
        from .data import load_jsonl
        from .registry import get_dataset

        info = get_dataset(args.dataset)
        kwargs = _parse_overrides(args.param)
        synth = get_synthesizer(args.model, tabular=info.tabular, **kwargs)

        train_records = load_jsonl(paths["data_dir"] / "train.jsonl")
        print(f"Training {args.model} on {args.dataset} ({len(train_records)} records)")
        synth.fit(train_records)
        synth.save(paths["checkpoint_dir"])
        print(f"Model saved to {paths['checkpoint_dir']}")


def cmd_sample(args: argparse.Namespace) -> None:
    from .data import load_jsonl, save_jsonl
    from .registry import get_dataset

    paths = _resolve_paths(args)
    info = get_dataset(args.dataset)

    _require_data(paths, args)
    _require_model(paths, args)

    if args.model == "origami":
        from .sample import sample_dataset

        sample_dataset(
            args.dataset,
            checkpoint_dir=paths["checkpoint_dir"],
            samples_dir=paths["samples_dir"],
            num_workers=args.num_workers,
            tabular=info.tabular,
            data_dir=paths["data_dir"],
            replicates=args.replicates,
        )
    else:
        from .baselines import get_synthesizer

        paths["samples_dir"].mkdir(parents=True, exist_ok=True)
        n_train = len(load_jsonl(paths["data_dir"] / "train.jsonl"))
        synth = type(get_synthesizer(args.model, tabular=info.tabular)).load(
            paths["checkpoint_dir"], tabular=info.tabular
        )

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
            print(f"Replicate {i}/{args.replicates}: saved {len(records)} records to {output_path}")


def cmd_eval(args: argparse.Namespace) -> None:
    from .eval import evaluate_dataset

    paths = _resolve_paths(args)

    _require_data(paths, args)
    _require_samples(paths, args)

    evaluate_dataset(
        args.dataset,
        data_dir=paths["data_dir"],
        samples_dir=paths["samples_dir"],
        report_dir=paths["report_dir"],
        dcr=args.dcr,
    )


def cmd_all(args: argparse.Namespace) -> None:
    """Run the full pipeline: data -> train -> sample -> eval."""
    cmd_data(args)
    cmd_train(args)
    cmd_sample(args)
    cmd_eval(args)


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
            "--model", default="origami", choices=MODEL_NAMES,
            help="Synthesizer model (default: origami)",
        )
        p.add_argument(
            "--dcr", action="store_true", help="DCR mode: 50/50 split, privacy eval only"
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
    p_train.set_defaults(func=cmd_train)

    # sample
    p_sample = subparsers.add_parser("sample", help="Generate synthetic data")
    add_common_args(p_sample)
    p_sample.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    p_sample.add_argument(
        "-R", "--replicates", type=int, default=1,
        help="Number of independent sampling rounds (default: 1)",
    )
    p_sample.set_defaults(func=cmd_sample)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate synthetic data")
    add_common_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    # all
    p_all = subparsers.add_parser("all", help="Run full pipeline: data -> train -> sample -> eval")
    add_common_args(p_all)
    p_all.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    p_all.add_argument(
        "-R", "--replicates", type=int, default=1,
        help="Number of independent sampling rounds (default: 1)",
    )
    p_all.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config/model parameter (e.g. --param epochs=300)",
    )
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
