"""CLI entry point for origami-jsynth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .registry import DATASET_NAMES


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Compute standard paths from CLI arguments."""
    base = Path(args.output_dir) / args.dataset
    if args.dcr:
        base = base / "dcr"
    return {
        "data_dir": base / "data",
        "checkpoint_dir": base / "checkpoints",
        "samples_dir": base / "samples",
        "report_dir": base / "report",
    }


def _config_path(dataset: str) -> Path:
    """Get the config YAML path for a dataset."""
    return Path(__file__).parent.parent / "configs" / f"{dataset}.yaml"


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
    has_model = (
        (checkpoint_dir / "final.pt").exists()
        or (checkpoint_dir / "best.pt").exists()
        or list(checkpoint_dir.glob("epoch_*.pt"))
        if checkpoint_dir.exists()
        else False
    )
    if not has_model:
        print(
            f"Error: No trained model found in {checkpoint_dir}\n\n"
            f"Run this first:\n"
            f"  origami-jsynth train --dataset {args.dataset}{_dcr_flag(args)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_samples(paths: dict[str, Path], args: argparse.Namespace) -> None:
    """Check that synthetic samples exist, exit with helpful message if not."""
    synthetic_path = paths["samples_dir"] / "synthetic.jsonl"
    if not synthetic_path.exists():
        print(
            f"Error: Synthetic data not found at {synthetic_path}\n\n"
            f"Run this first:\n"
            f"  origami-jsynth sample --dataset {args.dataset}{_dcr_flag(args)}",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_data(args: argparse.Namespace) -> None:
    from .data import prepare_dataset

    prepare_dataset(
        args.dataset,
        Path(args.output_dir),
        dcr=args.dcr,
    )


def cmd_train(args: argparse.Namespace) -> None:
    from .train import train_dataset

    paths = _resolve_paths(args)
    config_path = _config_path(args.dataset)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    _require_data(paths, args)

    train_dataset(
        args.dataset,
        data_dir=paths["data_dir"],
        checkpoint_dir=paths["checkpoint_dir"],
        config_path=config_path,
        overrides=args.param,
    )


def cmd_sample(args: argparse.Namespace) -> None:
    from .registry import get_dataset
    from .sample import sample_dataset

    paths = _resolve_paths(args)
    info = get_dataset(args.dataset)

    _require_data(paths, args)
    _require_model(paths, args)

    sample_dataset(
        args.dataset,
        checkpoint_dir=paths["checkpoint_dir"],
        samples_dir=paths["samples_dir"],
        num_workers=args.num_workers,
        tabular=info.tabular,
        data_dir=paths["data_dir"],
    )


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
            "--dcr", action="store_true", help="DCR mode: 50/50 split, privacy eval only"
        )

    # data
    p_data = subparsers.add_parser("data", help="Download and prepare dataset")
    add_common_args(p_data)
    p_data.set_defaults(func=cmd_data)

    # train
    p_train = subparsers.add_parser("train", help="Train Origami model")
    add_common_args(p_train)
    p_train.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config value (e.g. --param training.num_epochs=2)",
    )
    p_train.set_defaults(func=cmd_train)

    # sample
    p_sample = subparsers.add_parser("sample", help="Generate synthetic data")
    add_common_args(p_sample)
    p_sample.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
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
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config value (e.g. --param training.num_epochs=2)",
    )
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
