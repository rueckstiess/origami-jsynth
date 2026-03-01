#!/usr/bin/env python3
"""Convert a MostlyAI Engine workspace into CLI-compatible checkpoint files.

After killing a long-running ``origami-jsynth all`` run mid-training, the
workspace directory already contains trained model weights but the adapter's
``save()`` was never called.  This script reconstructs the two files that
``MostlyAIAdapter.load()`` expects:

    <checkpoint_dir>/model.pkl          — pickled TabularARGN instance
    <checkpoint_dir>/preprocess_state.pkl — PreprocessingState for record conversion

Usage:
    python scripts/recover_mostlyai_checkpoint.py \
        --dataset ddxplus --dcr

After running, you can resume the pipeline from sampling:

    origami-jsynth sample --dataset ddxplus --model mostlyai --dcr -R 10
    origami-jsynth eval   --dataset ddxplus --model mostlyai --dcr
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from origami_jsynth.baselines._preprocessing import PreprocessingState, records_to_dataframe
from origami_jsynth.data import load_jsonl
from origami_jsynth.registry import get_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. ddxplus)")
    parser.add_argument("--dcr", action="store_true", help="DCR mode (matches origami-jsynth --dcr)")
    parser.add_argument("--output-dir", default="./results", help="Base output directory (default: ./results)")
    args = parser.parse_args()

    dataset_name = f"{args.dataset}_dcr" if args.dcr else args.dataset
    base = Path(args.output_dir) / dataset_name
    checkpoint_dir = base / "mostlyai" / "checkpoints"
    workspace_dir = checkpoint_dir / "mostlyai_workspace"
    data_dir = base / "data"

    # --- Validate paths -----------------------------------------------------------
    if not workspace_dir.exists():
        sys.exit(f"Error: workspace not found at {workspace_dir}")

    model_weights = workspace_dir / "ModelStore" / "model-data" / "model-weights.pt"
    if not model_weights.exists():
        sys.exit(f"Error: no model weights found at {model_weights}")

    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        sys.exit(f"Error: training data not found at {train_path}")

    # --- Reconstruct PreprocessingState ------------------------------------------
    info = get_dataset(args.dataset)
    train_records = load_jsonl(train_path)
    print(f"Loaded {len(train_records)} training records from {train_path}")

    _, state = records_to_dataframe(train_records, tabular=info.tabular)
    state_path = checkpoint_dir / "preprocess_state.pkl"
    state.save(state_path)
    print(f"Saved PreprocessingState to {state_path}")

    # --- Reconstruct TabularARGN -------------------------------------------------
    from mostlyai.engine import TabularARGN

    model = TabularARGN(
        workspace_dir=str(workspace_dir),
        verbose=1,
    )
    model._fitted = True

    model_path = checkpoint_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved TabularARGN to {model_path}")

    print(f"\nDone. You can now run:\n  origami-jsynth sample --dataset {args.dataset} --model mostlyai{' --dcr' if args.dcr else ''}")


if __name__ == "__main__":
    main()
