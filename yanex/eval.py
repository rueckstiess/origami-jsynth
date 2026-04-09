"""Yanex stage 4: evaluate synthetic replicates against real data.

Loads synthetic replicates from the sample dependency, runs evaluation
for each replicate, aggregates results, and logs summary metrics to yanex.

Usage:
    yanex run yanex/eval.py \\
        -D samples=<sample-experiment-id> \\
        -n "adult-1-eval"
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import yanex
except ImportError as e:
    raise SystemExit(
        "yanex is not installed. Install it with: pip install yanex\n"
        "This script must be run via: yanex run yanex/eval.py"
    ) from e

from origami_jsynth.eval import evaluate_dataset

# =============================================================================
# Dependencies: data dir transitively, samples dir directly
# =============================================================================

yanex.assert_dependency("sample.py", "samples")

dataset: str = yanex.get_graph().get_param("dataset")
try:
    dcr: bool = yanex.get_graph().get_param("dcr")
except:
    dcr = False
all_deps = {dep.script_path.stem: dep for dep in yanex.get_dependencies(transitive=True)}
data_dir: Path = all_deps["data"].artifacts_dir
samples_dir: Path = yanex.get_dependency("samples").artifacts_dir
report_dir: Path = yanex.get_artifacts_dir()

print(f"Dataset: {dataset}")
print(f"DCR mode: {dcr}")
print(f"Data dir: {data_dir}")
print(f"Samples dir: {samples_dir}")
print(f"Report dir: {report_dir}")

# =============================================================================
# Evaluate
# =============================================================================

evaluate_dataset(dataset, data_dir, samples_dir, report_dir, dcr=dcr)

with open(report_dir / "agg_results.json") as f:
    agg = json.load(f)

yanex.log_metrics({k: v["mean"] for k, v in agg["metrics"].items()})

print("\nEvaluation complete.")
