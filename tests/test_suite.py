"""Tests for origami_jsynth.suite."""

import json
from contextlib import nullcontext
from unittest.mock import patch

from origami_jsynth.suite import (
    SKIP_OOM,
    SUITE_DATASETS,
    SUITE_MODELS,
    _build_args,
    _build_combos,
    _is_combo_complete,
    run_full_suite,
)

_NOOP_SYNC = patch(
    "origami_jsynth.suite.RemoteSync",
    lambda **kw: nullcontext(),
)


class TestComboGeneration:
    def test_no_oom_combos_in_list(self):
        combos = _build_combos()
        for model, dataset, _dcr in combos:
            assert (model, dataset) not in SKIP_OOM

    def test_each_non_oom_pair_has_base_and_dcr(self):
        combos = _build_combos()
        non_oom_pairs = {
            (m, d) for m in SUITE_MODELS for d in SUITE_DATASETS if (m, d) not in SKIP_OOM
        }
        for model, dataset in non_oom_pairs:
            assert (model, dataset, False) in combos, f"missing base: {model}+{dataset}"
            assert (model, dataset, True) in combos, f"missing DCR: {model}+{dataset}"

    def test_combo_count(self):
        combos = _build_combos()
        non_oom = len(SUITE_MODELS) * len(SUITE_DATASETS) - len(SKIP_OOM)
        assert len(combos) == non_oom * 2  # base + DCR

    def test_all_models_and_datasets_represented(self):
        combos = _build_combos()
        models = {m for m, _, _ in combos}
        datasets = {d for _, d, _ in combos}
        assert models == set(SUITE_MODELS)
        # Some datasets may be fully OOM'd out — check that non-OOM datasets appear
        expected_datasets = {
            d for d in SUITE_DATASETS if any((m, d) not in SKIP_OOM for m in SUITE_MODELS)
        }
        assert datasets == expected_datasets


class TestIsComboComplete:
    def test_incomplete_when_no_file(self, tmp_path):
        assert not _is_combo_complete(str(tmp_path), "tvae", "adult", False)

    def test_complete_when_agg_results_exists(self, tmp_path):
        report_dir = tmp_path / "adult" / "tvae" / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "agg_results.json").write_text("{}")
        assert _is_combo_complete(str(tmp_path), "tvae", "adult", False)

    def test_dcr_uses_suffixed_dir(self, tmp_path):
        report_dir = tmp_path / "adult_dcr" / "tvae" / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "agg_results.json").write_text("{}")
        assert _is_combo_complete(str(tmp_path), "tvae", "adult", True)
        # Base should still be incomplete
        assert not _is_combo_complete(str(tmp_path), "tvae", "adult", False)


class TestBuildArgs:
    def test_basic_args(self):
        args = _build_args("tvae", "adult", False, "./results", 10, 4, False)
        assert args.model == "tvae"
        assert args.dataset == "adult"
        assert args.dcr is False
        assert args.remote is None
        assert args.replicates == 10
        assert args.num_workers == 4
        assert args.no_wandb is False
        assert args.param == []
        assert args.max_minutes is None

    def test_global_max_minutes(self):
        args = _build_args("tvae", "adult", False, "./results", 10, 4, False, max_minutes=1440)
        assert args.max_minutes == 1440

    def test_per_combo_max_minutes_overrides_global(self):
        """Per-combo V100_OVERRIDES max_minutes takes precedence over global."""
        # Currently no per-combo max_minutes in V100_OVERRIDES, so global wins.
        args = _build_args("tabdiff", "yelp", False, "./results", 10, 4, False, max_minutes=60)
        assert args.max_minutes == 60

    def test_v100_overrides_applied(self):
        args = _build_args("tabdiff", "yelp", False, "./results", 10, 4, False)
        assert "batch_size=512" in args.param
        assert "check_val_every=20" in args.param
        assert "sample_batch_size=512" in args.param

    def test_no_duplicate_params(self):
        args = _build_args("tabdiff", "yelp", False, "./results", 10, 4, False)
        assert len(args.param) == len(set(args.param))

    def test_remote_always_none(self):
        """Remote is always None — the outer RemoteSync handles S3."""
        for model in SUITE_MODELS:
            args = _build_args(model, "adult", False, "./results", 1, 1, True)
            assert args.remote is None


class TestRunFullSuite:
    def test_resume_skips_completed(self, tmp_path):
        """Already-complete combos should be marked skipped_done, not re-run."""
        report_dir = tmp_path / "adult" / "tvae" / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "agg_results.json").write_text(json.dumps({"metrics": {}}))

        call_log = []

        def mock_cmd_all(args):
            call_log.append((args.model, args.dataset, args.dcr))

        with _NOOP_SYNC, patch("origami_jsynth.suite.cmd_all", mock_cmd_all):
            status = run_full_suite(
                output_dir=str(tmp_path),
                replicates=1,
                num_workers=1,
                no_wandb=True,
            )

        assert status[("tvae", "adult", False)] == "skipped_done"
        assert ("tvae", "adult", False) not in call_log

    def test_oom_combos_skipped(self, tmp_path):
        with _NOOP_SYNC, patch("origami_jsynth.suite.cmd_all", lambda args: None):
            status = run_full_suite(
                output_dir=str(tmp_path),
                replicates=1,
                num_workers=1,
                no_wandb=True,
            )

        for model, dataset in SKIP_OOM:
            assert status[(model, dataset, False)] == "skipped_oom"
            assert status[(model, dataset, True)] == "skipped_oom"

    def test_failure_continues_to_next(self, tmp_path):
        """A failing combo should not stop the suite."""
        call_count = {"n": 0}

        def mock_cmd_all(args):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated failure")

        with _NOOP_SYNC, patch("origami_jsynth.suite.cmd_all", mock_cmd_all):
            status = run_full_suite(
                output_dir=str(tmp_path),
                replicates=1,
                num_workers=1,
                no_wandb=True,
            )

        failed = [k for k, v in status.items() if v == "failed"]
        completed = [k for k, v in status.items() if v == "completed"]
        assert len(failed) == 1
        assert len(completed) > 0

    def test_sys_exit_caught(self, tmp_path):
        """sys.exit(1) from _require_* helpers should be caught."""
        first_call = {"done": False}

        def mock_cmd_all(args):
            if not first_call["done"]:
                first_call["done"] = True
                raise SystemExit(1)

        with _NOOP_SYNC, patch("origami_jsynth.suite.cmd_all", mock_cmd_all):
            status = run_full_suite(
                output_dir=str(tmp_path),
                replicates=1,
                num_workers=1,
                no_wandb=True,
            )

        failed = [k for k, v in status.items() if v == "failed"]
        assert len(failed) == 1
