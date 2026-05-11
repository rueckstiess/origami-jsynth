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
    _namespace_to_argv,
    run_full_suite,
)

_NOOP_SYNC = patch(
    "origami_jsynth.suite.RemoteSync",
    lambda **kw: nullcontext(),
)


class TestComboGeneration:
    def test_no_oom_combos_in_list(self):
        for dcr in (False, True):
            for reverse in (False, True):
                combos = _build_combos(dcr, reverse=reverse)
                for model, dataset, _dcr in combos:
                    assert (model, dataset) not in SKIP_OOM

    def test_base_combos_all_have_dcr_false(self):
        combos = _build_combos(False)
        assert all(dcr is False for _, _, dcr in combos)

    def test_dcr_combos_all_have_dcr_true(self):
        combos = _build_combos(True)
        assert all(dcr is True for _, _, dcr in combos)

    def test_each_non_oom_pair_present_in_each_mode(self):
        non_oom_pairs = {
            (m, d) for m in SUITE_MODELS for d in SUITE_DATASETS if (m, d) not in SKIP_OOM
        }
        for dcr in (False, True):
            combos = _build_combos(dcr)
            combo_pairs = {(m, d) for m, d, _ in combos}
            assert combo_pairs == non_oom_pairs

    def test_combo_count(self):
        non_oom = len(SUITE_MODELS) * len(SUITE_DATASETS) - len(SKIP_OOM)
        assert len(_build_combos(False)) == non_oom
        assert len(_build_combos(True)) == non_oom

    def test_reverse_produces_same_set_different_order(self):
        fwd = _build_combos(False, reverse=False)
        rev = _build_combos(False, reverse=True)
        assert set(fwd) == set(rev)
        assert fwd != rev
        assert fwd[0] == rev[-1]  # first forward == last reversed

    def test_all_models_and_datasets_represented(self):
        combos = _build_combos(False)
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

        def mock_run(args):
            call_log.append((args.model, args.dataset, args.dcr))
            return 0

        with _NOOP_SYNC, patch("origami_jsynth.suite._run_combo_subprocess", mock_run):
            status = run_full_suite(
                output_dir=str(tmp_path),
                replicates=1,
                num_workers=1,
                no_wandb=True,
            )

        assert status[("tvae", "adult", False)] == "skipped_done"
        assert ("tvae", "adult", False) not in call_log

    def test_oom_combos_skipped_base(self, tmp_path):
        with _NOOP_SYNC, patch("origami_jsynth.suite._run_combo_subprocess", lambda args: 0):
            status = run_full_suite(
                dcr=False,
                output_dir=str(tmp_path),
                replicates=1,
                num_workers=1,
                no_wandb=True,
            )

        for model, dataset in SKIP_OOM:
            assert status[(model, dataset, False)] == "skipped_oom"

    def test_oom_combos_skipped_dcr(self, tmp_path):
        with _NOOP_SYNC, patch("origami_jsynth.suite._run_combo_subprocess", lambda args: 0):
            status = run_full_suite(
                dcr=True,
                output_dir=str(tmp_path),
                replicates=1,
                num_workers=1,
                no_wandb=True,
            )

        for model, dataset in SKIP_OOM:
            assert status[(model, dataset, True)] == "skipped_oom"

    def test_dcr_flag_passed_to_subprocess(self, tmp_path):
        """run_full_suite(dcr=True) must pass dcr=True to every subprocess invocation."""
        seen_dcr = set()

        def mock_run(args):
            seen_dcr.add(args.dcr)
            return 0

        with _NOOP_SYNC, patch("origami_jsynth.suite._run_combo_subprocess", mock_run):
            run_full_suite(
                dcr=True,
                output_dir=str(tmp_path),
                replicates=1,
                num_workers=1,
                no_wandb=True,
            )

        assert seen_dcr == {True}

    def test_failure_continues_to_next(self, tmp_path):
        """A failing combo should not stop the suite."""
        call_count = {"n": 0}

        def mock_run(args):
            call_count["n"] += 1
            return 1 if call_count["n"] == 1 else 0

        with _NOOP_SYNC, patch("origami_jsynth.suite._run_combo_subprocess", mock_run):
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

    def test_nonzero_exit_marks_failed(self, tmp_path):
        """A subprocess exit code != 0 should be marked as failed."""
        call_count = {"n": 0}

        def mock_run(args):
            call_count["n"] += 1
            return 2 if call_count["n"] == 1 else 0

        with _NOOP_SYNC, patch("origami_jsynth.suite._run_combo_subprocess", mock_run):
            status = run_full_suite(
                output_dir=str(tmp_path),
                replicates=1,
                num_workers=1,
                no_wandb=True,
            )

        failed = [k for k, v in status.items() if v == "failed"]
        assert len(failed) == 1


class TestNamespaceToArgv:
    def _ns(self, **overrides):
        defaults = dict(
            dataset="adult",
            model="origami",
            dcr=False,
            output_dir="results",
            replicates=1,
            num_workers=4,
            param=[],
            max_minutes=None,
            no_wandb=False,
        )
        defaults.update(overrides)
        return _build_args(
            defaults["model"],
            defaults["dataset"],
            defaults["dcr"],
            defaults["output_dir"],
            defaults["replicates"],
            defaults["num_workers"],
            defaults["no_wandb"],
            defaults["max_minutes"],
        )

    def test_required_args_present(self):
        argv = _namespace_to_argv(self._ns())
        assert argv[0] == "all"
        assert "--dataset" in argv and argv[argv.index("--dataset") + 1] == "adult"
        assert "--model" in argv and argv[argv.index("--model") + 1] == "origami"
        assert "--output-dir" in argv
        assert "--num-workers" in argv
        assert "--replicates" in argv

    def test_dcr_flag_added_when_true(self):
        assert "--dcr" in _namespace_to_argv(self._ns(dcr=True))
        assert "--dcr" not in _namespace_to_argv(self._ns(dcr=False))

    def test_no_wandb_flag_added_when_true(self):
        assert "--no-wandb" in _namespace_to_argv(self._ns(no_wandb=True))
        assert "--no-wandb" not in _namespace_to_argv(self._ns(no_wandb=False))

    def test_max_minutes_only_when_set(self):
        argv = _namespace_to_argv(self._ns(max_minutes=None))
        assert "--max-minutes" not in argv
        argv = _namespace_to_argv(self._ns(max_minutes=30.0))
        assert "--max-minutes" in argv
        assert argv[argv.index("--max-minutes") + 1] == "30.0"

    def test_params_repeated(self):
        # _build_args dedupes via dict.fromkeys; pass distinct items.
        ns = self._ns()
        ns.param = ["foo=1", "bar.baz=2"]
        argv = _namespace_to_argv(ns)
        param_indices = [i for i, x in enumerate(argv) if x == "--param"]
        assert len(param_indices) == 2
        values = [argv[i + 1] for i in param_indices]
        assert values == ["foo=1", "bar.baz=2"]

    def test_remote_not_passed_to_child(self):
        """Outer RemoteSync handles S3 sync; child must not receive --remote."""
        argv = _namespace_to_argv(self._ns())
        assert "--remote" not in argv
