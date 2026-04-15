"""Tests for origami_jsynth._logging.TeeLogger and CLI log-dir derivation."""

import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from origami_jsynth.cli import _derive_log_dir


PREFIX_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+\-]\d{2}:\d{2} \S+\] "
)


class TestDeriveLogDir:
    def _ns(self, **kwargs):
        defaults = dict(dataset=None, model=None, output_dir=None, dcr=False)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_full_args_returns_path(self):
        ns = self._ns(dataset="adult", model="origami", output_dir="results")
        assert _derive_log_dir(ns) == Path("results/adult/origami")

    def test_dcr_appends_suffix(self):
        ns = self._ns(dataset="adult", model="tvae", output_dir="r", dcr=True)
        assert _derive_log_dir(ns) == Path("r/adult_dcr/tvae")

    def test_missing_dataset_returns_none(self):
        ns = self._ns(model="origami", output_dir="results")
        assert _derive_log_dir(ns) is None

    def test_missing_model_returns_none(self):
        ns = self._ns(dataset="adult", output_dir="results")
        assert _derive_log_dir(ns) is None

    def test_missing_output_dir_returns_none(self):
        ns = self._ns(dataset="adult", model="origami")
        assert _derive_log_dir(ns) is None


def _run_tee_in_subprocess(tmp_path: Path, body: str) -> tuple[str, str, int]:
    """Run a snippet inside a child python process under TeeLogger.

    Body runs inside `with TeeLogger(log_dir, 'test'):`. Returns
    (child_stdout, child_stderr, returncode).
    """
    script = textwrap.dedent(
        f"""
        import os, sys
        from pathlib import Path
        from origami_jsynth._logging import TeeLogger

        log_dir = Path({str(tmp_path)!r})
        with TeeLogger(log_dir, "test"):
        {textwrap.indent(textwrap.dedent(body), "            ")}
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    return proc.stdout, proc.stderr, proc.returncode


class TestTeeLogger:
    def test_stdout_tees_to_terminal_and_log(self, tmp_path):
        out, err, rc = _run_tee_in_subprocess(
            tmp_path, "print('hello stdout')"
        )
        assert rc == 0
        assert "hello stdout" in out

        log = (tmp_path / "stdout.log").read_text()
        assert "=== Run started" in log
        # Find the line and check the prefix.
        match = next((ln for ln in log.splitlines() if "hello stdout" in ln), None)
        assert match is not None
        assert PREFIX_RE.match(match) and match.endswith("hello stdout")

    def test_stderr_tees_to_terminal_and_log(self, tmp_path):
        out, err, rc = _run_tee_in_subprocess(
            tmp_path, "import sys; print('boom', file=sys.stderr)"
        )
        assert rc == 0
        assert "boom" in err

        log = (tmp_path / "stderr.log").read_text()
        match = next((ln for ln in log.splitlines() if "boom" in ln), None)
        assert match is not None
        assert PREFIX_RE.match(match)

    def test_captures_raw_fd_writes(self, tmp_path):
        """C-level writes (os.write) must be captured, not just sys.stdout."""
        out, _, rc = _run_tee_in_subprocess(
            tmp_path, "import os; os.write(1, b'raw fd write\\n')"
        )
        assert rc == 0
        assert "raw fd write" in out
        log = (tmp_path / "stdout.log").read_text()
        assert "raw fd write" in log

    def test_sys_exit_does_not_drop_terminal_output(self, tmp_path):
        """Regression: SystemExit during TeeLogger must not swallow terminal output."""
        out, err, rc = _run_tee_in_subprocess(
            tmp_path,
            "import sys; print('error msg', file=sys.stderr); sys.exit(2)",
        )
        assert rc == 2
        assert "error msg" in err
        log = (tmp_path / "stderr.log").read_text()
        assert "error msg" in log

    def test_appends_with_run_header(self, tmp_path):
        for i in range(2):
            _run_tee_in_subprocess(tmp_path, f"print('run {i}')")
        log = (tmp_path / "stdout.log").read_text()
        headers = [ln for ln in log.splitlines() if ln.startswith("=== Run started")]
        assert len(headers) == 2
        assert "run 0" in log and "run 1" in log
