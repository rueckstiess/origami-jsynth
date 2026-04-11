"""Periodic background sync of local results to a remote S3 bucket."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


class RemoteSync:
    """Context manager that periodically syncs a local directory to S3.

    Uses ``aws s3 sync`` via subprocess — works on Mac and Linux with no
    Python dependencies beyond stdlib.

    If *remote_url* is ``None``, the context manager is a no-op.
    """

    def __init__(
        self,
        local_dir: str | Path,
        remote_url: str | None,
        interval_seconds: float = 300,
    ) -> None:
        self.local_dir = str(local_dir)
        self.remote_url = remote_url
        self.interval_seconds = interval_seconds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    def _preflight(self) -> None:
        """Verify that the aws CLI is available and can write to the bucket."""
        if shutil.which("aws") is None:
            raise RuntimeError(
                "Remote sync requires the AWS CLI but 'aws' was not found on PATH.\n"
                "Install it with:  pip install awscli   (or see "
                "https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)\n"
                "Then configure credentials with:  aws configure"
            )

        # Write a small test object to verify credentials + bucket access
        test_key = f"{self.remote_url.rstrip('/')}/_sync_preflight"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("preflight\n")
            tmp = f.name

        try:
            cp = subprocess.run(
                ["aws", "s3", "cp", tmp, test_key, "--quiet"],
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0:
                raise RuntimeError(
                    f"Remote sync preflight failed — could not write to {self.remote_url}\n"
                    f"aws s3 cp stderr: {cp.stderr.strip()}\n"
                    "Check your AWS credentials (aws configure) and bucket permissions."
                )
            # Clean up the test object
            subprocess.run(
                ["aws", "s3", "rm", test_key, "--quiet"],
                capture_output=True,
            )
        finally:
            Path(tmp).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        """Run aws s3 sync (best-effort, never raises)."""
        try:
            # Newer botocore enables CRC checksums by default, wrapping
            # uploads in a non-seekable AwsChunkedWrapper that can't be
            # rewound on retry.  Use env var (works across CLI versions).
            env = {**os.environ, "AWS_REQUEST_CHECKSUM_CALCULATION": "when_required"}
            cp = subprocess.run(
                ["aws", "s3", "sync", self.local_dir, self.remote_url],
                env=env,
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0:
                print(
                    f"[RemoteSync] warning: sync failed: {cp.stderr.strip()}",
                    file=sys.stderr,
                )
            else:
                print(f"[RemoteSync] synced {self.local_dir} -> {self.remote_url}")
        except Exception as exc:
            print(f"[RemoteSync] warning: sync error: {exc}", file=sys.stderr)

    def _tick(self) -> None:
        """Timer callback: sync, then schedule the next tick."""
        self._sync()
        with self._lock:
            if self._timer is not None:  # not cancelled
                self._timer = threading.Timer(self.interval_seconds, self._tick)
                self._timer.daemon = True
                self._timer.start()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> RemoteSync:
        if self.remote_url is None:
            return self
        self._preflight()
        self._sync()
        self._timer = threading.Timer(self.interval_seconds, self._tick)
        self._timer.daemon = True
        self._timer.start()
        print(
            f"[RemoteSync] started — syncing every "
            f"{self.interval_seconds / 60:.0f} min to {self.remote_url}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        if self.remote_url is not None:
            print("[RemoteSync] final sync...")
            self._sync()
        return False
