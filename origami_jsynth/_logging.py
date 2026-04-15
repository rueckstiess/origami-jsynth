"""Per-combo log capture for CLI commands.

TeeLogger redirects fd 1 and fd 2 at the OS level so C-level writes (OpenMP
warnings, torch runtime errors, forked DataLoader worker output) are captured
alongside Python-level prints. Each log file line is prefixed with an
ISO-8601 timestamp and hostname; the original terminal output stays clean.
"""

from __future__ import annotations

import datetime
import os
import socket
import sys
import threading
from pathlib import Path
from typing import IO

_HOSTNAME = socket.gethostname()


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _prefix() -> bytes:
    return f"[{_now_iso()} {_HOSTNAME}] ".encode()


class TeeLogger:
    """Tee fd 1 / fd 2 to the original terminal and an enriched log file."""

    def __init__(self, log_dir: Path, cmd_name: str) -> None:
        self.log_dir = Path(log_dir)
        self.cmd_name = cmd_name
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._stdout_file: IO[bytes] | None = None
        self._stderr_file: IO[bytes] | None = None
        self._threads: list[threading.Thread] = []
        self._read_fds: list[int] = []

    def __enter__(self) -> TeeLogger:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._stdout_file = open(self.log_dir / "stdout.log", "ab")
        self._stderr_file = open(self.log_dir / "stderr.log", "ab")

        header = (
            f"=== Run started {_now_iso()} on {_HOSTNAME} "
            f"— cmd: {self.cmd_name} ===\n"
        ).encode()
        self._stdout_file.write(header)
        self._stdout_file.flush()
        self._stderr_file.write(header)
        self._stderr_file.flush()

        # Flush any buffered Python-level output before we swap fds.
        sys.stdout.flush()
        sys.stderr.flush()

        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        for target_fd, saved_fd, log_file in (
            (1, self._saved_stdout_fd, self._stdout_file),
            (2, self._saved_stderr_fd, self._stderr_file),
        ):
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, target_fd)
            os.close(write_fd)
            self._read_fds.append(read_fd)
            t = threading.Thread(
                target=self._reader,
                args=(read_fd, saved_fd, log_file),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Flush Python buffers, then restore the original fds. Closing the
        # write end of each pipe (via dup2) signals EOF to the reader threads.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # Restore fd 1 / fd 2 so subsequent writes hit the terminal again.
        # This also closes the pipe write ends, signalling EOF to the
        # reader threads. Keep the saved fds open until the threads have
        # drained the pipes — they're still using them as terminal_fd.
        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)

        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()

        if self._saved_stdout_fd is not None:
            os.close(self._saved_stdout_fd)
            self._saved_stdout_fd = None
        if self._saved_stderr_fd is not None:
            os.close(self._saved_stderr_fd)
            self._saved_stderr_fd = None

        for fd in self._read_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._read_fds.clear()

        if self._stdout_file is not None:
            self._stdout_file.close()
            self._stdout_file = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    @staticmethod
    def _reader(read_fd: int, terminal_fd: int, log_file: IO[bytes]) -> None:
        """Read bytes from the pipe, tee raw to terminal and prefixed to log."""
        buf = b""
        while True:
            try:
                chunk = os.read(read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break

            # Raw passthrough to terminal.
            try:
                os.write(terminal_fd, chunk)
            except OSError:
                pass

            # Split into lines for log file; buffer trailing partial line.
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    log_file.write(_prefix() + line + b"\n")
                    log_file.flush()
                except Exception:
                    pass

        if buf:
            try:
                log_file.write(_prefix() + buf + b"\n")
                log_file.flush()
            except Exception:
                pass
