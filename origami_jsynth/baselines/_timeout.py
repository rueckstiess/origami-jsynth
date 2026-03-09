"""Wall-clock timeout for opaque training loops."""

from __future__ import annotations

import threading


class TrainingTimeout:
    """Context manager that interrupts training after *max_seconds*.

    Used to enforce a wall-clock time limit on third-party training loops
    where we cannot inject an epoch-level check.  Fires a ``KeyboardInterrupt``
    via ``_thread.interrupt_main()`` when the deadline expires, then **suppresses**
    the resulting exception so that control returns normally to the caller
    (allowing checkpoint saves to proceed).

    If *max_seconds* is ``None``, the context manager is a no-op.
    """

    def __init__(self, max_seconds: float | None) -> None:
        self.max_seconds = max_seconds
        self._timer: threading.Timer | None = None
        self._triggered = False

    def _fire(self) -> None:
        import _thread

        self._triggered = True
        print(
            f"\nTimeout reached: {self.max_seconds / 60:.1f} min elapsed. "
            "Stopping training."
        )
        _thread.interrupt_main()

    def __enter__(self) -> TrainingTimeout:
        if self.max_seconds is not None:
            self._timer = threading.Timer(self.max_seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()
            print(f"Training timeout set: {self.max_seconds / 60:.1f} minutes")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        if self._timer is not None:
            self._timer.cancel()
        # Suppress KeyboardInterrupt only if we triggered it
        return exc_type is KeyboardInterrupt and self._triggered
