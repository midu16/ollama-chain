"""Terminal progress bar for pipeline execution.

Provides a thread-safe progress bar rendered on stderr using carriage-return
updates.  When used as a context manager, verbose internal logging is silently
captured and only the progress bar is displayed.  On unhandled errors the
captured log is dumped for debugging.
"""

import sys
import threading
import time

_BAR_WIDTH = 30


class _LogCapture:
    """File-like object that captures writes to a buffer."""

    def __init__(self, buffer: list, real):
        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_lock", threading.Lock())

    def write(self, text: str) -> int:
        with self._lock:
            if len(self._buffer) < 5000:
                self._buffer.append(text)
        return len(text)

    def flush(self):
        pass

    def __getattr__(self, name):
        return getattr(self._real, name)


class ProgressBar:
    """Thread-safe terminal progress bar with elapsed timer.

    Usage::

        with ProgressBar() as bar:
            set_progress(bar)
            bar.update(10, "Drafting...")
            ...
            bar.finish()

    While active, ``sys.stderr`` is redirected so only the bar is visible.
    """

    def __init__(self):
        self._real_stderr = sys.stderr
        self._pct: float = 0.0
        self._stage: str = ""
        self._lock = threading.Lock()
        self._last_len: int = 0
        self._log_buf: list[str] = []
        self._active = False
        self._stage_t: float = 0.0
        self._start_t: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- context manager --

    def __enter__(self):
        self._real_stderr = sys.stderr
        self._active = True
        self._start_t = time.monotonic()
        self._stage_t = self._start_t
        sys.stderr = _LogCapture(self._log_buf, self._real_stderr)
        self._stop.clear()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._active = False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        sys.stderr = self._real_stderr
        _benign = (KeyboardInterrupt, SystemExit)
        if exc_type and exc_type not in _benign:
            self._clear()
            w = self._real_stderr.write
            w("\n--- internal log ---\n")
            for chunk in self._log_buf[-200:]:
                w(chunk)
            w("--- end log ---\n\n")
        elif exc_type in _benign:
            self._clear()
            self._real_stderr.write("\n")
        self._log_buf.clear()
        return False

    # -- public API --

    def update(self, pct: float, stage: str = ""):
        with self._lock:
            self._pct = min(max(pct, 0.0), 100.0)
            if stage:
                self._stage = stage
                self._stage_t = time.monotonic()
        self._render()

    def finish(self):
        total = time.monotonic() - self._start_t
        self.update(100.0, f"Complete ({total:.1f}s total)")
        self._real_stderr.write("\n\n")
        self._real_stderr.flush()

    # -- internals --

    def _tick(self):
        while not self._stop.wait(1.0):
            if self._active:
                self._render()

    def _render(self):
        if not self._active:
            return
        with self._lock:
            filled = int(_BAR_WIDTH * self._pct / 100)
            bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
            elapsed = time.monotonic() - self._stage_t
            t = f" ({elapsed:.0f}s)" if elapsed >= 2 else ""
            line = f"\r  [{bar}] {self._pct:5.1f}%  {self._stage}{t}"
            pad = max(0, self._last_len - len(line))
            out = line + " " * pad
            self._last_len = len(line)
        try:
            self._real_stderr.write(out)
            self._real_stderr.flush()
        except Exception:
            pass

    def _clear(self):
        try:
            self._real_stderr.write("\r" + " " * (self._last_len + 5) + "\r")
            self._real_stderr.flush()
        except Exception:
            pass


# -- module-level singleton --

_current: ProgressBar | None = None


def set_progress(bar: ProgressBar | None):
    """Set the module-level progress bar instance."""
    global _current
    _current = bar


def progress_update(pct: float, stage: str = ""):
    """Update the active progress bar.  No-op if none is active."""
    bar = _current
    if bar and bar._active:
        bar.update(pct, stage)
