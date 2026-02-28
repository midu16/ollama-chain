"""Unit tests for the progress bar module."""

import sys
import threading
import time
from io import StringIO

import pytest

from ollama_chain.progress import (
    ProgressBar,
    _LogCapture,
    progress_update,
    set_progress,
)


class TestLogCapture:
    def test_captures_writes(self):
        buf = []
        cap = _LogCapture(buf, sys.stderr)
        cap.write("hello")
        assert buf == ["hello"]

    def test_returns_length(self):
        buf = []
        cap = _LogCapture(buf, sys.stderr)
        n = cap.write("abc")
        assert n == 3

    def test_flush_is_noop(self):
        buf = []
        cap = _LogCapture(buf, sys.stderr)
        cap.flush()

    def test_buffer_limit(self):
        buf = []
        cap = _LogCapture(buf, sys.stderr)
        for i in range(6000):
            cap.write(f"line {i}")
        assert len(buf) == 5000

    def test_thread_safety(self):
        buf = []
        cap = _LogCapture(buf, sys.stderr)
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    cap.write(f"t{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(buf) <= 5000


class TestProgressBar:
    def test_update_clamps_percentage(self):
        bar = ProgressBar()
        bar._active = True
        bar._real_stderr = StringIO()
        bar._start_t = time.monotonic()
        bar._stage_t = bar._start_t

        bar.update(-10, "test")
        assert bar._pct == 0.0

        bar.update(200, "test")
        assert bar._pct == 100.0

    def test_update_sets_stage(self):
        bar = ProgressBar()
        bar._active = True
        bar._real_stderr = StringIO()
        bar._start_t = time.monotonic()
        bar._stage_t = bar._start_t

        bar.update(50, "Halfway")
        assert bar._stage == "Halfway"

    def test_context_manager_restores_stderr(self):
        original = sys.stderr
        with ProgressBar() as bar:
            assert sys.stderr is not original
            bar.update(50, "testing")
        assert sys.stderr is original

    def test_finish_sets_100(self):
        bar = ProgressBar()
        bar._active = True
        bar._real_stderr = StringIO()
        bar._start_t = time.monotonic()
        bar._stage_t = bar._start_t
        bar.finish()
        assert bar._pct == 100.0

    def test_render_inactive_is_noop(self):
        bar = ProgressBar()
        bar._active = False
        bar._render()


class TestSetProgress:
    def test_set_and_update(self):
        bar = ProgressBar()
        bar._active = True
        bar._real_stderr = StringIO()
        bar._start_t = time.monotonic()
        bar._stage_t = bar._start_t

        set_progress(bar)
        progress_update(42, "test stage")
        assert bar._pct == 42.0
        assert bar._stage == "test stage"
        set_progress(None)

    def test_update_without_bar_is_noop(self):
        set_progress(None)
        progress_update(50, "should not crash")

    def test_update_with_inactive_bar_is_noop(self):
        bar = ProgressBar()
        bar._active = False
        set_progress(bar)
        progress_update(50, "ignored")
        assert bar._pct == 0.0
        set_progress(None)
