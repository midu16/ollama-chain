"""Unit tests for scheduler.py — PromptJob, Scheduler lifecycle."""

import asyncio
import time

import pytest

from ollama_chain.scheduler import (
    PromptJob,
    Scheduler,
    _DEFAULT_JOB_TIMEOUT,
    _get_available_memory_ratio,
)


# ---------------------------------------------------------------------------
# PromptJob
# ---------------------------------------------------------------------------

class TestPromptJob:
    def test_defaults(self):
        job = PromptJob(id="abc", prompt="test", mode="cascade")
        assert job.status == "queued"
        assert job.web_search is True
        assert job.max_iterations == 15
        assert job.timeout == _DEFAULT_JOB_TIMEOUT
        assert job.result is None
        assert job.error is None
        assert job.progress == []
        assert job.created_at > 0
        assert job.started_at is None
        assert job.completed_at is None

    def test_to_dict(self):
        job = PromptJob(id="xyz", prompt="q", mode="fast")
        d = job.to_dict()
        assert d["job_id"] == "xyz"
        assert d["status"] == "queued"
        assert d["timeout"] == _DEFAULT_JOB_TIMEOUT
        assert "result" in d
        assert "error" in d
        assert "progress" in d

    def test_custom_timeout(self):
        job = PromptJob(id="t", prompt="q", mode="fast", timeout=120)
        assert job.timeout == 120
        assert job.to_dict()["timeout"] == 120


# ---------------------------------------------------------------------------
# Scheduler — basic operations (synchronous parts)
# ---------------------------------------------------------------------------

class TestSchedulerSync:
    def test_get_nonexistent_job(self):
        s = Scheduler()
        assert s.get("nonexistent") is None

    def test_queue_size_initially_zero(self):
        s = Scheduler()
        assert s.queue_size == 0

    def test_active_count_initially_zero(self):
        s = Scheduler()
        assert s.active_count == 0

    def test_cancel_nonexistent_returns_false(self):
        s = Scheduler()
        assert s.cancel("nonexistent") is False

    def test_queue_position_not_found(self):
        s = Scheduler()
        assert s.queue_position("missing") == -1


# ---------------------------------------------------------------------------
# Scheduler — async operations
# ---------------------------------------------------------------------------

class TestSchedulerAsync:
    @pytest.fixture
    def scheduler(self):
        return Scheduler(max_concurrent=1, default_job_timeout=30)

    @pytest.mark.asyncio
    async def test_submit_creates_job(self, scheduler):
        job = await scheduler.submit("test prompt", "cascade")
        assert job.id
        assert job.prompt == "test prompt"
        assert job.mode == "cascade"
        assert job.status == "queued"

    @pytest.mark.asyncio
    async def test_get_submitted_job(self, scheduler):
        job = await scheduler.submit("q", "fast")
        retrieved = scheduler.get(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id

    @pytest.mark.asyncio
    async def test_queue_position(self, scheduler):
        job1 = await scheduler.submit("q1", "cascade")
        job2 = await scheduler.submit("q2", "cascade")
        assert scheduler.queue_position(job1.id) == 0
        assert scheduler.queue_position(job2.id) == 1

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, scheduler):
        job = await scheduler.submit("q", "cascade")
        assert scheduler.cancel(job.id) is True
        assert job.status == "cancelled"
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_cannot_cancel_completed_job(self, scheduler):
        job = await scheduler.submit("q", "cascade")
        job.status = "completed"
        assert scheduler.cancel(job.id) is False

    @pytest.mark.asyncio
    async def test_cannot_cancel_timed_out_job(self, scheduler):
        job = await scheduler.submit("q", "cascade")
        job.status = "timed_out"
        assert scheduler.cancel(job.id) is False

    @pytest.mark.asyncio
    async def test_submit_with_kwargs(self, scheduler):
        job = await scheduler.submit(
            "q", "agent",
            web_search=False,
            max_iterations=5,
            timeout=120,
        )
        assert job.web_search is False
        assert job.max_iterations == 5
        assert job.timeout == 120

    @pytest.mark.asyncio
    async def test_queue_size_increments(self, scheduler):
        await scheduler.submit("q1", "fast")
        await scheduler.submit("q2", "fast")
        assert scheduler.queue_size == 2


# ---------------------------------------------------------------------------
# _get_available_memory_ratio
# ---------------------------------------------------------------------------

class TestMemoryRatio:
    def test_returns_float_or_none(self):
        result = _get_available_memory_ratio()
        if result is not None:
            assert 0.0 <= result <= 1.0
