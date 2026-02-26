"""Unit tests for scheduler.py — PromptJob, Scheduler lifecycle."""

import asyncio
import time

import pytest

from ollama_chain.scheduler import (
    PromptJob,
    Scheduler,
    _DEFAULT_JOB_TIMEOUT,
    _DEFAULT_IDLE_TIMEOUT,
    _get_available_memory_ratio,
    _idle_timeout_for_mode,
    _timeout_suggestions,
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
        assert d["remaining_seconds"] is None
        assert "result" in d
        assert "error" in d
        assert "progress" in d

    def test_to_dict_running_shows_remaining(self):
        job = PromptJob(id="r", prompt="q", mode="fast")
        job.status = "running"
        job.started_at = time.time()
        job.effective_deadline = time.time() + 300
        d = job.to_dict()
        assert d["remaining_seconds"] is not None
        assert 290 <= d["remaining_seconds"] <= 300

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

    @pytest.mark.asyncio
    async def test_extend_timeout_nonexistent(self, scheduler):
        assert scheduler.extend_timeout("missing", 300) is None

    @pytest.mark.asyncio
    async def test_extend_timeout_queued_returns_none(self, scheduler):
        job = await scheduler.submit("q", "fast")
        assert job.status == "queued"
        assert scheduler.extend_timeout(job.id, 300) is None

    @pytest.mark.asyncio
    async def test_extend_timeout_running(self, scheduler):
        job = await scheduler.submit("q", "fast")
        job.status = "running"
        job.started_at = time.time()
        job.effective_deadline = time.time() + 100
        old_timeout = job.timeout
        result = scheduler.extend_timeout(job.id, 200)
        assert result is not None
        assert result["extended_by"] == 200
        assert result["remaining"] > 0
        assert job.timeout == old_timeout + 200

    @pytest.mark.asyncio
    async def test_extend_timeout_accumulates(self, scheduler):
        job = await scheduler.submit("q", "fast")
        job.status = "running"
        job.started_at = time.time()
        job.effective_deadline = time.time() + 100
        old_timeout = job.timeout
        scheduler.extend_timeout(job.id, 300)
        scheduler.extend_timeout(job.id, 300)
        assert job.timeout == old_timeout + 600


# ---------------------------------------------------------------------------
# _timeout_suggestions
# ---------------------------------------------------------------------------

class TestTimeoutSuggestions:
    def test_cascade_suggestions(self):
        s = _timeout_suggestions("cascade", 360, 360)
        assert "no output" in s
        assert "faster mode" in s

    def test_fast_mode_suggests_simpler_prompt(self):
        s = _timeout_suggestions("fast", 180, 180)
        assert "no output" in s
        assert "simpler prompt" in s

    def test_agent_suggestions(self):
        s = _timeout_suggestions("agent", 600, 600)
        assert "max_iterations" in s

    def test_includes_idle_info(self):
        s = _timeout_suggestions("cascade", 400, 360)
        assert "400s" in s
        assert "360s" in s


# ---------------------------------------------------------------------------
# _idle_timeout_for_mode
# ---------------------------------------------------------------------------

class TestIdleTimeoutForMode:
    def test_agent_has_longest_idle(self):
        assert _idle_timeout_for_mode("agent") >= 600

    def test_fast_uses_default(self):
        assert _idle_timeout_for_mode("fast") == _DEFAULT_IDLE_TIMEOUT

    def test_unknown_mode_uses_default(self):
        assert _idle_timeout_for_mode("nonexistent") == _DEFAULT_IDLE_TIMEOUT

    def test_agent_longer_than_default(self):
        assert _idle_timeout_for_mode("agent") > _DEFAULT_IDLE_TIMEOUT

    def test_cascade_longer_than_default(self):
        assert _idle_timeout_for_mode("cascade") > _DEFAULT_IDLE_TIMEOUT


# ---------------------------------------------------------------------------
# _get_available_memory_ratio
# ---------------------------------------------------------------------------

class TestMemoryRatio:
    def test_returns_float_or_none(self):
        result = _get_available_memory_ratio()
        if result is not None:
            assert 0.0 <= result <= 1.0
