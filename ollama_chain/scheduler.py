"""Memory-aware job scheduler for the ollama-chain API server.

Serialises prompt execution through an async queue so that only
``max_concurrent`` chain pipelines run at once (default: 1).  Before
dequeueing a job the scheduler checks available system memory and waits
if the ratio drops below a safety threshold, preventing OOM when
multiple prompts arrive concurrently.

Each job is executed as a subprocess (``python -m ollama_chain``),
which isolates model loading / unloading and makes cancellation safe.
"""

import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("ollama_chain.scheduler")

_DEFAULT_JOB_TIMEOUT = 600  # seconds — guaranteed minimum before idle checks begin
_WATCHDOG_INTERVAL = 10  # seconds between watchdog checks

_JOB_RETENTION_SECONDS = 3600  # keep completed jobs for 1 hour
_MAX_RETAINED_JOBS = 500  # hard cap on total jobs in memory
_EVICTION_INTERVAL = 120  # seconds between eviction sweeps

# Mode-specific idle timeouts: how long without ANY output (stdout or stderr)
# before the watchdog considers the job stuck.  Only checked after the base
# timeout elapses.  There is NO hard cap — as long as the process is producing
# output, the job runs indefinitely.
_MODE_IDLE_TIMEOUTS: dict[str, int] = {
    "agent": 600,      # long multi-model synthesis + tool calls
    "consensus": 480,  # N independent model calls + merge
    "cascade": 360,    # chains through all models with thinking
    "strong": 360,     # single large model with thinking
    "pipeline": 300,   # two-stage with thinking
    "verify": 300,     # two-stage with thinking
}
_DEFAULT_IDLE_TIMEOUT = 180  # fast, route, search, auto


def _idle_timeout_for_mode(mode: str) -> int:
    return _MODE_IDLE_TIMEOUTS.get(mode, _DEFAULT_IDLE_TIMEOUT)


# ---------------------------------------------------------------------------
# Job data model
# ---------------------------------------------------------------------------

@dataclass
class PromptJob:
    """A single prompt submitted for execution."""

    id: str
    prompt: str
    mode: str
    web_search: bool = True
    max_iterations: int = 15
    timeout: int = _DEFAULT_JOB_TIMEOUT
    status: str = "queued"  # queued | running | completed | failed | cancelled | timed_out
    result: str | None = None
    error: str | None = None
    progress: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    effective_deadline: float | None = None

    def to_dict(self) -> dict:
        remaining = None
        if self.status == "running" and self.effective_deadline is not None:
            remaining = max(0, self.effective_deadline - time.time())
        return {
            "job_id": self.id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "timeout": self.timeout,
            "remaining_seconds": remaining,
        }


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

_MIN_MEMORY_RATIO = 0.10


def _get_available_memory_ratio() -> float | None:
    """Return available/total memory ratio from /proc/meminfo, or None."""
    try:
        total = available = 0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1])
                if total and available:
                    return available / total
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Async job scheduler that gates chain execution on memory availability."""

    def __init__(self, max_concurrent: int = 1,
                 default_job_timeout: int = _DEFAULT_JOB_TIMEOUT):
        self._jobs: dict[str, PromptJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._max_concurrent = max_concurrent
        self._default_job_timeout = default_job_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_procs: dict[str, asyncio.subprocess.Process] = {}
        self._worker_tasks: list[asyncio.Task] = []
        self._insertion_order: list[str] = []
        self._eviction_task: asyncio.Task | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        loop = asyncio.get_event_loop()
        for i in range(self._max_concurrent):
            task = loop.create_task(self._worker())
            self._worker_tasks.append(task)
            logger.debug("Worker %d started", i)
        self._eviction_task = loop.create_task(self._eviction_loop())

    async def shutdown(self) -> None:
        logger.info("Shutting down %d worker(s)", len(self._worker_tasks))
        if self._eviction_task:
            self._eviction_task.cancel()
        for task in self._worker_tasks:
            task.cancel()
        for job_id, proc in list(self._active_procs.items()):
            await self._terminate_proc(job_id, proc)

    # -- public API ----------------------------------------------------------

    async def submit(self, prompt: str, mode: str, **kwargs) -> PromptJob:
        job = PromptJob(
            id=uuid.uuid4().hex[:12],
            prompt=prompt,
            mode=mode,
            web_search=kwargs.get("web_search", True),
            max_iterations=kwargs.get("max_iterations", 15),
            timeout=kwargs.get("timeout", self._default_job_timeout),
        )
        self._jobs[job.id] = job
        self._insertion_order.append(job.id)
        await self._queue.put(job.id)
        logger.info(
            "Job %s submitted: mode=%s  web_search=%s  timeout=%ds  prompt=%.100s",
            job.id, mode, job.web_search, job.timeout, prompt,
        )
        return job

    def get(self, job_id: str) -> PromptJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status in ("completed", "failed", "cancelled", "timed_out"):
            logger.debug("Cancel ignored for job %s (status=%s)", job_id, job.status if job else "N/A")
            return False
        prev = job.status
        job.status = "cancelled"
        job.completed_at = time.time()
        proc = self._active_procs.get(job_id)
        if proc:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        logger.info("Job %s cancelled (was %s)", job_id, prev)
        return True

    def extend_timeout(self, job_id: str, extra_seconds: int) -> dict | None:
        """Extend a running job's base timeout by *extra_seconds*.

        This pushes back the point at which idle-inactivity checks begin.
        There is no hard cap — the job already runs as long as it is active.
        """
        job = self._jobs.get(job_id)
        if not job or job.status != "running":
            return None
        job.timeout += extra_seconds
        idle_limit = _idle_timeout_for_mode(job.mode)
        base_deadline = (job.started_at or job.created_at) + job.timeout
        now = time.time()
        job.effective_deadline = max(base_deadline, now + idle_limit)
        remaining = max(0, job.effective_deadline - now)
        logger.info(
            "Job %s base timeout extended by %ds → timeout=%ds, "
            "estimated remaining %.0fs",
            job_id, extra_seconds, job.timeout, remaining,
        )
        job.progress.append(
            f"[scheduler] Timeout extended by {extra_seconds}s — "
            f"idle checking deferred until {job.timeout}s elapsed"
        )
        return {"extended_by": extra_seconds, "remaining": int(remaining), "timeout": job.timeout}

    def queue_position(self, job_id: str) -> int:
        """0-based position among queued jobs, or -1 if not queued."""
        pos = 0
        for jid in self._insertion_order:
            j = self._jobs.get(jid)
            if j and j.status == "queued":
                if jid == job_id:
                    return pos
                pos += 1
        return -1

    @property
    def queue_size(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "queued")

    @property
    def active_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "running")

    # -- eviction & cleanup --------------------------------------------------

    async def _eviction_loop(self) -> None:
        """Periodically evict completed jobs older than the retention window."""
        while True:
            try:
                await asyncio.sleep(_EVICTION_INTERVAL)
                self._evict_old_jobs()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("Eviction cycle error", exc_info=True)

    def _evict_old_jobs(self) -> None:
        now = time.time()
        terminal = frozenset({"completed", "failed", "cancelled", "timed_out"})
        evicted = 0
        for jid in list(self._insertion_order):
            job = self._jobs.get(jid)
            if not job:
                self._insertion_order.remove(jid)
                continue
            if job.status in terminal:
                age = now - (job.completed_at or job.created_at)
                if age > _JOB_RETENTION_SECONDS:
                    del self._jobs[jid]
                    self._insertion_order.remove(jid)
                    evicted += 1

        overflow = len(self._jobs) - _MAX_RETAINED_JOBS
        if overflow > 0:
            for jid in list(self._insertion_order):
                if overflow <= 0:
                    break
                job = self._jobs.get(jid)
                if job and job.status in terminal:
                    del self._jobs[jid]
                    self._insertion_order.remove(jid)
                    overflow -= 1
                    evicted += 1

        if evicted:
            logger.debug("Evicted %d old jobs (%d remaining)", evicted, len(self._jobs))

    async def _terminate_proc(self, job_id: str, proc: asyncio.subprocess.Process) -> None:
        """Terminate a subprocess, escalating to SIGKILL if needed."""
        try:
            proc.terminate()
            logger.debug("Terminated subprocess for job %s (pid=%s)", job_id, proc.pid)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                logger.warning("Force-killed subprocess for job %s", job_id)
            except ProcessLookupError:
                pass

    # -- internals -----------------------------------------------------------

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if not job or job.status == "cancelled":
                logger.debug("Skipping job %s (cancelled or missing)", job_id)
                self._queue.task_done()
                continue

            logger.debug("Worker dequeued job %s", job_id)
            await self._wait_for_memory(job)
            if job.status == "cancelled":
                logger.debug("Job %s cancelled while waiting for memory", job_id)
                self._queue.task_done()
                continue

            async with self._semaphore:
                await self._execute(job)
            self._queue.task_done()

    async def _wait_for_memory(self, job: PromptJob) -> None:
        while True:
            ratio = _get_available_memory_ratio()
            if ratio is None or ratio >= _MIN_MEMORY_RATIO:
                if ratio is not None:
                    logger.debug("Job %s: memory OK (%.1f%% available)", job.id, ratio * 100)
                return
            msg = (
                f"[scheduler] Memory low ({ratio:.0%} available), "
                f"waiting for resources..."
            )
            logger.warning("Job %s: %s", job.id, msg)
            job.progress.append(msg)
            await asyncio.sleep(5)
            if job.status == "cancelled":
                return

    async def _execute(self, job: PromptJob) -> None:
        job.status = "running"
        job.started_at = time.time()

        idle_limit = _idle_timeout_for_mode(job.mode)
        job.effective_deadline = job.started_at + job.timeout

        args = [sys.executable, "-m", "ollama_chain", "-m", job.mode]
        if not job.web_search:
            args.append("--no-search")
        if job.mode == "agent":
            args.extend(["--max-iterations", str(job.max_iterations)])
        args.append(job.prompt)

        logger.info(
            "Job %s running (base timeout=%ds, idle limit=%ds for mode '%s'): %s",
            job.id, job.timeout, idle_limit, job.mode, " ".join(args),
        )

        _MAX_STDOUT_BYTES = 10 * 1024 * 1024  # 10 MB cap

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            self._active_procs[job.id] = proc
            logger.debug("Job %s subprocess pid=%d", job.id, proc.pid)

            stdout_chunks: list[bytes] = []
            stdout_bytes = 0
            last_activity_time = time.time()

            async def _read_stdout() -> None:
                nonlocal last_activity_time, stdout_bytes
                assert proc.stdout
                async for line in proc.stdout:
                    if stdout_bytes < _MAX_STDOUT_BYTES:
                        stdout_chunks.append(line)
                        stdout_bytes += len(line)
                    last_activity_time = time.time()

            async def _read_stderr() -> None:
                nonlocal last_activity_time
                assert proc.stderr
                async for line in proc.stderr:
                    decoded = line.decode(errors="replace").strip()
                    if decoded:
                        if len(job.progress) < 5000:
                            job.progress.append(decoded)
                        last_activity_time = time.time()
                        logger.debug("Job %s [stderr]: %s", job.id, decoded)

            io_task = asyncio.ensure_future(
                asyncio.gather(_read_stdout(), _read_stderr())
            )

            timed_out = False
            past_base_logged = False

            while True:
                done, _ = await asyncio.wait(
                    {io_task}, timeout=_WATCHDOG_INTERVAL,
                )
                if done:
                    break
                if job.status == "cancelled":
                    break

                now = time.time()
                idle = now - last_activity_time
                # base_deadline recomputed each cycle so extend_timeout takes effect
                base_deadline = job.started_at + job.timeout

                # Rolling estimate for the UI: when the job would timeout
                # if no more output arrives.
                if now < base_deadline:
                    job.effective_deadline = base_deadline
                else:
                    job.effective_deadline = last_activity_time + idle_limit

                # Before the base deadline: never timeout (user asked for
                # at least this much time).
                if now < base_deadline:
                    continue

                # Past the base deadline: the ONLY timeout trigger is
                # prolonged inactivity.  No hard cap — if the process is
                # producing output, it runs as long as it needs.
                if idle >= idle_limit:
                    timed_out = True
                    logger.info(
                        "Job %s idle for %.0fs (limit %ds for mode '%s') "
                        "after %.0fs total, timing out",
                        job.id, idle, idle_limit, job.mode,
                        now - job.started_at,
                    )
                    break

                if not past_base_logged:
                    past_base_logged = True
                    logger.info(
                        "Job %s past base timeout (%ds) but still active — "
                        "will keep running (idle %.0fs / %ds limit)",
                        job.id, job.timeout, idle, idle_limit,
                    )
                    job.progress.append(
                        f"[scheduler] Past {job.timeout}s base timeout but "
                        f"still producing output — will keep running"
                    )

            if timed_out:
                elapsed = time.time() - job.started_at
                logger.error(
                    "Job %s timed out after %.1fs (base=%ds, idle=%.0fs/%ds), "
                    "killing subprocess",
                    job.id, elapsed, job.timeout,
                    time.time() - last_activity_time, idle_limit,
                )
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                await proc.wait()
                io_task.cancel()
                try:
                    await io_task
                except asyncio.CancelledError:
                    pass
                returncode = -1
            else:
                try:
                    await io_task
                except asyncio.CancelledError:
                    pass
                returncode = await proc.wait()

            if job.status == "cancelled":
                logger.info("Job %s subprocess finished after cancel (rc=%d)", job.id, returncode)
                return

            elapsed = time.time() - job.started_at

            if timed_out:
                partial = b"".join(stdout_chunks).decode(errors="replace").strip()
                idle_sec = int(time.time() - last_activity_time)
                suggestions = _timeout_suggestions(job.mode, idle_sec, idle_limit)
                job.error = (
                    f"Job timed out after {elapsed:.0f}s "
                    f"(idle: {idle_sec}s, idle limit: {idle_limit}s for mode '{job.mode}'). "
                    f"{suggestions}"
                )
                if partial:
                    job.result = partial
                    job.status = "timed_out"
                    logger.warning(
                        "Job %s timed out with partial result (%d bytes)",
                        job.id, len(partial),
                    )
                else:
                    job.status = "timed_out"
                    logger.error("Job %s timed out with no output", job.id)
            elif returncode == 0:
                job.result = b"".join(stdout_chunks).decode(errors="replace").strip()
                job.status = "completed"
                logger.info(
                    "Job %s completed in %.1fs (%d bytes)",
                    job.id, elapsed, len(job.result),
                )
            else:
                tail = job.progress[-5:] if job.progress else ["Chain execution failed"]
                job.error = "\n".join(tail)
                job.status = "failed"
                logger.error(
                    "Job %s failed (rc=%d, %.1fs): %s",
                    job.id, returncode, elapsed, job.error[:300],
                )
        except Exception as e:
            if job.status != "cancelled":
                job.error = str(e)
                job.status = "failed"
                logger.error("Job %s exception: %s", job.id, e, exc_info=True)
        finally:
            self._active_procs.pop(job.id, None)
            job.effective_deadline = None
            job.completed_at = time.time()


def _timeout_suggestions(mode: str, idle_sec: int, idle_limit: int) -> str:
    """Build a human-readable suggestion string for idle-timeout errors."""
    parts: list[str] = []

    parts.append(
        f"the process produced no output for {idle_sec}s "
        f"(limit: {idle_limit}s for '{mode}' mode)"
    )

    faster_modes = {
        "cascade": "fast, route, or verify",
        "consensus": "cascade or verify",
        "agent": "cascade (or reduce max_iterations)",
        "pipeline": "fast or route",
        "strong": "fast",
    }
    if mode in faster_modes:
        parts.append(f"try a faster mode: {faster_modes[mode]}")
    else:
        parts.append("try a simpler prompt or a faster mode")

    return "Suggestions: " + "; ".join(parts) + "."
