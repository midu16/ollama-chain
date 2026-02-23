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
import sys
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("ollama_chain.scheduler")

_DEFAULT_JOB_TIMEOUT = 600  # seconds (10 minutes)


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

    def to_dict(self) -> dict:
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

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        loop = asyncio.get_event_loop()
        for i in range(self._max_concurrent):
            task = loop.create_task(self._worker())
            self._worker_tasks.append(task)
            logger.debug("Worker %d started", i)

    async def shutdown(self) -> None:
        logger.info("Shutting down %d worker(s)", len(self._worker_tasks))
        for task in self._worker_tasks:
            task.cancel()
        for job_id, proc in self._active_procs.items():
            try:
                proc.terminate()
                logger.debug("Terminated subprocess for job %s", job_id)
            except ProcessLookupError:
                pass

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

        args = [sys.executable, "-m", "ollama_chain", "-m", job.mode]
        if not job.web_search:
            args.append("--no-search")
        if job.mode == "agent":
            args.extend(["--max-iterations", str(job.max_iterations)])
        args.append(job.prompt)

        logger.info("Job %s running (timeout=%ds): %s", job.id, job.timeout, " ".join(args))

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._active_procs[job.id] = proc
            logger.debug("Job %s subprocess pid=%d", job.id, proc.pid)

            stdout_chunks: list[bytes] = []

            async def _read_stdout() -> None:
                assert proc.stdout
                async for line in proc.stdout:
                    stdout_chunks.append(line)

            async def _read_stderr() -> None:
                assert proc.stderr
                async for line in proc.stderr:
                    decoded = line.decode(errors="replace").strip()
                    if decoded:
                        job.progress.append(decoded)
                        logger.debug("Job %s [stderr]: %s", job.id, decoded)

            timed_out = False
            try:
                await asyncio.wait_for(
                    asyncio.gather(_read_stdout(), _read_stderr()),
                    timeout=job.timeout,
                )
                returncode = await proc.wait()
            except asyncio.TimeoutError:
                timed_out = True
                elapsed = time.time() - job.started_at
                logger.error(
                    "Job %s timed out after %.1fs (limit=%ds), killing subprocess",
                    job.id, elapsed, job.timeout,
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                returncode = -1

            if job.status == "cancelled":
                logger.info("Job %s subprocess finished after cancel (rc=%d)", job.id, returncode)
                return

            elapsed = time.time() - job.started_at

            if timed_out:
                partial = b"".join(stdout_chunks).decode(errors="replace").strip()
                job.error = (
                    f"Job timed out after {elapsed:.0f}s (limit: {job.timeout}s). "
                    f"Try increasing the timeout or using a faster mode."
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
            job.completed_at = time.time()
