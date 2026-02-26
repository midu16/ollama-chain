"""HTTP API server for ollama-chain with memory-aware scheduling.

Start with::

    ollama-chain-server              # default 127.0.0.1:8585
    ollama-chain-server --port 9090  # custom port

Endpoints
---------
POST   /api/prompt                    Submit a prompt → {job_id, status, position}
GET    /api/prompt/{job_id}           Poll job status → full job dict
GET    /api/prompt/{job_id}/stream    SSE stream of progress + final result
PATCH  /api/prompt/{job_id}/timeout   Extend running job deadline → {extended_by, remaining}
DELETE /api/prompt/{job_id}           Cancel a running/queued job
GET    /api/models                    List available Ollama models
GET    /api/health                    Server health + queue stats

Logging
-------
All logs are written to ``.logs/ollama-chain-server.log`` at DEBUG level
(configurable via ``--log-dir``).  A summary INFO stream is also printed
to stderr.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time as _time

from aiohttp import web

from .chains import CLI_ONLY_MODES
from .scheduler import Scheduler

logger = logging.getLogger("ollama_chain.server")

scheduler = Scheduler()

_SSE_HEARTBEAT_INTERVAL = 15  # seconds between keepalive comments
_MAX_REQUEST_BODY = 64 * 1024  # 64 KB limit on JSON request bodies


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_FORMAT = (
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
)
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: str = ".logs") -> str:
    """Configure root logger with a DEBUG file handler and an INFO console handler.

    Returns the resolved path of the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "ollama-chain-server.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if root.handlers:
        root.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    root.addHandler(ch)

    logging.getLogger("aiohttp.access").setLevel(logging.DEBUG)

    return log_file


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        return web.Response(
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@web.middleware
async def request_logging_middleware(request: web.Request, handler):
    t0 = _time.monotonic()
    logger.debug(
        "%s %s  remote=%s",
        request.method, request.path, request.remote,
    )
    try:
        response = await handler(request)
        elapsed = (_time.monotonic() - t0) * 1000
        logger.debug(
            "%s %s  status=%d  %.1fms",
            request.method, request.path, response.status, elapsed,
        )
        return response
    except web.HTTPException as exc:
        elapsed = (_time.monotonic() - t0) * 1000
        logger.warning(
            "%s %s  status=%d  %.1fms  %s",
            request.method, request.path, exc.status, elapsed, exc.text,
        )
        raise
    except Exception as exc:
        elapsed = (_time.monotonic() - t0) * 1000
        logger.error(
            "%s %s  UNHANDLED  %.1fms  %s",
            request.method, request.path, elapsed, exc,
            exc_info=True,
        )
        raise


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def health(request: web.Request) -> web.Response:
    data = {
        "status": "ok",
        "queue_size": scheduler.queue_size,
        "active_jobs": scheduler.active_count,
    }
    logger.debug("Health check: %s", data)
    return web.json_response(data)


async def submit_prompt(request: web.Request) -> web.Response:
    if request.content_length and request.content_length > _MAX_REQUEST_BODY:
        raise web.HTTPRequestEntityTooLarge(
            max_size=_MAX_REQUEST_BODY,
            actual_size=request.content_length,
            text="Request body too large",
        )
    try:
        data = await request.json()
    except Exception:
        logger.warning("Invalid JSON body from %s", request.remote)
        raise web.HTTPBadRequest(text="Invalid JSON body")

    prompt = data.get("prompt", "").strip()
    if not prompt:
        logger.warning("Empty prompt from %s", request.remote)
        raise web.HTTPBadRequest(text="Missing 'prompt' field")

    mode = data.get("mode", "cascade")
    if mode in CLI_ONLY_MODES:
        logger.warning(
            "Rejected CLI-only mode '%s' from %s", mode, request.remote,
        )
        raise web.HTTPBadRequest(
            text=f"Mode '{mode}' is only available via the CLI"
        )
    ws = data.get("web_search", True)
    max_iter = data.get("max_iterations", 15)
    timeout = data.get("timeout", scheduler._default_job_timeout)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        timeout = scheduler._default_job_timeout
    timeout = max(int(timeout), 60)

    logger.info(
        "New prompt: mode=%s  web_search=%s  max_iter=%d  timeout=%ds  prompt=%.120s",
        mode, ws, max_iter, timeout, prompt,
    )

    job = await scheduler.submit(
        prompt, mode, web_search=ws, max_iterations=max_iter, timeout=timeout,
    )
    logger.info("Job %s queued (position %d)", job.id, scheduler.queue_position(job.id))
    return web.json_response(
        {
            "job_id": job.id,
            "status": job.status,
            "position": scheduler.queue_position(job.id),
        },
        status=202,
    )


async def get_job(request: web.Request) -> web.Response:
    job_id = request.match_info["job_id"]
    job = scheduler.get(job_id)
    if not job:
        logger.debug("Job %s not found", job_id)
        raise web.HTTPNotFound(text="Job not found")

    result = job.to_dict()
    result["position"] = scheduler.queue_position(job_id)
    logger.debug("Job %s polled: status=%s", job_id, job.status)
    return web.json_response(result)


_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "timed_out"})


async def stream_job(request: web.Request) -> web.StreamResponse:
    """Server-Sent Events stream for a job.

    If the job has already reached a terminal state, a JSON response
    identical to ``GET /api/prompt/{job_id}`` is returned instead of
    opening an SSE stream.  This prevents clients from entering an
    infinite reconnection loop when they miss the terminal event.

    Events emitted (SSE):
      queued    – {position}  while waiting in queue
      progress  – {line}      stderr lines from the chain subprocess
      complete  – {result}    final answer text
      timed_out – {error, partial_result}  job exceeded its timeout
      error     – {error}     if the chain failed
      cancelled – {}          if the job was cancelled
    """
    job_id = request.match_info["job_id"]
    job = scheduler.get(job_id)
    if not job:
        logger.debug("Stream requested for unknown job %s", job_id)
        raise web.HTTPNotFound(text="Job not found")

    if job.status in _TERMINAL_STATUSES:
        logger.info(
            "Job %s already terminal (%s), returning JSON instead of SSE",
            job_id, job.status,
        )
        result = job.to_dict()
        result["position"] = scheduler.queue_position(job_id)
        return web.json_response(result)

    logger.info("SSE stream opened for job %s (status=%s)", job_id, job.status)

    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    last_idx = 0
    last_write = _time.monotonic()
    _ACTIVE_STATUSES = ("queued", "running")
    while job.status in _ACTIVE_STATUSES:
        wrote_something = False
        while last_idx < len(job.progress):
            line = job.progress[last_idx]
            await response.write(
                f"event: progress\ndata: {json.dumps({'line': line})}\n\n".encode()
            )
            last_idx += 1
            wrote_something = True

        if job.status == "queued":
            pos = scheduler.queue_position(job_id)
            await response.write(
                f"event: queued\ndata: {json.dumps({'position': pos})}\n\n".encode()
            )
            wrote_something = True

        now = _time.monotonic()
        if wrote_something:
            last_write = now
        elif now - last_write >= _SSE_HEARTBEAT_INTERVAL:
            elapsed = now - (job.started_at or job.created_at)
            await response.write(
                f": keepalive {elapsed:.0f}s\n\n".encode()
            )
            last_write = now
            logger.debug("SSE heartbeat for job %s (%.0fs elapsed)", job_id, elapsed)

        sleep_interval = 1.0 if job.status == "queued" else 0.3
        await asyncio.sleep(sleep_interval)

    while last_idx < len(job.progress):
        line = job.progress[last_idx]
        await response.write(
            f"event: progress\ndata: {json.dumps({'line': line})}\n\n".encode()
        )
        last_idx += 1

    if job.status == "completed":
        await response.write(
            f"event: complete\ndata: {json.dumps({'result': job.result})}\n\n".encode()
        )
    elif job.status == "timed_out":
        payload = {"error": job.error or "Job timed out", "partial_result": job.result}
        await response.write(
            f"event: timed_out\ndata: {json.dumps(payload)}\n\n".encode()
        )
    elif job.status == "failed":
        await response.write(
            f"event: error\ndata: {json.dumps({'error': job.error})}\n\n".encode()
        )
    elif job.status == "cancelled":
        await response.write(
            f"event: cancelled\ndata: {json.dumps({})}\n\n".encode()
        )

    logger.info("SSE stream closed for job %s (final status=%s)", job_id, job.status)
    await response.write_eof()
    return response


async def extend_job_timeout(request: web.Request) -> web.Response:
    job_id = request.match_info["job_id"]
    try:
        data = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body")

    extra = data.get("extend_by", 300)
    if not isinstance(extra, (int, float)) or extra <= 0:
        raise web.HTTPBadRequest(text="'extend_by' must be a positive number (seconds)")
    extra = min(int(extra), 3600)

    result = scheduler.extend_timeout(job_id, extra)
    if result is None:
        logger.debug("Extend failed for job %s (not running or not found)", job_id)
        raise web.HTTPNotFound(text="Job not found or not running")

    logger.info("Job %s timeout extended: %s", job_id, result)
    return web.json_response({"job_id": job_id, **result})


async def cancel_job(request: web.Request) -> web.Response:
    job_id = request.match_info["job_id"]
    if not scheduler.cancel(job_id):
        logger.debug("Cancel failed for job %s (not found or finished)", job_id)
        raise web.HTTPNotFound(text="Job not found or already finished")
    logger.info("Job %s cancelled", job_id)
    return web.json_response({"job_id": job_id, "status": "cancelled"})


async def list_models(request: web.Request) -> web.Response:
    from .models import discover_models, model_names

    try:
        models = discover_models()
        names = model_names(models)
    except SystemExit:
        logger.error("Cannot reach Ollama (model discovery failed)")
        return web.json_response({"models": [], "error": "Cannot reach Ollama"})
    except Exception as e:
        logger.error("Model discovery error: %s", e, exc_info=True)
        return web.json_response({"models": [], "error": str(e)})

    logger.debug("Models listed: %s", names)
    return web.json_response({"models": names})


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

async def on_startup(app: web.Application) -> None:
    logger.info("Scheduler starting (%d worker(s))", scheduler._max_concurrent)
    scheduler.start()


async def on_shutdown(app: web.Application) -> None:
    logger.info("Scheduler shutting down")
    await scheduler.shutdown()
    logger.info("Server stopped")


def create_app(max_concurrent: int = 1,
               default_job_timeout: int = 600) -> web.Application:
    global scheduler
    scheduler = Scheduler(
        max_concurrent=max_concurrent,
        default_job_timeout=default_job_timeout,
    )

    app = web.Application(
        middlewares=[cors_middleware, request_logging_middleware],
        client_max_size=_MAX_REQUEST_BODY,
    )
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/api/health", health)
    app.router.add_post("/api/prompt", submit_prompt)
    app.router.add_get("/api/prompt/{job_id}", get_job)
    app.router.add_get("/api/prompt/{job_id}/stream", stream_job)
    app.router.add_patch("/api/prompt/{job_id}/timeout", extend_job_timeout)
    app.router.add_delete("/api/prompt/{job_id}", cancel_job)
    app.router.add_get("/api/models", list_models)

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ollama-chain-server",
        description="Run the ollama-chain API server with memory-aware scheduling.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8585,
        help="Listen port (default: 8585)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=1,
        help="Max concurrent chain executions (default: 1)",
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=600,
        metavar="SECS",
        help="Default per-job timeout in seconds (default: 600). Clients can override per-request up to 3600.",
    )
    parser.add_argument(
        "--log-dir",
        default=".logs",
        metavar="DIR",
        help="Directory for log files (default: .logs)",
    )
    args = parser.parse_args()

    log_file = setup_logging(log_dir=args.log_dir)

    logger.info(
        "ollama-chain API server starting on http://%s:%d",
        args.host, args.port,
    )
    logger.info("Max concurrent jobs: %d", args.max_concurrent)
    logger.info("Default job timeout: %ds", args.job_timeout)
    logger.info("Log file: %s (DEBUG level)", log_file)

    app = create_app(max_concurrent=args.max_concurrent,
                     default_job_timeout=args.job_timeout)
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        print=None,
        keepalive_timeout=75,
    )
