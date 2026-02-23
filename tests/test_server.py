"""Unit tests for server.py — API endpoints, middleware, logging."""

import asyncio
import json
import os
import tempfile

import pytest

from ollama_chain.server import (
    _TERMINAL_STATUSES,
    create_app,
    setup_logging,
)
from ollama_chain.scheduler import PromptJob, Scheduler


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_creates_log_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            log_file = setup_logging(log_dir=log_dir)
            assert os.path.isdir(log_dir)
            assert log_file.endswith("ollama-chain-server.log")
            assert os.path.isfile(log_file)


# ---------------------------------------------------------------------------
# Terminal statuses
# ---------------------------------------------------------------------------

class TestTerminalStatuses:
    def test_all_terminal_statuses(self):
        assert "completed" in _TERMINAL_STATUSES
        assert "failed" in _TERMINAL_STATUSES
        assert "cancelled" in _TERMINAL_STATUSES
        assert "timed_out" in _TERMINAL_STATUSES

    def test_non_terminal_statuses(self):
        assert "queued" not in _TERMINAL_STATUSES
        assert "running" not in _TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# API endpoints (using aiohttp_client fixture from pytest-aiohttp)
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    return create_app(max_concurrent=1, default_job_timeout=30)


@pytest.mark.asyncio
async def test_health(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/api/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert "queue_size" in data
    assert "active_jobs" in data


@pytest.mark.asyncio
async def test_submit_valid(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/prompt",
        json={"prompt": "What is TCP?", "mode": "cascade"},
    )
    assert resp.status == 202
    data = await resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_submit_empty_prompt(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post("/api/prompt", json={"prompt": ""})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_submit_missing_prompt(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post("/api/prompt", json={"mode": "fast"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_submit_invalid_json(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/prompt",
        data=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_submit_cli_only_pcap_rejected(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/prompt", json={"prompt": "test", "mode": "pcap"},
    )
    assert resp.status == 400
    text = await resp.text()
    assert "CLI" in text


@pytest.mark.asyncio
async def test_submit_cli_only_k8s_rejected(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/prompt", json={"prompt": "test", "mode": "k8s"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_submit_default_mode(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post("/api/prompt", json={"prompt": "test"})
    assert resp.status == 202


@pytest.mark.asyncio
async def test_get_existing_job(aiohttp_client, app):
    client = await aiohttp_client(app)
    submit = await client.post("/api/prompt", json={"prompt": "test"})
    job_id = (await submit.json())["job_id"]

    resp = await client.get(f"/api/prompt/{job_id}")
    assert resp.status == 200
    data = await resp.json()
    assert data["job_id"] == job_id
    assert "status" in data
    assert "position" in data


@pytest.mark.asyncio
async def test_get_nonexistent_job(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/api/prompt/nonexistent")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_cancel_queued_job(aiohttp_client, app):
    client = await aiohttp_client(app)
    submit = await client.post("/api/prompt", json={"prompt": "test"})
    job_id = (await submit.json())["job_id"]

    resp = await client.delete(f"/api/prompt/{job_id}")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_nonexistent(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.delete("/api/prompt/nonexistent")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_stream_terminal_returns_json(aiohttp_client, app):
    from ollama_chain.server import scheduler as sched
    client = await aiohttp_client(app)
    submit = await client.post("/api/prompt", json={"prompt": "test"})
    job_id = (await submit.json())["job_id"]

    job = sched.get(job_id)
    job.status = "completed"
    job.result = "the answer"

    resp = await client.get(f"/api/prompt/{job_id}/stream")
    assert resp.status == 200
    ct = resp.headers.get("Content-Type", "")
    assert "application/json" in ct
    data = await resp.json()
    assert data["status"] == "completed"
    assert data["result"] == "the answer"


@pytest.mark.asyncio
async def test_stream_timed_out_returns_json(aiohttp_client, app):
    from ollama_chain.server import scheduler as sched
    client = await aiohttp_client(app)
    submit = await client.post("/api/prompt", json={"prompt": "test"})
    job_id = (await submit.json())["job_id"]

    job = sched.get(job_id)
    job.status = "timed_out"
    job.error = "Timed out"
    job.result = "partial"

    resp = await client.get(f"/api/prompt/{job_id}/stream")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "timed_out"


@pytest.mark.asyncio
async def test_stream_nonexistent_job(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/api/prompt/nonexistent/stream")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_cors_header_present(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/api/health")
    assert resp.headers.get("Access-Control-Allow-Origin") == "*"


@pytest.mark.asyncio
async def test_options_request(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.options("/api/prompt")
    assert resp.status == 200
    assert "Access-Control-Allow-Methods" in resp.headers
