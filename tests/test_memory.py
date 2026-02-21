"""Unit tests for the memory module — no Ollama required."""

import json
import tempfile
from pathlib import Path

import pytest

from ollama_chain.memory import (
    CONTENT_ERROR,
    CONTENT_FACT,
    CONTENT_TEXT,
    CONTENT_TOOL_OUTPUT,
    MemoryEntry,
    PersistentMemory,
    SessionMemory,
)


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

class TestMemoryEntry:
    def test_default_content_type(self):
        e = MemoryEntry(role="user", content="hello")
        assert e.content_type == CONTENT_TEXT
        assert e.tags == []

    def test_custom_content_type(self):
        e = MemoryEntry(
            role="tool", content="output", content_type=CONTENT_TOOL_OUTPUT,
            tags=["shell"],
        )
        assert e.content_type == CONTENT_TOOL_OUTPUT
        assert "shell" in e.tags

    def test_timestamp_auto(self):
        e = MemoryEntry(role="user", content="x")
        assert e.timestamp > 0


# ---------------------------------------------------------------------------
# SessionMemory
# ---------------------------------------------------------------------------

class TestSessionMemory:
    def _make_session(self) -> SessionMemory:
        s = SessionMemory(session_id="test", goal="Test goal")
        s.plan = [
            {"id": 1, "description": "Step one", "tool": "shell",
             "status": "completed", "depends_on": []},
            {"id": 2, "description": "Step two", "tool": "web_search",
             "status": "pending", "depends_on": [1]},
        ]
        return s

    def test_add_and_history(self):
        s = SessionMemory(session_id="t", goal="g")
        s.add("user", "hello")
        s.add("assistant", "world")
        assert len(s.history) == 2
        assert s.history[0].role == "user"

    def test_add_tool_output(self):
        s = SessionMemory(session_id="t", goal="g")
        s.add_tool_output("shell", "ok", step_id=1, success=True)
        assert len(s.history) == 1
        assert s.history[0].content_type == CONTENT_TOOL_OUTPUT
        assert "shell" in s.history[0].tags

    def test_add_fact_dedup(self):
        s = SessionMemory(session_id="t", goal="g")
        s.add_fact("OS: Fedora 43")
        s.add_fact("OS: Fedora 43")
        assert len(s.facts) == 1
        assert s.facts[0] == "OS: Fedora 43"

    def test_add_error(self):
        s = SessionMemory(session_id="t", goal="g")
        s.add_error("timeout", step_id=5)
        assert s.history[0].content_type == CONTENT_ERROR
        assert s.history[0].metadata["step_id"] == 5

    def test_summarize_includes_plan_and_facts(self):
        s = self._make_session()
        s.facts = ["OS: Fedora"]
        text = s.summarize()
        assert "Test goal" in text
        assert "[done]" in text
        assert "[ ]" in text
        assert "Fedora" in text

    def test_get_context_window(self):
        s = SessionMemory(session_id="t", goal="g")
        s.add("user", "q1")
        s.add("assistant", "a1")
        window = s.get_context_window(max_entries=5)
        assert len(window) >= 2
        roles = [m["role"] for m in window]
        assert "user" in roles or "assistant" in roles

    def test_get_structured_context(self):
        s = self._make_session()
        s.facts = ["OS: Fedora 43", "Kernel: 6.18"]
        s.tool_results = [
            {"step": 1, "tool": "shell", "success": True,
             "output": "NAME=Fedora", "args": {}},
        ]
        s.add_error("step 99 failed", step_id=99)
        ctx = s.get_structured_context(
            current_step={"description": "Search for Fedora CVEs"},
        )
        assert "Fedora" in ctx
        assert "[done]" in ctx

    def test_get_relevant_history(self):
        s = SessionMemory(session_id="t", goal="g")
        s.add("user", "alpha beta gamma")
        s.add("user", "delta epsilon")
        s.add("assistant", "alpha result")
        relevant = s.get_relevant_history(["alpha"], max_entries=2)
        assert len(relevant) <= 2
        assert any("alpha" in e.content for e in relevant)

    def test_counters(self):
        s = self._make_session()
        assert s.completed_step_count() == 1
        assert s.failed_step_count() == 0
        assert len(s.pending_steps()) == 1


# ---------------------------------------------------------------------------
# PersistentMemory
# ---------------------------------------------------------------------------

class TestPersistentMemory:
    def test_facts_round_trip(self, tmp_path):
        mem = PersistentMemory(memory_dir=tmp_path)
        mem.store_fact("fact1")
        mem.store_fact("fact2")
        mem.store_fact("fact1")
        assert mem.get_facts() == ["fact1", "fact2"]

        mem2 = PersistentMemory(memory_dir=tmp_path)
        assert mem2.get_facts() == ["fact1", "fact2"]

    def test_session_summary(self, tmp_path):
        mem = PersistentMemory(memory_dir=tmp_path)
        mem.store_session_summary("s1", "goal1", "done")
        sessions = mem.get_recent_sessions()
        assert len(sessions) == 1
        assert sessions[0]["goal"] == "goal1"

    def test_clear(self, tmp_path):
        mem = PersistentMemory(memory_dir=tmp_path)
        mem.store_fact("f")
        mem.store_session_summary("s", "g", "d")
        mem.clear()
        assert mem.get_facts() == []
        assert mem.get_recent_sessions() == []

    def test_get_relevant_context(self, tmp_path):
        mem = PersistentMemory(memory_dir=tmp_path)
        mem.store_fact("OS: Ubuntu")
        mem.store_session_summary("s1", "find CVEs", "done 3/3")
        ctx = mem.get_relevant_context("CVEs")
        assert "Ubuntu" in ctx
        assert "find CVEs" in ctx

    def test_empty_context(self, tmp_path):
        mem = PersistentMemory(memory_dir=tmp_path)
        assert mem.get_relevant_context("anything") == ""
