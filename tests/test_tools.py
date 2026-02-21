"""Unit tests for the tools module — no Ollama required."""

import os
import tempfile

import pytest

from ollama_chain.tools import (
    TOOL_FALLBACKS,
    TOOL_REGISTRY,
    ToolResult,
    execute_tool,
    execute_tool_with_retry,
    format_tool_descriptions,
    _adapt_args_for_fallback,
)


class TestToolResult:
    def test_defaults(self):
        r = ToolResult(success=True, output="ok", tool_name="test")
        assert r.duration_ms == 0.0
        assert r.retries_used == 0
        assert r.error_detail == ""

    def test_with_metadata(self):
        r = ToolResult(
            success=False, output="err", tool_name="shell",
            duration_ms=123.4, retries_used=2, error_detail="timeout",
        )
        assert r.duration_ms == 123.4
        assert r.retries_used == 2
        assert r.error_detail == "timeout"


class TestExecuteTool:
    def test_shell_echo(self):
        r = execute_tool("shell", {"command": "echo hello"})
        assert r.success
        assert "hello" in r.output

    def test_shell_failure(self):
        r = execute_tool("shell", {"command": "false"})
        assert not r.success

    def test_python_eval(self):
        r = execute_tool("python_eval", {"code": "2 ** 10"})
        assert r.success
        assert r.output == "1024"

    def test_python_eval_error(self):
        r = execute_tool("python_eval", {"code": "1/0"})
        assert not r.success
        assert "Error" in r.output

    def test_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content123")
        r = execute_tool("read_file", {"path": str(f)})
        assert r.success
        assert "content123" in r.output

    def test_read_file_missing(self):
        r = execute_tool("read_file", {"path": "/nonexistent/xyz.txt"})
        assert not r.success
        assert r.error_detail

    def test_write_file(self, tmp_path):
        p = str(tmp_path / "out.txt")
        r = execute_tool("write_file", {"path": p, "content": "data"})
        assert r.success
        assert os.path.exists(p)
        assert open(p).read() == "data"

    def test_append_file(self, tmp_path):
        p = str(tmp_path / "app.txt")
        open(p, "w").write("A")
        r = execute_tool("append_file", {"path": p, "content": "B"})
        assert r.success
        assert open(p).read() == "AB"

    def test_list_dir(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        r = execute_tool("list_dir", {"path": str(tmp_path)})
        assert r.success
        assert "a.txt" in r.output
        assert "b.txt" in r.output

    def test_unknown_tool(self):
        r = execute_tool("nonexistent", {})
        assert not r.success
        assert "Unknown tool" in r.output

    def test_bad_args(self):
        r = execute_tool("shell", {"bad_param": "x"})
        assert not r.success
        assert "Invalid arguments" in r.output or "error" in r.output.lower()


class TestExecuteToolWithRetry:
    def test_success(self):
        r = execute_tool_with_retry("python_eval", {"code": "3 + 4"})
        assert r.success
        assert r.output == "7"
        assert r.duration_ms >= 0

    def test_shell_retry(self):
        r = execute_tool_with_retry("shell", {"command": "echo retry_test"})
        assert r.success
        assert "retry_test" in r.output


class TestFallbacks:
    def test_fallback_mapping_exists(self):
        assert "web_search" in TOOL_FALLBACKS
        assert "read_file" in TOOL_FALLBACKS

    def test_adapt_search_args(self):
        result = _adapt_args_for_fallback(
            "web_search", "web_search_news", {"query": "test"},
        )
        assert result == {"query": "test"}

    def test_adapt_read_to_shell(self):
        result = _adapt_args_for_fallback(
            "read_file", "shell", {"path": "/etc/hostname"},
        )
        assert result is not None
        assert "/etc/hostname" in result["command"]

    def test_adapt_unknown(self):
        result = _adapt_args_for_fallback("python_eval", "shell", {"code": "1"})
        assert result is None


class TestFormatDescriptions:
    def test_all_tools_listed(self):
        desc = format_tool_descriptions()
        for name in TOOL_REGISTRY:
            assert name in desc

    def test_has_parameters(self):
        desc = format_tool_descriptions()
        assert "Parameters:" in desc
