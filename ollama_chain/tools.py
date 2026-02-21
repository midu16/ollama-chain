"""Tool registry for agent environmental interaction.

Each tool wraps a real operation (shell, filesystem, web, computation) behind a
uniform ToolResult interface so the agent loop can invoke any of them by name.

Resilience features (Gap 3):
  - Per-tool retry with configurable attempts
  - Fallback chains: when a tool fails, alternatives are tried automatically
  - Structured error metadata on ToolResult
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

from .search import web_search, web_search_news, format_search_results


# ---------------------------------------------------------------------------
# Result / Tool dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    success: bool
    output: str
    tool_name: str
    duration_ms: float = 0.0
    retries_used: int = 0
    error_detail: str = ""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    function: Callable[..., ToolResult]
    max_retries: int = 1
    retry_delay: float = 1.0


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_shell(command: str, timeout: int = 30) -> ToolResult:
    """Execute a shell command and return combined stdout/stderr."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return ToolResult(
            success=result.returncode == 0,
            output=output.strip() or "(no output)",
            tool_name="shell",
            error_detail="" if result.returncode == 0 else f"exit {result.returncode}",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            False, f"Command timed out after {timeout}s", "shell",
            error_detail="timeout",
        )
    except Exception as e:
        return ToolResult(False, str(e), "shell", error_detail=type(e).__name__)


def tool_read_file(path: str) -> ToolResult:
    """Read a file and return its contents (truncated at 50 kB)."""
    try:
        path = os.path.expanduser(path)
        with open(path) as f:
            content = f.read()
        if len(content) > 50_000:
            content = content[:50_000] + "\n... [truncated, file too large]"
        return ToolResult(True, content or "(empty file)", "read_file")
    except Exception as e:
        return ToolResult(
            False, str(e), "read_file", error_detail=type(e).__name__,
        )


def tool_write_file(path: str, content: str) -> ToolResult:
    """Write content to a file, creating parent directories as needed."""
    try:
        path = os.path.expanduser(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return ToolResult(True, f"Written {len(content)} bytes to {path}", "write_file")
    except Exception as e:
        return ToolResult(
            False, str(e), "write_file", error_detail=type(e).__name__,
        )


def tool_append_file(path: str, content: str) -> ToolResult:
    """Append content to an existing file."""
    try:
        path = os.path.expanduser(path)
        with open(path, "a") as f:
            f.write(content)
        return ToolResult(True, f"Appended {len(content)} bytes to {path}", "append_file")
    except Exception as e:
        return ToolResult(
            False, str(e), "append_file", error_detail=type(e).__name__,
        )


def tool_list_dir(path: str = ".") -> ToolResult:
    """List directory contents."""
    try:
        path = os.path.expanduser(path)
        entries = sorted(os.listdir(path))
        return ToolResult(True, "\n".join(entries) or "(empty directory)", "list_dir")
    except Exception as e:
        return ToolResult(
            False, str(e), "list_dir", error_detail=type(e).__name__,
        )


def tool_web_search_tool(query: str) -> ToolResult:
    """Search the web via DuckDuckGo."""
    results = web_search(query, max_results=5)
    formatted = format_search_results(results)
    if not formatted:
        return ToolResult(
            False, "No search results found.", "web_search",
            error_detail="empty_results",
        )
    return ToolResult(True, formatted, "web_search")


def tool_web_search_news_tool(query: str) -> ToolResult:
    """Search recent news via DuckDuckGo."""
    results = web_search_news(query, max_results=5)
    formatted = format_search_results(results)
    if not formatted:
        return ToolResult(
            False, "No news results found.", "web_search_news",
            error_detail="empty_results",
        )
    return ToolResult(True, formatted, "web_search_news")


def tool_python_eval(code: str) -> ToolResult:
    """Evaluate a Python expression in a restricted namespace."""
    try:
        import math

        allowed_builtins = {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "chr": chr, "dict": dict, "enumerate": enumerate, "float": float,
            "hex": hex, "int": int, "len": len, "list": list, "map": map,
            "max": max, "min": min, "oct": oct, "ord": ord, "pow": pow,
            "range": range, "repr": repr, "reversed": reversed, "round": round,
            "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
            "type": type, "zip": zip,
        }
        namespace = {"__builtins__": allowed_builtins, "math": math}
        result = eval(code, namespace)
        return ToolResult(True, str(result), "python_eval")
    except Exception as e:
        return ToolResult(
            False, f"Error: {e}", "python_eval", error_detail=type(e).__name__,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Tool] = {
    "shell": Tool(
        name="shell",
        description=(
            "Execute a shell command. Use for system commands, "
            "checking processes, network info, package management, etc."
        ),
        parameters={
            "command": "The shell command to execute",
            "timeout": "(optional) Timeout in seconds, default 30",
        },
        function=tool_shell,
        max_retries=2,
        retry_delay=1.0,
    ),
    "read_file": Tool(
        name="read_file",
        description="Read the contents of a file.",
        parameters={"path": "Absolute or relative path to the file"},
        function=tool_read_file,
    ),
    "write_file": Tool(
        name="write_file",
        description="Write content to a file. Creates parent dirs if needed.",
        parameters={"path": "Path to write to", "content": "Content to write"},
        function=tool_write_file,
    ),
    "append_file": Tool(
        name="append_file",
        description="Append content to an existing file.",
        parameters={"path": "Path to the file", "content": "Content to append"},
        function=tool_append_file,
    ),
    "list_dir": Tool(
        name="list_dir",
        description="List files and directories at a given path.",
        parameters={"path": "(optional) Directory path, defaults to '.'"},
        function=tool_list_dir,
    ),
    "web_search": Tool(
        name="web_search",
        description="Search the web using DuckDuckGo. Returns titles, URLs, snippets.",
        parameters={"query": "The search query"},
        function=tool_web_search_tool,
        max_retries=2,
        retry_delay=2.0,
    ),
    "web_search_news": Tool(
        name="web_search_news",
        description="Search recent news articles via DuckDuckGo.",
        parameters={"query": "The news search query"},
        function=tool_web_search_news_tool,
        max_retries=2,
        retry_delay=2.0,
    ),
    "python_eval": Tool(
        name="python_eval",
        description=(
            "Evaluate a Python expression. Supports math, string ops, "
            "list comprehensions. No imports or side effects."
        ),
        parameters={"code": "Python expression to evaluate"},
        function=tool_python_eval,
    ),
}

TOOL_FALLBACKS: dict[str, list[str]] = {
    "web_search": ["web_search_news"],
    "web_search_news": ["web_search"],
    "read_file": ["shell"],
}


# ---------------------------------------------------------------------------
# Execution with retry + fallback (Gap 3)
# ---------------------------------------------------------------------------

def execute_tool(name: str, args: dict) -> ToolResult:
    """Look up a tool by name and execute it (single attempt, no retry)."""
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return ToolResult(False, f"Unknown tool: {name}", name, error_detail="unknown_tool")
    try:
        return tool.function(**args)
    except TypeError as e:
        return ToolResult(
            False, f"Invalid arguments for {name}: {e}", name,
            error_detail="bad_args",
        )
    except Exception as e:
        return ToolResult(
            False, f"Tool {name} error: {e}", name,
            error_detail=type(e).__name__,
        )


def execute_tool_with_retry(name: str, args: dict) -> ToolResult:
    """Execute a tool with per-tool retry logic and fallback chain.

    1. Retry the primary tool up to tool.max_retries times
    2. If all retries fail, try each fallback tool in order
    3. Return the first successful result, or the last failure
    """
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return ToolResult(False, f"Unknown tool: {name}", name, error_detail="unknown_tool")

    t0 = time.monotonic()
    last_result: ToolResult | None = None

    for attempt in range(1, tool.max_retries + 1):
        try:
            result = tool.function(**args)
            result.retries_used = attempt - 1
            result.duration_ms = (time.monotonic() - t0) * 1000
            if result.success:
                return result
            last_result = result
        except TypeError as e:
            return ToolResult(
                False, f"Invalid arguments for {name}: {e}", name,
                error_detail="bad_args",
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as e:
            last_result = ToolResult(
                False, f"Tool {name} error: {e}", name,
                error_detail=type(e).__name__,
                retries_used=attempt,
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        if attempt < tool.max_retries:
            print(
                f"[tool] {name} attempt {attempt} failed, "
                f"retrying in {tool.retry_delay}s...",
                file=sys.stderr,
            )
            time.sleep(tool.retry_delay)

    fallbacks = TOOL_FALLBACKS.get(name, [])
    for fb_name in fallbacks:
        fb_tool = TOOL_REGISTRY.get(fb_name)
        if not fb_tool:
            continue
        fb_args = _adapt_args_for_fallback(name, fb_name, args)
        if fb_args is None:
            continue
        print(
            f"[tool] {name} failed, trying fallback '{fb_name}'...",
            file=sys.stderr,
        )
        try:
            result = fb_tool.function(**fb_args)
            result.duration_ms = (time.monotonic() - t0) * 1000
            result.tool_name = f"{name}>{fb_name}"
            if result.success:
                return result
        except Exception:
            continue

    if last_result:
        last_result.duration_ms = (time.monotonic() - t0) * 1000
        return last_result
    return ToolResult(
        False, f"Tool {name} exhausted all retries and fallbacks", name,
        duration_ms=(time.monotonic() - t0) * 1000,
        error_detail="all_retries_exhausted",
    )


def _adapt_args_for_fallback(
    original: str, fallback: str, args: dict,
) -> dict | None:
    """Translate arguments from the original tool to the fallback tool."""
    if original in ("web_search", "web_search_news") and fallback in (
        "web_search", "web_search_news",
    ):
        return {"query": args.get("query", "")}
    if original == "read_file" and fallback == "shell":
        path = args.get("path", "")
        return {"command": f"cat {path}"} if path else None
    return None


def format_tool_descriptions() -> str:
    """Format all available tools into a description block for the LLM."""
    lines = []
    for name, tool in TOOL_REGISTRY.items():
        params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
        lines.append(f"- {name}: {tool.description}")
        lines.append(f"  Parameters: {params}")
    return "\n".join(lines)
