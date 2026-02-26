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
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Callable

from .search import (
    web_search, web_search_news, format_search_results,
    github_search, github_search_issues, stackoverflow_search,
    docs_search,
)

_WEB_TOOL_TIMEOUT = 20  # hard timeout for web search tool functions


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

_MAX_SHELL_OUTPUT = 100_000  # 100 KB cap on shell command output


def tool_shell(command: str, timeout: int = 30) -> ToolResult:
    """Execute a shell command and return combined stdout/stderr."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout,
        )
        stdout = result.stdout
        stderr = result.stderr
        if len(stdout) > _MAX_SHELL_OUTPUT:
            stdout = stdout[:_MAX_SHELL_OUTPUT] + "\n... [output truncated]"
        if len(stderr) > _MAX_SHELL_OUTPUT:
            stderr = stderr[:_MAX_SHELL_OUTPUT] + "\n... [stderr truncated]"
        output = stdout
        if stderr:
            output += f"\n[stderr] {stderr}"
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


def _run_search_with_timeout(fn, tool_name: str, *args, **kwargs) -> ToolResult:
    """Run a search function with a hard timeout to prevent hanging."""
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn, *args, **kwargs)
            results = future.result(timeout=_WEB_TOOL_TIMEOUT)
    except TimeoutError:
        return ToolResult(
            False, f"Search timed out after {_WEB_TOOL_TIMEOUT}s", tool_name,
            error_detail="timeout",
        )
    except Exception as e:
        return ToolResult(
            False, f"Search failed: {e}", tool_name,
            error_detail=type(e).__name__,
        )
    formatted = format_search_results(results) if isinstance(results, list) else ""
    if not formatted:
        return ToolResult(
            False, "No results found.", tool_name,
            error_detail="empty_results",
        )
    return ToolResult(True, formatted, tool_name)


def tool_web_search_tool(query: str) -> ToolResult:
    """Search the web via DuckDuckGo."""
    return _run_search_with_timeout(web_search, "web_search", query, max_results=5)


def tool_web_search_news_tool(query: str) -> ToolResult:
    """Search recent news via DuckDuckGo."""
    return _run_search_with_timeout(web_search_news, "web_search_news", query, max_results=5)


def tool_github_search(query: str) -> ToolResult:
    """Search GitHub repositories and issues."""
    def _combined_github(q: str) -> list:
        repos = github_search(q, max_results=3)
        issues = github_search_issues(q, max_results=3)
        return repos + issues
    return _run_search_with_timeout(_combined_github, "github_search", query)


def tool_stackoverflow_search(query: str) -> ToolResult:
    """Search Stack Overflow for Q&A."""
    return _run_search_with_timeout(stackoverflow_search, "stackoverflow_search", query, max_results=5)


def tool_docs_search(query: str) -> ToolResult:
    """Search trusted documentation sites."""
    return _run_search_with_timeout(docs_search, "docs_search", query, max_results=5)


_EVAL_BLOCKED_KEYWORDS = frozenset({
    "import", "exec", "eval", "__import__", "compile",
    "open", "globals", "locals", "getattr", "setattr", "delattr",
    "__builtins__", "__class__", "__subclasses__",
    "os.", "sys.", "subprocess", "shutil",
})


def tool_python_eval(code: str) -> ToolResult:
    """Evaluate a Python expression in a restricted namespace."""
    code_lower = code.lower().replace(" ", "")
    for blocked in _EVAL_BLOCKED_KEYWORDS:
        if blocked.replace(" ", "") in code_lower:
            return ToolResult(
                False,
                f"Blocked: '{blocked}' is not allowed in eval",
                "python_eval",
                error_detail="blocked_keyword",
            )

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
        output = str(result)
        if len(output) > 50_000:
            output = output[:50_000] + "\n... [output truncated]"
        return ToolResult(True, output, "python_eval")
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
    "github_search": Tool(
        name="github_search",
        description=(
            "Search GitHub repositories and issues. Returns repo names, stars, "
            "descriptions, and relevant issues/discussions."
        ),
        parameters={"query": "The search query for GitHub"},
        function=tool_github_search,
        max_retries=2,
        retry_delay=2.0,
    ),
    "stackoverflow_search": Tool(
        name="stackoverflow_search",
        description=(
            "Search Stack Overflow for programming Q&A. Returns questions "
            "with scores, answer counts, and tags."
        ),
        parameters={"query": "The search query for Stack Overflow"},
        function=tool_stackoverflow_search,
        max_retries=2,
        retry_delay=2.0,
    ),
    "docs_search": Tool(
        name="docs_search",
        description=(
            "Search trusted official documentation sites (kubernetes.io, "
            "docs.redhat.com, MDN, python.org, kernel.org, IETF, W3C, etc.)."
        ),
        parameters={"query": "The search query for documentation"},
        function=tool_docs_search,
        max_retries=2,
        retry_delay=2.0,
    ),
}

TOOL_FALLBACKS: dict[str, list[str]] = {
    "web_search": ["web_search_news", "docs_search"],
    "web_search_news": ["web_search"],
    "github_search": ["web_search"],
    "stackoverflow_search": ["web_search"],
    "docs_search": ["web_search"],
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


_SEARCH_TOOLS = frozenset({
    "web_search", "web_search_news", "github_search",
    "stackoverflow_search", "docs_search",
})


def _adapt_args_for_fallback(
    original: str, fallback: str, args: dict,
) -> dict | None:
    """Translate arguments from the original tool to the fallback tool."""
    if original in _SEARCH_TOOLS and fallback in _SEARCH_TOOLS:
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
