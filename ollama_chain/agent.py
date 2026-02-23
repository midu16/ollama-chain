"""Autonomous agent with planning, memory, tool use, and dynamic control flow.

Addresses all six identified gaps:
  1. Dynamic reasoning loop — continuous plan revision on new facts, not just failures
  2. Full LLM integration  — structured context, parsed responses, memory updates
  3. Robust tool errors    — retry with fallback chains, structured error metadata
  4. Multi-modal memory    — content-typed MemoryEntry for text/tool/fact/error
  5. Concurrency           — parallel execution of independent plan steps
  6. Structured prompts    — prioritised facts, plan status, sectioned formatting
"""

import json
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import (
    SOURCE_GUIDANCE,
    ask,
    build_structured_prompt,
    chat_with_retry,
    model_supports_thinking,
    sanitize_messages,
)
from .memory import SessionMemory, PersistentMemory
from .planner import (
    decompose_goal,
    replan,
    should_replan,
    detect_parallel_groups,
)
from .router import (
    RouteDecision,
    build_fallback_chain,
    optimize_routing,
    route_query,
    select_models_for_step,
)
from .validation import validate_step
from .tools import (
    execute_tool_with_retry,
    format_tool_descriptions,
    ToolResult,
)

MAX_ITERATIONS = 15
MAX_PARALLEL = 3


# ---------------------------------------------------------------------------
# Model fallback
# ---------------------------------------------------------------------------

def _agent_chat(
    all_models: list[str],
    messages: list[dict],
    preferred_models: list[str] | None = None,
    thinking: bool = False,
) -> tuple[str, str] | None:
    """Try models from strongest to weakest.  Return (model, text) or None.

    When *preferred_models* is given, try those first (in order) before
    falling back to the remaining models strongest-first.

    *thinking*: when ``True`` the model is allowed to reason internally.
    When ``False``, thinking-capable models receive a ``/no_think``
    prefix on the last user message to suppress chain-of-thought.
    """
    if preferred_models:
        seen = set(preferred_models)
        models_to_try = list(preferred_models) + [
            m for m in reversed(all_models) if m not in seen
        ]
    else:
        models_to_try = list(reversed(all_models))
    for model in models_to_try:
        try:
            chat_msgs = list(messages)
            if not thinking and model_supports_thinking(model):
                last = chat_msgs[-1].copy()
                last["content"] = "/no_think\n" + last["content"]
                chat_msgs[-1] = last
            resp = chat_with_retry(model=model, messages=chat_msgs, retries=2)
            text = resp["message"]["content"]
            if "<think>" in text:
                end = text.find("</think>")
                if end != -1:
                    text = text[end + len("</think>"):].strip()
            return model, text
        except Exception as e:
            print(
                f"[agent]   {model} unavailable ({e}), trying next...",
                file=sys.stderr,
            )
    return None


# ---------------------------------------------------------------------------
# Auto-execution: run tools directly from plan hints when models fail
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS = (
    "rm -rf", "rm -r /", "mkfs", "dd if=", ":(){ ", "> /dev/sd",
    "chmod -r 777 /", "format c:",
)


def _is_safe_command(cmd: str) -> bool:
    cmd_lower = cmd.strip().lower()
    return not any(p in cmd_lower for p in _DANGEROUS_PATTERNS)


def _extract_quoted_strings(text: str) -> list[str]:
    results: list[str] = []
    for pat in (r"'([^']{2,})'", r'"([^"]{2,})"', r'`([^`]{2,})`'):
        results.extend(re.findall(pat, text))
    return results


def _build_search_query(desc: str, facts: list[str]) -> str:
    query = desc
    for prefix in (
        "Search the web for ", "Search for ", "Look up ",
        "Find information on ", "Research ", "Find ",
    ):
        if query.lower().startswith(prefix.lower()):
            query = query[len(prefix):]
            break

    os_fact = ""
    for fact in facts:
        if fact.startswith("OS:"):
            os_fact = fact[3:].strip()
            break

    if os_fact:
        for placeholder in (
            "the identified OS version", "the identified OS",
            "the OS version", "the current OS",
        ):
            if placeholder.lower() in query.lower():
                query = re.sub(
                    re.escape(placeholder), os_fact, query, flags=re.IGNORECASE,
                )

    return query.strip().rstrip(".")


def _auto_execute_step(step: dict, session: SessionMemory) -> ToolResult | None:
    """Execute a step's tool directly from plan hints — no LLM required."""
    tool = step.get("tool", "none")
    desc = step.get("description", "")

    if tool == "none":
        return None

    print(f"[agent]   Auto-executing '{tool}' from plan hints...", file=sys.stderr)

    if tool == "shell":
        commands = [c for c in _extract_quoted_strings(desc) if _is_safe_command(c)]
        if commands:
            cmd = " && ".join(commands[:3])
            return execute_tool_with_retry("shell", {"command": cmd})
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in (
            "os version", "operating system", "os-release", "distribution",
            "uname", "system version",
        )):
            return execute_tool_with_retry("shell", {
                "command": "cat /etc/os-release 2>/dev/null; echo '---'; uname -a",
            })
        if "kernel" in desc_lower:
            return execute_tool_with_retry("shell", {"command": "uname -r"})
        if "package" in desc_lower or "installed" in desc_lower:
            return execute_tool_with_retry("shell", {
                "command": (
                    "rpm -qa --qf '%{NAME}-%{VERSION}\\n' 2>/dev/null | head -40 || "
                    "dpkg -l 2>/dev/null | tail -40"
                ),
            })
        if "process" in desc_lower:
            return execute_tool_with_retry("shell", {"command": "ps aux --sort=-%mem | head -20"})
        if "disk" in desc_lower:
            return execute_tool_with_retry("shell", {"command": "df -h"})
        if "network" in desc_lower or "ip " in desc_lower:
            return execute_tool_with_retry("shell", {"command": "ip -brief addr"})
        return None

    if tool == "web_search":
        query = _build_search_query(desc, session.facts)
        return execute_tool_with_retry("web_search", {"query": query})

    if tool == "web_search_news":
        query = _build_search_query(desc, session.facts)
        return execute_tool_with_retry("web_search_news", {"query": query})

    if tool == "github_search":
        query = _build_search_query(desc, session.facts)
        return execute_tool_with_retry("github_search", {"query": query})

    if tool == "stackoverflow_search":
        query = _build_search_query(desc, session.facts)
        return execute_tool_with_retry("stackoverflow_search", {"query": query})

    if tool == "docs_search":
        query = _build_search_query(desc, session.facts)
        return execute_tool_with_retry("docs_search", {"query": query})

    if tool == "read_file":
        paths = re.findall(r'(/[\w./\-]+)', desc)
        if paths:
            return execute_tool_with_retry("read_file", {"path": paths[0]})
        return None

    if tool == "list_dir":
        paths = re.findall(r'(/[\w./\-]+)', desc)
        return execute_tool_with_retry("list_dir", {"path": paths[0] if paths else "."})

    if tool == "python_eval":
        codes = _extract_quoted_strings(desc)
        if codes:
            return execute_tool_with_retry("python_eval", {"code": codes[0]})
        return None

    return None


# ---------------------------------------------------------------------------
# Fact extraction from tool output
# ---------------------------------------------------------------------------

def _extract_facts_from_output(tool_name: str, output: str) -> list[str]:
    facts: list[str] = []

    if tool_name in ("list_dir", "list_dir>shell"):
        entries = [e.strip() for e in output.split() if e.strip()]
        if entries:
            visible = [e for e in entries if not e.startswith(".")]
            dirs = [e for e in visible if not "." in e or e.endswith("/")]
            files = [e for e in visible if "." in e and not e.endswith("/")]
            listing = ", ".join(visible[:30])
            facts.append(f"Directory contents: {listing}")
            if files:
                facts.append(f"Files found: {', '.join(files[:20])}")
            py_files = [e for e in entries if e.endswith(".py")]
            if py_files:
                facts.append(f"Python files: {', '.join(py_files[:20])}")
        return facts

    if tool_name != "shell":
        return facts

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("PRETTY_NAME="):
            facts.append(f"OS: {line.split('=', 1)[1].strip('\"')}")
        elif line.startswith("VERSION_ID="):
            facts.append(f"OS Version ID: {line.split('=', 1)[1].strip('\"')}")
        elif line.startswith("ID=") and not line.startswith("ID_LIKE"):
            facts.append(f"OS ID: {line.split('=', 1)[1].strip('\"')}")
        elif line.startswith("HOME_URL="):
            facts.append(f"OS Home URL: {line.split('=', 1)[1].strip('\"')}")

    kern_match = re.search(r'Linux \S+ (\d+\.\d+\.\S+)', output)
    if kern_match:
        facts.append(f"Kernel: {kern_match.group(1)}")

    return facts


# ---------------------------------------------------------------------------
# Prompt construction (Gap 6: structured prompts)
# ---------------------------------------------------------------------------

def _build_system_prompt(
    session: SessionMemory,
    persistent_ctx: str,
    current_step: dict | None = None,
) -> str:
    tool_docs = format_tool_descriptions()
    structured_ctx = session.get_structured_context(current_step=current_step)

    return build_structured_prompt(
        sections=[
            ("Role", (
                "You are an autonomous AI agent that solves problems step by step "
                "using planning, memory, and tools.\n"
                f"{SOURCE_GUIDANCE}"
            )),
            ("Available Tools", tool_docs),
            ("Response Format", (
                "Respond with EXACTLY ONE of these blocks:\n\n"
                "1. To use a tool:\n"
                "<tool_call>\n"
                '{"name": "tool_name", "args": {"param": "value"}}\n'
                "</tool_call>\n\n"
                "2. To give your final answer:\n"
                "<final_answer>\n"
                "Your complete answer here.\n"
                "</final_answer>\n\n"
                "3. To store a fact for memory:\n"
                "<store_fact>\n"
                "The fact to remember.\n"
                "</store_fact>\n\n"
                "You may include <store_fact> alongside <tool_call> or <final_answer>."
            )),
            ("Session State", structured_ctx),
            ("Long-Term Memory", persistent_ctx),
        ],
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict:
    result: dict = {"type": "reasoning", "content": text}

    tool_match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL,
    )
    if tool_match:
        try:
            call = json.loads(tool_match.group(1))
            result = {
                "type": "tool_call",
                "name": call.get("name", ""),
                "args": call.get("args", {}),
                "raw": text,
            }
        except json.JSONDecodeError:
            result = {"type": "malformed_tool_call", "content": tool_match.group(1)}

    answer_match = re.search(
        r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL,
    )
    if answer_match:
        result = {"type": "final_answer", "content": answer_match.group(1)}

    facts = re.findall(r"<store_fact>\s*(.*?)\s*</store_fact>", text, re.DOTALL)
    if facts:
        result["facts"] = facts

    return result


# ---------------------------------------------------------------------------
# Single-step execution (used by both serial and parallel paths)
# ---------------------------------------------------------------------------

def _execute_step(
    step: dict,
    session: SessionMemory,
    persistent: PersistentMemory,
    persistent_ctx: str,
    all_models: list[str],
    iteration: int,
    max_iterations: int,
    query_complexity: str = "complex",
) -> str | None:
    """Execute one plan step.  Returns a final-answer string, or None to continue."""
    step_id = step["id"]
    step_desc = step["description"]
    step_tool = step.get("tool", "none")

    if not _validate_before_execution(step, session):
        step["status"] = "failed"
        session.add_error(
            f"Step {step_id} skipped: validation failed",
            step_id=step_id,
        )
        return None

    step["status"] = "in_progress"

    total_steps = len(session.plan)
    print(
        f"[agent] Step {step_id}/{total_steps} "
        f"(iter {iteration}/{max_iterations}): {step_desc[:70]}",
        file=sys.stderr,
    )

    # -- Build messages with structured context (Gap 6) --
    system_prompt = _build_system_prompt(session, persistent_ctx, current_step=step)
    remaining_tool_steps = [
        s for s in session.plan
        if s["status"] == "pending" and s["id"] != step_id
        and s.get("tool", "none") != "none"
    ]
    if remaining_tool_steps:
        step_instruction = (
            "Execute this step using the suggested tool. "
            "Do NOT provide <final_answer> — there are more steps to complete."
        )
    else:
        step_instruction = (
            "Execute this step. Use a tool if helpful, or provide "
            "<final_answer> if the goal is fully addressed."
        )
    step_prompt = (
        f"Current step: {step_id}/{total_steps}. {step_desc}\n"
        f"Suggested tool: {step_tool}\n\n"
        f"{step_instruction}"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for entry in session.history[-10:]:
        role = entry.role if entry.role in ("user", "assistant") else "assistant"
        messages.append({"role": role, "content": entry.content})
    messages.append({"role": "user", "content": step_prompt})
    messages = sanitize_messages(messages)

    preferred = step.get("preferred_models") or select_models_for_step(
        step, all_models, query_complexity,
    )
    step_needs_thinking = (
        step_tool == "none"
        and query_complexity != "simple"
    )
    chat_result = _agent_chat(
        all_models, messages,
        preferred_models=preferred,
        thinking=step_needs_thinking,
    )

    if chat_result is not None:
        model_used, raw = chat_result
        print(f"[agent]   Model: {model_used}", file=sys.stderr)
        parsed = _parse_response(raw)

        # Store facts
        for fact in parsed.get("facts", []):
            session.add_fact(fact)
            persistent.store_fact(fact)
            print(f"[agent]   Stored fact: {fact[:60]}", file=sys.stderr)

        if parsed["type"] == "tool_call":
            _handle_tool_call(parsed, step, session, persistent)

        elif parsed["type"] == "final_answer":
            remaining_tool_steps = [
                s for s in session.plan
                if s["status"] == "pending" and s["id"] != step_id
                and s.get("tool", "none") != "none"
            ]
            if remaining_tool_steps:
                print(
                    f"[agent]   Deferring final answer — "
                    f"{len(remaining_tool_steps)} tool steps remain",
                    file=sys.stderr,
                )
                session.add("assistant", parsed["content"])
                step["status"] = "completed"
            else:
                session.add("assistant", parsed["content"])
                for s in session.plan:
                    if s["status"] in ("pending", "in_progress"):
                        s["status"] = "completed"
                return parsed["content"]

        else:
            session.add("assistant", raw)
            step["status"] = "completed"

    else:
        # -- All models failed: auto-execute from plan hints --
        auto_result = _auto_execute_step(step, session)
        if auto_result is not None:
            _handle_auto_result(auto_result, step_tool, step, session, persistent)
        elif step_tool == "none":
            print(
                "[agent]   Skipping reasoning step (models unavailable)",
                file=sys.stderr,
            )
            step["status"] = "completed"
        else:
            print(
                f"[agent]   Could not auto-execute '{step_tool}' — marking failed",
                file=sys.stderr,
            )
            step["status"] = "failed"
            session.add_error(
                f"Step {step_id} failed: all models unavailable and "
                f"auto-execution not possible.",
                step_id=step_id,
            )

    return None


# ---------------------------------------------------------------------------
# Parallel step execution (Gap 5)
# ---------------------------------------------------------------------------

def _execute_parallel_group(
    group: list[dict],
    session: SessionMemory,
    persistent: PersistentMemory,
    persistent_ctx: str,
    all_models: list[str],
    iteration: int,
    max_iterations: int,
    query_complexity: str = "complex",
) -> str | None:
    """Execute a group of independent steps concurrently.

    Returns a final-answer string if any step produced one, else None.
    """
    if len(group) == 1:
        return _execute_step(
            group[0], session, persistent, persistent_ctx,
            all_models, iteration, max_iterations,
            query_complexity=query_complexity,
        )

    step_ids = ", ".join(str(s["id"]) for s in group)
    print(
        f"[agent] Executing steps [{step_ids}] in parallel...",
        file=sys.stderr,
    )

    final_answer = None
    with ThreadPoolExecutor(max_workers=min(len(group), MAX_PARALLEL)) as pool:
        futures = {
            pool.submit(
                _execute_step,
                step, session, persistent, persistent_ctx,
                all_models, iteration, max_iterations,
                query_complexity,
            ): step
            for step in group
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None and final_answer is None:
                    final_answer = result
            except Exception as e:
                step = futures[future]
                step["status"] = "failed"
                session.add_error(f"Parallel step {step['id']} error: {e}")
                print(
                    f"[agent]   Step {step['id']} exception: {e}",
                    file=sys.stderr,
                )

    return final_answer


# ---------------------------------------------------------------------------
# Pre-execution validation
# ---------------------------------------------------------------------------

def _validate_before_execution(
    step: dict,
    session: SessionMemory,
) -> bool:
    """Validate a step before execution.  Returns True if step is safe to run."""
    completed_ids = {s["id"] for s in session.plan if s["status"] == "completed"}
    warnings = validate_step(step, completed_steps=completed_ids)
    if not warnings:
        return True

    for w in warnings:
        print(f"[agent]   Validation: {w}", file=sys.stderr)

    if any("Unknown tool" in w for w in warnings):
        step["tool"] = "none"
        print("[agent]   Downgraded to reasoning step", file=sys.stderr)

    if any("Unmet" in w for w in warnings):
        print("[agent]   Skipping — unmet dependencies", file=sys.stderr)
        return False

    return True


# ---------------------------------------------------------------------------
# Monitoring and replanning
# ---------------------------------------------------------------------------

def monitor_and_replan(
    goal: str,
    session: SessionMemory,
    current_group: list[dict],
    fast_model: str,
    facts_at_last_replan: int,
) -> tuple[bool, int]:
    """Evaluate execution state and replan if needed.

    Checks three triggers (in priority order):
      1. Discovery steps completed with new facts → force replan
      2. New facts that may invalidate remaining steps → ask LLM
      3. Accumulated failures → force replan

    Returns (did_replan, updated_facts_checkpoint).
    """
    new_facts_since = session.facts[facts_at_last_replan:]

    # --- Trigger 1: discovery steps with new facts ---
    discovery_tools = {"list_dir", "list_dir>shell"}
    group_was_discovery = any(
        s.get("tool") in discovery_tools
        or (s.get("tool") == "shell" and any(
            kw in s.get("description", "").lower()
            for kw in ("list", "ls ", "find ", "tree")
        ))
        for s in current_group
    )

    if group_was_discovery and new_facts_since:
        print(
            "[agent] Discovery step completed — re-planning with "
            "actual file listing...",
            file=sys.stderr,
        )
        observations = _build_observations(session)
        session.plan = replan(goal, session.plan, observations, fast_model)
        _print_revised_plan(session)
        return True, len(session.facts)

    # --- Trigger 2: new facts may invalidate remaining steps ---
    if new_facts_since:
        try:
            needs_replan = should_replan(
                goal, session.plan, new_facts_since, fast_model,
            )
        except Exception:
            needs_replan = False

        if needs_replan:
            print(
                "[agent] New facts triggered re-plan...",
                file=sys.stderr,
            )
            observations = _build_observations(session)
            session.plan = replan(goal, session.plan, observations, fast_model)
            _print_revised_plan(session)
            return True, len(session.facts)

    # --- Trigger 3: accumulated failures ---
    if session.failed_step_count() > 0:
        print(
            f"[agent] Re-planning ({session.failed_step_count()} failed)...",
            file=sys.stderr,
        )
        observations = _build_observations(session)
        session.plan = replan(goal, session.plan, observations, fast_model)
        _print_revised_plan(session)
        return True, len(session.facts)

    return False, facts_at_last_replan


# ---------------------------------------------------------------------------
# Agent loop (Gap 1: dynamic reasoning)
# ---------------------------------------------------------------------------

def run_agent(
    goal: str,
    all_models: list[str],
    *,
    web_search: bool = True,
    fast: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    session_id = uuid.uuid4().hex[:8]

    session = SessionMemory(session_id=session_id, goal=goal)
    persistent = PersistentMemory()
    persistent_ctx = persistent.get_relevant_context(goal)

    # ---- Phase 0: Route the query ------------------------------------------
    routing = route_query(
        goal, all_models,
        fast_model=fast_name,
        use_llm=(len(all_models) > 1),
        web_search=web_search,
    )

    # ---- Phase 1: Planning ------------------------------------------------
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[agent] Session {session_id}", file=sys.stderr)
    print(f"[agent] Goal: {goal}", file=sys.stderr)
    print(
        f"[agent] Routing: complexity={routing.complexity} "
        f"strategy={routing.strategy}",
        file=sys.stderr,
    )
    print(f"{'='*60}", file=sys.stderr)
    print(f"[agent] Planning with {fast_name}...", file=sys.stderr)

    plan = decompose_goal(
        goal, fast_name,
        context=persistent_ctx,
        complexity_hint=routing.complexity,
    )
    session.plan = plan

    print(f"[agent] Plan ({len(plan)} steps):", file=sys.stderr)
    for step in plan:
        deps = step.get("depends_on", [])
        dep_str = f" (after {deps})" if deps else ""
        print(
            f"  {step['id']}. {step['description']} "
            f"[{step.get('tool', 'none')}]{dep_str}",
            file=sys.stderr,
        )
    print(file=sys.stderr)

    session.add("assistant", f"Plan created with {len(plan)} steps.")

    # ---- Phase 1b: Optimize routing for each step -------------------------
    optimize_routing(plan, all_models, routing.complexity)

    # ---- Phase 2: Execution loop ------------------------------------------
    iteration = 0
    facts_at_last_replan = len(session.facts)
    failed_models: set[str] = set()

    while iteration < max_iterations:
        pending = session.pending_steps()
        if not pending:
            break

        # --- Identify parallelisable groups (Gap 5) ---
        groups = detect_parallel_groups(session.plan)
        if not groups:
            break
        current_group = groups[0]

        iteration += 1

        answer = _execute_parallel_group(
            current_group, session, persistent, persistent_ctx,
            all_models, iteration, max_iterations,
            query_complexity=routing.complexity,
        )
        if answer is not None:
            _finalize_session(session, persistent)
            return answer

        # --- Monitor and replan if needed ---
        did_replan, facts_at_last_replan = monitor_and_replan(
            goal, session, current_group, fast_name,
            facts_at_last_replan,
        )
        if did_replan:
            optimize_routing(
                session.plan, all_models, routing.complexity,
                failed_models=failed_models,
            )

    # ---- Phase 3: Final synthesis -----------------------------------------
    _print_step_summary(session)

    print(
        f"[agent] Synthesizing final answer with {strong_name}...",
        file=sys.stderr,
    )

    collected = "\n\n".join(
        f"[Step {tr['step']} -- {tr['tool']}]\n{tr['output']}"
        for tr in session.tool_results
    )
    facts_block = "\n".join(f"- {f}" for f in session.facts) if session.facts else "None"

    final_prompt = build_structured_prompt(
        sections=[
            ("Role", (
                "You are an expert assistant. Based on the research and tool "
                "results below, provide a comprehensive, well-structured answer.\n"
                f"{SOURCE_GUIDANCE}"
            )),
            ("Goal", goal),
            ("Discovered Facts", facts_block),
            ("Tool Results", collected),
        ],
        instructions=(
            "Provide a thorough, accurate, well-cited answer. "
            "Use the discovered facts and tool results as primary evidence."
        ),
    )

    final_answer = _synthesize_final(final_prompt, all_models)
    _finalize_session(session, persistent)
    return final_answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_observations(session: SessionMemory) -> str:
    """Build an observations string from recent tool results and facts."""
    parts: list[str] = []
    for tr in session.tool_results[-5:]:
        status = "OK" if tr["success"] else "FAILED"
        parts.append(
            f"Step {tr['step']} ({tr['tool']}): {status} — {tr['output'][:100]}"
        )
    if session.facts:
        parts.append("Discovered facts: " + "; ".join(session.facts[-5:]))
    return "\n".join(parts)


def _handle_tool_call(
    parsed: dict,
    step: dict,
    session: SessionMemory,
    persistent: PersistentMemory,
):
    tool_name = parsed["name"]
    tool_args = parsed["args"]
    args_preview = json.dumps(tool_args)[:80]
    print(f"[agent]   Tool: {tool_name}({args_preview})", file=sys.stderr)

    result = execute_tool_with_retry(tool_name, tool_args)
    _record_tool_result(result, tool_name, tool_args, step, session, persistent)


def _handle_auto_result(
    result: ToolResult,
    tool_name: str,
    step: dict,
    session: SessionMemory,
    persistent: PersistentMemory,
):
    _record_tool_result(
        result, tool_name, {},
        step, session, persistent,
    )


def _record_tool_result(
    result: ToolResult,
    tool_name: str,
    tool_args: dict,
    step: dict,
    session: SessionMemory,
    persistent: PersistentMemory,
):
    truncated = result.output
    if len(truncated) > 3000:
        truncated = truncated[:3000] + "\n... [truncated]"

    status = "OK" if result.success else "FAILED"
    timing = f" [{result.duration_ms:.0f}ms]" if result.duration_ms else ""
    retries = f" (retries: {result.retries_used})" if result.retries_used else ""
    print(
        f"[agent]   Result ({status}{timing}{retries}): "
        f"{truncated[:120].replace(chr(10), ' ')}",
        file=sys.stderr,
    )

    session.add("assistant", f"Used tool {tool_name}: {json.dumps(tool_args)}")
    session.add_tool_output(tool_name, truncated, step["id"], result.success)
    session.tool_results.append({
        "step": step["id"],
        "tool": tool_name,
        "args": tool_args,
        "success": result.success,
        "output": truncated,
        "duration_ms": result.duration_ms,
        "error_detail": result.error_detail,
    })

    step["status"] = "completed" if result.success else "failed"

    if result.success:
        auto_facts = _extract_facts_from_output(tool_name, result.output)
        for fact in auto_facts:
            if fact not in session.facts:
                session.add_fact(fact)
                persistent.store_fact(fact)
                print(f"[agent]   Extracted fact: {fact}", file=sys.stderr)


def _synthesize_final(prompt: str, all_models: list[str]) -> str:
    """Cascade-refine the final answer through all available models.

    Uses the same smallest-to-largest chaining strategy as the cascade
    mode so the agent benefits from multi-model accuracy:
      1. Smallest model drafts from collected evidence (no thinking)
      2. Each intermediate model reviews with thinking + lower temp
      3. Largest model produces the definitive answer with thinking
    """
    n = len(all_models)

    # --- Stage 1: first model drafts (no thinking — speed) ---
    print(
        f"[agent]   Cascade synthesis 1/{n}: drafting with {all_models[0]}...",
        file=sys.stderr,
    )
    try:
        current = ask(prompt, model=all_models[0])
    except Exception as e:
        print(f"[agent]   {all_models[0]} failed ({e}), trying others...", file=sys.stderr)
        for model in all_models[1:]:
            try:
                return ask(prompt, model=model, thinking=True, temperature=0.3)
            except Exception:
                continue
        return "(Agent could not generate a final answer — all models unavailable.)"

    if n == 1:
        return current

    # --- Stages 2..N-1: intermediate models refine (thinking + low temp) ---
    for i, model in enumerate(all_models[1:-1], start=2):
        print(
            f"[agent]   Cascade synthesis {i}/{n}: refining with {model} +think...",
            file=sys.stderr,
        )
        try:
            current = ask(
                f"You are a reviewer improving the accuracy of an answer.\n"
                f"{SOURCE_GUIDANCE}\n\n"
                f"Original goal and context:\n{prompt[:1500]}\n\n"
                f"Current answer:\n{current}\n\n"
                f"Instructions:\n"
                f"- Fix any factual errors\n"
                f"- Add missing important information\n"
                f"- Strengthen source references (add [Source: ...] citations)\n"
                f"- Remove unsupported speculation\n"
                f"- Preserve what is already correct\n"
                f"Output ONLY the improved answer.",
                model=model,
                thinking=True,
                temperature=0.4,
            )
        except Exception as e:
            print(f"[agent]   {model} unavailable ({e}), skipping...", file=sys.stderr)

    # --- Final stage: strongest model produces definitive answer ---
    print(
        f"[agent]   Cascade synthesis {n}/{n}: final answer with {all_models[-1]} +think...",
        file=sys.stderr,
    )
    try:
        return ask(
            f"You are the final reviewer producing the definitive answer.\n"
            f"{SOURCE_GUIDANCE}\n\n"
            f"Original goal and context:\n{prompt[:1500]}\n\n"
            f"Draft answer (refined by {n - 1} model(s)):\n{current}\n\n"
            f"Instructions:\n"
            f"- Verify all factual claims and correct any remaining errors\n"
            f"- Ensure key claims have [Source: ...] citations\n"
            f"- Produce a clean, well-structured final answer\n"
            f"Output ONLY the final authoritative answer.",
            model=all_models[-1],
            thinking=True,
            temperature=0.3,
        )
    except Exception as e:
        print(
            f"[agent]   {all_models[-1]} failed ({e}), using last draft",
            file=sys.stderr,
        )
        return current


def _print_revised_plan(session: SessionMemory):
    """Print the revised plan after a re-plan."""
    pending = session.pending_steps()
    print(f"[agent] Revised plan ({len(pending)} pending steps):", file=sys.stderr)
    for step in session.plan:
        if step["status"] == "completed":
            continue
        deps = step.get("depends_on", [])
        dep_str = f" (after {deps})" if deps else ""
        print(
            f"  {step['id']}. {step['description'][:70]} "
            f"[{step.get('tool', 'none')}]{dep_str}",
            file=sys.stderr,
        )
    print(file=sys.stderr)


def _print_step_summary(session: SessionMemory):
    print(f"\n{'='*60}", file=sys.stderr)
    print("[agent] Step summary:", file=sys.stderr)
    for step in session.plan:
        marker = {
            "completed": "OK",
            "failed": "FAIL",
            "pending": "SKIP",
            "in_progress": "...",
        }.get(step["status"], "?")
        print(
            f"  [{marker:>4}] {step['id']}. {step['description'][:60]}",
            file=sys.stderr,
        )
    if session.facts:
        print(f"\n[agent] Facts discovered ({len(session.facts)}):", file=sys.stderr)
        for f in session.facts:
            print(f"  - {f}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


def _finalize_session(session: SessionMemory, persistent: PersistentMemory):
    completed = session.completed_step_count()
    total = len(session.plan)
    summary = (
        f"Completed {completed}/{total} steps. "
        f"Tools used: {len(session.tool_results)}. "
        f"Facts learned: {len(session.facts)}."
    )
    persistent.store_session_summary(session.session_id, session.goal, summary)
    print("[agent] Session saved to memory.", file=sys.stderr)
    session.clear()
