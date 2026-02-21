"""Planning module for LLM-driven goal decomposition.

The planner uses a fast model to break a high-level goal into an ordered
list of concrete steps, each optionally tagged with a tool to invoke and
a dependency list.

Dynamic reasoning (Gap 1):
  - should_replan() evaluates whether new facts invalidate remaining steps
  - assess_progress() provides lightweight progress evaluation
  - detect_parallel_groups() identifies independent steps for concurrency
"""

import json
import re
import sys

from .common import chat_with_retry
from .validation import validate_and_fix_plan, detect_circular_deps


def decompose_goal(
    goal: str, model: str, context: str = "",
    complexity_hint: str = "",
) -> list[dict]:
    """
    Break *goal* into an ordered list of steps with dependency tracking.

    When *complexity_hint* is provided by the router (``"simple"``,
    ``"moderate"``, or ``"complex"``), the planner uses it to calibrate
    plan granularity — simple goals get fewer, coarser steps while complex
    goals get more fine-grained decomposition.

    Returns::
        [{"id": 1, "description": "...", "tool": "...",
          "depends_on": [], "status": "pending"}, ...]
    """
    context_block = f"\n\nRelevant context:\n{context}" if context else ""

    granularity_guidance = ""
    if complexity_hint == "simple":
        granularity_guidance = (
            "\nThis is a simple goal — keep the plan short (1-3 steps). "
            "Avoid unnecessary decomposition.\n"
        )
    elif complexity_hint == "complex":
        granularity_guidance = (
            "\nThis is a complex goal — create a thorough plan with enough "
            "steps to cover all aspects. Mark independent steps so they "
            "can run in parallel.\n"
        )

    prompt = (
        "/no_think\n"
        "You are a planning agent. Decompose the following goal into a concrete, "
        "ordered list of steps. Each step should be a specific, actionable task.\n\n"
        + granularity_guidance +
        "For each step, specify:\n"
        "- A clear description of what to do\n"
        "- Which tool to use (if any): shell, read_file, write_file, list_dir, "
        "web_search, web_search_news, python_eval, or 'none' for pure reasoning\n"
        "- depends_on: list of step IDs this step requires (empty if independent)\n\n"
        "Respond with ONLY a JSON array of objects. Example:\n"
        '[{"id": 1, "description": "Get OS version", "tool": "shell", "depends_on": []},\n'
        ' {"id": 2, "description": "Search for CVEs for that OS", "tool": "web_search", "depends_on": [1]},\n'
        ' {"id": 3, "description": "Summarize findings", "tool": "none", "depends_on": [1, 2]}]\n\n'
        f"Goal: {goal}"
        f"{context_block}"
    )

    response = chat_with_retry(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = _strip_think(response["message"]["content"])
    plan = _parse_plan(content, goal)
    plan, fix_warnings = validate_and_fix_plan(plan)
    for w in fix_warnings:
        print(f"[planner] Auto-fix: {w}", file=sys.stderr)

    cycles = detect_circular_deps(plan)
    if cycles:
        for a, b in cycles:
            print(
                f"[planner] Breaking circular dep: step {a} ↔ {b}",
                file=sys.stderr,
            )
            step = next((s for s in plan if s["id"] == a), None)
            if step:
                step["depends_on"] = [
                    d for d in step["depends_on"] if d != b
                ]

    return plan


def replan(
    goal: str,
    current_plan: list[dict],
    observations: str,
    model: str,
) -> list[dict]:
    """Produce a revised plan incorporating observations from execution so far."""
    plan_text = "\n".join(
        f"  {s['id']}. [{s['status']}] {s['description']}"
        for s in current_plan
    )

    prompt = (
        "/no_think\n"
        "You are a planning agent. The original plan needs revision based on "
        "new observations.\n\n"
        f"Goal: {goal}\n\n"
        f"Current plan:\n{plan_text}\n\n"
        f"Observations:\n{observations}\n\n"
        "Produce an updated plan as a JSON array. Keep completed steps as-is. "
        "Modify, remove, or add pending steps as needed. "
        "Include depends_on for each step.\n"
        "Respond with ONLY the JSON array."
    )

    response = chat_with_retry(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = _strip_think(response["message"]["content"])
    plan = _parse_plan(content, goal)
    plan, fix_warnings = validate_and_fix_plan(plan)
    for w in fix_warnings:
        print(f"[planner] Auto-fix (replan): {w}", file=sys.stderr)
    return plan


# ---------------------------------------------------------------------------
# Dynamic reasoning triggers (Gap 1)
# ---------------------------------------------------------------------------

def should_replan(
    goal: str,
    plan: list[dict],
    new_facts: list[str],
    model: str,
) -> bool:
    """Ask the LLM whether newly discovered facts require a plan revision.

    Returns True if the plan should be revised, False otherwise.
    This is a lightweight check (fast model, short prompt) run after steps
    that discover significant new information.
    """
    if not new_facts:
        return False

    pending = [s for s in plan if s["status"] == "pending"]
    if not pending:
        return False

    pending_desc = "\n".join(f"  - {s['description']}" for s in pending[:5])
    facts_desc = "\n".join(f"  - {f}" for f in new_facts[-5:])

    prompt = (
        "/no_think\n"
        "You are evaluating whether a plan needs revision.\n\n"
        f"Goal: {goal}\n\n"
        f"New facts just discovered:\n{facts_desc}\n\n"
        f"Remaining planned steps:\n{pending_desc}\n\n"
        "Do these new facts make any remaining step obsolete, incorrect, "
        "or in need of modification? Answer with ONLY 'yes' or 'no'."
    )

    try:
        response = chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            retries=1,
        )
        answer = _strip_think(response["message"]["content"]).strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False


def assess_progress(
    goal: str,
    plan: list[dict],
    facts: list[str],
    model: str,
) -> str:
    """Lightweight progress assessment — returns a brief evaluation string."""
    completed = [s for s in plan if s["status"] == "completed"]
    pending = [s for s in plan if s["status"] == "pending"]
    failed = [s for s in plan if s["status"] == "failed"]

    summary = (
        f"Goal: {goal}\n"
        f"Steps completed: {len(completed)}/{len(plan)}\n"
        f"Steps failed: {len(failed)}\n"
        f"Steps remaining: {len(pending)}\n"
        f"Facts discovered: {len(facts)}\n"
    )
    if facts:
        summary += "Key facts:\n" + "\n".join(f"  - {f}" for f in facts[-5:])

    prompt = (
        "/no_think\n"
        "Briefly assess whether the agent is on track to achieve the goal. "
        "Reply in 1-2 sentences.\n\n"
        f"{summary}"
    )

    try:
        response = chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            retries=1,
        )
        return _strip_think(response["message"]["content"]).strip()
    except Exception:
        return f"{len(completed)}/{len(plan)} steps done, {len(facts)} facts."


# ---------------------------------------------------------------------------
# Concurrency support (Gap 5)
# ---------------------------------------------------------------------------

def detect_parallel_groups(plan: list[dict]) -> list[list[dict]]:
    """Identify groups of pending steps that can execute concurrently.

    Steps are grouped when they:
      - Are all pending
      - Have all their dependencies satisfied (completed)
      - Don't depend on each other

    Returns a list of groups.  Each group is a list of steps that can
    run in parallel.  Groups themselves must run sequentially.
    """
    completed_ids = {s["id"] for s in plan if s["status"] == "completed"}
    pending = [s for s in plan if s["status"] == "pending"]

    groups: list[list[dict]] = []
    remaining = list(pending)

    while remaining:
        ready = []
        not_ready = []
        for step in remaining:
            deps = set(step.get("depends_on", []))
            if deps <= completed_ids:
                ready.append(step)
            else:
                not_ready.append(step)

        if not ready:
            ready = [remaining[0]]
            not_ready = remaining[1:]

        groups.append(ready)
        for s in ready:
            completed_ids.add(s["id"])
        remaining = not_ready

    return groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Remove <think>...</think> wrapper if present."""
    if "<think>" in text:
        end = text.find("</think>")
        if end != -1:
            return text[end + len("</think>"):].strip()
    return text


def _parse_plan(text: str, goal: str) -> list[dict]:
    """Extract a JSON plan from LLM output, with a one-step fallback."""
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            plan = json.loads(json_match.group())
            if isinstance(plan, list) and all(isinstance(s, dict) for s in plan):
                for i, step in enumerate(plan):
                    step.setdefault("id", i + 1)
                    step.setdefault("status", "pending")
                    step.setdefault("tool", "none")
                    step.setdefault("description", f"Step {i + 1}")
                    step.setdefault("depends_on", [])
                    if not isinstance(step["depends_on"], list):
                        step["depends_on"] = []
                return plan
        except json.JSONDecodeError:
            pass

    return [
        {"id": 1, "description": goal, "tool": "none",
         "status": "pending", "depends_on": []},
    ]
