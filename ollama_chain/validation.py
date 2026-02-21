"""Validation functions for plan steps and model cascade sequences."""

from .tools import TOOL_REGISTRY

VALID_TOOLS = frozenset(TOOL_REGISTRY.keys()) | frozenset({"none"})


class ValidationError(Exception):
    """Raised when a plan step or cascade configuration is invalid."""


def validate_step(
    step: dict,
    completed_steps: set[int] | None = None,
) -> list[str]:
    """Validate a single plan step.  Returns a list of warnings (empty = valid)."""
    warnings: list[str] = []
    step_id = step.get("id", "?")

    tool = step.get("tool", "none")
    if tool not in VALID_TOOLS:
        warnings.append(f"Unknown tool '{tool}' in step {step_id}")

    if not step.get("description"):
        warnings.append(f"Step {step_id} has no description")

    deps = step.get("depends_on", [])
    if not isinstance(deps, list):
        warnings.append(f"Step {step_id} depends_on is not a list")
    elif completed_steps is not None:
        unmet = [d for d in deps if d not in completed_steps]
        if unmet:
            warnings.append(
                f"Unmet dependencies {unmet} for step {step_id}"
            )

    return warnings


def validate_plan(plan: list[dict]) -> list[str]:
    """Validate an entire plan.  Returns a list of all warnings."""
    warnings: list[str] = []
    all_ids = {s.get("id") for s in plan}
    seen_ids: set[int] = set()

    for step in plan:
        step_id = step.get("id")
        if step_id in seen_ids:
            warnings.append(f"Duplicate step id: {step_id}")
        seen_ids.add(step_id)

        warnings.extend(validate_step(step))

        for dep in step.get("depends_on", []):
            if dep not in all_ids:
                warnings.append(
                    f"Step {step_id} depends on non-existent step {dep}"
                )

    return warnings


def validate_and_fix_plan(plan: list[dict]) -> tuple[list[dict], list[str]]:
    """Validate a plan and auto-repair common issues.

    Fixes:
      - Dangling dependency refs (removed silently)
      - Unknown tool names (replaced with 'none')
      - Missing descriptions (filled with 'Step N')
      - Non-list depends_on (replaced with empty list)
      - Missing status (set to 'pending')

    Returns (fixed_plan, warnings) where warnings describe what was fixed.
    """
    warnings: list[str] = []
    all_ids = {s.get("id") for s in plan}

    for step in plan:
        step_id = step.get("id", "?")

        if not step.get("description"):
            step["description"] = f"Step {step_id}"
            warnings.append(f"Step {step_id}: added placeholder description")

        tool = step.get("tool", "none")
        if tool not in VALID_TOOLS:
            warnings.append(
                f"Step {step_id}: replaced unknown tool '{tool}' with 'none'"
            )
            step["tool"] = "none"

        deps = step.get("depends_on", [])
        if not isinstance(deps, list):
            step["depends_on"] = []
            warnings.append(f"Step {step_id}: reset invalid depends_on to []")
        else:
            dangling = [d for d in deps if d not in all_ids]
            if dangling:
                step["depends_on"] = [d for d in deps if d in all_ids]
                warnings.append(
                    f"Step {step_id}: removed dangling deps {dangling}"
                )

        step.setdefault("status", "pending")

    return plan, warnings


def detect_circular_deps(plan: list[dict]) -> list[tuple[int, int]]:
    """Detect circular dependencies in a plan.

    Returns a list of (step_a, step_b) pairs that form cycles.
    """
    adj: dict[int, set[int]] = {}
    for step in plan:
        sid = step.get("id", 0)
        adj[sid] = set(step.get("depends_on", []))

    cycles: list[tuple[int, int]] = []
    visited: set[int] = set()
    path: set[int] = set()

    def _dfs(node: int) -> bool:
        visited.add(node)
        path.add(node)
        for dep in adj.get(node, set()):
            if dep in path:
                cycles.append((node, dep))
                return True
            if dep not in visited and _dfs(dep):
                return True
        path.discard(node)
        return False

    for sid in adj:
        if sid not in visited:
            _dfs(sid)

    return cycles


def validate_model_sequence(models: list[str]) -> list[str]:
    """Validate a model cascade sequence.  Returns a list of warnings."""
    warnings: list[str] = []
    if not models:
        warnings.append("Empty model sequence")
    if len(models) != len(set(models)):
        warnings.append("Duplicate models in sequence")
    return warnings
