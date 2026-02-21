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


def validate_model_sequence(models: list[str]) -> list[str]:
    """Validate a model cascade sequence.  Returns a list of warnings."""
    warnings: list[str] = []
    if not models:
        warnings.append("Empty model sequence")
    if len(models) != len(set(models)):
        warnings.append("Duplicate models in sequence")
    return warnings
