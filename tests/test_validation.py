"""Unit tests for the validation module — no Ollama required."""

import pytest

from ollama_chain.validation import (
    ValidationError,
    detect_circular_deps,
    validate_and_fix_plan,
    validate_model_sequence,
    validate_plan,
    validate_step,
)


class TestValidateStep:
    def test_valid_step(self):
        step = {"id": 1, "tool": "shell", "description": "Run cmd", "depends_on": []}
        warnings = validate_step(step)
        assert warnings == []

    def test_unknown_tool(self):
        step = {"id": 1, "tool": "nonexistent", "description": "X", "depends_on": []}
        warnings = validate_step(step)
        assert any("Unknown tool" in w for w in warnings)

    def test_missing_description(self):
        step = {"id": 1, "tool": "shell", "description": "", "depends_on": []}
        warnings = validate_step(step)
        assert any("no description" in w for w in warnings)

    def test_none_tool_valid(self):
        step = {"id": 1, "tool": "none", "description": "Think", "depends_on": []}
        assert validate_step(step) == []

    def test_unmet_dependencies(self):
        step = {"id": 2, "tool": "shell", "description": "X", "depends_on": [1]}
        warnings = validate_step(step, completed_steps=set())
        assert any("Unmet" in w for w in warnings)

    def test_met_dependencies(self):
        step = {"id": 2, "tool": "shell", "description": "X", "depends_on": [1]}
        warnings = validate_step(step, completed_steps={1})
        assert warnings == []

    def test_invalid_depends_on_type(self):
        step = {"id": 1, "tool": "shell", "description": "X", "depends_on": "bad"}
        warnings = validate_step(step)
        assert any("not a list" in w for w in warnings)

    def test_all_real_tools_valid(self):
        from ollama_chain.tools import TOOL_REGISTRY

        for tool_name in TOOL_REGISTRY:
            step = {"id": 1, "tool": tool_name, "description": "X", "depends_on": []}
            assert validate_step(step) == [], f"{tool_name} should be valid"


class TestValidatePlan:
    def test_valid_plan(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell", "depends_on": []},
            {"id": 2, "description": "B", "tool": "none", "depends_on": [1]},
        ]
        assert validate_plan(plan) == []

    def test_duplicate_ids(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell", "depends_on": []},
            {"id": 1, "description": "B", "tool": "shell", "depends_on": []},
        ]
        warnings = validate_plan(plan)
        assert any("Duplicate" in w for w in warnings)

    def test_dangling_dependency(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell", "depends_on": [99]},
        ]
        warnings = validate_plan(plan)
        assert any("non-existent" in w for w in warnings)

    def test_empty_plan(self):
        assert validate_plan([]) == []


class TestValidateAndFixPlan:
    def test_clean_plan_unchanged(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell", "depends_on": [], "status": "pending"},
            {"id": 2, "description": "B", "tool": "none", "depends_on": [1], "status": "pending"},
        ]
        fixed, warnings = validate_and_fix_plan(plan)
        assert warnings == []
        assert fixed == plan

    def test_fixes_unknown_tool(self):
        plan = [
            {"id": 1, "description": "Do X", "tool": "magic_wand", "depends_on": []},
        ]
        fixed, warnings = validate_and_fix_plan(plan)
        assert fixed[0]["tool"] == "none"
        assert any("magic_wand" in w for w in warnings)

    def test_fixes_missing_description(self):
        plan = [
            {"id": 1, "tool": "shell", "description": "", "depends_on": []},
        ]
        fixed, warnings = validate_and_fix_plan(plan)
        assert fixed[0]["description"] == "Step 1"
        assert any("placeholder" in w for w in warnings)

    def test_fixes_dangling_deps(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell", "depends_on": [99]},
        ]
        fixed, warnings = validate_and_fix_plan(plan)
        assert 99 not in fixed[0]["depends_on"]
        assert any("dangling" in w for w in warnings)

    def test_fixes_non_list_depends_on(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell", "depends_on": "bad"},
        ]
        fixed, warnings = validate_and_fix_plan(plan)
        assert fixed[0]["depends_on"] == []
        assert any("invalid" in w.lower() for w in warnings)

    def test_adds_default_status(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell", "depends_on": []},
        ]
        fixed, warnings = validate_and_fix_plan(plan)
        assert fixed[0]["status"] == "pending"

    def test_preserves_existing_status(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell", "depends_on": [], "status": "completed"},
        ]
        fixed, _ = validate_and_fix_plan(plan)
        assert fixed[0]["status"] == "completed"

    def test_multiple_fixes(self):
        plan = [
            {"id": 1, "description": "", "tool": "nonexistent", "depends_on": [99]},
        ]
        fixed, warnings = validate_and_fix_plan(plan)
        assert len(warnings) == 3
        assert fixed[0]["tool"] == "none"
        assert fixed[0]["description"] == "Step 1"
        assert fixed[0]["depends_on"] == []


class TestDetectCircularDeps:
    def test_no_cycles(self):
        plan = [
            {"id": 1, "depends_on": []},
            {"id": 2, "depends_on": [1]},
            {"id": 3, "depends_on": [2]},
        ]
        assert detect_circular_deps(plan) == []

    def test_direct_cycle(self):
        plan = [
            {"id": 1, "depends_on": [2]},
            {"id": 2, "depends_on": [1]},
        ]
        cycles = detect_circular_deps(plan)
        assert len(cycles) > 0

    def test_self_loop(self):
        plan = [
            {"id": 1, "depends_on": [1]},
        ]
        cycles = detect_circular_deps(plan)
        assert len(cycles) > 0

    def test_no_deps(self):
        plan = [
            {"id": 1, "depends_on": []},
            {"id": 2, "depends_on": []},
        ]
        assert detect_circular_deps(plan) == []

    def test_empty_plan(self):
        assert detect_circular_deps([]) == []


class TestValidateModelSequence:
    def test_valid_sequence(self):
        assert validate_model_sequence(["a", "b", "c"]) == []

    def test_empty_sequence(self):
        warnings = validate_model_sequence([])
        assert any("Empty" in w for w in warnings)

    def test_duplicate_models(self):
        warnings = validate_model_sequence(["a", "a", "b"])
        assert any("Duplicate" in w for w in warnings)

    def test_single_model(self):
        assert validate_model_sequence(["a"]) == []
