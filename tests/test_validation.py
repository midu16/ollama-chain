"""Unit tests for the validation module — no Ollama required."""

import pytest

from ollama_chain.validation import (
    ValidationError,
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
