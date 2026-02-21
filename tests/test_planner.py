"""Unit tests for the planner module — no Ollama required."""

import pytest

from ollama_chain.planner import _parse_plan, detect_parallel_groups


class TestParsePlan:
    def test_valid_json(self):
        text = '[{"id": 1, "description": "Do A", "tool": "shell"}]'
        plan = _parse_plan(text, "goal")
        assert len(plan) == 1
        assert plan[0]["description"] == "Do A"
        assert plan[0]["status"] == "pending"
        assert plan[0]["depends_on"] == []

    def test_defaults_filled(self):
        text = '[{"description": "Step"}]'
        plan = _parse_plan(text, "goal")
        assert plan[0]["id"] == 1
        assert plan[0]["tool"] == "none"
        assert plan[0]["status"] == "pending"
        assert plan[0]["depends_on"] == []

    def test_multiple_steps(self):
        text = """[
            {"id": 1, "description": "A", "tool": "shell", "depends_on": []},
            {"id": 2, "description": "B", "tool": "web_search", "depends_on": [1]},
            {"id": 3, "description": "C", "tool": "none", "depends_on": [1, 2]}
        ]"""
        plan = _parse_plan(text, "goal")
        assert len(plan) == 3
        assert plan[2]["depends_on"] == [1, 2]

    def test_invalid_json_fallback(self):
        plan = _parse_plan("not json at all", "my goal")
        assert len(plan) == 1
        assert plan[0]["description"] == "my goal"

    def test_garbage_around_json(self):
        text = 'Here is the plan:\n[{"id":1,"description":"X","tool":"shell"}]\nDone.'
        plan = _parse_plan(text, "g")
        assert len(plan) == 1
        assert plan[0]["description"] == "X"

    def test_bad_depends_on_type(self):
        text = '[{"id": 1, "description": "A", "depends_on": "invalid"}]'
        plan = _parse_plan(text, "g")
        assert plan[0]["depends_on"] == []


class TestDetectParallelGroups:
    def test_all_independent(self):
        plan = [
            {"id": 1, "description": "A", "status": "pending", "depends_on": []},
            {"id": 2, "description": "B", "status": "pending", "depends_on": []},
            {"id": 3, "description": "C", "status": "pending", "depends_on": []},
        ]
        groups = detect_parallel_groups(plan)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_linear_chain(self):
        plan = [
            {"id": 1, "description": "A", "status": "pending", "depends_on": []},
            {"id": 2, "description": "B", "status": "pending", "depends_on": [1]},
            {"id": 3, "description": "C", "status": "pending", "depends_on": [2]},
        ]
        groups = detect_parallel_groups(plan)
        assert len(groups) == 3
        assert all(len(g) == 1 for g in groups)

    def test_diamond(self):
        plan = [
            {"id": 1, "description": "A", "status": "completed", "depends_on": []},
            {"id": 2, "description": "B", "status": "pending", "depends_on": [1]},
            {"id": 3, "description": "C", "status": "pending", "depends_on": [1]},
            {"id": 4, "description": "D", "status": "pending", "depends_on": [2, 3]},
        ]
        groups = detect_parallel_groups(plan)
        assert len(groups[0]) == 2
        ids = {s["id"] for s in groups[0]}
        assert ids == {2, 3}
        assert groups[1][0]["id"] == 4

    def test_completed_skipped(self):
        plan = [
            {"id": 1, "description": "A", "status": "completed", "depends_on": []},
            {"id": 2, "description": "B", "status": "pending", "depends_on": [1]},
        ]
        groups = detect_parallel_groups(plan)
        assert len(groups) == 1
        assert groups[0][0]["id"] == 2

    def test_empty_plan(self):
        groups = detect_parallel_groups([])
        assert groups == []

    def test_all_completed(self):
        plan = [
            {"id": 1, "description": "A", "status": "completed", "depends_on": []},
        ]
        groups = detect_parallel_groups(plan)
        assert groups == []
