"""Unit tests for the planner module — no Ollama required."""

from unittest.mock import patch

import pytest

from ollama_chain.planner import (
    _parse_plan,
    _strip_think,
    assess_progress,
    decompose_goal,
    detect_parallel_groups,
    replan,
    should_replan,
)


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


# ---------------------------------------------------------------------------
# _strip_think
# ---------------------------------------------------------------------------

class TestStripThink:
    def test_removes_think_block(self):
        text = "<think>internal reasoning</think>\nactual answer"
        assert _strip_think(text) == "actual answer"

    def test_no_think_block(self):
        text = "just plain text"
        assert _strip_think(text) == "just plain text"

    def test_unclosed_think_block(self):
        text = "<think>no closing tag"
        assert _strip_think(text) == "<think>no closing tag"

    def test_empty_think_block(self):
        text = "<think></think>result"
        assert _strip_think(text) == "result"


# ---------------------------------------------------------------------------
# decompose_goal (mocked LLM)
# ---------------------------------------------------------------------------

class TestDecomposeGoal:
    @patch("ollama_chain.planner.chat_with_retry")
    def test_basic_decomposition(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": (
                '[{"id": 1, "description": "Check OS", "tool": "shell", "depends_on": []},'
                ' {"id": 2, "description": "Summarize", "tool": "none", "depends_on": [1]}]'
            )}
        }
        plan = decompose_goal("Find my OS version", "fast:7b")
        assert len(plan) == 2
        assert plan[0]["tool"] == "shell"
        assert plan[1]["depends_on"] == [1]

    @patch("ollama_chain.planner.chat_with_retry")
    def test_invalid_json_falls_back(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": "I don't know how to make JSON"}
        }
        plan = decompose_goal("do something", "fast:7b")
        assert len(plan) == 1
        assert plan[0]["description"] == "do something"

    @patch("ollama_chain.planner.chat_with_retry")
    def test_simple_complexity_hint(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": '[{"id":1,"description":"Quick answer","tool":"none"}]'}
        }
        plan = decompose_goal("What is DNS?", "fast:7b", complexity_hint="simple")
        assert len(plan) == 1

    @patch("ollama_chain.planner.chat_with_retry")
    def test_strips_think_from_response(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": (
                '<think>let me think</think>'
                '[{"id":1,"description":"Step one","tool":"shell"}]'
            )}
        }
        plan = decompose_goal("goal", "fast:7b")
        assert len(plan) == 1
        assert plan[0]["description"] == "Step one"


# ---------------------------------------------------------------------------
# replan (mocked LLM)
# ---------------------------------------------------------------------------

class TestReplan:
    @patch("ollama_chain.planner.chat_with_retry")
    def test_replan_produces_updated_plan(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": (
                '[{"id":1,"description":"Done","tool":"shell","status":"completed","depends_on":[]},'
                ' {"id":2,"description":"New step","tool":"web_search","status":"pending","depends_on":[1]}]'
            )}
        }
        current = [
            {"id": 1, "description": "A", "tool": "shell", "status": "completed", "depends_on": []},
            {"id": 2, "description": "B", "tool": "none", "status": "failed", "depends_on": [1]},
        ]
        new_plan = replan("goal", current, "step 2 failed because X", "fast:7b")
        assert len(new_plan) == 2
        pending = [s for s in new_plan if s["status"] == "pending"]
        assert len(pending) >= 1


# ---------------------------------------------------------------------------
# should_replan (mocked LLM)
# ---------------------------------------------------------------------------

class TestShouldReplan:
    def test_no_new_facts_returns_false(self):
        plan = [{"id": 1, "description": "A", "status": "pending"}]
        assert should_replan("goal", plan, [], "fast:7b") is False

    def test_no_pending_steps_returns_false(self):
        plan = [{"id": 1, "description": "A", "status": "completed"}]
        assert should_replan("goal", plan, ["new fact"], "fast:7b") is False

    @patch("ollama_chain.planner.chat_with_retry")
    def test_llm_says_yes(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "yes"}}
        plan = [{"id": 1, "description": "Search X", "status": "pending"}]
        assert should_replan("goal", plan, ["OS: Fedora 43"], "fast:7b") is True

    @patch("ollama_chain.planner.chat_with_retry")
    def test_llm_says_no(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "no"}}
        plan = [{"id": 1, "description": "Search X", "status": "pending"}]
        assert should_replan("goal", plan, ["fact"], "fast:7b") is False

    @patch("ollama_chain.planner.chat_with_retry", side_effect=Exception("fail"))
    def test_llm_failure_returns_false(self, mock_chat):
        plan = [{"id": 1, "description": "A", "status": "pending"}]
        assert should_replan("goal", plan, ["fact"], "fast:7b") is False


# ---------------------------------------------------------------------------
# assess_progress (mocked LLM)
# ---------------------------------------------------------------------------

class TestAssessProgress:
    @patch("ollama_chain.planner.chat_with_retry")
    def test_returns_assessment(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "On track, 2/3 done."}}
        plan = [
            {"id": 1, "status": "completed"},
            {"id": 2, "status": "completed"},
            {"id": 3, "status": "pending"},
        ]
        result = assess_progress("goal", plan, ["fact1"], "fast:7b")
        assert "track" in result.lower() or "2" in result

    @patch("ollama_chain.planner.chat_with_retry", side_effect=Exception("fail"))
    def test_fallback_on_error(self, mock_chat):
        plan = [{"id": 1, "status": "completed"}, {"id": 2, "status": "pending"}]
        result = assess_progress("goal", plan, [], "fast:7b")
        assert "1/2" in result
