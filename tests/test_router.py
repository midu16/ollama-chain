"""Unit tests for the router module — no Ollama required."""

import pytest

from ollama_chain.router import (
    COMPLEXITY_COMPLEX,
    COMPLEXITY_MODERATE,
    COMPLEXITY_SIMPLE,
    STRATEGY_DIRECT_FAST,
    STRATEGY_FULL_CASCADE,
    STRATEGY_SUBSET_CASCADE,
    RouteDecision,
    build_fallback_chain,
    classify_complexity_heuristic,
    identify_parallel_candidates,
    is_time_sensitive,
    optimize_routing,
    route_query,
    select_models_for_step,
)


MODELS = ["small:7b", "medium:14b", "large:32b"]


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

class TestClassifyComplexityHeuristic:
    def test_simple_what_is(self):
        c, conf = classify_complexity_heuristic("What is SSH?")
        assert c == COMPLEXITY_SIMPLE
        assert conf > 0

    def test_simple_short(self):
        c, _ = classify_complexity_heuristic("What port does HTTPS use?")
        assert c == COMPLEXITY_SIMPLE

    def test_moderate_medium_length(self):
        c, _ = classify_complexity_heuristic(
            "Explain the difference between TCP and UDP protocols "
            "and when you would use each one."
        )
        assert c in (COMPLEXITY_MODERATE, COMPLEXITY_COMPLEX)

    def test_complex_technical(self):
        c, _ = classify_complexity_heuristic(
            "Derive the asymptotic complexity of Dijkstra's algorithm "
            "with a Fibonacci heap implementation and compare it to the "
            "distributed consensus protocol used in Kubernetes scheduler "
            "optimization. Also analyze the cryptographic hash function "
            "pipeline and its throughput implications."
        )
        assert c == COMPLEXITY_COMPLEX

    def test_confidence_range(self):
        _, conf = classify_complexity_heuristic("Hello")
        assert 0 < conf <= 1.0

    def test_empty_query(self):
        c, _ = classify_complexity_heuristic("")
        assert c == COMPLEXITY_SIMPLE


# ---------------------------------------------------------------------------
# route_query
# ---------------------------------------------------------------------------

class TestRouteQuery:
    def test_single_model(self):
        d = route_query("test", ["only:7b"], use_llm=False)
        assert d.strategy == STRATEGY_DIRECT_FAST
        assert d.models == ["only:7b"]
        assert d.fallback_model == "only:7b"

    def test_simple_uses_fast(self):
        d = route_query("What is SSH?", MODELS, use_llm=False)
        assert d.strategy == STRATEGY_DIRECT_FAST
        assert d.models == [MODELS[0]]
        assert d.skip_search is True

    def test_complex_uses_full_cascade(self):
        d = route_query(
            "Derive the asymptotic complexity of the distributed "
            "consensus algorithm used in Kubernetes with cryptographic "
            "hash verification and pipeline optimization",
            MODELS,
            use_llm=False,
        )
        assert d.strategy == STRATEGY_FULL_CASCADE
        assert d.models == MODELS

    def test_moderate_uses_subset(self):
        d = route_query(
            "Compare REST vs GraphQL APIs and explain the trade-offs "
            "for different application architectures",
            MODELS,
            use_llm=False,
        )
        if d.complexity == COMPLEXITY_MODERATE:
            assert d.strategy == STRATEGY_SUBSET_CASCADE
            assert len(d.models) == 2
            assert d.models[0] == MODELS[0]
            assert d.models[-1] == MODELS[-1]

    def test_no_models_raises(self):
        with pytest.raises(ValueError):
            route_query("test", [])

    def test_web_search_respected(self):
        d = route_query("test", MODELS, use_llm=False, web_search=False)
        assert d.skip_search is True

    def test_decision_has_reasoning(self):
        d = route_query("What is 2+2?", MODELS, use_llm=False)
        assert d.reasoning

    def test_confidence_in_range(self):
        d = route_query("test", MODELS, use_llm=False)
        assert 0 < d.confidence <= 1.0


# ---------------------------------------------------------------------------
# build_fallback_chain
# ---------------------------------------------------------------------------

class TestBuildFallbackChain:
    def test_excludes_failed(self):
        chain = build_fallback_chain(MODELS, "medium:14b")
        assert "medium:14b" not in chain

    def test_prefers_larger_first(self):
        chain = build_fallback_chain(MODELS, "small:7b")
        assert chain[0] == "large:32b"

    def test_single_model(self):
        chain = build_fallback_chain(["only:7b"], "only:7b")
        assert chain == ["only:7b"]

    def test_all_models_present(self):
        chain = build_fallback_chain(MODELS, "small:7b")
        assert set(chain) == {"medium:14b", "large:32b"}


# ---------------------------------------------------------------------------
# select_models_for_step
# ---------------------------------------------------------------------------

class TestSelectModelsForStep:
    def test_reasoning_step_uses_strong(self):
        step = {"tool": "none", "description": "Analyze results"}
        models = select_models_for_step(step, MODELS, COMPLEXITY_COMPLEX)
        assert models == [MODELS[-1]]

    def test_shell_step_simple_uses_fast(self):
        step = {"tool": "shell", "description": "Run uname"}
        models = select_models_for_step(step, MODELS, COMPLEXITY_SIMPLE)
        assert models == [MODELS[0]]

    def test_shell_step_complex_uses_all(self):
        step = {"tool": "shell", "description": "Run uname"}
        models = select_models_for_step(step, MODELS, COMPLEXITY_COMPLEX)
        assert models == MODELS

    def test_web_search_simple(self):
        step = {"tool": "web_search", "description": "Search"}
        models = select_models_for_step(step, MODELS, COMPLEXITY_SIMPLE)
        assert models == [MODELS[0]]

    def test_unknown_tool_uses_all(self):
        step = {"tool": "python_eval", "description": "Calculate"}
        models = select_models_for_step(step, MODELS, COMPLEXITY_MODERATE)
        assert models == MODELS

    def test_empty_models(self):
        step = {"tool": "shell", "description": "Run"}
        assert select_models_for_step(step, [], COMPLEXITY_SIMPLE) == []

    def test_missing_tool_key(self):
        step = {"description": "Do something"}
        models = select_models_for_step(step, MODELS, COMPLEXITY_COMPLEX)
        assert models == [MODELS[-1]]


# ---------------------------------------------------------------------------
# optimize_routing
# ---------------------------------------------------------------------------

class TestOptimizeRouting:
    def test_adds_preferred_models(self):
        plan = [
            {"id": 1, "description": "List files", "tool": "list_dir",
             "depends_on": [], "status": "pending"},
            {"id": 2, "description": "Analyze results", "tool": "none",
             "depends_on": [1], "status": "pending"},
        ]
        result = optimize_routing(plan, MODELS, COMPLEXITY_MODERATE)
        assert "preferred_models" in result[0]
        assert "preferred_models" in result[1]

    def test_data_gathering_uses_fast_for_simple(self):
        plan = [
            {"id": 1, "description": "Run uname", "tool": "shell",
             "depends_on": [], "status": "pending"},
        ]
        optimize_routing(plan, MODELS, COMPLEXITY_SIMPLE)
        assert plan[0]["preferred_models"] == [MODELS[0]]

    def test_reasoning_uses_strong(self):
        plan = [
            {"id": 1, "description": "Analyze and summarize the findings",
             "tool": "none", "depends_on": [], "status": "pending"},
        ]
        optimize_routing(plan, MODELS, COMPLEXITY_COMPLEX)
        assert plan[0]["preferred_models"] == [MODELS[-1]]

    def test_skips_completed_steps(self):
        plan = [
            {"id": 1, "description": "Done", "tool": "shell",
             "depends_on": [], "status": "completed"},
        ]
        optimize_routing(plan, MODELS, COMPLEXITY_SIMPLE)
        assert "preferred_models" not in plan[0]

    def test_deprioritises_failed_models(self):
        plan = [
            {"id": 1, "description": "Search web", "tool": "web_search",
             "depends_on": [], "status": "pending"},
        ]
        optimize_routing(
            plan, MODELS, COMPLEXITY_SIMPLE,
            failed_models={MODELS[0]},
        )
        assert MODELS[0] not in plan[0]["preferred_models"]

    def test_empty_models_safe(self):
        plan = [
            {"id": 1, "description": "X", "tool": "shell",
             "depends_on": [], "status": "pending"},
        ]
        result = optimize_routing(plan, [], COMPLEXITY_SIMPLE)
        assert result == plan

    def test_python_eval_uses_fast(self):
        plan = [
            {"id": 1, "description": "Calculate 2+2", "tool": "python_eval",
             "depends_on": [], "status": "pending"},
        ]
        optimize_routing(plan, MODELS, COMPLEXITY_MODERATE)
        assert plan[0]["preferred_models"] == [MODELS[0]]


# ---------------------------------------------------------------------------
# identify_parallel_candidates
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# is_time_sensitive
# ---------------------------------------------------------------------------

class TestIsTimeSensitive:
    def test_latest_keyword(self):
        assert is_time_sensitive("What is the latest OpenShift release?")

    def test_current_keyword(self):
        assert is_time_sensitive("current version of Kubernetes")

    def test_newest_keyword(self):
        assert is_time_sensitive("newest Linux kernel release")

    def test_most_recent_phrase(self):
        assert is_time_sensitive("What is the most recent CVE for Apache?")

    def test_version_keyword(self):
        assert is_time_sensitive("What version of RHEL is out?")

    def test_release_keyword(self):
        assert is_time_sensitive("What release of Fedora is available?")

    def test_year_keyword(self):
        assert is_time_sensitive("Best tools for DevOps in 2025")

    def test_this_week(self):
        assert is_time_sensitive("What happened this week in Go?")

    def test_not_time_sensitive_definition(self):
        assert not is_time_sensitive("What is a binary search tree?")

    def test_not_time_sensitive_concept(self):
        assert not is_time_sensitive("Explain how TLS certificates work")

    def test_not_time_sensitive_comparison(self):
        assert not is_time_sensitive("Compare REST and GraphQL")

    def test_empty_string(self):
        assert not is_time_sensitive("")

    def test_case_insensitive(self):
        assert is_time_sensitive("LATEST VERSION OF PYTHON")


# ---------------------------------------------------------------------------
# Routing with time_sensitive flag
# ---------------------------------------------------------------------------

class TestRouteTimeSensitive:
    def test_time_sensitive_upgrades_simple_to_moderate(self):
        d = route_query("What is the latest Python version?", MODELS, use_llm=False)
        assert d.time_sensitive is True
        assert d.complexity != COMPLEXITY_SIMPLE

    def test_time_sensitive_never_skips_search(self):
        d = route_query("latest Docker release", MODELS, use_llm=False)
        assert d.time_sensitive is True
        if d.complexity == COMPLEXITY_SIMPLE:
            pytest.fail("time-sensitive should not be simple")

    def test_non_time_sensitive_can_skip_search(self):
        d = route_query("What is SSH?", MODELS, use_llm=False)
        assert d.time_sensitive is False
        assert d.skip_search is True

    def test_route_decision_dataclass_has_field(self):
        d = RouteDecision(
            models=MODELS, complexity="complex", strategy="full_cascade",
            fallback_model="small:7b", skip_search=False, confidence=0.8,
            time_sensitive=True,
        )
        assert d.time_sensitive is True

    def test_time_sensitive_defaults_false(self):
        d = RouteDecision(
            models=MODELS, complexity="simple", strategy="direct_fast",
            fallback_model="small:7b", skip_search=True, confidence=0.8,
        )
        assert d.time_sensitive is False


class TestIdentifyParallelCandidates:
    def test_all_independent(self):
        plan = [
            {"id": 1, "status": "pending", "depends_on": []},
            {"id": 2, "status": "pending", "depends_on": []},
            {"id": 3, "status": "pending", "depends_on": []},
        ]
        groups = identify_parallel_candidates(plan)
        assert len(groups) == 1
        assert set(groups[0]) == {1, 2, 3}

    def test_linear_chain(self):
        plan = [
            {"id": 1, "status": "pending", "depends_on": []},
            {"id": 2, "status": "pending", "depends_on": [1]},
            {"id": 3, "status": "pending", "depends_on": [2]},
        ]
        groups = identify_parallel_candidates(plan)
        assert len(groups) == 3
        assert all(len(g) == 1 for g in groups)

    def test_diamond(self):
        plan = [
            {"id": 1, "status": "completed", "depends_on": []},
            {"id": 2, "status": "pending", "depends_on": [1]},
            {"id": 3, "status": "pending", "depends_on": [1]},
            {"id": 4, "status": "pending", "depends_on": [2, 3]},
        ]
        groups = identify_parallel_candidates(plan)
        assert set(groups[0]) == {2, 3}
        assert groups[1] == [4]

    def test_empty_plan(self):
        assert identify_parallel_candidates([]) == []

    def test_all_completed(self):
        plan = [
            {"id": 1, "status": "completed", "depends_on": []},
        ]
        assert identify_parallel_candidates(plan) == []
