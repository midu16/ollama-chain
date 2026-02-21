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
