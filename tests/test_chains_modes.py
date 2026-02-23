"""Unit tests for all chain modes — mocked LLM calls, no Ollama required."""

from unittest.mock import patch, call

import pytest

from ollama_chain.chains import (
    CLI_ONLY_MODES,
    _enrich_with_search,
    _inject_search_context,
    chain_cascade,
    chain_consensus,
    chain_fast,
    chain_pipeline,
    chain_route,
    chain_search,
    chain_strong,
    chain_verify,
)


MODELS = ["small:7b", "medium:14b", "large:32b"]


# ---------------------------------------------------------------------------
# CLI_ONLY_MODES
# ---------------------------------------------------------------------------

class TestCLIOnlyModes:
    def test_pcap_is_cli_only(self):
        assert "pcap" in CLI_ONLY_MODES

    def test_k8s_is_cli_only(self):
        assert "k8s" in CLI_ONLY_MODES

    def test_cascade_is_not_cli_only(self):
        assert "cascade" not in CLI_ONLY_MODES

    def test_agent_is_not_cli_only(self):
        assert "agent" not in CLI_ONLY_MODES


# ---------------------------------------------------------------------------
# _enrich_with_search / _inject_search_context
# ---------------------------------------------------------------------------

class TestSearchHelpers:
    @patch("ollama_chain.chains.search_for_query", return_value="result1\nresult2")
    def test_enrich_with_search_enabled(self, mock_search):
        ctx = _enrich_with_search("query", "fast:7b", True)
        assert "SEARCH RESULTS" in ctx
        assert "result1" in ctx

    def test_enrich_with_search_disabled(self):
        ctx = _enrich_with_search("query", "fast:7b", False)
        assert ctx == ""

    @patch("ollama_chain.chains.search_for_query", return_value="")
    def test_enrich_with_search_no_results(self, mock_search):
        ctx = _enrich_with_search("query", "fast:7b", True)
        assert ctx == ""

    def test_inject_search_context_with_data(self):
        result = _inject_search_context("prompt", "\nSEARCH")
        assert result == "prompt\nSEARCH"

    def test_inject_search_context_empty(self):
        result = _inject_search_context("prompt", "")
        assert result == "prompt"


# ---------------------------------------------------------------------------
# chain_cascade — complexity-driven thinking
# ---------------------------------------------------------------------------

class TestCascadeComplexity:
    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_simple_complexity_no_thinking(self, _s, mock_ask):
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        chain_cascade("q", MODELS, web_search=False, complexity="simple")
        for c in mock_ask.call_args_list:
            assert c.kwargs.get("thinking", False) is False

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_moderate_complexity_final_only_thinking(self, _s, mock_ask):
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        chain_cascade("q", MODELS, web_search=False, complexity="moderate")
        # draft: no thinking, review: no thinking, final: thinking
        calls = mock_ask.call_args_list
        assert calls[0].kwargs.get("thinking", False) is False  # draft
        assert calls[1].kwargs.get("thinking", False) is False  # review
        assert calls[2].kwargs.get("thinking") is True          # final

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_complex_complexity_review_and_final_thinking(self, _s, mock_ask):
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        chain_cascade("q", MODELS, web_search=False, complexity="complex")
        calls = mock_ask.call_args_list
        assert calls[0].kwargs.get("thinking", False) is False  # draft
        assert calls[1].kwargs.get("thinking") is True          # review
        assert calls[2].kwargs.get("thinking") is True          # final

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_none_complexity_defaults_to_complex(self, _s, mock_ask):
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        chain_cascade("q", MODELS, web_search=False, complexity=None)
        calls = mock_ask.call_args_list
        assert calls[1].kwargs.get("thinking") is True  # review
        assert calls[2].kwargs.get("thinking") is True  # final

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_complex_passes_temperature_for_review(self, _s, mock_ask):
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        chain_cascade("q", MODELS, web_search=False, complexity="complex")
        calls = mock_ask.call_args_list
        assert calls[1].kwargs.get("temperature") == 0.4  # review
        assert calls[2].kwargs.get("temperature") == 0.3  # final


# ---------------------------------------------------------------------------
# chain_route
# ---------------------------------------------------------------------------

class TestChainRoute:
    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_simple_routes_to_fast(self, _s, mock_ask):
        mock_ask.side_effect = ["2", "fast answer"]
        result = chain_route("simple q", MODELS, web_search=False)
        assert result == "fast answer"
        assert mock_ask.call_count == 2

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_complex_routes_to_strong(self, _s, mock_ask):
        mock_ask.side_effect = ["5", "strong answer"]
        result = chain_route("complex q", MODELS, web_search=False)
        assert result == "strong answer"
        # Second call should use strong model with thinking
        second_call = mock_ask.call_args_list[1]
        assert second_call.kwargs.get("thinking") is True

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_non_numeric_defaults_complex(self, _s, mock_ask):
        mock_ask.side_effect = ["not a number", "strong answer"]
        result = chain_route("q", MODELS, web_search=False)
        assert result == "strong answer"


# ---------------------------------------------------------------------------
# chain_pipeline
# ---------------------------------------------------------------------------

class TestChainPipeline:
    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_three_stage_pipeline(self, _s, mock_ask):
        mock_ask.side_effect = ["key points", "networking", "deep analysis"]
        result = chain_pipeline("explain TCP", MODELS, web_search=False)
        assert result == "deep analysis"
        assert mock_ask.call_count == 3
        # Final call uses thinking
        assert mock_ask.call_args_list[2].kwargs.get("thinking") is True


# ---------------------------------------------------------------------------
# chain_verify
# ---------------------------------------------------------------------------

class TestChainVerify:
    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_draft_then_verify(self, _s, mock_ask):
        mock_ask.side_effect = ["draft", "verified"]
        result = chain_verify("q", MODELS, web_search=False)
        assert result == "verified"
        assert mock_ask.call_count == 2
        assert mock_ask.call_args_list[1].kwargs.get("thinking") is True


# ---------------------------------------------------------------------------
# chain_consensus
# ---------------------------------------------------------------------------

class TestChainConsensus:
    @patch("ollama_chain.chains.model_supports_thinking", return_value=False)
    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_all_models_answer_then_merge(self, _s, mock_ask, _cap):
        mock_ask.side_effect = ["ans1", "ans2", "ans3", "merged"]
        result = chain_consensus("q", MODELS, web_search=False)
        assert result == "merged"
        assert mock_ask.call_count == 4  # 3 answers + 1 merge


# ---------------------------------------------------------------------------
# chain_search
# ---------------------------------------------------------------------------

class TestChainSearch:
    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains.search_for_query", return_value="search results")
    def test_search_with_results(self, _s, mock_ask):
        mock_ask.return_value = "synthesized"
        result = chain_search("q", MODELS, web_search=True)
        assert result == "synthesized"
        assert mock_ask.call_args.kwargs.get("thinking") is True

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains.search_for_query", return_value="")
    def test_search_no_results_fallback(self, _s, mock_ask):
        mock_ask.return_value = "fallback answer"
        result = chain_search("q", MODELS, web_search=True)
        assert result == "fallback answer"


# ---------------------------------------------------------------------------
# chain_fast / chain_strong
# ---------------------------------------------------------------------------

class TestChainFastStrong:
    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_fast_no_thinking(self, _s, mock_ask):
        mock_ask.return_value = "fast"
        result = chain_fast("q", MODELS, web_search=False)
        assert result == "fast"
        assert mock_ask.call_args.kwargs.get("thinking", False) is False

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_strong_with_thinking(self, _s, mock_ask):
        mock_ask.return_value = "strong"
        result = chain_strong("q", MODELS, web_search=False)
        assert result == "strong"
        assert mock_ask.call_args.kwargs.get("thinking") is True
        assert mock_ask.call_args.kwargs.get("temperature") == 0.3


# ---------------------------------------------------------------------------
# chain_auto (complexity propagation)
# ---------------------------------------------------------------------------

class TestChainAutoComplexity:
    @patch("ollama_chain.chains.route_query")
    @patch("ollama_chain.chains.chain_cascade")
    def test_auto_passes_complexity_to_cascade(self, mock_cascade, mock_route):
        from ollama_chain.router import RouteDecision
        mock_route.return_value = RouteDecision(
            models=MODELS, complexity="complex",
            strategy="full_cascade", fallback_model="small:7b",
            skip_search=False, confidence=0.7, reasoning="test",
        )
        mock_cascade.return_value = "result"
        from ollama_chain.chains import chain_auto
        chain_auto("q", MODELS, web_search=True)
        _, kwargs = mock_cascade.call_args
        assert kwargs["complexity"] == "complex"

    @patch("ollama_chain.chains.route_query")
    @patch("ollama_chain.chains.chain_cascade")
    def test_auto_subset_passes_complexity(self, mock_cascade, mock_route):
        from ollama_chain.router import RouteDecision
        mock_route.return_value = RouteDecision(
            models=["small:7b", "large:32b"], complexity="moderate",
            strategy="subset_cascade", fallback_model="small:7b",
            skip_search=False, confidence=0.6, reasoning="test",
        )
        mock_cascade.return_value = "result"
        from ollama_chain.chains import chain_auto
        chain_auto("q", MODELS, web_search=True)
        _, kwargs = mock_cascade.call_args
        assert kwargs["complexity"] == "moderate"
