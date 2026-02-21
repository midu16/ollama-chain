"""Unit tests for cascade error handling and fallback — no Ollama required."""

from unittest.mock import MagicMock, patch

import pytest

from ollama_chain.chains import chain_cascade, chain_auto
from ollama_chain.router import RouteDecision, STRATEGY_FULL_CASCADE


MODELS = ["small:7b", "medium:14b", "large:32b"]


class TestCascadeErrorHandling:
    """Verify that the cascade gracefully handles model failures."""

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_normal_cascade(self, _search, mock_ask):
        """All models succeed — normal flow."""
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        result = chain_cascade("test query", MODELS, web_search=False)
        assert result == "final"
        assert mock_ask.call_count == 3

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_intermediate_model_fails(self, _search, mock_ask):
        """Middle model fails — cascade skips it and continues."""
        mock_ask.side_effect = [
            "draft",
            Exception("medium model unavailable"),
            "final",
        ]
        result = chain_cascade("test query", MODELS, web_search=False)
        assert result == "final"

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_draft_model_falls_back(self, _search, mock_ask):
        """First model fails — cascade falls back to next model for draft."""
        mock_ask.side_effect = [
            Exception("small model down"),
            "draft from medium",
            "final",
        ]
        result = chain_cascade("test query", MODELS, web_search=False)
        assert result == "final"

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_final_model_falls_back(self, _search, mock_ask):
        """Final model fails — cascade falls back to previous model."""
        mock_ask.side_effect = [
            "draft",
            "reviewed",
            Exception("large model down"),
            "fallback final",
        ]
        result = chain_cascade("test query", MODELS, web_search=False)
        assert result == "fallback final"

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_all_models_fail_during_draft(self, _search, mock_ask):
        """All models fail during draft — raises RuntimeError."""
        mock_ask.side_effect = Exception("all down")
        with pytest.raises(RuntimeError, match="All models failed"):
            chain_cascade("test", MODELS, web_search=False)

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_all_review_and_final_fail_returns_draft(self, _search, mock_ask):
        """Draft succeeds but all subsequent models fail — returns draft."""
        mock_ask.side_effect = [
            "draft answer",
            Exception("medium fail"),
            Exception("large fail"),
            Exception("medium fallback fail"),
        ]
        result = chain_cascade("test", MODELS, web_search=False)
        assert result == "draft answer"

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_single_model_cascade(self, _search, mock_ask):
        """Single model — no review or final stage."""
        mock_ask.return_value = "answer"
        result = chain_cascade("test", ["only:7b"], web_search=False)
        assert result == "answer"
        assert mock_ask.call_count == 1

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_two_model_cascade(self, _search, mock_ask):
        """Two models — draft + final, no intermediate review."""
        mock_ask.side_effect = ["draft", "final"]
        result = chain_cascade("test", ["small:7b", "large:32b"], web_search=False)
        assert result == "final"
        assert mock_ask.call_count == 2


class TestChainAuto:
    """Verify that auto mode uses the router and dispatches correctly."""

    @patch("ollama_chain.chains.route_query")
    @patch("ollama_chain.chains.chain_fast")
    def test_simple_routes_to_fast(self, mock_fast, mock_route):
        mock_route.return_value = RouteDecision(
            models=["small:7b"],
            complexity="simple",
            strategy="direct_fast",
            fallback_model="large:32b",
            skip_search=True,
            confidence=0.8,
            reasoning="test",
        )
        mock_fast.return_value = "fast answer"
        result = chain_auto("What is SSH?", MODELS, web_search=True)
        assert result == "fast answer"
        mock_fast.assert_called_once()

    @patch("ollama_chain.chains.route_query")
    @patch("ollama_chain.chains.chain_cascade")
    def test_complex_routes_to_full_cascade(self, mock_cascade, mock_route):
        mock_route.return_value = RouteDecision(
            models=MODELS,
            complexity="complex",
            strategy="full_cascade",
            fallback_model="small:7b",
            skip_search=False,
            confidence=0.7,
            reasoning="test",
        )
        mock_cascade.return_value = "cascade answer"
        result = chain_auto("complex question", MODELS, web_search=True)
        assert result == "cascade answer"
        mock_cascade.assert_called_once()

    @patch("ollama_chain.chains.route_query")
    @patch("ollama_chain.chains.chain_cascade")
    def test_moderate_routes_to_subset(self, mock_cascade, mock_route):
        subset = ["small:7b", "large:32b"]
        mock_route.return_value = RouteDecision(
            models=subset,
            complexity="moderate",
            strategy="subset_cascade",
            fallback_model="small:7b",
            skip_search=False,
            confidence=0.6,
            reasoning="test",
        )
        mock_cascade.return_value = "subset answer"
        result = chain_auto("moderate question", MODELS)
        assert result == "subset answer"
        args, kwargs = mock_cascade.call_args
        assert args[1] == subset
