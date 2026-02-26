"""Unit tests for common.py — model capability detection, ask(), retry, helpers."""

from unittest.mock import MagicMock, patch, call

import pytest

from ollama_chain.common import (
    _DEFAULT_KEEP_ALIVE,
    _SOURCE_MARKERS,
    SOURCE_GUIDANCE,
    ask,
    build_structured_prompt,
    chat_with_retry,
    ensure_sources,
    format_prompt_section,
    model_supports_thinking,
    sanitize_messages,
    unload_all_models,
    unload_model,
)


# ---------------------------------------------------------------------------
# model_supports_thinking
# ---------------------------------------------------------------------------

class TestModelSupportsThinking:
    def setup_method(self):
        model_supports_thinking.cache_clear()

    @patch("ollama_chain.common.ollama_client")
    def test_thinking_model_via_api(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.capabilities = ["completion", "tools", "thinking"]
        mock_client.show.return_value = mock_resp

        assert model_supports_thinking("qwen3:14b") is True
        mock_client.show.assert_called_once_with("qwen3:14b")

    @patch("ollama_chain.common.ollama_client")
    def test_non_thinking_model_via_api(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.capabilities = ["completion", "tools"]
        mock_client.show.return_value = mock_resp

        assert model_supports_thinking("mistral-large:123b") is False

    @patch("ollama_chain.common.ollama_client")
    def test_fallback_to_name_heuristic_qwen3(self, mock_client):
        mock_client.show.side_effect = Exception("connection refused")
        assert model_supports_thinking("qwen3:8b") is True

    @patch("ollama_chain.common.ollama_client")
    def test_fallback_to_name_heuristic_deepseek(self, mock_client):
        mock_client.show.side_effect = Exception("connection refused")
        assert model_supports_thinking("deepseek-r1:70b") is True

    @patch("ollama_chain.common.ollama_client")
    def test_fallback_to_name_heuristic_unknown(self, mock_client):
        mock_client.show.side_effect = Exception("connection refused")
        assert model_supports_thinking("llama3:8b") is False

    @patch("ollama_chain.common.ollama_client")
    def test_result_is_cached(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.capabilities = ["thinking"]
        mock_client.show.return_value = mock_resp

        model_supports_thinking("test:1b")
        model_supports_thinking("test:1b")
        mock_client.show.assert_called_once()

    @patch("ollama_chain.common.ollama_client")
    def test_none_capabilities(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.capabilities = None
        mock_client.show.return_value = mock_resp

        assert model_supports_thinking("unknown:7b") is False

    @patch("ollama_chain.common.ollama_client")
    def test_qwq_fallback(self, mock_client):
        mock_client.show.side_effect = Exception("fail")
        assert model_supports_thinking("qwq:32b") is True


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

class TestAsk:
    @patch("ollama_chain.common.model_supports_thinking", return_value=True)
    @patch("ollama_chain.common.chat_with_retry")
    def test_no_think_prefix_when_thinking_false(self, mock_chat, mock_cap):
        mock_chat.return_value = {"message": {"content": "answer"}}
        result = ask("question", "qwen3:14b", thinking=False)
        sent_prompt = mock_chat.call_args[1]["messages"][0]["content"]
        assert sent_prompt.startswith("/no_think\n")
        assert result == "answer"

    @patch("ollama_chain.common.model_supports_thinking", return_value=True)
    @patch("ollama_chain.common.chat_with_retry")
    def test_no_prefix_when_thinking_true(self, mock_chat, mock_cap):
        mock_chat.return_value = {"message": {"content": "answer"}}
        result = ask("question", "qwen3:14b", thinking=True)
        sent_prompt = mock_chat.call_args[1]["messages"][0]["content"]
        assert not sent_prompt.startswith("/no_think")

    @patch("ollama_chain.common.model_supports_thinking", return_value=False)
    @patch("ollama_chain.common.chat_with_retry")
    def test_no_prefix_for_non_thinking_model(self, mock_chat, mock_cap):
        mock_chat.return_value = {"message": {"content": "answer"}}
        ask("question", "mistral-large:123b", thinking=False)
        sent_prompt = mock_chat.call_args[1]["messages"][0]["content"]
        assert not sent_prompt.startswith("/no_think")

    @patch("ollama_chain.common.model_supports_thinking", return_value=False)
    @patch("ollama_chain.common.chat_with_retry")
    def test_no_prefix_for_non_thinking_model_thinking_true(self, mock_chat, mock_cap):
        mock_chat.return_value = {"message": {"content": "answer"}}
        ask("question", "mistral-large:123b", thinking=True)
        sent_prompt = mock_chat.call_args[1]["messages"][0]["content"]
        assert not sent_prompt.startswith("/no_think")

    @patch("ollama_chain.common.model_supports_thinking", return_value=True)
    @patch("ollama_chain.common.chat_with_retry")
    def test_strips_think_tags(self, mock_chat, mock_cap):
        mock_chat.return_value = {
            "message": {"content": "<think>reasoning here</think>The real answer"}
        }
        result = ask("q", "qwen3:14b", thinking=True)
        assert result == "The real answer"
        assert "<think>" not in result

    @patch("ollama_chain.common.model_supports_thinking", return_value=True)
    @patch("ollama_chain.common.chat_with_retry")
    def test_temperature_passed_as_options(self, mock_chat, mock_cap):
        mock_chat.return_value = {"message": {"content": "ans"}}
        ask("q", "qwen3:14b", temperature=0.3)
        _, kwargs = mock_chat.call_args
        assert kwargs["options"] == {"temperature": 0.3}

    @patch("ollama_chain.common.model_supports_thinking", return_value=True)
    @patch("ollama_chain.common.chat_with_retry")
    def test_no_options_when_temperature_none(self, mock_chat, mock_cap):
        mock_chat.return_value = {"message": {"content": "ans"}}
        ask("q", "qwen3:14b")
        _, kwargs = mock_chat.call_args
        assert kwargs.get("options") is None

    @patch("ollama_chain.common.model_supports_thinking", return_value=True)
    @patch("ollama_chain.common.chat_with_retry")
    def test_no_unclosed_think_tag(self, mock_chat, mock_cap):
        mock_chat.return_value = {
            "message": {"content": "<think>no closing tag but still answer"}
        }
        result = ask("q", "m", thinking=True)
        assert "<think>" in result


# ---------------------------------------------------------------------------
# chat_with_retry
# ---------------------------------------------------------------------------

class TestChatWithRetry:
    @patch("ollama_chain.common.ollama_client")
    def test_success_first_attempt(self, mock_client):
        mock_client.chat.return_value = {"message": {"content": "ok"}}
        result = chat_with_retry("model", [{"role": "user", "content": "q"}])
        assert result["message"]["content"] == "ok"

    @patch("ollama_chain.common.ollama_client")
    def test_retry_on_transient_error(self, mock_client):
        mock_client.chat.side_effect = [
            ConnectionError("connection reset"),
            {"message": {"content": "ok"}},
        ]
        result = chat_with_retry(
            "model", [{"role": "user", "content": "q"}],
            retries=2, keep_alive="1m",
        )
        assert result["message"]["content"] == "ok"
        assert mock_client.chat.call_count == 2

    @patch("ollama_chain.common.ollama_client")
    def test_non_retryable_error_raises_immediately(self, mock_client):
        mock_client.chat.side_effect = ValueError("invalid model")
        with pytest.raises(ValueError, match="invalid model"):
            chat_with_retry("model", [{"role": "user", "content": "q"}])
        assert mock_client.chat.call_count == 1

    @patch("ollama_chain.common.ollama_client")
    def test_all_retries_exhausted(self, mock_client):
        mock_client.chat.side_effect = ConnectionError("connection timeout")
        with pytest.raises(ConnectionError):
            chat_with_retry(
                "model", [{"role": "user", "content": "q"}],
                retries=2, keep_alive="1m",
            )
        assert mock_client.chat.call_count == 2

    @patch("ollama_chain.common.ollama_client")
    def test_default_keep_alive(self, mock_client):
        mock_client.chat.return_value = {"message": {"content": "ok"}}
        chat_with_retry("model", [{"role": "user", "content": "q"}])
        _, kwargs = mock_client.chat.call_args
        assert kwargs["keep_alive"] == _DEFAULT_KEEP_ALIVE

    @patch("ollama_chain.common.ollama_client")
    def test_options_passed_through(self, mock_client):
        mock_client.chat.return_value = {"message": {"content": "ok"}}
        chat_with_retry(
            "model", [{"role": "user", "content": "q"}],
            options={"temperature": 0.4},
        )
        _, kwargs = mock_client.chat.call_args
        assert kwargs["options"] == {"temperature": 0.4}

    @patch("ollama_chain.common.ollama_client")
    def test_no_options_by_default(self, mock_client):
        mock_client.chat.return_value = {"message": {"content": "ok"}}
        chat_with_retry("model", [{"role": "user", "content": "q"}])
        _, kwargs = mock_client.chat.call_args
        assert "options" not in kwargs


# ---------------------------------------------------------------------------
# sanitize_messages
# ---------------------------------------------------------------------------

class TestSanitizeMessages:
    def test_alternating_roles_unchanged(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 2

    def test_consecutive_same_role_merged(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 2
        assert "a" in result[0]["content"]
        assert "b" in result[0]["content"]

    def test_empty_list(self):
        assert sanitize_messages([]) == []

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = sanitize_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_does_not_mutate_original(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        original_first = msgs[0]["content"]
        sanitize_messages(msgs)
        assert msgs[0]["content"] == original_first

    def test_three_consecutive_merged(self):
        msgs = [
            {"role": "assistant", "content": "x"},
            {"role": "assistant", "content": "y"},
            {"role": "assistant", "content": "z"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 1
        assert "x" in result[0]["content"]
        assert "z" in result[0]["content"]


# ---------------------------------------------------------------------------
# unload helpers
# ---------------------------------------------------------------------------

class TestUnloadHelpers:
    @patch("ollama_chain.common.ollama_client")
    def test_unload_model(self, mock_client):
        unload_model("test:7b")
        mock_client.generate.assert_called_once_with(
            model="test:7b", prompt="", keep_alive=0,
        )

    @patch("ollama_chain.common.ollama_client")
    def test_unload_model_error_ignored(self, mock_client):
        mock_client.generate.side_effect = Exception("fail")
        unload_model("test:7b")  # should not raise

    @patch("ollama_chain.common.unload_model")
    def test_unload_all_deduplicates(self, mock_unload):
        unload_all_models(["a", "b", "a", "c", "b"])
        assert mock_unload.call_count == 3
        mock_unload.assert_any_call("a")
        mock_unload.assert_any_call("b")
        mock_unload.assert_any_call("c")

    @patch("ollama_chain.common.unload_model")
    def test_unload_all_empty(self, mock_unload):
        unload_all_models([])
        mock_unload.assert_not_called()


# ---------------------------------------------------------------------------
# format_prompt_section / build_structured_prompt
# ---------------------------------------------------------------------------

class TestPromptFormatting:
    def test_format_prompt_section(self):
        result = format_prompt_section("role", "You are helpful")
        assert result == "=== ROLE ===\nYou are helpful"

    def test_build_structured_prompt_basic(self):
        result = build_structured_prompt(
            [("Title", "Body")], instructions="Do it well",
        )
        assert "=== TITLE ===" in result
        assert "Body" in result
        assert "Do it well" in result

    def test_build_structured_prompt_skips_empty(self):
        result = build_structured_prompt(
            [("A", "content"), ("B", ""), ("C", "  ")],
        )
        assert "=== A ===" in result
        assert "=== B ===" not in result
        assert "=== C ===" not in result

    def test_build_structured_prompt_no_instructions(self):
        result = build_structured_prompt([("A", "b")])
        assert "=== A ===" in result


# ---------------------------------------------------------------------------
# ensure_sources
# ---------------------------------------------------------------------------

class TestEnsureSources:
    def test_already_has_sources_section(self):
        answer = "The answer.\n\n## Sources\n1. Source one"
        result = ensure_sources(answer, "query", "model")
        assert result == answer

    def test_already_has_references_section(self):
        answer = "The answer.\n\n## References\n1. Ref"
        result = ensure_sources(answer, "query", "model")
        assert result == answer

    def test_already_has_bold_sources(self):
        answer = "The answer.\n\n**Sources**\n1. Src"
        result = ensure_sources(answer, "query", "model")
        assert result == answer

    @patch("ollama_chain.common.ask")
    def test_appends_sources_when_missing(self, mock_ask):
        mock_ask.return_value = "## Sources\n1. Added source"
        answer = "An answer without sources."
        result = ensure_sources(answer, "query", "model")
        assert "## Sources" in result
        assert "Added source" in result

    @patch("ollama_chain.common.ask")
    def test_returns_original_on_failure(self, mock_ask):
        mock_ask.side_effect = Exception("model down")
        answer = "An answer without sources."
        result = ensure_sources(answer, "query", "model")
        assert result == answer

    @patch("ollama_chain.common.ask")
    def test_returns_original_on_empty_response(self, mock_ask):
        mock_ask.return_value = "   "
        answer = "An answer without sources."
        result = ensure_sources(answer, "query", "model")
        assert result == answer

    def test_case_insensitive_detection(self):
        answer = "Answer.\n\n### SOURCES\n1. Src"
        result = ensure_sources(answer, "q", "m")
        assert result == answer
