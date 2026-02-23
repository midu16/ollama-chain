"""Unit tests for models.py — model discovery, selection, formatting."""

from unittest.mock import MagicMock, patch

import pytest

from ollama_chain.models import (
    ModelInfo,
    _get,
    _parse_param_size,
    list_models_table,
    pick_models,
)


# ---------------------------------------------------------------------------
# _parse_param_size
# ---------------------------------------------------------------------------

class TestParseParamSize:
    def test_billions(self):
        assert _parse_param_size("14.8B") == 14.8

    def test_millions(self):
        assert _parse_param_size("350M") == pytest.approx(0.35)

    def test_raw_number(self):
        assert _parse_param_size("7.0") == 7.0

    def test_empty_string(self):
        assert _parse_param_size("") == 0.0

    def test_invalid(self):
        assert _parse_param_size("abc") == 0.0

    def test_lowercase(self):
        assert _parse_param_size("8b") == 8.0

    def test_with_whitespace(self):
        assert _parse_param_size("  32.8B  ") == 32.8


# ---------------------------------------------------------------------------
# _get helper
# ---------------------------------------------------------------------------

class TestGetHelper:
    def test_dict_access(self):
        assert _get({"key": "val"}, "key") == "val"

    def test_dict_default(self):
        assert _get({}, "missing", "fallback") == "fallback"

    def test_object_access(self):
        obj = MagicMock()
        obj.name = "test"
        assert _get(obj, "name") == "test"

    def test_object_default(self):
        obj = MagicMock(spec=[])
        assert _get(obj, "missing", "fb") == "fb"


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_creation(self):
        m = ModelInfo(
            name="qwen3:14b",
            parameter_size=14.8,
            quantization="Q4_K_M",
            family="qwen3",
            size_bytes=9_300_000_000,
        )
        assert m.name == "qwen3:14b"
        assert m.parameter_size == 14.8
        assert m.family == "qwen3"


# ---------------------------------------------------------------------------
# pick_models
# ---------------------------------------------------------------------------

class TestPickModels:
    def test_single_model(self):
        m = ModelInfo("a", 7.0, "Q4", "qwen", 5_000_000_000)
        fast, strong = pick_models([m])
        assert fast == strong == m

    def test_two_models(self):
        small = ModelInfo("s", 7.0, "Q4", "qwen", 5_000_000_000)
        large = ModelInfo("l", 70.0, "Q4", "qwen", 40_000_000_000)
        fast, strong = pick_models([small, large])
        assert fast == small
        assert strong == large

    def test_many_models(self):
        models = [
            ModelInfo(f"m{i}", float(i), "Q4", "f", i * 1_000_000_000)
            for i in range(1, 6)
        ]
        fast, strong = pick_models(models)
        assert fast == models[0]
        assert strong == models[-1]


# ---------------------------------------------------------------------------
# list_models_table
# ---------------------------------------------------------------------------

class TestListModelsTable:
    def test_table_format(self):
        models = [
            ModelInfo("qwen3:8b", 8.2, "Q4_K_M", "qwen3", 5_200_000_000),
            ModelInfo("mistral:123b", 122.6, "Q4_K_M", "mistral", 73_000_000_000),
        ]
        table = list_models_table(models)
        assert "qwen3:8b" in table
        assert "mistral:123b" in table
        assert "Q4_K_M" in table
        assert "#" in table.splitlines()[0]

    def test_empty_list(self):
        table = list_models_table([])
        lines = table.strip().splitlines()
        assert len(lines) == 2  # header + separator


# ---------------------------------------------------------------------------
# discover_models (mocked)
# ---------------------------------------------------------------------------

class TestDiscoverModels:
    @patch("ollama_chain.models.ollama")
    def test_discovery(self, mock_ollama):
        mock_ollama.list.return_value = {
            "models": [
                {
                    "model": "qwen3:14b",
                    "size": 9_300_000_000,
                    "details": {
                        "parameter_size": "14.8B",
                        "quantization_level": "Q4_K_M",
                        "family": "qwen3",
                    },
                },
                {
                    "model": "qwen3:8b",
                    "size": 5_200_000_000,
                    "details": {
                        "parameter_size": "8.2B",
                        "quantization_level": "Q4_K_M",
                        "family": "qwen3",
                    },
                },
            ]
        }
        from ollama_chain.models import discover_models
        models = discover_models()
        assert len(models) == 2
        assert models[0].name == "qwen3:8b"  # sorted smallest first
        assert models[1].name == "qwen3:14b"

    @patch("ollama_chain.models.ollama")
    def test_empty_models(self, mock_ollama):
        mock_ollama.list.return_value = {"models": []}
        from ollama_chain.models import discover_models
        models = discover_models()
        assert models == []

    @patch("ollama_chain.models.ollama")
    def test_missing_details(self, mock_ollama):
        mock_ollama.list.return_value = {
            "models": [{"model": "test:1b", "size": 1000, "details": {}}]
        }
        from ollama_chain.models import discover_models
        models = discover_models()
        assert len(models) == 1
        assert models[0].quantization == "unknown"


# ---------------------------------------------------------------------------
# ensure_memory_available (mocked)
# ---------------------------------------------------------------------------

class TestEnsureMemoryAvailable:
    @patch("ollama_chain.common.unload_model")
    @patch("ollama_chain.models._get_memory_info", return_value=(100_000_000_000, 50_000_000_000))
    @patch("ollama_chain.models.ollama")
    def test_no_eviction_when_memory_ok(self, mock_ollama, _mem, mock_unload):
        mock_ollama.ps.return_value = {
            "models": [
                {"model": "stale:7b", "size": 5_000_000_000},
            ]
        }
        from ollama_chain.models import ensure_memory_available
        ensure_memory_available(["needed:14b"])
        mock_unload.assert_not_called()

    @patch("ollama_chain.common.unload_model")
    @patch("ollama_chain.models._get_memory_info", return_value=(100_000_000_000, 5_000_000_000))
    @patch("ollama_chain.models.ollama")
    def test_eviction_when_memory_low(self, mock_ollama, _mem, mock_unload):
        mock_ollama.ps.return_value = {
            "models": [
                {"model": "stale:7b", "size": 5_000_000_000},
            ]
        }
        from ollama_chain.models import ensure_memory_available
        ensure_memory_available(["needed:14b"])
        mock_unload.assert_called_once_with("stale:7b")

    @patch("ollama_chain.common.unload_model")
    @patch("ollama_chain.models._get_memory_info", return_value=(100_000_000_000, 5_000_000_000))
    @patch("ollama_chain.models.ollama")
    def test_no_eviction_of_needed_models(self, mock_ollama, _mem, mock_unload):
        mock_ollama.ps.return_value = {
            "models": [
                {"model": "needed:14b", "size": 9_000_000_000},
            ]
        }
        from ollama_chain.models import ensure_memory_available
        ensure_memory_available(["needed:14b"])
        mock_unload.assert_not_called()
