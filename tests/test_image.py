"""Tests for the image generation module."""

import base64
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ollama_chain.image import (
    _IMAGE_CAPABILITY,
    _PROMPT_ENHANCE_TEMPLATE,
    chain_image,
    discover_image_models,
    enhance_prompt,
    generate_image,
    image_model_names,
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscoverImageModels:
    @patch("ollama_chain.image.ollama_client")
    def test_finds_image_capable_models(self, mock_ollama):
        mock_ollama.list.return_value = {
            "models": [
                {"model": "x/flux2-klein:4b", "size": 5_700_000_000},
                {"model": "qwen3:8b", "size": 5_200_000_000},
            ],
        }
        mock_ollama.show.side_effect = lambda name: {
            "x/flux2-klein:4b": {
                "capabilities": ["image"],
                "details": {
                    "family": "Flux2KleinPipeline",
                    "parameter_size": "8.0B",
                    "quantization_level": "FP4",
                },
            },
            "qwen3:8b": {
                "capabilities": ["completion", "tools", "thinking"],
                "details": {
                    "family": "qwen3",
                    "parameter_size": "8.2B",
                    "quantization_level": "Q4_K_M",
                },
            },
        }[name]

        from ollama_chain.image import _discover_image_models
        models = _discover_image_models()
        assert len(models) == 1
        assert models[0]["name"] == "x/flux2-klein:4b"

    @patch("ollama_chain.image.ollama_client")
    def test_returns_empty_when_no_image_models(self, mock_ollama):
        mock_ollama.list.return_value = {"models": [
            {"model": "qwen3:8b", "size": 5_200_000_000},
        ]}
        mock_ollama.show.return_value = {
            "capabilities": ["completion", "tools"],
            "details": {"family": "qwen3"},
        }

        from ollama_chain.image import _discover_image_models
        models = _discover_image_models()
        assert models == []

    @patch("ollama_chain.image.ollama_client")
    def test_handles_ollama_connection_error(self, mock_ollama):
        mock_ollama.list.side_effect = ConnectionError("offline")

        from ollama_chain.image import _discover_image_models
        models = _discover_image_models()
        assert models == []

    @patch("ollama_chain.image._discover_image_models")
    def test_image_model_names(self, mock_discover):
        mock_discover.return_value = [
            {"name": "x/flux2-klein:4b"},
            {"name": "x/z-image-turbo:latest"},
        ]
        # Bypass cache by calling _discover directly through image_model_names
        with patch("ollama_chain.image.discover_image_models", return_value=mock_discover.return_value):
            names = image_model_names()
        assert names == ["x/flux2-klein:4b", "x/z-image-turbo:latest"]


# ---------------------------------------------------------------------------
# Prompt enhancement
# ---------------------------------------------------------------------------


class TestEnhancePrompt:
    @patch("ollama_chain.image.ask")
    def test_enhances_short_prompt(self, mock_ask):
        mock_ask.return_value = (
            "A fluffy orange tabby cat wearing a miniature astronaut suit, "
            "standing on the lunar surface with Earth visible in the background, "
            "dramatic rim lighting, photorealistic, shot on Hasselblad"
        )
        result = enhance_prompt("cat on the moon", "qwen3:8b")
        assert len(result) > len("cat on the moon")
        assert "cat" in result.lower()
        mock_ask.assert_called_once()
        call_prompt = mock_ask.call_args[0][0]
        assert "cat on the moon" in call_prompt

    @patch("ollama_chain.image.ask")
    def test_falls_back_on_error(self, mock_ask):
        mock_ask.side_effect = RuntimeError("model error")
        result = enhance_prompt("a red ball", "qwen3:8b")
        assert result == "a red ball"

    @patch("ollama_chain.image.ask")
    def test_falls_back_when_enhancement_shorter(self, mock_ask):
        mock_ask.return_value = "hi"
        result = enhance_prompt("a very detailed description of a landscape", "qwen3:8b")
        assert result == "a very detailed description of a landscape"

    def test_template_contains_key_instructions(self):
        assert "diffusion" in _PROMPT_ENHANCE_TEMPLATE.lower()
        assert "lighting" in _PROMPT_ENHANCE_TEMPLATE.lower()
        assert "{description}" in _PROMPT_ENHANCE_TEMPLATE


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


class TestGenerateImage:
    @patch("ollama_chain.image.ollama_client")
    def test_saves_image_to_disk(self, mock_ollama):
        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        fake_b64 = base64.b64encode(fake_png).decode()
        mock_ollama.generate.return_value = {
            "images": [fake_b64],
            "done": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_image(
                "a red circle", "x/flux2-klein:4b",
                output_dir=tmpdir,
            )
            assert len(paths) == 1
            assert os.path.exists(paths[0])
            assert paths[0].endswith(".png")
            with open(paths[0], "rb") as f:
                content = f.read()
            assert content == fake_png

    @patch("ollama_chain.image.ollama_client")
    def test_saves_multiple_images(self, mock_ollama):
        fake_png = b"\x89PNG" + b"\x00" * 50
        fake_b64 = base64.b64encode(fake_png).decode()
        mock_ollama.generate.return_value = {
            "images": [fake_b64, fake_b64],
            "done": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_image(
                "two cats", "x/flux2-klein:4b",
                output_dir=tmpdir,
            )
            assert len(paths) == 2
            assert all(os.path.exists(p) for p in paths)

    @patch("ollama_chain.image.ollama_client")
    def test_raises_on_no_images(self, mock_ollama):
        mock_ollama.generate.return_value = {"images": [], "done": True}

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="did not return any images"):
                generate_image(
                    "test", "x/flux2-klein:4b",
                    output_dir=tmpdir,
                )

    @patch("ollama_chain.image.ollama_client")
    def test_raises_clear_error_on_mlx_failure(self, mock_ollama):
        mock_ollama.generate.side_effect = RuntimeError(
            "mlx runner failed: Failed to load libmlxc library"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="macOS-only"):
                generate_image(
                    "test", "x/flux2-klein:4b",
                    output_dir=tmpdir,
                )

    @patch("ollama_chain.image.ollama_client")
    def test_handles_pydantic_response(self, mock_ollama):
        """Handles both dict and pydantic-model responses."""
        fake_png = b"\x89PNG" + b"\x00" * 30
        fake_b64 = base64.b64encode(fake_png).decode()

        response_obj = MagicMock()
        response_obj.__contains__ = lambda self, key: False
        response_obj.__getitem__ = MagicMock(side_effect=KeyError)
        type(response_obj).images = [fake_b64]
        mock_ollama.generate.return_value = response_obj

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_image(
                "test", "x/flux2-klein:4b",
                output_dir=tmpdir,
            )
            assert len(paths) == 1


# ---------------------------------------------------------------------------
# Chain entry point
# ---------------------------------------------------------------------------


class TestChainImage:
    @patch("ollama_chain.image.generate_image")
    @patch("ollama_chain.image.enhance_prompt")
    @patch("ollama_chain.image.discover_image_models")
    def test_full_pipeline(self, mock_discover, mock_enhance, mock_gen):
        mock_discover.return_value = [
            {"name": "x/flux2-klein:4b", "size_bytes": 5_700_000_000},
        ]
        mock_enhance.return_value = "Enhanced: a detailed cat"
        mock_gen.return_value = ["/tmp/image_test.png"]

        result = chain_image(
            "a cat", ["qwen3:8b"],
            output_dir="/tmp",
        )
        assert "Generated" in result
        assert "/tmp/image_test.png" in result
        assert "Enhanced: a detailed cat" in result
        mock_enhance.assert_called_once_with("a cat", "qwen3:8b")
        mock_gen.assert_called_once()

    @patch("ollama_chain.image.discover_image_models")
    def test_raises_when_no_image_models(self, mock_discover):
        mock_discover.return_value = []
        with pytest.raises(RuntimeError, match="No image generation models"):
            chain_image("a cat", ["qwen3:8b"])

    @patch("ollama_chain.image.generate_image")
    @patch("ollama_chain.image.enhance_prompt")
    @patch("ollama_chain.image.discover_image_models")
    def test_handles_generation_failure_gracefully(
        self, mock_discover, mock_enhance, mock_gen,
    ):
        mock_discover.return_value = [
            {"name": "x/flux2-klein:4b", "size_bytes": 5_700_000_000},
        ]
        mock_enhance.return_value = "enhanced prompt"
        mock_gen.side_effect = RuntimeError("macOS-only")

        result = chain_image("a cat", ["qwen3:8b"])
        assert "failed" in result.lower()

    @patch("ollama_chain.image.generate_image")
    @patch("ollama_chain.image.enhance_prompt")
    @patch("ollama_chain.image.discover_image_models")
    def test_generates_with_multiple_models(
        self, mock_discover, mock_enhance, mock_gen,
    ):
        mock_discover.return_value = [
            {"name": "x/flux2-klein:4b", "size_bytes": 5_700_000_000},
            {"name": "x/z-image-turbo:latest", "size_bytes": 12_000_000_000},
        ]
        mock_enhance.return_value = "enhanced"
        mock_gen.side_effect = [
            ["/tmp/img1.png"],
            ["/tmp/img2.png"],
        ]

        result = chain_image("a cat", ["qwen3:8b"])
        assert "2 image(s)" in result
        assert "2 model(s)" in result
        assert mock_gen.call_count == 2


# ---------------------------------------------------------------------------
# Model filtering (models.py integration)
# ---------------------------------------------------------------------------


class TestImageModelFiltering:
    @patch("ollama_chain.models.ollama")
    def test_image_models_excluded_from_text_cascade(self, mock_ollama):
        mock_ollama.list.return_value = {
            "models": [
                {
                    "model": "qwen3:8b",
                    "size": 5_200_000_000,
                    "details": {
                        "parameter_size": "8.2B",
                        "quantization_level": "Q4_K_M",
                        "family": "qwen3",
                    },
                },
                {
                    "model": "x/flux2-klein:4b",
                    "size": 5_700_000_000,
                    "details": {
                        "parameter_size": "8.0B",
                        "quantization_level": "FP4",
                        "family": "",
                    },
                },
            ],
        }
        mock_ollama.show.side_effect = lambda name: {
            "x/flux2-klein:4b": {"capabilities": ["image"]},
            "qwen3:8b": {"capabilities": ["completion", "tools", "thinking"]},
        }[name]

        from ollama_chain.models import discover_models, model_names, _model_cache
        import ollama_chain.models as models_mod
        models_mod._model_cache = None

        result = discover_models(_force=True)
        names = model_names(result)
        assert "qwen3:8b" in names
        assert "x/flux2-klein:4b" not in names

    def test_image_capability_constant(self):
        assert _IMAGE_CAPABILITY == "image"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestImageCLI:
    def test_image_mode_in_cli_choices(self):
        from ollama_chain.cli import main
        import argparse

        with patch("sys.argv", ["ollama-chain", "-m", "image", "--help"]):
            with pytest.raises(SystemExit):
                main()

    def test_cli_only_modes_includes_image(self):
        from ollama_chain.chains import CLI_ONLY_MODES
        assert "image" in CLI_ONLY_MODES
