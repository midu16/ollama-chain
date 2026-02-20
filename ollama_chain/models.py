"""Auto-discover and manage locally available Ollama models."""

import sys
from dataclasses import dataclass

import ollama


@dataclass
class ModelInfo:
    name: str
    parameter_size: float  # billions
    quantization: str
    family: str
    size_bytes: int


def _parse_param_size(raw: str) -> float:
    """Convert '14.8B' or '32.8B' to a float in billions."""
    raw = raw.strip().upper()
    multiplier = {"B": 1, "M": 0.001, "K": 0.000001}
    for suffix, mult in multiplier.items():
        if raw.endswith(suffix):
            try:
                return float(raw[:-1]) * mult
            except ValueError:
                return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def discover_models() -> list[ModelInfo]:
    """Query Ollama API and return sorted list of available models (smallest first)."""
    try:
        response = ollama.list()
    except Exception as e:
        print(f"Error: cannot reach Ollama at localhost:11434 — {e}", file=sys.stderr)
        sys.exit(1)

    models = []
    for m in response.get("models", []):
        details = m.get("details", {})
        models.append(ModelInfo(
            name=m["name"],
            parameter_size=_parse_param_size(details.get("parameter_size", "0")),
            quantization=details.get("quantization_level", "unknown"),
            family=details.get("family", "unknown"),
            size_bytes=m.get("size", 0),
        ))

    models.sort(key=lambda m: m.parameter_size)
    return models


def model_names(models: list[ModelInfo]) -> list[str]:
    """Return ordered list of model names (smallest to largest)."""
    if not models:
        print("Error: no models found in Ollama. Pull one first: ollama pull qwen3:14b", file=sys.stderr)
        sys.exit(1)
    return [m.name for m in models]


def pick_models(models: list[ModelInfo]) -> tuple[ModelInfo, ModelInfo]:
    """Pick fastest (smallest) and strongest (largest) model from available set."""
    if not models:
        print("Error: no models found in Ollama. Pull one first: ollama pull qwen3:14b", file=sys.stderr)
        sys.exit(1)
    if len(models) == 1:
        return models[0], models[0]
    return models[0], models[-1]


def list_models_table(models: list[ModelInfo]) -> str:
    """Format models as a human-readable table with cascade order."""
    lines = [f"{'#':<4} {'MODEL':<25} {'PARAMS':<10} {'QUANT':<10} {'FAMILY':<10} {'SIZE':<10}"]
    lines.append("-" * 69)
    for i, m in enumerate(models, 1):
        size_gb = m.size_bytes / (1024 ** 3)
        lines.append(
            f"{i:<4} {m.name:<25} {m.parameter_size:<10.1f}B {m.quantization:<10} "
            f"{m.family:<10} {size_gb:.1f} GB"
        )
    return "\n".join(lines)
