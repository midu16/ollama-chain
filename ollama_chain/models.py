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


def _get(obj, key, default=None):
    """Access a field from either a dict or a pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def discover_models() -> list[ModelInfo]:
    """Query Ollama API and return sorted list of available models (smallest first)."""
    try:
        response = ollama.list()
    except Exception as e:
        print(f"Error: cannot reach Ollama at localhost:11434 — {e}", file=sys.stderr)
        sys.exit(1)

    raw_models = _get(response, "models", None) or []

    models = []
    for m in raw_models:
        details = _get(m, "details", {}) or {}
        name = _get(m, "model", None) or _get(m, "name", "unknown")
        models.append(ModelInfo(
            name=name,
            parameter_size=_parse_param_size(
                _get(details, "parameter_size", "0") or "0",
            ),
            quantization=_get(details, "quantization_level", "unknown") or "unknown",
            family=_get(details, "family", "unknown") or "unknown",
            size_bytes=_get(m, "size", 0) or 0,
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


def _get_memory_info() -> tuple[int, int]:
    """Return (total_bytes, available_bytes) from /proc/meminfo.

    Falls back to (0, 0) on non-Linux or if the file is unreadable.
    """
    total = available = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                if total and available:
                    break
    except Exception:
        pass
    return total, available


_LOW_MEMORY_RATIO = 0.15


def ensure_memory_available(needed_models: list[str]):
    """Check Ollama for stale loaded models and unload them if memory is tight.

    Queries ``ollama ps`` to find models still resident in memory from a
    previous run.  Models not required for this run are considered stale.
    If system memory is under pressure (available < 15 % of total), stale
    models are evicted so the upcoming run has room.

    When system memory info is unavailable (non-Linux), stale models that
    exceed 2 GB are evicted unconditionally as a safety fallback.
    """
    from .common import unload_model

    try:
        ps_result = ollama.ps()
    except Exception:
        return

    loaded = _get(ps_result, "models", []) or []
    if not loaded:
        return

    needed_set = set(needed_models)
    stale: list[str] = []
    stale_bytes = 0
    total_loaded_bytes = 0

    for m in loaded:
        name = _get(m, "model") or _get(m, "name", "")
        size = int(_get(m, "size", 0) or 0)
        total_loaded_bytes += size
        if name not in needed_set:
            stale.append(name)
            stale_bytes += size

    if not stale:
        return

    total_ram, available_ram = _get_memory_info()

    if total_ram > 0:
        if available_ram > total_ram * _LOW_MEMORY_RATIO:
            return
    else:
        if stale_bytes < 2 * 1024 ** 3:
            return

    stale_gb = stale_bytes / (1024 ** 3)
    loaded_gb = total_loaded_bytes / (1024 ** 3)
    if total_ram > 0:
        avail_gb = available_ram / (1024 ** 3)
        print(
            f"[memory] System RAM low ({avail_gb:.1f} GB free) — "
            f"{loaded_gb:.1f} GB held by Ollama, "
            f"{stale_gb:.1f} GB from {len(stale)} stale model(s)",
            file=sys.stderr,
        )
    else:
        print(
            f"[memory] {stale_gb:.1f} GB held by {len(stale)} stale model(s) "
            f"— unloading",
            file=sys.stderr,
        )

    for name in stale:
        unload_model(name)
        print(f"[memory] Unloaded {name}", file=sys.stderr)


def list_models_table(models: list[ModelInfo]) -> str:
    """Format models as a human-readable table with cascade order."""
    lines = [f"{'#':<4} {'MODEL':<30} {'PARAMS':<10} {'QUANT':<10} {'FAMILY':<12} {'SIZE':<10}"]
    lines.append("-" * 76)
    for i, m in enumerate(models, 1):
        size_gb = m.size_bytes / (1024 ** 3)
        params = f"{m.parameter_size:.1f}B"
        lines.append(
            f"{i:<4} {m.name:<30} {params:<10} {m.quantization:<10} "
            f"{m.family:<12} {size_gb:.1f} GB"
        )
    return "\n".join(lines)
