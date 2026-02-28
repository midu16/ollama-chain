"""Image generation mode — text-to-image via Ollama diffusion models.

Discovers locally installed models with the ``image`` capability,
uses a text LLM to enhance the user's description into a detailed
diffusion-optimised prompt, then generates the image and saves it
to disk.
"""

import base64
import os
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import ollama as ollama_client

from .common import ask
from .models import _get
from .progress import progress_update


_IMAGE_CAPABILITY = "image"

_PROMPT_ENHANCE_TEMPLATE = (
    "/no_think\n"
    "You are an expert prompt engineer for text-to-image diffusion models. "
    "Rewrite the user's description into a highly detailed image generation prompt. "
    "Add photographic details: lighting, composition, camera angle, style, colors, "
    "textures, mood, and atmosphere. Keep quoted text exactly as-is (for text rendering). "
    "Output ONLY the enhanced prompt, nothing else.\n\n"
    "User description: {description}"
)

_DEFAULT_WIDTH = 1024
_DEFAULT_HEIGHT = 1024


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _discover_image_models_cached(cache_key: float) -> list[dict]:
    """Internal cached discovery keyed by a TTL-based key."""
    return _discover_image_models()


def _discover_image_models() -> list[dict]:
    """Query Ollama for all installed models with the ``image`` capability."""
    try:
        response = ollama_client.list()
    except Exception as e:
        print(f"[image] Cannot reach Ollama: {e}", file=sys.stderr)
        return []

    raw_models = _get(response, "models", None) or []
    image_models = []

    for m in raw_models:
        name = _get(m, "model", None) or _get(m, "name", "")
        if not name:
            continue
        try:
            info = ollama_client.show(name)
            caps = _get(info, "capabilities", None) or []
            if _IMAGE_CAPABILITY in caps:
                details = _get(info, "details", {}) or {}
                image_models.append({
                    "name": name,
                    "family": _get(details, "family", "unknown") or "unknown",
                    "parameter_size": _get(details, "parameter_size", "") or "",
                    "quantization": _get(details, "quantization_level", "") or "",
                    "size_bytes": _get(m, "size", 0) or 0,
                })
        except Exception:
            continue

    image_models.sort(key=lambda m: m.get("size_bytes", 0))
    return image_models


def discover_image_models() -> list[dict]:
    """Return cached list of image-capable models (30s TTL)."""
    cache_key = int(time.monotonic() / 30)
    return _discover_image_models_cached(cache_key)


def image_model_names() -> list[str]:
    """Return names of all installed image-capable models."""
    return [m["name"] for m in discover_image_models()]


# ---------------------------------------------------------------------------
# Prompt enhancement
# ---------------------------------------------------------------------------

def enhance_prompt(
    description: str,
    text_model: str,
) -> str:
    """Use a text LLM to expand a short description into a detailed prompt.

    The enhanced prompt includes photographic details (lighting, style,
    composition) that produce better results from diffusion models.
    """
    try:
        enhanced = ask(
            _PROMPT_ENHANCE_TEMPLATE.format(description=description),
            model=text_model,
        )
        if enhanced and len(enhanced.strip()) > len(description):
            return enhanced.strip()
    except Exception as e:
        print(
            f"[image] Prompt enhancement failed ({e}), using original",
            file=sys.stderr,
        )
    return description


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    model: str,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    output_dir: str | None = None,
) -> list[str]:
    """Generate an image using an Ollama image-capable model.

    Returns a list of file paths to the saved PNG images.
    """
    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.replace("/", "_").replace(":", "_")

    try:
        response = ollama_client.generate(
            model=model,
            prompt=prompt,
            stream=False,
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "mlx" in err_msg or "libmlxc" in err_msg or "libcuda" in err_msg:
            raise RuntimeError(
                "Image generation requires Ollama's MLX runner which is "
                "currently macOS-only. Linux and Windows support is coming "
                "in a future Ollama release."
            ) from e
        raise

    if isinstance(response, dict):
        images_data = response.get("images") or []
    else:
        images_data = getattr(response, "images", None) or []
    if not images_data:
        raise RuntimeError(
            f"Model {model} did not return any images. "
            f"Ensure it has the 'image' capability."
        )

    saved_paths: list[str] = []
    for i, img_b64 in enumerate(images_data):
        suffix = f"_{i}" if len(images_data) > 1 else ""
        filename = f"image_{timestamp}_{model_short}{suffix}.png"
        filepath = out_dir / filename

        img_bytes = base64.b64decode(img_b64)
        filepath.write_bytes(img_bytes)
        saved_paths.append(str(filepath))

    return saved_paths


# ---------------------------------------------------------------------------
# Chain entry point
# ---------------------------------------------------------------------------

def chain_image(
    query: str,
    all_models: list[str],
    *,
    web_search: bool = False,
    fast: str | None = None,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    output_dir: str | None = None,
) -> str:
    """Image generation chain mode.

    1. Discovers installed models with the ``image`` capability
    2. Uses the fastest text LLM to enhance the user's description
    3. Generates images with each available image model
    4. Saves results as PNG files

    The *all_models* parameter is the standard text LLM list (used for
    prompt enhancement only). Image models are discovered separately
    via ``ollama.show()`` capability detection.
    """
    img_models = discover_image_models()
    if not img_models:
        raise RuntimeError(
            "No image generation models found. Pull one first:\n"
            "  ollama pull x/flux2-klein:4b\n"
            "  ollama pull x/z-image-turbo"
        )

    img_names = [m["name"] for m in img_models]
    fast_text = fast or all_models[0] if all_models else None

    progress_update(5, "Discovering image models...")
    print(
        f"[image] Found {len(img_models)} image model(s): "
        + ", ".join(img_names),
        file=sys.stderr,
    )

    # --- Enhance prompt with text LLM ---
    enhanced = query
    if fast_text:
        progress_update(10, f"Enhancing prompt with {fast_text}...")
        print(
            f"[image] Enhancing prompt with {fast_text}...",
            file=sys.stderr,
        )
        enhanced = enhance_prompt(query, fast_text)
        if enhanced != query:
            print(f"[image] Enhanced prompt: {enhanced[:120]}...", file=sys.stderr)

    # --- Generate with each image model ---
    all_paths: list[str] = []
    step_pct = 80.0 / max(len(img_models), 1)

    for i, img_model in enumerate(img_names):
        pct = 15 + step_pct * i
        progress_update(pct, f"Generating with {img_model}...")
        print(
            f"[image {i + 1}/{len(img_names)}] Generating with {img_model}...",
            file=sys.stderr,
        )

        try:
            paths = generate_image(
                enhanced,
                img_model,
                width=width,
                height=height,
                output_dir=output_dir,
            )
            all_paths.extend(paths)
            for p in paths:
                print(f"[image] Saved: {p}", file=sys.stderr)
        except RuntimeError as e:
            print(f"[image] {img_model} failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[image] {img_model} error: {e}", file=sys.stderr)

    if not all_paths:
        return (
            "Image generation failed for all models. This may be because:\n"
            "- Image generation currently requires macOS (MLX runner)\n"
            "- Linux and Windows support is coming in a future Ollama release\n\n"
            f"Models attempted: {', '.join(img_names)}\n"
            f"Prompt used: {enhanced}"
        )

    # --- Format output ---
    result_parts = [
        f"## Generated Image{'s' if len(all_paths) > 1 else ''}",
        "",
        f"**Prompt**: {query}",
    ]
    if enhanced != query:
        result_parts.append(f"**Enhanced prompt**: {enhanced}")
    result_parts.append("")

    for path in all_paths:
        filename = os.path.basename(path)
        result_parts.append(f"- `{path}`")

    result_parts.extend([
        "",
        f"Generated {len(all_paths)} image(s) using "
        f"{len(img_models)} model(s).",
    ])

    return "\n".join(result_parts)
