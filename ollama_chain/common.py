"""Shared utilities used by both chains and the agent."""

import sys
import time

import ollama as ollama_client

SOURCE_GUIDANCE = (
    "When making factual claims, reference authoritative public sources such as: "
    "official documentation (e.g. Red Hat, kernel.org, Mozilla MDN, Microsoft Docs), "
    "standards bodies (IEEE, IETF RFCs, ISO, W3C, NIST), "
    "peer-reviewed publications, and official project repositories. "
    "Format references as: [Source: <name>, <url or identifier>]. "
    "If you are uncertain about a claim and cannot back it with a source, say so explicitly."
)

_RETRYABLE_FRAGMENTS = ("EOF", "eof", "connection", "timeout", "reset", "broken pipe")
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds; doubles each attempt


# ---------------------------------------------------------------------------
# Ollama call helpers
# ---------------------------------------------------------------------------

def chat_with_retry(
    model: str,
    messages: list[dict],
    *,
    retries: int = MAX_RETRIES,
    keep_alive: str = "5m",
) -> dict:
    """Call ollama.chat with automatic retry on transient errors."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return ollama_client.chat(
                model=model,
                messages=messages,
                keep_alive=keep_alive,
            )
        except Exception as e:
            last_err = e
            err_text = str(e).lower()
            retryable = any(frag in err_text for frag in _RETRYABLE_FRAGMENTS)
            if not retryable or attempt == retries:
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"[ollama] Transient error (attempt {attempt}/{retries}): {e} "
                f"— retrying in {delay}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise last_err  # unreachable, keeps type checkers happy


def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages that share the same role.

    Ollama (and many chat-format models) expects strictly alternating
    user/assistant turns.  Adjacent entries with the same role are
    concatenated.
    """
    if not messages:
        return messages
    merged: list[dict] = [messages[0].copy()]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg.copy())
    return merged


def unload_model(model: str):
    """Tell Ollama to unload a model from memory immediately."""
    try:
        ollama_client.generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass


def unload_all_models(models: list[str]):
    """Unload all specified models from Ollama to free system memory."""
    seen: set[str] = set()
    for model in models:
        if model not in seen:
            seen.add(model)
            unload_model(model)


def ask(prompt: str, model: str, thinking: bool = False) -> str:
    """Send a prompt to an Ollama model and return the response text."""
    if not thinking:
        prompt = "/no_think\n" + prompt
    response = chat_with_retry(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response["message"]["content"]
    if "<think>" in content:
        end = content.find("</think>")
        if end != -1:
            content = content[end + len("</think>"):].strip()
    return content


# ---------------------------------------------------------------------------
# Prompt formatting helpers (Gap 6)
# ---------------------------------------------------------------------------

def format_prompt_section(title: str, body: str) -> str:
    """Format a named section for inclusion in an LLM prompt."""
    return f"=== {title.upper()} ===\n{body}"


def build_structured_prompt(
    sections: list[tuple[str, str]],
    instructions: str = "",
) -> str:
    """Build a multi-section prompt from (title, body) pairs.

    Produces a cleanly-formatted prompt with clear section boundaries
    that helps the LLM parse distinct information blocks.
    """
    parts: list[str] = []
    for title, body in sections:
        if body and body.strip():
            parts.append(format_prompt_section(title, body))
    if instructions:
        parts.append(f"\n{instructions}")
    return "\n\n".join(parts)
