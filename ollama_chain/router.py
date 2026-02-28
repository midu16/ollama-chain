"""Query router for adaptive model selection and cascade optimization.

Classifies query complexity using heuristics or a fast LLM call, then
produces a RouteDecision that tells the chain layer which models to use,
what fallback order to prefer, and whether web search is worthwhile.
"""

import re
import sys
from dataclasses import dataclass, field

from .common import chat_with_retry


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

COMPLEXITY_SIMPLE = "simple"
COMPLEXITY_MODERATE = "moderate"
COMPLEXITY_COMPLEX = "complex"

STRATEGY_DIRECT_FAST = "direct_fast"
STRATEGY_DIRECT_STRONG = "direct_strong"
STRATEGY_SUBSET_CASCADE = "subset_cascade"
STRATEGY_FULL_CASCADE = "full_cascade"


@dataclass
class RouteDecision:
    """Output of the router — consumed by chains, agent, and planner."""

    models: list[str]
    complexity: str
    strategy: str
    fallback_model: str
    skip_search: bool
    confidence: float
    reasoning: str = ""
    time_sensitive: bool = False


# ---------------------------------------------------------------------------
# Complexity classification
# ---------------------------------------------------------------------------

_TECHNICAL_TERMS = frozenset({
    "algorithm", "complexity", "architecture", "protocol", "implementation",
    "optimization", "concurrent", "distributed", "cryptographic", "asymptotic",
    "theorem", "proof", "derive", "polynomial", "heuristic", "latency",
    "throughput", "consensus", "replication", "sharding", "serialization",
    "deadlock", "mutex", "semaphore", "pipeline", "microservice",
    "kubernetes", "docker", "kernel", "syscall", "interrupt", "scheduler",
    "garbage", "collector", "jit", "compiler", "ast", "parser", "lexer",
    "tcp", "udp", "tls", "ssl", "dns", "bgp", "ospf", "ipsec",
    "encryption", "decryption", "hash", "signature", "certificate",
    "cve", "vulnerability", "exploit", "mitigation", "firewall",
})

_SIMPLE_PREFIXES = (
    "what is", "what are", "who is", "when did", "where is",
    "define ", "what port", "what does", "how many",
)

_TIME_SENSITIVE_KEYWORDS = frozenset({
    "latest", "current", "newest", "most recent", "today",
    "this week", "this month", "this year", "right now",
    "up to date", "up-to-date", "as of", "2024", "2025", "2026",
    "release", "version", "update", "patch", "announcement",
    "just released", "recently", "new release",
})

_TIME_SENSITIVE_PATTERNS = (
    "what is the latest", "what is the current", "what is the newest",
    "what is the most recent", "what version", "which version",
    "what release", "most current", "latest version",
    "current version", "newest version", "latest release",
    "current release", "newest release",
)


def is_time_sensitive(query: str) -> bool:
    """Detect whether a query requires current/up-to-date information.

    Time-sensitive queries need fresh web/news search results and models
    must be strongly instructed to defer to search evidence over training
    data.
    """
    q = query.lower().strip()
    if any(q.startswith(p) or p in q for p in _TIME_SENSITIVE_PATTERNS):
        return True
    words = q.split()
    for kw in _TIME_SENSITIVE_KEYWORDS:
        if " " in kw:
            if kw in q:
                return True
        elif kw in words:
            return True
    return False


def classify_complexity_heuristic(query: str) -> tuple[str, float]:
    """Classify query complexity without an LLM call."""
    query_lower = query.lower().strip()
    words = query_lower.split()
    word_count = len(words)

    if word_count <= 6 and any(query_lower.startswith(p) for p in _SIMPLE_PREFIXES):
        return COMPLEXITY_SIMPLE, 0.80

    tech_count = sum(
        1 for w in words if w.strip(",.?!:;()") in _TECHNICAL_TERMS
    )
    multi_question = query.count("?") > 1
    has_conjunctions = any(c in query_lower for c in (
        " and also ", " additionally ", " furthermore ", " moreover ",
    ))

    score = 0.0
    score += min(word_count / 12, 2.5)
    score += tech_count * 0.6
    score += 1.0 if multi_question else 0.0
    score += 0.5 if has_conjunctions else 0.0

    if score <= 1.5:
        return COMPLEXITY_SIMPLE, 0.70
    if score <= 3.5:
        return COMPLEXITY_MODERATE, 0.60
    return COMPLEXITY_COMPLEX, 0.65


def classify_complexity_llm(
    query: str, fast_model: str,
) -> tuple[str, float]:
    """Classify query complexity using the fast LLM."""
    try:
        response = chat_with_retry(
            model=fast_model,
            messages=[{"role": "user", "content": (
                "/no_think\n"
                "Rate the complexity of answering this query.\n"
                "Reply with ONLY one word: simple, moderate, or complex.\n\n"
                f"Query: {query}"
            )}],
            retries=1,
        )
        raw = response["message"]["content"]
        if "<think>" in raw:
            end = raw.find("</think>")
            if end != -1:
                raw = raw[end + 8:]
        content = raw.strip().lower()

        for level in (COMPLEXITY_SIMPLE, COMPLEXITY_MODERATE, COMPLEXITY_COMPLEX):
            if level in content:
                return level, 0.85
        return COMPLEXITY_MODERATE, 0.50
    except Exception:
        return classify_complexity_heuristic(query)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_query(
    query: str,
    all_models: list[str],
    *,
    fast_model: str | None = None,
    use_llm: bool = True,
    web_search: bool = True,
) -> RouteDecision:
    """Determine optimal model selection and strategy for *query*."""
    if not all_models:
        raise ValueError("No models available for routing")

    fast = fast_model or all_models[0]
    strong = all_models[-1]
    n = len(all_models)
    ts = is_time_sensitive(query)

    if use_llm and n > 1:
        complexity, confidence = classify_complexity_llm(query, fast)
    else:
        complexity, confidence = classify_complexity_heuristic(query)

    # Time-sensitive queries are at least moderate complexity (need search)
    if ts and complexity == COMPLEXITY_SIMPLE:
        complexity = COMPLEXITY_MODERATE
        confidence = max(confidence - 0.1, 0.5)

    if n == 1:
        return RouteDecision(
            models=all_models,
            complexity=complexity,
            strategy=STRATEGY_DIRECT_FAST,
            fallback_model=all_models[0],
            skip_search=not web_search,
            confidence=confidence,
            reasoning="Single model available",
            time_sensitive=ts,
        )

    if complexity == COMPLEXITY_SIMPLE:
        return RouteDecision(
            models=[fast],
            complexity=COMPLEXITY_SIMPLE,
            strategy=STRATEGY_DIRECT_FAST,
            fallback_model=strong,
            skip_search=True,
            confidence=confidence,
            reasoning="Simple query — fast model sufficient",
            time_sensitive=ts,
        )

    if complexity == COMPLEXITY_MODERATE:
        subset = [fast, strong] if n > 2 else all_models
        return RouteDecision(
            models=subset,
            complexity=COMPLEXITY_MODERATE,
            strategy=STRATEGY_SUBSET_CASCADE,
            fallback_model=fast,
            skip_search=not web_search,
            confidence=confidence,
            reasoning="Moderate query — subset cascade (fast + strong)",
            time_sensitive=ts,
        )

    return RouteDecision(
        models=all_models,
        complexity=COMPLEXITY_COMPLEX,
        strategy=STRATEGY_FULL_CASCADE,
        fallback_model=fast,
        skip_search=not web_search,
        confidence=confidence,
        reasoning="Complex query — full cascade through all models",
        time_sensitive=ts,
    )


# ---------------------------------------------------------------------------
# Plan-level routing optimization
# ---------------------------------------------------------------------------

_DATA_GATHERING_TOOLS = frozenset({
    "shell", "read_file", "list_dir", "web_search", "web_search_news",
})

_REASONING_KEYWORDS = frozenset({
    "analyze", "synthesize", "summarize", "evaluate", "compare",
    "explain", "propose", "recommend", "conclude", "reason",
    "design", "architect", "optimize", "review", "assess",
})


def optimize_routing(
    plan: list[dict],
    all_models: list[str],
    complexity: str,
    *,
    failed_models: set[str] | None = None,
) -> list[dict]:
    """Assign preferred model lists to each plan step.

    Enriches each step with a ``preferred_models`` key containing an
    ordered list of models best suited for that step, considering:
      - Tool type (data-gathering vs reasoning)
      - Step description keywords
      - Overall query complexity
      - Previously failed models (deprioritised)

    Returns the plan with ``preferred_models`` added to each step.
    """
    if not all_models:
        return plan

    fast = all_models[0]
    strong = all_models[-1]
    failed = failed_models or set()

    for step in plan:
        if step.get("status") == "completed":
            continue

        tool = step.get("tool", "none")
        desc_lower = step.get("description", "").lower()

        if tool in _DATA_GATHERING_TOOLS and complexity != COMPLEXITY_COMPLEX:
            preferred = [fast]
        elif tool == "none":
            has_reasoning = any(kw in desc_lower for kw in _REASONING_KEYWORDS)
            if has_reasoning:
                preferred = [strong]
            elif complexity == COMPLEXITY_SIMPLE:
                preferred = [fast]
            else:
                preferred = [strong]
        elif tool == "python_eval":
            preferred = [fast]
        else:
            preferred = list(all_models)

        if failed:
            filtered = [m for m in preferred if m not in failed]
            if not filtered:
                filtered = [m for m in all_models if m not in failed]
            preferred = filtered or preferred

        step["preferred_models"] = preferred

    return plan


def identify_parallel_candidates(plan: list[dict]) -> list[list[int]]:
    """Identify groups of step IDs that can execute concurrently.

    Unlike ``detect_parallel_groups`` in the planner (which returns step
    dicts), this returns ID groups suitable for routing decisions — letting
    the router assign lighter models to parallel data-gathering batches.
    """
    completed_ids = {s["id"] for s in plan if s["status"] == "completed"}
    pending = [s for s in plan if s["status"] == "pending"]

    groups: list[list[int]] = []
    remaining = list(pending)

    while remaining:
        ready_ids: list[int] = []
        not_ready: list[dict] = []
        for step in remaining:
            deps = set(step.get("depends_on", []))
            if deps <= completed_ids:
                ready_ids.append(step["id"])
            else:
                not_ready.append(step)

        if not ready_ids:
            ready_ids = [remaining[0]["id"]]
            not_ready = remaining[1:]

        groups.append(ready_ids)
        completed_ids.update(ready_ids)
        remaining = not_ready

    return groups


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def build_fallback_chain(
    all_models: list[str], failed_model: str,
) -> list[str]:
    """Return models ordered for fallback, excluding *failed_model*.

    Prefers larger models first (they are more likely to handle edge cases).
    """
    remaining = [m for m in all_models if m != failed_model]
    return list(reversed(remaining)) if remaining else list(all_models)


def select_models_for_step(
    step: dict,
    all_models: list[str],
    query_complexity: str,
) -> list[str]:
    """Pick the right model subset for a specific plan step.

    Data-gathering steps (shell, read_file, …) can use the fast model
    for simple queries.  Reasoning / synthesis steps always prefer the
    strongest available model.
    """
    if not all_models:
        return []

    tool = step.get("tool", "none")

    if tool == "none":
        return [all_models[-1]]

    if tool in ("shell", "read_file", "list_dir", "web_search", "web_search_news"):
        if query_complexity == COMPLEXITY_SIMPLE:
            return [all_models[0]]
        return all_models

    return all_models
