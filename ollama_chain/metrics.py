"""Metrics for evaluating prompt quality before sending to LLM models.

Provides a suite of heuristic-based metrics that score prompts on
multiple dimensions — clarity, specificity, structure, actionability,
and context sufficiency — and produces improvement suggestions.

These metrics run locally without any LLM calls, making them fast
enough to gate every prompt in the chain pipeline.
"""

import math
import re
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Scoring result types
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    """A single metric score with explanation."""
    name: str
    score: float          # 0.0 – 1.0
    weight: float         # contribution to overall score
    explanation: str = ""


@dataclass
class PromptMetrics:
    """Aggregated quality evaluation of a prompt."""
    prompt: str
    scores: list[MetricScore] = field(default_factory=list)
    overall_score: float = 0.0
    grade: str = ""       # A / B / C / D / F
    suggestions: list[str] = field(default_factory=list)
    word_count: int = 0
    evaluation_ms: float = 0.0

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [
            f"Prompt Quality: {self.grade} ({self.overall_score:.0f}/100)  "
            f"[{self.word_count} words, evaluated in {self.evaluation_ms:.1f}ms]",
        ]
        for s in self.scores:
            bar_len = int(s.score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            pct = s.score * 100
            lines.append(f"  {s.name:<22} {bar} {pct:5.1f}%  {s.explanation}")
        if self.suggestions:
            lines.append("\n  Suggestions:")
            for i, sug in enumerate(self.suggestions, 1):
                lines.append(f"    {i}. {sug}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual metric evaluators
# ---------------------------------------------------------------------------

_QUESTION_WORDS = frozenset({
    "what", "how", "why", "when", "where", "who", "which",
    "explain", "describe", "compare", "analyze", "list",
    "define", "summarize", "evaluate", "discuss",
})

_VAGUE_WORDS = frozenset({
    "stuff", "things", "something", "anything", "whatever",
    "somehow", "kind of", "sort of", "maybe", "probably",
    "a lot", "very", "really", "basically", "just",
})

_TECHNICAL_TERMS = frozenset({
    "algorithm", "complexity", "architecture", "protocol", "implementation",
    "optimization", "concurrent", "distributed", "cryptographic", "asymptotic",
    "theorem", "proof", "derive", "polynomial", "heuristic", "latency",
    "throughput", "consensus", "replication", "sharding", "serialization",
    "deadlock", "mutex", "semaphore", "pipeline", "microservice",
    "kubernetes", "docker", "kernel", "syscall", "interrupt", "scheduler",
    "tcp", "udp", "tls", "ssl", "dns", "bgp", "ospf", "ipsec",
    "encryption", "decryption", "hash", "signature", "certificate",
    "api", "database", "sql", "nosql", "rest", "graphql", "grpc",
    "regression", "classification", "neural", "transformer", "embedding",
    "container", "orchestration", "deployment", "ci/cd", "monitoring",
    "authentication", "authorization", "oauth", "jwt", "rbac",
})

_STRUCTURAL_MARKERS = (
    r"^\d+[\.\)]\s",        # numbered list
    r"^[-*]\s",             # bullet points
    r"^#{1,6}\s",           # markdown headers
    r"\n\n",                # paragraph breaks
    r"```",                 # code blocks
    r":\s*\n",              # key-value style
)

_CONSTRAINT_PATTERNS = (
    r"\b(must|should|shall|require|need to|has to|ensure)\b",
    r"\b(at least|at most|no more than|no less than|exactly|within)\b",
    r"\b(maximum|minimum|limit|constraint|restrict|bound)\b",
    r"\b(format|output|return|respond|reply)\b.{0,20}\b(as|in|with|using)\b",
    r"\b(step[- ]by[- ]step|one by one|sequentially|in order)\b",
    r"\b(include|exclude|avoid|do not|don't|never|always)\b",
)

_DELIMITER_PATTERNS = (
    r"===",                   # section delimiters
    r"---",                   # horizontal rules
    r"```",                   # code fences
    r"<\w+>.*?</\w+>",       # XML-style tags
    r"\|.*\|.*\|",           # table-style pipes
    r"^\s*#{1,6}\s",         # markdown headers as section delimiters
    r'"""',                   # triple-quote blocks
)

_COT_PATTERNS = (
    r"\b(step[- ]by[- ]step|think.*through|reason.*through|show.*reasoning)\b",
    r"\b(let'?s think|walk.*through|break.*down|work.*through)\b",
    r"\b(first.*then.*finally|explain.*reasoning|show.*work)\b",
    r"\b(why.*because|derive|prove|logical|implication)\b",
)

_FEW_SHOT_PATTERNS = (
    r"\b(example|e\.g\.|for instance|such as|sample|demo)\b",
    r"(input|output)\s*:",
    r"(q|question|query)\s*:.*\n.*(a|answer|response)\s*:",
    r"```\w*\n.*```",         # code blocks act as examples
)

_DECOMPOSITION_PATTERNS = (
    r"^\s*\d+[\.\)]\s",       # numbered steps
    r"\b(first|second|third|then|next|finally|lastly)\b",
    r"\b(step\s*\d|phase\s*\d|part\s*\d)\b",
    r"\b(break.*down|decompose|sub-?tasks?|subtasks?)\b",
    r"\b(and also|additionally|furthermore|moreover)\b",
)


def _score_clarity(prompt: str, words: list[str]) -> MetricScore:
    """Evaluate how clear and unambiguous the prompt is."""
    score = 0.5
    word_count = len(words)
    prompt_lower = prompt.lower()

    if word_count < 3:
        score = 0.15
        explanation = "Too short to convey clear intent"
    elif word_count > 500:
        score = max(0.3, score - 0.2)
        explanation = "Very long — may dilute focus"
    else:
        explanation = "Reasonable length"
        if 10 <= word_count <= 200:
            score += 0.15

    has_question = "?" in prompt
    first_word = words[0].lower().rstrip(",:") if words else ""
    has_directive = first_word in _QUESTION_WORDS
    if has_question or has_directive:
        score += 0.15
        explanation = "Clear question or directive"

    vague_count = sum(1 for w in words if w.lower().strip(",.?!") in _VAGUE_WORDS)
    vague_ratio = vague_count / max(word_count, 1)
    if vague_ratio > 0.15:
        score -= 0.2
        explanation = "High proportion of vague/filler words"
    elif vague_count == 0 and word_count >= 5:
        score += 0.1

    sentences = re.split(r'[.!?]+', prompt)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        score += 0.05

    return MetricScore(
        name="Clarity",
        score=max(0.0, min(1.0, score)),
        weight=0.20,
        explanation=explanation,
    )


def _score_specificity(prompt: str, words: list[str]) -> MetricScore:
    """Evaluate how specific and detailed the prompt is."""
    score = 0.4
    word_count = len(words)
    prompt_lower = prompt.lower()

    tech_count = sum(
        1 for w in words if w.lower().strip(",.?!:;()") in _TECHNICAL_TERMS
    )
    tech_ratio = tech_count / max(word_count, 1)
    if tech_ratio > 0.1:
        score += 0.25
        explanation = f"Good technical depth ({tech_count} domain terms)"
    elif tech_count > 0:
        score += 0.1
        explanation = f"Some technical terms ({tech_count})"
    else:
        explanation = "No domain-specific terminology detected"

    has_numbers = bool(re.search(r'\b\d+\b', prompt))
    has_names = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', prompt))
    has_versions = bool(re.search(r'\b\d+\.\d+', prompt))
    has_urls = bool(re.search(r'https?://', prompt))
    specifics = sum([has_numbers, has_names, has_versions, has_urls])
    score += specifics * 0.08

    if word_count >= 15:
        score += 0.1

    return MetricScore(
        name="Specificity",
        score=max(0.0, min(1.0, score)),
        weight=0.18,
        explanation=explanation,
    )


def _score_structure(prompt: str, words: list[str]) -> MetricScore:
    """Evaluate prompt organization and formatting."""
    score = 0.4
    markers_found = 0

    for pattern in _STRUCTURAL_MARKERS:
        if re.search(pattern, prompt, re.MULTILINE):
            markers_found += 1

    if markers_found >= 3:
        score = 0.95
        explanation = "Well-structured with multiple formatting elements"
    elif markers_found >= 2:
        score = 0.8
        explanation = "Good structure with formatting"
    elif markers_found >= 1:
        score = 0.6
        explanation = "Some structural formatting"
    else:
        line_count = prompt.count("\n") + 1
        if line_count == 1 and len(words) > 30:
            score = 0.25
            explanation = "Long single block — consider breaking into sections"
        elif line_count > 1:
            score = 0.5
            explanation = "Multi-line but minimal formatting"
        else:
            explanation = "Single-line prompt (fine for simple queries)"

    return MetricScore(
        name="Structure",
        score=max(0.0, min(1.0, score)),
        weight=0.12,
        explanation=explanation,
    )


def _score_actionability(prompt: str, words: list[str]) -> MetricScore:
    """Evaluate whether the prompt clearly requests a specific action/output."""
    score = 0.3
    prompt_lower = prompt.lower()

    constraint_count = 0
    for pattern in _CONSTRAINT_PATTERNS:
        matches = re.findall(pattern, prompt_lower)
        constraint_count += len(matches)

    if constraint_count >= 4:
        score = 0.95
        explanation = f"Highly actionable with {constraint_count} constraints/directives"
    elif constraint_count >= 2:
        score = 0.75
        explanation = f"Good actionability ({constraint_count} directives)"
    elif constraint_count >= 1:
        score = 0.55
        explanation = "Some direction provided"
    else:
        has_question = "?" in prompt
        first_word = words[0].lower().rstrip(",:") if words else ""
        if has_question or first_word in _QUESTION_WORDS:
            score = 0.5
            explanation = "Has a question but lacks output constraints"
        else:
            explanation = "No clear action or output format requested"

    return MetricScore(
        name="Actionability",
        score=max(0.0, min(1.0, score)),
        weight=0.15,
        explanation=explanation,
    )


def _score_context_sufficiency(prompt: str, words: list[str]) -> MetricScore:
    """Evaluate whether the prompt provides enough context for a good answer."""
    score = 0.4
    word_count = len(words)
    prompt_lower = prompt.lower()

    context_signals = 0

    if re.search(r'\b(background|context|given|assuming|suppose|consider)\b', prompt_lower):
        context_signals += 1
    if re.search(r'\b(for example|e\.g\.|i\.e\.|such as|like)\b', prompt_lower):
        context_signals += 1
    if re.search(r'\b(because|since|due to|reason|purpose|goal)\b', prompt_lower):
        context_signals += 1
    if re.search(r'\b(versus|vs\.?|compared to|difference between|trade-?offs?)\b', prompt_lower):
        context_signals += 1
    if re.search(r'\b(use case|scenario|situation|environment|production|development)\b', prompt_lower):
        context_signals += 1

    if context_signals >= 3:
        score = 0.9
        explanation = "Rich context with background, examples, and scope"
    elif context_signals >= 2:
        score = 0.7
        explanation = "Good context provided"
    elif context_signals >= 1:
        score = 0.55
        explanation = "Some context present"
    else:
        if word_count >= 20:
            score = 0.4
            explanation = "Lengthy but lacks explicit context framing"
        elif word_count >= 8:
            score = 0.35
            explanation = "Minimal context — consider adding background or constraints"
        else:
            score = 0.2
            explanation = "Very little context for the model to work with"

    return MetricScore(
        name="Context Sufficiency",
        score=max(0.0, min(1.0, score)),
        weight=0.15,
        explanation=explanation,
    )


def _score_delimiter_usage(prompt: str, words: list[str]) -> MetricScore:
    """Evaluate use of delimiters to separate prompt sections.

    Per the Prompt Engineering Guide, delimiters (===, ---, ```, XML tags)
    help the model distinguish between instructions, context, and data.
    """
    score = 0.35
    word_count = len(words)
    delimiter_count = 0

    for pattern in _DELIMITER_PATTERNS:
        if re.search(pattern, prompt, re.MULTILINE):
            delimiter_count += 1

    if word_count < 15:
        score = 0.6
        explanation = "Short prompt — delimiters not necessary"
    elif delimiter_count >= 3:
        score = 0.95
        explanation = f"Excellent delimiter usage ({delimiter_count} types)"
    elif delimiter_count >= 2:
        score = 0.8
        explanation = f"Good delimiter usage ({delimiter_count} types)"
    elif delimiter_count >= 1:
        score = 0.6
        explanation = "Some delimiters used"
    elif word_count > 40:
        score = 0.2
        explanation = "Long prompt with no delimiters — model may confuse data with instructions"
    else:
        explanation = "No delimiters (acceptable for shorter prompts)"

    return MetricScore(
        name="Delimiter Usage",
        score=max(0.0, min(1.0, score)),
        weight=0.05,
        explanation=explanation,
    )


def _score_cot_readiness(prompt: str, words: list[str]) -> MetricScore:
    """Evaluate whether the prompt leverages chain-of-thought reasoning.

    CoT prompting (Wei et al. 2022) encourages step-by-step reasoning
    and is especially effective for arithmetic, logic, and analysis tasks.
    """
    prompt_lower = prompt.lower()
    word_count = len(words)

    needs_reasoning = bool(re.search(
        r"\b(why|explain|analyze|compare|derive|prove|evaluate|"
        r"reason|trade-?offs?|implications?|consequences?|assess)\b",
        prompt_lower,
    ))

    cot_signals = 0
    for pattern in _COT_PATTERNS:
        if re.search(pattern, prompt_lower):
            cot_signals += 1

    if not needs_reasoning:
        score = 0.7
        explanation = "Factual/simple query — CoT not required"
    elif cot_signals >= 2:
        score = 0.95
        explanation = f"Strong CoT framing ({cot_signals} reasoning directives)"
    elif cot_signals == 1:
        score = 0.7
        explanation = "Some step-by-step guidance present"
    else:
        score = 0.3
        explanation = "Reasoning task but no CoT guidance — add 'think step by step'"

    return MetricScore(
        name="Chain-of-Thought",
        score=max(0.0, min(1.0, score)),
        weight=0.05,
        explanation=explanation,
    )


def _score_few_shot_readiness(prompt: str, words: list[str]) -> MetricScore:
    """Evaluate whether the prompt includes examples (few-shot pattern).

    Few-shot prompting (Brown et al. 2020) provides demonstrations that
    guide the model toward the desired output format and quality.
    """
    prompt_lower = prompt.lower()
    word_count = len(words)

    needs_examples = bool(re.search(
        r"\b(classify|categorize|label|extract|parse|convert|"
        r"translate|format|transform|rewrite|generate)\b",
        prompt_lower,
    ))

    example_signals = 0
    for pattern in _FEW_SHOT_PATTERNS:
        if re.search(pattern, prompt_lower, re.MULTILINE | re.DOTALL):
            example_signals += 1

    if not needs_examples:
        score = 0.65
        explanation = "Task type doesn't strongly benefit from examples"
    elif example_signals >= 2:
        score = 0.95
        explanation = f"Good few-shot setup ({example_signals} example patterns)"
    elif example_signals == 1:
        score = 0.7
        explanation = "Has some example content"
    else:
        score = 0.3
        explanation = "Extraction/classification task with no examples — add 1-2 demos"

    return MetricScore(
        name="Few-Shot Readiness",
        score=max(0.0, min(1.0, score)),
        weight=0.05,
        explanation=explanation,
    )


def _score_task_decomposition(prompt: str, words: list[str]) -> MetricScore:
    """Evaluate whether complex prompts are broken into subtasks.

    The Prompt Engineering Guide recommends decomposing complex operations
    into simpler subtasks to improve clarity and accuracy.
    """
    prompt_lower = prompt.lower()
    word_count = len(words)

    is_complex = (
        word_count > 30
        or prompt.count("?") > 1
        or bool(re.search(r"\b(and also|additionally|furthermore|moreover)\b", prompt_lower))
    )

    decomp_signals = 0
    for pattern in _DECOMPOSITION_PATTERNS:
        if re.search(pattern, prompt_lower, re.MULTILINE):
            decomp_signals += 1

    if not is_complex:
        score = 0.7
        explanation = "Simple task — decomposition not needed"
    elif decomp_signals >= 3:
        score = 0.95
        explanation = f"Well-decomposed ({decomp_signals} structural signals)"
    elif decomp_signals >= 2:
        score = 0.75
        explanation = "Some task breakdown present"
    elif decomp_signals >= 1:
        score = 0.55
        explanation = "Partial decomposition — could be more explicit"
    else:
        score = 0.2
        explanation = "Complex prompt with no task breakdown — split into numbered steps"

    return MetricScore(
        name="Task Decomposition",
        score=max(0.0, min(1.0, score)),
        weight=0.05,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_prompt(prompt: str) -> PromptMetrics:
    """Evaluate prompt quality across all dimensions.

    Returns a PromptMetrics with individual scores, an overall score
    (0-100), a letter grade, and actionable improvement suggestions.
    All evaluation is heuristic-based — no LLM calls required.
    """
    t0 = time.monotonic()

    words = prompt.split()
    word_count = len(words)

    scores = [
        _score_clarity(prompt, words),
        _score_specificity(prompt, words),
        _score_structure(prompt, words),
        _score_actionability(prompt, words),
        _score_context_sufficiency(prompt, words),
        _score_delimiter_usage(prompt, words),
        _score_cot_readiness(prompt, words),
        _score_few_shot_readiness(prompt, words),
        _score_task_decomposition(prompt, words),
    ]

    total_weight = sum(s.weight for s in scores)
    overall = sum(s.score * s.weight for s in scores) / max(total_weight, 0.01)
    overall_100 = overall * 100

    grade = _letter_grade(overall_100)
    suggestions = _generate_suggestions(scores, prompt, words)

    elapsed_ms = (time.monotonic() - t0) * 1000

    return PromptMetrics(
        prompt=prompt,
        scores=scores,
        overall_score=overall_100,
        grade=grade,
        suggestions=suggestions,
        word_count=word_count,
        evaluation_ms=elapsed_ms,
    )


def _letter_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 65:
        return "C"
    if score >= 50:
        return "D"
    return "F"


def _generate_suggestions(
    scores: list[MetricScore], prompt: str, words: list[str],
) -> list[str]:
    """Produce actionable improvement suggestions based on weak scores."""
    suggestions: list[str] = []
    score_map = {s.name: s.score for s in scores}

    if score_map.get("Clarity", 1) < 0.5:
        if len(words) < 5:
            suggestions.append(
                "Expand your prompt — add more detail about what you need"
            )
        else:
            suggestions.append(
                "Reduce vague language (stuff, things, maybe) and be direct"
            )

    if score_map.get("Specificity", 1) < 0.5:
        suggestions.append(
            "Add domain-specific terms, version numbers, or concrete examples"
        )

    if score_map.get("Structure", 1) < 0.5:
        if len(words) > 30:
            suggestions.append(
                "Break your prompt into sections with bullet points or numbered steps"
            )

    if score_map.get("Actionability", 1) < 0.5:
        suggestions.append(
            "Specify the desired output format or add constraints "
            "(e.g. 'list 5 items', 'respond in JSON', 'step by step')"
        )

    if score_map.get("Context Sufficiency", 1) < 0.5:
        suggestions.append(
            "Add background context — mention your use case, environment, "
            "or what you've already tried"
        )

    if score_map.get("Delimiter Usage", 1) < 0.4:
        suggestions.append(
            "Use delimiters (===, ---, ```, or XML tags) to separate "
            "context, instructions, and data sections"
        )

    if score_map.get("Chain-of-Thought", 1) < 0.5:
        suggestions.append(
            "Add 'Think through this step by step' or 'Show your reasoning' "
            "to elicit chain-of-thought reasoning"
        )

    if score_map.get("Few-Shot Readiness", 1) < 0.5:
        suggestions.append(
            "Include 1-2 input/output examples to demonstrate the "
            "expected response pattern (few-shot prompting)"
        )

    if score_map.get("Task Decomposition", 1) < 0.6:
        suggestions.append(
            "Break your complex request into numbered subtasks so the "
            "model can address each part systematically"
        )

    if "?" not in prompt and len(words) >= 3 and not any(
        words[0].lower().rstrip(",:") == w for w in _QUESTION_WORDS
    ):
        suggestions.append(
            "Consider framing as a question or starting with an action verb "
            "(Explain, Compare, List, Analyze)"
        )

    return suggestions


# ---------------------------------------------------------------------------
# Chain-mode quality alignment
# ---------------------------------------------------------------------------

_MODE_IDEAL_RANGES: dict[str, tuple[int, int]] = {
    "fast": (3, 80),
    "strong": (10, 300),
    "cascade": (10, 400),
    "auto": (3, 400),
    "verify": (10, 300),
    "consensus": (10, 300),
    "pipeline": (10, 300),
    "route": (3, 300),
    "search": (5, 100),
    "agent": (15, 500),
    "pcap": (0, 200),
}


def evaluate_mode_alignment(
    prompt: str, mode: str,
) -> tuple[float, str]:
    """Score how well a prompt fits the chosen chain mode.

    Returns (score 0-1, explanation).
    """
    words = prompt.split()
    word_count = len(words)
    lo, hi = _MODE_IDEAL_RANGES.get(mode, (3, 400))

    if lo <= word_count <= hi:
        length_fit = 1.0
    elif word_count < lo:
        length_fit = max(0.2, word_count / max(lo, 1))
    else:
        length_fit = max(0.3, 1.0 - (word_count - hi) / max(hi, 1) * 0.5)

    prompt_lower = prompt.lower()
    tech_count = sum(
        1 for w in words if w.lower().strip(",.?!:;()") in _TECHNICAL_TERMS
    )

    if mode == "fast" and tech_count > 5:
        return (
            max(0.2, length_fit - 0.4),
            f"Highly technical prompt ({tech_count} terms) may be better "
            f"suited for 'cascade' or 'strong' mode",
        )
    if mode == "fast" and word_count > 80:
        return (
            max(0.3, length_fit - 0.2),
            "Long prompt — consider 'cascade' or 'strong' for thorough analysis",
        )

    if mode in ("cascade", "consensus") and word_count < 8 and tech_count == 0:
        return (
            max(0.4, length_fit - 0.2),
            "Simple query — 'fast' or 'auto' mode would be more efficient",
        )

    if mode == "agent" and word_count < 10:
        return (
            max(0.3, length_fit - 0.2),
            "Agent mode works best with detailed goals — add more context",
        )

    if mode == "search" and "?" not in prompt:
        return (
            max(0.5, length_fit - 0.1),
            "Search mode works best with question-form queries",
        )

    return (length_fit, f"Prompt length fits '{mode}' mode well")


# ---------------------------------------------------------------------------
# Comparative metrics for chain outputs
# ---------------------------------------------------------------------------

@dataclass
class ResponseMetrics:
    """Lightweight metrics collected after a chain produces output."""
    response_length: int
    word_count: int
    sentence_count: int
    has_citations: bool
    citation_count: int
    response_time_ms: float
    compression_ratio: float   # response_words / prompt_words

    def summary(self) -> str:
        lines = [
            f"Response: {self.word_count} words, "
            f"{self.sentence_count} sentences, "
            f"{self.response_time_ms:.0f}ms",
        ]
        if self.has_citations:
            lines.append(f"  Citations: {self.citation_count} source references")
        lines.append(
            f"  Expansion ratio: {self.compression_ratio:.1f}x "
            f"(response / prompt)"
        )
        return "\n".join(lines)


def evaluate_response(
    prompt: str, response: str, response_time_ms: float = 0.0,
) -> ResponseMetrics:
    """Evaluate the quality characteristics of a model response."""
    prompt_words = len(prompt.split())
    resp_words = len(response.split())
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    citation_patterns = [
        r'\[Source:.*?\]',
        r'\[(?:RFC|ISO|IEEE|NIST|W3C)\s*\d+.*?\]',
        r'https?://\S+',
        r'\[\d+\]',
    ]
    citation_count = 0
    for pat in citation_patterns:
        citation_count += len(re.findall(pat, response))

    return ResponseMetrics(
        response_length=len(response),
        word_count=resp_words,
        sentence_count=len(sentences),
        has_citations=citation_count > 0,
        citation_count=citation_count,
        response_time_ms=response_time_ms,
        compression_ratio=resp_words / max(prompt_words, 1),
    )
