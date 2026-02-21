"""LLM-driven prompt optimizer based on prompt engineering best practices.

Takes a raw user prompt, evaluates it with heuristic metrics, classifies
its intent, and feeds both into an optimization meta-prompt that rewrites
the query using techniques from the Prompt Engineering Guide:

  - Specificity & clarity (remove ambiguity, add precision)
  - Structured inputs/outputs (delimiters, format specs)
  - Task decomposition (break monolithic prompts into subtasks)
  - Few-shot prompting (add input-output examples when beneficial)
  - Chain-of-thought (elicit step-by-step reasoning)
  - ReAct pattern (interleave reasoning and action for tool-use tasks)

The optimizer requires one LLM call (via the fast model) and produces
a rewritten prompt that scores higher on all quality dimensions.
"""

import re
import sys

from .common import ask, build_structured_prompt
from .metrics import PromptMetrics, evaluate_prompt, evaluate_mode_alignment


# ---------------------------------------------------------------------------
# Intent classification (heuristic — no LLM call)
# ---------------------------------------------------------------------------

INTENT_FACTUAL = "factual"
INTENT_REASONING = "reasoning"
INTENT_COMPARISON = "comparison"
INTENT_CODE = "code"
INTENT_CREATIVE = "creative"
INTENT_CLASSIFICATION = "classification"
INTENT_MULTI_STEP = "multi_step"
INTENT_HOWTO = "howto"
INTENT_ANALYSIS = "analysis"

_INTENT_PATTERNS: list[tuple[str, str]] = [
    (INTENT_COMPARISON, r"\b(compare|versus|vs\.?|difference between|trade-?offs?|pros and cons|advantages|disadvantages)\b"),
    (INTENT_CREATIVE, r"\b(write a story|poem|creative|imagine|brainstorm|generate ideas|come up with|invent)\b"),
    (INTENT_CODE, r"\b(code|implement|function|class|script|program|debug|refactor|snippet|syntax|write a?\s*(?:python|java|bash|sql|go|rust))\b"),
    (INTENT_CLASSIFICATION, r"\b(classify|categorize|label|identify which|sort into|group|bucket|is this a)\b"),
    (INTENT_MULTI_STEP, r"\b(first.*then|step\s*\d|multi-?step|pipeline|workflow|process for|procedure|plan to)\b"),
    (INTENT_HOWTO, r"\b(how (?:do|to|can|should)|setup|configure|install|deploy|build|create|migrate)\b"),
    (INTENT_ANALYSIS, r"\b(analyze|evaluate|assess|audit|review|investigate|examine|diagnose|root cause)\b"),
    (INTENT_REASONING, r"\b(why|explain|reason|because|prove|derive|logic|theorem|implication|therefore|consequence)\b"),
    (INTENT_FACTUAL, r"\b(what is|what are|who is|when did|where is|define|what does|what port|how many|list)\b"),
]


def classify_intent(prompt: str) -> str:
    """Classify the primary intent of a prompt without LLM calls."""
    prompt_lower = prompt.lower()

    if prompt.count("?") > 2 or bool(re.search(r"\b(and also|additionally|furthermore|then)\b", prompt_lower)):
        words = prompt.split()
        if len(words) > 25:
            return INTENT_MULTI_STEP

    for intent, pattern in _INTENT_PATTERNS:
        if re.search(pattern, prompt_lower):
            return intent

    return INTENT_FACTUAL


# ---------------------------------------------------------------------------
# Technique selection based on intent + metrics
# ---------------------------------------------------------------------------

def _select_techniques(
    intent: str,
    metrics: PromptMetrics,
    mode: str,
) -> list[str]:
    """Decide which optimization techniques to apply."""
    techniques: list[str] = []
    score_map = {s.name: s.score for s in metrics.scores}

    techniques.append("specificity_and_clarity")

    if score_map.get("Structure", 1) < 0.6 and metrics.word_count > 15:
        techniques.append("structured_io")

    if intent in (INTENT_MULTI_STEP, INTENT_HOWTO, INTENT_ANALYSIS):
        techniques.append("task_decomposition")

    if intent in (INTENT_REASONING, INTENT_ANALYSIS, INTENT_COMPARISON):
        techniques.append("chain_of_thought")

    if intent in (INTENT_CLASSIFICATION, INTENT_CODE):
        techniques.append("few_shot")

    if mode == "agent":
        techniques.append("react_pattern")

    if score_map.get("Structure", 1) < 0.5 and metrics.word_count > 20:
        techniques.append("delimiters")

    if score_map.get("Actionability", 1) < 0.6:
        techniques.append("output_specification")

    if score_map.get("Context Sufficiency", 1) < 0.5:
        techniques.append("context_enrichment")

    return techniques


# ---------------------------------------------------------------------------
# Technique instruction blocks
# ---------------------------------------------------------------------------

_TECHNIQUE_INSTRUCTIONS: dict[str, str] = {
    "specificity_and_clarity": (
        "SPECIFICITY & CLARITY: Remove ambiguity and vague language. "
        "Replace generic words (stuff, things, something) with precise "
        "domain terms. Make the desired outcome explicit."
    ),
    "structured_io": (
        "STRUCTURED INPUT/OUTPUT: Add clear section boundaries using "
        "delimiters (===, ---, ###). If the prompt has multiple parts, "
        "organize them with labeled sections. Specify the expected output "
        "format (bullet list, table, JSON, numbered steps, etc.)."
    ),
    "task_decomposition": (
        "TASK DECOMPOSITION: If the prompt asks for a complex operation, "
        "break it into numbered subtasks. Each subtask should be specific "
        "and independently verifiable. Order them logically so each builds "
        "on the previous result."
    ),
    "chain_of_thought": (
        "CHAIN-OF-THOUGHT: Add an instruction to reason step by step. "
        "For reasoning or analytical prompts, include a directive like "
        "'Think through this step by step' or 'Show your reasoning for "
        "each point before giving the final answer.'"
    ),
    "few_shot": (
        "FEW-SHOT EXAMPLES: Add 1-2 brief input/output examples that "
        "demonstrate the expected pattern. Use a clear format like:\n"
        "  Input: <example input>\n"
        "  Output: <example output>\n"
        "This guides the model toward the desired response format and "
        "quality. Keep examples concise and representative."
    ),
    "react_pattern": (
        "ReAct PATTERN: For agent/tool-use tasks, frame the prompt to "
        "encourage interleaved reasoning and action. Structure as:\n"
        "  Thought: what to figure out next\n"
        "  Action: what tool/step to take\n"
        "  Observation: what was learned\n"
        "This helps the model plan, act, and adjust dynamically."
    ),
    "delimiters": (
        "DELIMITERS: Use clear separators (===, ---, ```, or XML-style "
        "tags) to delineate different sections of the prompt — context, "
        "instructions, constraints, examples. This prevents the model "
        "from confusing data with directives."
    ),
    "output_specification": (
        "OUTPUT SPECIFICATION: Add explicit constraints on the response "
        "format, length, or style. Examples: 'Respond in exactly 5 bullet "
        "points', 'Format as a markdown table', 'Include code examples', "
        "'Limit to 200 words'. This reduces ambiguity about what a good "
        "answer looks like."
    ),
    "context_enrichment": (
        "CONTEXT ENRICHMENT: Add relevant background information, specify "
        "the target audience, mention the use case or environment, and "
        "state any assumptions. Example: 'I am deploying on Ubuntu 22.04 "
        "with 16GB RAM' or 'This is for a beginner audience'."
    ),
}


# ---------------------------------------------------------------------------
# Meta-prompt construction
# ---------------------------------------------------------------------------

def _build_meta_prompt(
    original: str,
    metrics: PromptMetrics,
    intent: str,
    techniques: list[str],
    mode: str,
) -> str:
    """Construct the optimization meta-prompt sent to the fast LLM."""

    weak_areas = []
    for s in metrics.scores:
        if s.score < 0.6:
            weak_areas.append(f"- {s.name}: {s.score*100:.0f}% — {s.explanation}")

    weak_block = "\n".join(weak_areas) if weak_areas else "No critical weaknesses."

    technique_block = "\n\n".join(
        f"{i}. {_TECHNIQUE_INSTRUCTIONS[t]}"
        for i, t in enumerate(techniques, 1)
    )

    return build_structured_prompt(
        sections=[
            ("Role", (
                "You are a prompt engineering expert. Your task is to rewrite "
                "and optimize the user's prompt to maximize the quality of "
                "the LLM response it will produce.\n"
                "Apply established prompt engineering best practices from "
                "the Prompt Engineering Guide (promptingguide.ai)."
            )),
            ("Original Prompt", original),
            ("Quality Assessment", (
                f"Overall score: {metrics.overall_score:.0f}/100 "
                f"(grade: {metrics.grade})\n"
                f"Intent classified as: {intent}\n"
                f"Target execution mode: {mode}\n"
                f"Word count: {metrics.word_count}\n\n"
                f"Weak areas to improve:\n{weak_block}"
            )),
            ("Techniques To Apply", technique_block),
            ("Rules", (
                "1. Preserve the original intent — do NOT change what is being asked\n"
                "2. Do NOT answer the question — only rewrite the prompt\n"
                "3. Keep the rewritten prompt self-contained (no references to 'the original')\n"
                "4. If the original is already high quality, make only minor refinements\n"
                "5. Do NOT add fictional context or false assumptions\n"
                "6. The optimized prompt should work standalone without any preamble\n"
                "7. Match the complexity to the task — don't over-engineer simple queries"
            )),
        ],
        instructions=(
            "Output ONLY the rewritten, optimized prompt. "
            "Do not include explanations, commentary, or meta-text. "
            "Do not wrap it in quotes or code blocks."
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_prompt(
    prompt: str,
    model: str,
    *,
    mode: str = "cascade",
    show_progress: bool = True,
) -> tuple[str, PromptMetrics, PromptMetrics]:
    """Evaluate, optimize, and re-evaluate a prompt.

    Uses the heuristic metrics to identify weaknesses, classifies the
    prompt intent, selects applicable optimization techniques, and sends
    a meta-prompt to the LLM for rewriting.

    Returns (optimized_prompt, before_metrics, after_metrics).
    """
    before = evaluate_prompt(prompt)
    intent = classify_intent(prompt)
    techniques = _select_techniques(intent, before, mode)

    if show_progress:
        print(
            f"[optimize] Intent: {intent} | "
            f"Techniques: {', '.join(techniques)} | "
            f"Before: {before.grade} ({before.overall_score:.0f}/100)",
            file=sys.stderr,
        )

    meta_prompt = _build_meta_prompt(prompt, before, intent, techniques, mode)

    if show_progress:
        print(f"[optimize] Rewriting with {model}...", file=sys.stderr)

    optimized = ask(meta_prompt, model=model, thinking=True)

    optimized = _clean_optimizer_output(optimized, prompt)

    after = evaluate_prompt(optimized)

    if show_progress:
        delta = after.overall_score - before.overall_score
        direction = "+" if delta >= 0 else ""
        print(
            f"[optimize] After: {after.grade} ({after.overall_score:.0f}/100) "
            f"[{direction}{delta:.0f}]",
            file=sys.stderr,
        )

    return optimized, before, after


def _clean_optimizer_output(output: str, original: str) -> str:
    """Strip common LLM artifacts from the optimizer output."""
    output = output.strip()

    if output.startswith("```") and output.endswith("```"):
        lines = output.split("\n")
        output = "\n".join(lines[1:-1]).strip()

    for prefix in (
        "Here is the optimized prompt:",
        "Here's the optimized prompt:",
        "Optimized prompt:",
        "Rewritten prompt:",
        "Here is the rewritten prompt:",
        "Here's the rewritten prompt:",
    ):
        if output.lower().startswith(prefix.lower()):
            output = output[len(prefix):].strip()

    if (output.startswith('"') and output.endswith('"')) or \
       (output.startswith("'") and output.endswith("'")):
        output = output[1:-1].strip()

    if not output or len(output) < 5:
        return original

    return output


def format_optimization_report(
    original: str,
    optimized: str,
    before: PromptMetrics,
    after: PromptMetrics,
    mode: str,
) -> str:
    """Format a before/after comparison report for CLI display."""
    intent = classify_intent(original)
    techniques = _select_techniques(intent, before, mode)

    lines = [
        f"{'='*60}",
        "PROMPT OPTIMIZATION REPORT",
        f"{'='*60}",
        "",
        f"Intent: {intent}",
        f"Techniques applied: {', '.join(techniques)}",
        "",
        "--- BEFORE ---",
        before.summary(),
        "",
        "--- AFTER ---",
        after.summary(),
        "",
        f"{'─'*60}",
        f"Score change: {before.overall_score:.0f} → {after.overall_score:.0f} "
        f"({after.overall_score - before.overall_score:+.0f})",
        f"Grade change: {before.grade} → {after.grade}",
        f"{'─'*60}",
        "",
        "--- ORIGINAL PROMPT ---",
        original,
        "",
        "--- OPTIMIZED PROMPT ---",
        optimized,
        f"{'='*60}",
    ]
    return "\n".join(lines)
