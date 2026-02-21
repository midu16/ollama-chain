"""Unit tests for the optimizer module — no Ollama required.

Tests cover intent classification, technique selection, meta-prompt
construction, output cleaning, and report formatting.  The actual
LLM-based optimize_prompt() is not tested here since it requires a
running Ollama instance.
"""

import pytest

from ollama_chain.optimizer import (
    INTENT_ANALYSIS,
    INTENT_CLASSIFICATION,
    INTENT_CODE,
    INTENT_COMPARISON,
    INTENT_CREATIVE,
    INTENT_FACTUAL,
    INTENT_HOWTO,
    INTENT_MULTI_STEP,
    INTENT_REASONING,
    _build_meta_prompt,
    _clean_optimizer_output,
    _select_techniques,
    classify_intent,
    format_optimization_report,
)
from ollama_chain.metrics import evaluate_prompt


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    def test_factual_what_is(self):
        assert classify_intent("What is a binary search tree?") == INTENT_FACTUAL

    def test_factual_define(self):
        assert classify_intent("Define polymorphism") == INTENT_FACTUAL

    def test_factual_list(self):
        assert classify_intent("List the planets in the solar system") == INTENT_FACTUAL

    def test_comparison(self):
        assert classify_intent("Compare REST vs GraphQL APIs") == INTENT_COMPARISON

    def test_comparison_tradeoffs(self):
        assert classify_intent(
            "What are the trade-offs between TCP and UDP?"
        ) == INTENT_COMPARISON

    def test_comparison_pros_cons(self):
        assert classify_intent(
            "Pros and cons of microservices vs monolith"
        ) == INTENT_COMPARISON

    def test_code(self):
        assert classify_intent("Write a Python function to sort a list") == INTENT_CODE

    def test_code_debug(self):
        assert classify_intent("Debug this function that crashes on empty input") == INTENT_CODE

    def test_classification(self):
        assert classify_intent("Classify these log entries by severity") == INTENT_CLASSIFICATION

    def test_multi_step_explicit(self):
        assert classify_intent(
            "First gather the system info, then analyze the logs, "
            "then write a report about the findings and also suggest fixes. "
            "Additionally check the network configuration and finally summarize."
        ) == INTENT_MULTI_STEP

    def test_howto(self):
        assert classify_intent("How to configure Nginx as a reverse proxy") == INTENT_HOWTO

    def test_howto_deploy(self):
        assert classify_intent("Deploy a Django app to Kubernetes") == INTENT_HOWTO

    def test_creative(self):
        assert classify_intent("Write a story about a robot learning to code") == INTENT_CREATIVE

    def test_analysis(self):
        assert classify_intent("Analyze the root cause of this memory leak") == INTENT_ANALYSIS

    def test_reasoning(self):
        assert classify_intent("Why does TCP use a three-way handshake?") == INTENT_REASONING

    def test_reasoning_explain(self):
        assert classify_intent(
            "Explain the implications of CAP theorem"
        ) == INTENT_REASONING

    def test_empty_defaults_factual(self):
        assert classify_intent("hello") == INTENT_FACTUAL

    def test_short_defaults_factual(self):
        assert classify_intent("DNS") == INTENT_FACTUAL


# ---------------------------------------------------------------------------
# Technique selection
# ---------------------------------------------------------------------------

class TestSelectTechniques:
    def test_always_includes_specificity(self):
        metrics = evaluate_prompt("test query")
        techniques = _select_techniques(INTENT_FACTUAL, metrics, "cascade")
        assert "specificity_and_clarity" in techniques

    def test_reasoning_gets_cot(self):
        metrics = evaluate_prompt("Why does TCP need a three-way handshake?")
        techniques = _select_techniques(INTENT_REASONING, metrics, "cascade")
        assert "chain_of_thought" in techniques

    def test_comparison_gets_cot(self):
        metrics = evaluate_prompt("Compare REST vs GraphQL")
        techniques = _select_techniques(INTENT_COMPARISON, metrics, "cascade")
        assert "chain_of_thought" in techniques

    def test_classification_gets_few_shot(self):
        metrics = evaluate_prompt("Classify these items")
        techniques = _select_techniques(INTENT_CLASSIFICATION, metrics, "cascade")
        assert "few_shot" in techniques

    def test_code_gets_few_shot(self):
        metrics = evaluate_prompt("Write a function to parse JSON")
        techniques = _select_techniques(INTENT_CODE, metrics, "cascade")
        assert "few_shot" in techniques

    def test_multi_step_gets_decomposition(self):
        metrics = evaluate_prompt("First do X, then do Y")
        techniques = _select_techniques(INTENT_MULTI_STEP, metrics, "cascade")
        assert "task_decomposition" in techniques

    def test_howto_gets_decomposition(self):
        metrics = evaluate_prompt("How to set up Docker on Ubuntu?")
        techniques = _select_techniques(INTENT_HOWTO, metrics, "cascade")
        assert "task_decomposition" in techniques

    def test_agent_mode_gets_react(self):
        metrics = evaluate_prompt("Find all log files and analyze them")
        techniques = _select_techniques(INTENT_ANALYSIS, metrics, "agent")
        assert "react_pattern" in techniques

    def test_non_agent_no_react(self):
        metrics = evaluate_prompt("Find all log files and analyze them")
        techniques = _select_techniques(INTENT_ANALYSIS, metrics, "cascade")
        assert "react_pattern" not in techniques

    def test_low_actionability_gets_output_spec(self):
        metrics = evaluate_prompt("networking")
        techniques = _select_techniques(INTENT_FACTUAL, metrics, "cascade")
        assert "output_specification" in techniques

    def test_low_context_gets_enrichment(self):
        metrics = evaluate_prompt("explain it")
        techniques = _select_techniques(INTENT_FACTUAL, metrics, "cascade")
        assert "context_enrichment" in techniques


# ---------------------------------------------------------------------------
# Meta-prompt construction
# ---------------------------------------------------------------------------

class TestBuildMetaPrompt:
    def test_contains_original(self):
        original = "What is DNS?"
        metrics = evaluate_prompt(original)
        techniques = _select_techniques(INTENT_FACTUAL, metrics, "cascade")
        prompt = _build_meta_prompt(original, metrics, INTENT_FACTUAL, techniques, "cascade")
        assert original in prompt

    def test_contains_techniques(self):
        original = "networking stuff"
        metrics = evaluate_prompt(original)
        techniques = _select_techniques(INTENT_FACTUAL, metrics, "cascade")
        prompt = _build_meta_prompt(original, metrics, INTENT_FACTUAL, techniques, "cascade")
        assert "SPECIFICITY" in prompt

    def test_contains_rules(self):
        original = "test"
        metrics = evaluate_prompt(original)
        techniques = ["specificity_and_clarity"]
        prompt = _build_meta_prompt(original, metrics, INTENT_FACTUAL, techniques, "cascade")
        assert "Preserve the original intent" in prompt
        assert "Do NOT answer the question" in prompt

    def test_contains_quality_assessment(self):
        original = "Explain something"
        metrics = evaluate_prompt(original)
        techniques = ["specificity_and_clarity"]
        prompt = _build_meta_prompt(original, metrics, INTENT_FACTUAL, techniques, "cascade")
        assert f"grade: {metrics.grade}" in prompt

    def test_cot_technique_included_when_selected(self):
        original = "Why does X happen?"
        metrics = evaluate_prompt(original)
        techniques = ["specificity_and_clarity", "chain_of_thought"]
        prompt = _build_meta_prompt(original, metrics, INTENT_REASONING, techniques, "cascade")
        assert "CHAIN-OF-THOUGHT" in prompt
        assert "step by step" in prompt.lower()

    def test_few_shot_technique_included(self):
        original = "Classify these items"
        metrics = evaluate_prompt(original)
        techniques = ["specificity_and_clarity", "few_shot"]
        prompt = _build_meta_prompt(
            original, metrics, INTENT_CLASSIFICATION, techniques, "cascade",
        )
        assert "FEW-SHOT" in prompt
        assert "Input:" in prompt

    def test_react_technique_included(self):
        original = "Find files and analyze them"
        metrics = evaluate_prompt(original)
        techniques = ["specificity_and_clarity", "react_pattern"]
        prompt = _build_meta_prompt(original, metrics, INTENT_ANALYSIS, techniques, "agent")
        assert "ReAct" in prompt
        assert "Thought:" in prompt


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------

class TestCleanOptimizerOutput:
    def test_strips_code_fences(self):
        output = "```\nOptimized prompt here\n```"
        assert _clean_optimizer_output(output, "original") == "Optimized prompt here"

    def test_strips_preamble(self):
        output = "Here is the optimized prompt:\n\nThe actual prompt"
        assert _clean_optimizer_output(output, "original") == "The actual prompt"

    def test_strips_quotes(self):
        output = '"The optimized prompt text"'
        assert _clean_optimizer_output(output, "original") == "The optimized prompt text"

    def test_strips_single_quotes(self):
        output = "'The optimized prompt text'"
        assert _clean_optimizer_output(output, "original") == "The optimized prompt text"

    def test_preserves_normal_output(self):
        output = "What are the key differences between TCP and UDP protocols?"
        assert _clean_optimizer_output(output, "original") == output

    def test_falls_back_on_empty(self):
        assert _clean_optimizer_output("", "original") == "original"

    def test_falls_back_on_too_short(self):
        assert _clean_optimizer_output("hi", "original") == "original"

    def test_strips_whitespace(self):
        output = "  \n  The prompt  \n  "
        assert _clean_optimizer_output(output, "original") == "The prompt"

    def test_case_insensitive_preamble(self):
        output = "OPTIMIZED PROMPT:\nThe actual content"
        assert _clean_optimizer_output(output, "original") == "The actual content"


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

class TestFormatOptimizationReport:
    def test_report_is_string(self):
        before = evaluate_prompt("test")
        after = evaluate_prompt("Explain the concept of DNS resolution step by step")
        report = format_optimization_report(
            "test",
            "Explain the concept of DNS resolution step by step",
            before, after, "cascade",
        )
        assert isinstance(report, str)

    def test_report_contains_sections(self):
        before = evaluate_prompt("stuff")
        after = evaluate_prompt("Explain how HTTPS encryption works")
        report = format_optimization_report(
            "stuff", "Explain how HTTPS encryption works",
            before, after, "cascade",
        )
        assert "BEFORE" in report
        assert "AFTER" in report
        assert "ORIGINAL PROMPT" in report
        assert "OPTIMIZED PROMPT" in report
        assert "Score change" in report
        assert "Grade change" in report

    def test_report_contains_techniques(self):
        before = evaluate_prompt("stuff")
        after = evaluate_prompt("Explain DNS")
        report = format_optimization_report(
            "stuff", "Explain DNS", before, after, "cascade",
        )
        assert "Techniques applied" in report

    def test_report_shows_both_prompts(self):
        original = "tell me about stuff"
        optimized = "Explain the key concepts of container orchestration"
        before = evaluate_prompt(original)
        after = evaluate_prompt(optimized)
        report = format_optimization_report(
            original, optimized, before, after, "cascade",
        )
        assert original in report
        assert optimized in report


# ---------------------------------------------------------------------------
# Integration: metrics → optimizer pipeline
# ---------------------------------------------------------------------------

class TestMetricsOptimizerIntegration:
    """Verify that the metrics evaluation feeds correctly into the
    optimizer's technique selection and meta-prompt construction."""

    def test_low_quality_prompt_selects_many_techniques(self):
        prompt = "stuff"
        metrics = evaluate_prompt(prompt)
        intent = classify_intent(prompt)
        techniques = _select_techniques(intent, metrics, "cascade")
        assert len(techniques) >= 3

    def test_high_quality_prompt_selects_fewer_techniques(self):
        prompt = (
            "=== TASK ===\n"
            "Compare TCP and UDP protocols step by step.\n\n"
            "--- REQUIREMENTS ---\n"
            "1. Include latency analysis\n"
            "2. Show code examples in Python\n"
            "3. Format as a numbered list\n\n"
            "For example, consider the reliability trade-offs."
        )
        metrics = evaluate_prompt(prompt)
        intent = classify_intent(prompt)
        techniques = _select_techniques(intent, metrics, "cascade")
        low_metrics = evaluate_prompt("stuff")
        low_techniques = _select_techniques(INTENT_FACTUAL, low_metrics, "cascade")
        assert len(techniques) <= len(low_techniques)

    def test_meta_prompt_adapts_to_intent(self):
        prompt_code = "Write a function to sort a list"
        prompt_reason = "Why does TCP use a three-way handshake?"

        m_code = evaluate_prompt(prompt_code)
        m_reason = evaluate_prompt(prompt_reason)

        t_code = _select_techniques(INTENT_CODE, m_code, "cascade")
        t_reason = _select_techniques(INTENT_REASONING, m_reason, "cascade")

        mp_code = _build_meta_prompt(prompt_code, m_code, INTENT_CODE, t_code, "cascade")
        mp_reason = _build_meta_prompt(prompt_reason, m_reason, INTENT_REASONING, t_reason, "cascade")

        assert "FEW-SHOT" in mp_code
        assert "CHAIN-OF-THOUGHT" in mp_reason

    def test_agent_mode_injects_react(self):
        prompt = "Find all log files and analyze their contents"
        metrics = evaluate_prompt(prompt)
        intent = classify_intent(prompt)
        techniques = _select_techniques(intent, metrics, "agent")
        mp = _build_meta_prompt(prompt, metrics, intent, techniques, "agent")
        assert "ReAct" in mp
        assert "agent" in mp

    def test_all_technique_instructions_resolvable(self):
        from ollama_chain.optimizer import _TECHNIQUE_INSTRUCTIONS
        prompt = "test"
        metrics = evaluate_prompt(prompt)
        for intent_const in [INTENT_FACTUAL, INTENT_REASONING, INTENT_COMPARISON,
                             INTENT_CODE, INTENT_CREATIVE, INTENT_CLASSIFICATION,
                             INTENT_MULTI_STEP, INTENT_HOWTO, INTENT_ANALYSIS]:
            for mode in ("cascade", "agent", "fast"):
                techniques = _select_techniques(intent_const, metrics, mode)
                for t in techniques:
                    assert t in _TECHNIQUE_INSTRUCTIONS, (
                        f"Technique '{t}' from intent={intent_const}, mode={mode} "
                        f"not in _TECHNIQUE_INSTRUCTIONS"
                    )


# ---------------------------------------------------------------------------
# Intent edge cases
# ---------------------------------------------------------------------------

class TestIntentEdgeCases:
    def test_mixed_intent_comparison_wins(self):
        assert classify_intent(
            "Compare the code implementations of quicksort vs mergesort"
        ) == INTENT_COMPARISON

    def test_multi_step_long_prompt(self):
        prompt = (
            "First install the dependencies, then configure the database, "
            "and also set up the web server, additionally create the API routes, "
            "furthermore write the tests and finally deploy to production. "
            "Also check the monitoring and review the security settings."
        )
        assert classify_intent(prompt) == INTENT_MULTI_STEP

    def test_single_word_defaults(self):
        assert classify_intent("networking") == INTENT_FACTUAL

    def test_question_marks_alone_not_multi_step(self):
        assert classify_intent("What? Who? Where?") != INTENT_MULTI_STEP


# ---------------------------------------------------------------------------
# Clean output edge cases
# ---------------------------------------------------------------------------

class TestCleanOutputEdgeCases:
    def test_nested_code_fences(self):
        output = "```\n```python\ndef hello():\n    pass\n```\n```"
        result = _clean_optimizer_output(output, "original")
        assert "def hello" in result

    def test_multiple_preamble_lines(self):
        output = (
            "Here is the optimized prompt:\n\n"
            "Explain how HTTPS encryption ensures data integrity"
        )
        result = _clean_optimizer_output(output, "original")
        assert result.startswith("Explain")

    def test_only_whitespace_fallback(self):
        assert _clean_optimizer_output("   \n  \t  ", "fallback") == "fallback"

    def test_very_short_output_fallback(self):
        assert _clean_optimizer_output("ok", "original prompt") == "original prompt"

    def test_unicode_preserved(self):
        output = "Explain how the Ñ character works in UTF-8 encoding"
        assert _clean_optimizer_output(output, "original") == output

    def test_multiline_preserved(self):
        output = "Step 1: Do X\nStep 2: Do Y\nStep 3: Do Z"
        assert _clean_optimizer_output(output, "original") == output


# ---------------------------------------------------------------------------
# Report formatting edge cases
# ---------------------------------------------------------------------------

class TestReportEdgeCases:
    def test_same_score_shows_zero_delta(self):
        metrics = evaluate_prompt("Explain TCP")
        report = format_optimization_report(
            "Explain TCP", "Explain TCP", metrics, metrics, "cascade",
        )
        assert "+0" in report

    def test_report_contains_intent(self):
        before = evaluate_prompt("Compare X and Y")
        after = evaluate_prompt("Compare X and Y in detail step by step")
        report = format_optimization_report(
            "Compare X and Y",
            "Compare X and Y in detail step by step",
            before, after, "cascade",
        )
        assert "comparison" in report

    def test_report_separator_lines(self):
        before = evaluate_prompt("test")
        after = evaluate_prompt("Explain DNS")
        report = format_optimization_report(
            "test", "Explain DNS", before, after, "cascade",
        )
        assert "=" * 60 in report
