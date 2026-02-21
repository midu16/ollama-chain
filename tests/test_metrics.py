"""Unit tests for the metrics module — no Ollama required."""

import pytest

from ollama_chain.metrics import (
    MetricScore,
    PromptMetrics,
    ResponseMetrics,
    evaluate_mode_alignment,
    evaluate_prompt,
    evaluate_response,
    _letter_grade,
)


# ---------------------------------------------------------------------------
# evaluate_prompt — overall behaviour
# ---------------------------------------------------------------------------

class TestEvaluatePrompt:
    def test_returns_prompt_metrics(self):
        result = evaluate_prompt("What is a binary search tree?")
        assert isinstance(result, PromptMetrics)

    def test_has_all_nine_scores(self):
        result = evaluate_prompt("Explain TCP vs UDP")
        assert len(result.scores) == 9
        names = {s.name for s in result.scores}
        assert names == {
            "Clarity", "Specificity", "Structure",
            "Actionability", "Context Sufficiency",
            "Delimiter Usage", "Chain-of-Thought",
            "Few-Shot Readiness", "Task Decomposition",
        }

    def test_overall_score_in_range(self):
        result = evaluate_prompt("How does encryption work?")
        assert 0 <= result.overall_score <= 100

    def test_grade_is_letter(self):
        result = evaluate_prompt("What is SSH?")
        assert result.grade in ("A", "B", "C", "D", "F")

    def test_word_count_tracked(self):
        result = evaluate_prompt("one two three four five")
        assert result.word_count == 5

    def test_evaluation_time_tracked(self):
        result = evaluate_prompt("Hello world")
        assert result.evaluation_ms >= 0

    def test_summary_is_string(self):
        result = evaluate_prompt("What is Python?")
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Prompt Quality" in summary

    def test_each_score_in_range(self):
        result = evaluate_prompt(
            "Explain the difference between TCP and UDP protocols, "
            "including use cases for each in production environments."
        )
        for score in result.scores:
            assert 0.0 <= score.score <= 1.0, f"{score.name} out of range"
            assert score.weight > 0
            assert score.explanation


# ---------------------------------------------------------------------------
# Clarity scoring
# ---------------------------------------------------------------------------

class TestClarity:
    def test_very_short_prompt_scores_low(self):
        result = evaluate_prompt("hi")
        clarity = next(s for s in result.scores if s.name == "Clarity")
        assert clarity.score < 0.4

    def test_clear_question_scores_well(self):
        result = evaluate_prompt(
            "What are the key differences between REST and GraphQL APIs?"
        )
        clarity = next(s for s in result.scores if s.name == "Clarity")
        assert clarity.score >= 0.5

    def test_vague_prompt_penalized(self):
        result = evaluate_prompt(
            "Tell me about stuff and things, maybe something about whatever"
        )
        clarity = next(s for s in result.scores if s.name == "Clarity")
        assert clarity.score < 0.6

    def test_directive_verb_boosts_clarity(self):
        result = evaluate_prompt("Explain how TLS handshake works")
        clarity = next(s for s in result.scores if s.name == "Clarity")
        assert clarity.score >= 0.5


# ---------------------------------------------------------------------------
# Specificity scoring
# ---------------------------------------------------------------------------

class TestSpecificity:
    def test_technical_terms_boost_score(self):
        result = evaluate_prompt(
            "Compare the TCP congestion control algorithm with the UDP "
            "protocol implementation for distributed microservice architectures"
        )
        specificity = next(s for s in result.scores if s.name == "Specificity")
        assert specificity.score >= 0.6

    def test_no_technical_terms(self):
        result = evaluate_prompt("Tell me something interesting")
        specificity = next(s for s in result.scores if s.name == "Specificity")
        assert specificity.score < 0.6

    def test_version_numbers_help(self):
        result = evaluate_prompt("What changed in Python 3.12 compared to 3.11?")
        specificity = next(s for s in result.scores if s.name == "Specificity")
        assert specificity.score >= 0.5

    def test_urls_help(self):
        result = evaluate_prompt(
            "Summarize the content at https://example.com/docs"
        )
        specificity = next(s for s in result.scores if s.name == "Specificity")
        assert specificity.score >= 0.5


# ---------------------------------------------------------------------------
# Structure scoring
# ---------------------------------------------------------------------------

class TestStructure:
    def test_well_structured_prompt(self):
        prompt = (
            "I need help with the following:\n\n"
            "1. Explain TCP handshake\n"
            "2. Compare with UDP\n"
            "3. Show example code\n\n"
            "Please format as markdown."
        )
        result = evaluate_prompt(prompt)
        structure = next(s for s in result.scores if s.name == "Structure")
        assert structure.score >= 0.6

    def test_single_line_simple(self):
        result = evaluate_prompt("What is DNS?")
        structure = next(s for s in result.scores if s.name == "Structure")
        assert structure.score >= 0.3

    def test_long_unstructured_penalized(self):
        long_prompt = "word " * 50
        result = evaluate_prompt(long_prompt.strip())
        structure = next(s for s in result.scores if s.name == "Structure")
        assert structure.score < 0.5


# ---------------------------------------------------------------------------
# Actionability scoring
# ---------------------------------------------------------------------------

class TestActionability:
    def test_many_constraints_score_high(self):
        prompt = (
            "List exactly 5 differences between REST and GraphQL. "
            "You must include code examples. Format the output as a "
            "markdown table. Ensure each row has at least 3 columns. "
            "Do not include opinions."
        )
        result = evaluate_prompt(prompt)
        actionability = next(
            s for s in result.scores if s.name == "Actionability"
        )
        assert actionability.score >= 0.7

    def test_no_constraints(self):
        result = evaluate_prompt("networking")
        actionability = next(
            s for s in result.scores if s.name == "Actionability"
        )
        assert actionability.score < 0.5

    def test_question_has_some_actionability(self):
        result = evaluate_prompt("How does BGP routing work?")
        actionability = next(
            s for s in result.scores if s.name == "Actionability"
        )
        assert actionability.score >= 0.4


# ---------------------------------------------------------------------------
# Context sufficiency scoring
# ---------------------------------------------------------------------------

class TestContextSufficiency:
    def test_rich_context(self):
        prompt = (
            "Given a production Kubernetes environment running on AWS, "
            "compare the trade-offs between using Istio versus Linkerd "
            "for service mesh, considering our use case of handling "
            "10k requests per second. For example, we need mTLS and "
            "traffic splitting."
        )
        result = evaluate_prompt(prompt)
        context = next(
            s for s in result.scores if s.name == "Context Sufficiency"
        )
        assert context.score >= 0.7

    def test_no_context(self):
        result = evaluate_prompt("Tell me about it")
        context = next(
            s for s in result.scores if s.name == "Context Sufficiency"
        )
        assert context.score < 0.4

    def test_some_context(self):
        result = evaluate_prompt(
            "Explain how DNS works because I'm debugging a network issue"
        )
        context = next(
            s for s in result.scores if s.name == "Context Sufficiency"
        )
        assert context.score >= 0.5


# ---------------------------------------------------------------------------
# Delimiter usage scoring
# ---------------------------------------------------------------------------

class TestDelimiterUsage:
    def test_delimiters_present(self):
        prompt = (
            "=== CONTEXT ===\nI'm working on a web app.\n\n"
            "--- QUESTION ---\nHow do I add authentication?\n\n"
            "```json\n{\"framework\": \"Django\"}\n```"
        )
        result = evaluate_prompt(prompt)
        delim = next(s for s in result.scores if s.name == "Delimiter Usage")
        assert delim.score >= 0.8

    def test_no_delimiters_long_prompt(self):
        prompt = "word " * 50
        result = evaluate_prompt(prompt.strip())
        delim = next(s for s in result.scores if s.name == "Delimiter Usage")
        assert delim.score < 0.4

    def test_short_prompt_ok_without_delimiters(self):
        result = evaluate_prompt("What is DNS?")
        delim = next(s for s in result.scores if s.name == "Delimiter Usage")
        assert delim.score >= 0.5


# ---------------------------------------------------------------------------
# Chain-of-thought scoring
# ---------------------------------------------------------------------------

class TestChainOfThought:
    def test_cot_framing_present(self):
        prompt = "Explain step by step how TLS handshake works. Show your reasoning."
        result = evaluate_prompt(prompt)
        cot = next(s for s in result.scores if s.name == "Chain-of-Thought")
        assert cot.score >= 0.7

    def test_reasoning_task_without_cot(self):
        prompt = "Why does TCP use a three-way handshake?"
        result = evaluate_prompt(prompt)
        cot = next(s for s in result.scores if s.name == "Chain-of-Thought")
        assert cot.score < 0.5

    def test_factual_task_no_penalty(self):
        prompt = "What port does HTTPS use?"
        result = evaluate_prompt(prompt)
        cot = next(s for s in result.scores if s.name == "Chain-of-Thought")
        assert cot.score >= 0.6


# ---------------------------------------------------------------------------
# Few-shot readiness scoring
# ---------------------------------------------------------------------------

class TestFewShotReadiness:
    def test_examples_present(self):
        prompt = (
            "Classify the following errors by severity.\n"
            "For example:\n"
            "  Input: 'Connection refused'\n"
            "  Output: 'High severity'\n\n"
            "Now classify: 'Disk full'"
        )
        result = evaluate_prompt(prompt)
        fs = next(s for s in result.scores if s.name == "Few-Shot Readiness")
        assert fs.score >= 0.7

    def test_classification_without_examples(self):
        prompt = "Classify these log entries by severity level"
        result = evaluate_prompt(prompt)
        fs = next(s for s in result.scores if s.name == "Few-Shot Readiness")
        assert fs.score < 0.5

    def test_non_classification_ok_without(self):
        prompt = "What is the capital of France?"
        result = evaluate_prompt(prompt)
        fs = next(s for s in result.scores if s.name == "Few-Shot Readiness")
        assert fs.score >= 0.5


# ---------------------------------------------------------------------------
# Task decomposition scoring
# ---------------------------------------------------------------------------

class TestTaskDecomposition:
    def test_decomposed_prompt(self):
        prompt = (
            "1. First, gather the system information\n"
            "2. Then, analyze the log files\n"
            "3. Finally, write a summary report"
        )
        result = evaluate_prompt(prompt)
        td = next(s for s in result.scores if s.name == "Task Decomposition")
        assert td.score >= 0.7

    def test_complex_without_decomposition(self):
        prompt = (
            "Gather the system information and also analyze the log files "
            "and additionally check the network configuration and furthermore "
            "write a report about all of it"
        )
        result = evaluate_prompt(prompt)
        td = next(s for s in result.scores if s.name == "Task Decomposition")
        assert td.score < 0.6

    def test_simple_no_penalty(self):
        result = evaluate_prompt("What is DNS?")
        td = next(s for s in result.scores if s.name == "Task Decomposition")
        assert td.score >= 0.6


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

class TestSuggestions:
    def test_short_prompt_gets_suggestions(self):
        result = evaluate_prompt("hi")
        assert len(result.suggestions) > 0

    def test_high_quality_prompt_fewer_suggestions(self):
        prompt = (
            "Explain step by step how the TCP three-way handshake works. "
            "Include the SYN, SYN-ACK, and ACK packets. Compare this with "
            "the UDP protocol. Format the response as a numbered list. "
            "Given a production environment handling 1000 concurrent "
            "connections, what are the trade-offs?"
        )
        result = evaluate_prompt(prompt)
        low_quality = evaluate_prompt("stuff")
        assert len(result.suggestions) <= len(low_quality.suggestions)

    def test_suggestions_are_strings(self):
        result = evaluate_prompt("hello")
        for s in result.suggestions:
            assert isinstance(s, str)
            assert len(s) > 10


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

class TestGrading:
    def test_excellent_prompt_gets_high_grade(self):
        prompt = (
            "=== TASK ===\n"
            "Explain step by step how the distributed consensus algorithm "
            "in Kubernetes scheduler optimization works.\n\n"
            "--- REQUIREMENTS ---\n"
            "1. Compare it with traditional approaches\n"
            "2. Given a production environment with 100 nodes, "
            "analyze the trade-offs\n"
            "3. Include references to relevant IETF RFCs\n\n"
            "Format as a numbered list with at least 5 points. "
            "For example, consider the latency implications. "
            "Think through each point step by step before answering."
        )
        result = evaluate_prompt(prompt)
        assert result.grade in ("A", "B")

    def test_minimal_prompt_gets_low_grade(self):
        result = evaluate_prompt("hi")
        assert result.grade in ("D", "F")


# ---------------------------------------------------------------------------
# Mode alignment
# ---------------------------------------------------------------------------

class TestModeAlignment:
    def test_simple_query_aligns_with_fast(self):
        score, _ = evaluate_mode_alignment("What is SSH?", "fast")
        assert score >= 0.5

    def test_simple_query_poor_for_agent(self):
        score, explanation = evaluate_mode_alignment("What is SSH?", "agent")
        assert score < 0.7
        assert "agent" in explanation.lower() or "context" in explanation.lower()

    def test_complex_query_fits_cascade(self):
        query = (
            "Analyze the TCP congestion control algorithm implementation "
            "in distributed microservice architectures with Kubernetes "
            "orchestration and compare the throughput optimization "
            "strategies for different deployment scenarios"
        )
        score, _ = evaluate_mode_alignment(query, "cascade")
        assert score >= 0.5

    def test_technical_query_poor_for_fast(self):
        query = (
            "Derive the asymptotic complexity of the distributed consensus "
            "algorithm with cryptographic hash verification and pipeline "
            "optimization for concurrent mutex deadlock prevention"
        )
        score, explanation = evaluate_mode_alignment(query, "fast")
        assert score < 0.7
        assert "cascade" in explanation.lower() or "strong" in explanation.lower()

    def test_returns_tuple(self):
        score, explanation = evaluate_mode_alignment("test", "auto")
        assert isinstance(score, float)
        assert isinstance(explanation, str)

    def test_score_in_range(self):
        for mode in ("fast", "strong", "cascade", "auto", "agent", "search"):
            score, _ = evaluate_mode_alignment("test query here", mode)
            assert 0.0 <= score <= 1.0, f"Score out of range for mode {mode}"

    def test_search_prefers_questions(self):
        score_q, _ = evaluate_mode_alignment(
            "What is the latest Linux kernel version?", "search"
        )
        score_s, _ = evaluate_mode_alignment(
            "Linux kernel version", "search"
        )
        assert score_q >= score_s

    def test_simple_query_poor_for_cascade(self):
        score, explanation = evaluate_mode_alignment("Hi", "cascade")
        assert score < 0.8
        assert "fast" in explanation.lower() or "auto" in explanation.lower() or "simple" in explanation.lower()


# ---------------------------------------------------------------------------
# Response metrics
# ---------------------------------------------------------------------------

class TestResponseMetrics:
    def test_basic_response_metrics(self):
        result = evaluate_response(
            "What is TCP?",
            "TCP is a connection-oriented protocol. It provides reliable delivery.",
            response_time_ms=150.0,
        )
        assert isinstance(result, ResponseMetrics)
        assert result.word_count > 0
        assert result.sentence_count >= 1
        assert result.response_time_ms == 150.0

    def test_citations_detected(self):
        result = evaluate_response(
            "What is TCP?",
            "TCP (RFC 793) is described in [Source: IETF, https://tools.ietf.org/rfc793].",
        )
        assert result.has_citations is True
        assert result.citation_count >= 1

    def test_no_citations(self):
        result = evaluate_response(
            "What is TCP?",
            "TCP is a protocol used for reliable data transfer.",
        )
        assert result.has_citations is False
        assert result.citation_count == 0

    def test_compression_ratio(self):
        result = evaluate_response(
            "short",
            "This is a much longer response with many more words than the prompt.",
        )
        assert result.compression_ratio > 1.0

    def test_summary_is_string(self):
        result = evaluate_response("test", "response text here", 100.0)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "words" in summary

    def test_url_citations_counted(self):
        result = evaluate_response(
            "test",
            "See https://example.com and https://docs.python.org for details.",
        )
        assert result.citation_count >= 2

    def test_numbered_citations_counted(self):
        result = evaluate_response(
            "test",
            "According to [1] and [2], this is correct [3].",
        )
        assert result.citation_count == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_prompt(self):
        result = evaluate_prompt("")
        assert result.word_count == 0
        assert result.overall_score >= 0
        assert result.grade in ("D", "F")

    def test_whitespace_only(self):
        result = evaluate_prompt("   \n\t  ")
        assert result.word_count == 0

    def test_very_long_prompt(self):
        prompt = "Explain " + "the algorithm " * 200
        result = evaluate_prompt(prompt)
        assert result.word_count > 100
        assert 0 <= result.overall_score <= 100

    def test_unicode_prompt(self):
        result = evaluate_prompt("Explain the Ñ character encoding in UTF-8")
        assert result.word_count > 0
        assert result.overall_score > 0

    def test_code_block_in_prompt(self):
        prompt = (
            "Explain what this code does:\n\n"
            "```python\ndef hello():\n    print('hello')\n```"
        )
        result = evaluate_prompt(prompt)
        structure = next(s for s in result.scores if s.name == "Structure")
        assert structure.score >= 0.5

    def test_empty_response(self):
        result = evaluate_response("test prompt", "")
        assert result.word_count == 0
        assert result.compression_ratio == 0.0


# ---------------------------------------------------------------------------
# Weight consistency
# ---------------------------------------------------------------------------

class TestWeightConsistency:
    def test_weights_sum_to_one(self):
        result = evaluate_prompt("test query")
        total = sum(s.weight for s in result.scores)
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_all_weights_positive(self):
        result = evaluate_prompt("test")
        for s in result.scores:
            assert s.weight > 0, f"{s.name} has non-positive weight {s.weight}"

    def test_core_metrics_dominate(self):
        result = evaluate_prompt("test")
        core = sum(
            s.weight for s in result.scores
            if s.name in ("Clarity", "Specificity", "Actionability",
                          "Context Sufficiency", "Structure")
        )
        advanced = sum(
            s.weight for s in result.scores
            if s.name in ("Delimiter Usage", "Chain-of-Thought",
                          "Few-Shot Readiness", "Task Decomposition")
        )
        assert core > advanced


# ---------------------------------------------------------------------------
# Score monotonicity — better prompts score higher
# ---------------------------------------------------------------------------

class TestScoreMonotonicity:
    def test_detailed_beats_vague(self):
        vague = evaluate_prompt("tell me about stuff")
        detailed = evaluate_prompt(
            "Explain how the TCP three-way handshake establishes "
            "a reliable connection between client and server"
        )
        assert detailed.overall_score > vague.overall_score

    def test_structured_beats_flat(self):
        flat = evaluate_prompt(
            "explain DNS how it works what are the record types "
            "and how does caching work and what about DNSSEC"
        )
        structured = evaluate_prompt(
            "Explain DNS resolution:\n\n"
            "1. How does recursive lookup work?\n"
            "2. What are the main record types (A, AAAA, CNAME, MX)?\n"
            "3. How does caching work at each level?\n"
            "4. What does DNSSEC add?\n\n"
            "Format each answer as a short paragraph."
        )
        assert structured.overall_score > flat.overall_score

    def test_constrained_beats_open(self):
        open_ended = evaluate_prompt("Tell me about databases")
        constrained = evaluate_prompt(
            "Compare SQL and NoSQL databases. You must include "
            "at least 3 differences. Format as a markdown table. "
            "Include a use case recommendation for each."
        )
        assert constrained.overall_score > open_ended.overall_score

    def test_contextual_beats_bare(self):
        bare = evaluate_prompt("How to fix the error?")
        contextual = evaluate_prompt(
            "How to fix a 'connection refused' error when connecting "
            "to PostgreSQL 15 on Ubuntu 22.04? The service is running "
            "but only listening on 127.0.0.1. I need it to accept "
            "connections from the application server at 10.0.1.5."
        )
        assert contextual.overall_score > bare.overall_score


# ---------------------------------------------------------------------------
# Letter grade boundaries
# ---------------------------------------------------------------------------

class TestLetterGrade:
    @pytest.mark.parametrize("score,expected", [
        (95, "A"), (90, "A"), (89.9, "B"), (80, "B"),
        (79.9, "C"), (65, "C"), (64.9, "D"), (50, "D"),
        (49.9, "F"), (0, "F"),
    ])
    def test_grade_boundaries(self, score, expected):
        assert _letter_grade(score) == expected


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

class TestSummaryFormatting:
    def test_summary_contains_all_metric_names(self):
        result = evaluate_prompt("Explain TCP step by step")
        summary = result.summary()
        for s in result.scores:
            assert s.name in summary

    def test_summary_contains_bar_chart(self):
        result = evaluate_prompt("What is DNS?")
        summary = result.summary()
        assert "█" in summary or "░" in summary

    def test_summary_contains_grade_and_score(self):
        result = evaluate_prompt("test")
        summary = result.summary()
        assert result.grade in summary
        assert f"{result.overall_score:.0f}" in summary

    def test_response_summary_has_expansion_ratio(self):
        resp = evaluate_response("q", "a longer response than the prompt", 50.0)
        summary = resp.summary()
        assert "Expansion ratio" in summary
        assert "words" in summary


# ---------------------------------------------------------------------------
# Mode alignment — all modes
# ---------------------------------------------------------------------------

class TestModeAlignmentAllModes:
    @pytest.mark.parametrize("mode", [
        "fast", "strong", "cascade", "auto", "verify",
        "consensus", "pipeline", "route", "search", "agent", "pcap",
    ])
    def test_known_mode(self, mode):
        score, explanation = evaluate_mode_alignment("test query here", mode)
        assert 0.0 <= score <= 1.0
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_unknown_mode_uses_defaults(self):
        score, explanation = evaluate_mode_alignment("test", "unknown_mode")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Suggestion coverage
# ---------------------------------------------------------------------------

class TestSuggestionCoverage:
    def test_delimiter_suggestion_for_long_unstructured(self):
        prompt = "word " * 60
        result = evaluate_prompt(prompt.strip())
        has_delimiter_sug = any("delimiter" in s.lower() for s in result.suggestions)
        assert has_delimiter_sug

    def test_cot_suggestion_for_reasoning_without_cot(self):
        result = evaluate_prompt("Why does this algorithm have O(n log n) complexity?")
        has_cot_sug = any("step" in s.lower() or "reasoning" in s.lower()
                          for s in result.suggestions)
        assert has_cot_sug

    def test_few_shot_suggestion_for_classification(self):
        result = evaluate_prompt("Classify these errors by severity")
        has_fs_sug = any("example" in s.lower() or "few-shot" in s.lower()
                         for s in result.suggestions)
        assert has_fs_sug

    def test_decomposition_suggestion_for_complex(self):
        result = evaluate_prompt(
            "Gather logs and also check network and additionally "
            "review firewall rules and furthermore analyze performance "
            "metrics and write a comprehensive report"
        )
        has_decomp_sug = any("subtask" in s.lower() or "numbered" in s.lower()
                             or "break" in s.lower() for s in result.suggestions)
        assert has_decomp_sug


# ---------------------------------------------------------------------------
# Multi-citation response patterns
# ---------------------------------------------------------------------------

class TestResponseCitationPatterns:
    def test_rfc_citations(self):
        result = evaluate_response("q", "See [RFC 793] and [RFC 791] for details.")
        assert result.citation_count >= 2

    def test_source_tag_citations(self):
        result = evaluate_response(
            "q",
            "[Source: MDN, https://developer.mozilla.org] "
            "[Source: W3C, https://w3.org]",
        )
        assert result.citation_count >= 2

    def test_mixed_citation_types(self):
        result = evaluate_response(
            "q",
            "Per [1], see https://example.com and [Source: NIST, SP-800].",
        )
        assert result.citation_count >= 3

    def test_long_response_metrics(self):
        long_resp = "This is a sentence. " * 100
        result = evaluate_response("short query", long_resp, 5000.0)
        assert result.sentence_count >= 50
        assert result.compression_ratio > 10.0
        assert result.response_time_ms == 5000.0
