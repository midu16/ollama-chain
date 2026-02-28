"""Code quality and accuracy tests for ollama-chain.

Tests that the codebase follows accuracy best practices:
  - Time-sensitive query detection works correctly
  - Search grounding instructions are applied
  - Prompts contain required accuracy directives
  - Chain modes properly propagate search context
  - Fact extraction from search results functions correctly
  - Temperature controls are correct for accuracy-critical paths
  - No dangerous code patterns exist
"""

import ast
import importlib
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Locate package sources
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "ollama_chain"
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PYTHON_FILES = sorted(PACKAGE_DIR.glob("*.py"))
TEST_FILES = sorted(TESTS_DIR.glob("test_*.py"))

# Modules that we'll import for deeper checks
from ollama_chain.router import (
    RouteDecision,
    is_time_sensitive,
    classify_complexity_heuristic,
    route_query,
)
from ollama_chain.chains import (
    _GROUNDING_INSTRUCTION,
    _TEMP_FINAL,
    _TEMP_GROUNDED,
    _TEMP_REVIEW,
    _enrich_with_search,
    _extract_key_facts_from_search,
    _inject_search_context,
)
from ollama_chain.common import SOURCE_GUIDANCE
from ollama_chain.validation import (
    validate_plan,
    validate_step,
    validate_and_fix_plan,
    detect_circular_deps,
)


# ===================================================================
# 1. Time-Sensitive Query Detection
# ===================================================================

class TestTimeSensitiveDetection:
    """Verify that is_time_sensitive correctly classifies queries."""

    @pytest.mark.parametrize("query", [
        "What is the latest OpenShift release?",
        "What is the most current Kubernetes version?",
        "What is the newest Linux kernel release?",
        "latest version of Python",
        "current release of Fedora",
        "What version of RHEL was released most recently?",
        "What is the latest stable release of Docker?",
        "What is the most recent CVE for Apache?",
        "What was released this week in Go?",
        "What is the current LTS version of Ubuntu?",
    ])
    def test_detects_time_sensitive_queries(self, query):
        assert is_time_sensitive(query), f"Should detect as time-sensitive: {query!r}"

    @pytest.mark.parametrize("query", [
        "What is a binary search tree?",
        "Explain the TCP three-way handshake",
        "How does DNS work?",
        "Compare REST and GraphQL",
        "What is the Big O notation for quicksort?",
        "Explain how TLS certificates work",
        "What is Kubernetes?",
        "Define polymorphism in OOP",
    ])
    def test_non_time_sensitive_queries(self, query):
        assert not is_time_sensitive(query), f"Should NOT be time-sensitive: {query!r}"


# ===================================================================
# 2. Search Grounding Instructions
# ===================================================================

class TestGroundingInstructions:
    """Verify grounding instruction content is robust."""

    def test_grounding_instruction_contains_override_directive(self):
        assert "training data" in _GROUNDING_INSTRUCTION.lower()

    def test_grounding_instruction_contains_search_preference(self):
        assert "search results" in _GROUNDING_INSTRUCTION.lower()

    def test_grounding_instruction_mentions_contradiction_handling(self):
        assert "contradict" in _GROUNDING_INSTRUCTION.lower()

    def test_grounding_instruction_requires_citations(self):
        assert "citation" in _GROUNDING_INSTRUCTION.lower() or "cite" in _GROUNDING_INSTRUCTION.lower()

    def test_source_guidance_requires_sources_section(self):
        assert "## sources" in SOURCE_GUIDANCE.lower() or "## Sources" in SOURCE_GUIDANCE


# ===================================================================
# 3. Fact Extraction from Search Results
# ===================================================================

class TestFactExtraction:
    """Verify _extract_key_facts_from_search works correctly."""

    def test_extracts_version_numbers(self):
        search_ctx = "OpenShift 4.17 is the latest release. RHEL 9.4 stable."
        facts = _extract_key_facts_from_search(search_ctx)
        assert "4.17" in facts
        assert "9.4" in facts

    def test_extracts_dates(self):
        search_ctx = "Released on January 15, 2025. Updated 2025-02-01."
        facts = _extract_key_facts_from_search(search_ctx)
        assert "January 15, 2025" in facts or "2025-02-01" in facts

    def test_returns_empty_for_empty_input(self):
        assert _extract_key_facts_from_search("") == ""

    def test_returns_empty_for_no_facts(self):
        assert _extract_key_facts_from_search("No factual data here.") == ""

    def test_deduplicates_facts(self):
        search_ctx = "Version 4.17. Release 4.17. OpenShift 4.17."
        facts = _extract_key_facts_from_search(search_ctx)
        assert facts.count("4.17") == 1


# ===================================================================
# 4. Search Context Injection
# ===================================================================

class TestSearchContextInjection:
    def test_inject_appends_context(self):
        result = _inject_search_context("base prompt", "\nSEARCH DATA")
        assert result == "base prompt\nSEARCH DATA"

    def test_inject_no_context(self):
        result = _inject_search_context("base prompt", "")
        assert result == "base prompt"

    @patch("ollama_chain.chains.search_for_query", return_value="")
    def test_enrich_returns_empty_when_disabled(self, _):
        assert _enrich_with_search("q", "model", False) == ""

    @patch("ollama_chain.chains.search_for_query", return_value="results here")
    def test_enrich_includes_search_header(self, _):
        ctx = _enrich_with_search("q", "model", True)
        assert "SEARCH RESULTS" in ctx

    @patch("ollama_chain.chains.search_for_query", return_value="results here")
    @patch("ollama_chain.chains.web_search_news", return_value=[
        MagicMock(title="News", url="http://x", snippet="body", source="news"),
    ])
    @patch("ollama_chain.chains.format_search_results", return_value="[1] news item")
    def test_enrich_includes_news_when_requested(self, _fmt, _news, _search):
        ctx = _enrich_with_search("latest OpenShift release", "model", True, include_news=True)
        assert "RECENT NEWS" in ctx


# ===================================================================
# 5. Temperature Controls
# ===================================================================

class TestTemperatureControls:
    def test_review_temp_is_moderate(self):
        assert 0.3 <= _TEMP_REVIEW <= 0.5

    def test_final_temp_is_low(self):
        assert 0.2 <= _TEMP_FINAL <= 0.4

    def test_grounded_temp_is_lowest(self):
        assert _TEMP_GROUNDED <= _TEMP_FINAL
        assert _TEMP_GROUNDED <= _TEMP_REVIEW

    def test_grounded_temp_is_very_low(self):
        assert _TEMP_GROUNDED <= 0.3


# ===================================================================
# 6. Route Decision Quality
# ===================================================================

class TestRouteDecisionQuality:
    def test_time_sensitive_never_simple(self):
        """Time-sensitive queries should never be classified as simple."""
        with patch("ollama_chain.router.classify_complexity_llm", return_value=("simple", 0.8)):
            decision = route_query(
                "What is the latest OpenShift release?",
                ["small:7b", "large:32b"],
                fast_model="small:7b",
            )
        assert decision.complexity != "simple"
        assert decision.time_sensitive is True

    def test_time_sensitive_flag_set(self):
        with patch("ollama_chain.router.classify_complexity_llm", return_value=("complex", 0.9)):
            decision = route_query(
                "What is the latest Kubernetes version?",
                ["small:7b", "large:32b"],
                fast_model="small:7b",
            )
        assert decision.time_sensitive is True

    def test_non_time_sensitive_flag_unset(self):
        with patch("ollama_chain.router.classify_complexity_llm", return_value=("simple", 0.8)):
            decision = route_query(
                "What is a binary search tree?",
                ["small:7b", "large:32b"],
                fast_model="small:7b",
            )
        assert decision.time_sensitive is False

    def test_single_model_sets_time_sensitive(self):
        with patch("ollama_chain.router.classify_complexity_heuristic", return_value=("simple", 0.8)):
            decision = route_query(
                "latest Python release",
                ["only:7b"],
            )
        assert decision.time_sensitive is True


# ===================================================================
# 7. Source File Quality Checks
# ===================================================================

class TestSourceFileQuality:
    """Static analysis checks on all source files."""

    def test_all_python_files_have_docstrings(self):
        missing = []
        for f in PYTHON_FILES:
            if f.name.startswith("__"):
                continue
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                missing.append(f"SYNTAX ERROR: {f.name}")
                continue
            docstring = ast.get_docstring(tree)
            if not docstring:
                missing.append(f.name)
        assert not missing, f"Files missing module docstrings: {missing}"

    def test_no_bare_except_in_production_code(self):
        """Bare except clauses hide bugs; all should catch specific exceptions."""
        violations = []
        for f in PYTHON_FILES:
            if f.name.startswith("__"):
                continue
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    violations.append(f"{f.name}:{node.lineno}")
        assert not violations, f"Bare except clauses found: {violations}"

    def test_no_hardcoded_localhost_ports(self):
        """URLs with hardcoded ports should use constants or config.

        Exceptions: Ollama default (:11434), server bind address (:8585)
        which are intentional defaults.
        """
        known_ports = {":11434", ":8585"}
        violations = []
        for f in PYTHON_FILES:
            if f.name.startswith("__"):
                continue
            text = f.read_text()
            matches = re.findall(
                r'(localhost:\d+|127\.0\.0\.1:\d+)',
                text,
            )
            for m in matches:
                if not any(p in m for p in known_ports):
                    violations.append(f"{f.name}: {m}")
        assert not violations, f"Hardcoded ports found: {violations}"

    def test_all_source_files_parse(self):
        """Every .py file must be valid Python."""
        errors = []
        for f in PYTHON_FILES:
            try:
                ast.parse(f.read_text())
            except SyntaxError as e:
                errors.append(f"{f.name}: {e}")
        assert not errors, f"Syntax errors found: {errors}"

    def test_no_print_to_stdout_in_chain_logic(self):
        """Chain/agent logic should use stderr for logging, not stdout.

        Only the CLI's final print(result) should go to stdout.
        """
        files_to_check = [
            "chains.py", "agent.py", "router.py", "planner.py",
            "search.py", "tools.py", "common.py",
        ]
        violations = []
        for name in files_to_check:
            f = PACKAGE_DIR / name
            if not f.exists():
                continue
            text = f.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "print(" in stripped and "file=sys.stderr" not in stripped:
                    if "print(f" in stripped or 'print("' in stripped or "print('" in stripped:
                        violations.append(f"{name}:{i}: {stripped[:80]}")
        assert not violations, (
            f"print() to stdout in non-CLI code (should use file=sys.stderr):\n"
            + "\n".join(violations)
        )


# ===================================================================
# 8. Cascade Chain Accuracy Structure
# ===================================================================

class TestCascadeAccuracyStructure:
    """Verify the cascade chain uses correct prompt patterns for accuracy."""

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="")
    def test_cascade_calls_all_models(self, _s, mock_ask):
        models = ["small:7b", "medium:14b", "large:32b"]
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        from ollama_chain.chains import chain_cascade
        chain_cascade("test query", models, web_search=False)
        assert mock_ask.call_count == 3

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="search data here")
    def test_cascade_includes_search_in_all_stages(self, _s, mock_ask):
        models = ["small:7b", "medium:14b", "large:32b"]
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        from ollama_chain.chains import chain_cascade
        chain_cascade("test query", models, web_search=True)
        for c in mock_ask.call_args_list:
            prompt = c.args[0] if c.args else c.kwargs.get("prompt", "")
            assert "search" in prompt.lower() or "SEARCH" in prompt

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="OpenShift 4.17 released")
    @patch("ollama_chain.chains.is_time_sensitive", return_value=True)
    def test_cascade_uses_grounding_for_time_sensitive(self, _ts, _s, mock_ask):
        models = ["small:7b", "large:32b"]
        mock_ask.side_effect = ["draft", "final"]
        from ollama_chain.chains import chain_cascade
        chain_cascade("latest OpenShift release?", models, web_search=True)
        first_prompt = mock_ask.call_args_list[0].args[0]
        assert "CRITICAL" in first_prompt or "training data" in first_prompt.lower()

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="search results")
    @patch("ollama_chain.chains.is_time_sensitive", return_value=True)
    def test_cascade_uses_low_temp_for_time_sensitive(self, _ts, _s, mock_ask):
        models = ["small:7b", "large:32b"]
        mock_ask.side_effect = ["draft", "final"]
        from ollama_chain.chains import chain_cascade
        chain_cascade("latest release?", models, web_search=True)
        for c in mock_ask.call_args_list:
            temp = c.kwargs.get("temperature")
            if temp is not None:
                assert temp <= 0.3, f"Expected low temp for time-sensitive, got {temp}"


# ===================================================================
# 9. Validation Module Quality
# ===================================================================

class TestValidationQuality:
    def test_validate_step_catches_unknown_tool(self):
        step = {"id": 1, "description": "test", "tool": "nonexistent_tool"}
        warnings = validate_step(step)
        assert any("Unknown tool" in w for w in warnings)

    def test_validate_step_catches_missing_description(self):
        step = {"id": 1, "tool": "shell"}
        warnings = validate_step(step)
        assert any("no description" in w for w in warnings)

    def test_validate_step_catches_unmet_deps(self):
        step = {"id": 2, "description": "test", "tool": "shell", "depends_on": [1]}
        warnings = validate_step(step, completed_steps=set())
        assert any("Unmet" in w for w in warnings)

    def test_validate_step_passes_valid_step(self):
        step = {"id": 1, "description": "run ls", "tool": "shell", "depends_on": []}
        warnings = validate_step(step, completed_steps=set())
        assert warnings == []

    def test_validate_plan_catches_circular_deps(self):
        plan = [
            {"id": 1, "description": "a", "tool": "none", "depends_on": [2]},
            {"id": 2, "description": "b", "tool": "none", "depends_on": [1]},
        ]
        cycles = detect_circular_deps(plan)
        assert len(cycles) > 0

    def test_validate_and_fix_repairs_unknown_tool(self):
        plan = [
            {"id": 1, "description": "test", "tool": "magic_tool", "depends_on": [], "status": "pending"},
        ]
        fixed, warnings = validate_and_fix_plan(plan)
        assert fixed[0]["tool"] == "none"
        assert any("unknown tool" in w.lower() for w in warnings)

    def test_validate_and_fix_repairs_dangling_deps(self):
        plan = [
            {"id": 1, "description": "a", "tool": "none", "depends_on": [99], "status": "pending"},
        ]
        fixed, warnings = validate_and_fix_plan(plan)
        assert 99 not in fixed[0]["depends_on"]

    def test_validate_plan_catches_duplicate_ids(self):
        plan = [
            {"id": 1, "description": "a", "tool": "none", "depends_on": []},
            {"id": 1, "description": "b", "tool": "none", "depends_on": []},
        ]
        warnings = validate_plan(plan)
        assert any("Duplicate" in w for w in warnings)


# ===================================================================
# 10. Tool Registry Quality
# ===================================================================

class TestToolRegistryQuality:
    def test_all_tools_have_descriptions(self):
        from ollama_chain.tools import TOOL_REGISTRY
        for name, tool in TOOL_REGISTRY.items():
            assert tool.description, f"Tool {name} has no description"
            assert len(tool.description) > 10, f"Tool {name} description too short"

    def test_all_tools_have_parameters(self):
        from ollama_chain.tools import TOOL_REGISTRY
        for name, tool in TOOL_REGISTRY.items():
            assert isinstance(tool.parameters, dict), f"Tool {name} has invalid parameters"

    def test_all_tools_have_callable_functions(self):
        from ollama_chain.tools import TOOL_REGISTRY
        for name, tool in TOOL_REGISTRY.items():
            assert callable(tool.function), f"Tool {name} function is not callable"

    def test_all_fallback_tools_exist(self):
        from ollama_chain.tools import TOOL_REGISTRY, TOOL_FALLBACKS
        for tool, fallbacks in TOOL_FALLBACKS.items():
            assert tool in TOOL_REGISTRY, f"Fallback source {tool} not in registry"
            for fb in fallbacks:
                assert fb in TOOL_REGISTRY, f"Fallback target {fb} not in registry"

    def test_destructive_commands_blocked(self):
        from ollama_chain.tools import tool_shell
        result = tool_shell("rm -rf /")
        assert not result.success
        assert "destructive" in result.output.lower() or "Blocked" in result.output

    def test_shell_timeout_works(self):
        from ollama_chain.tools import tool_shell
        result = tool_shell("sleep 5", timeout=1)
        assert not result.success
        assert "timed out" in result.output.lower()


# ===================================================================
# 11. Common Module Quality
# ===================================================================

class TestCommonModuleQuality:
    def test_source_guidance_not_empty(self):
        assert len(SOURCE_GUIDANCE) > 50

    def test_source_guidance_mentions_standards_bodies(self):
        for org in ("IETF", "IEEE", "ISO", "W3C"):
            assert org in SOURCE_GUIDANCE

    def test_source_guidance_mentions_official_docs(self):
        for doc in ("Red Hat", "kernel.org", "MDN"):
            assert doc in SOURCE_GUIDANCE

    def test_sanitize_messages_merges_consecutive(self):
        from ollama_chain.common import sanitize_messages
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
            {"role": "assistant", "content": "hi"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 2
        assert "hello" in result[0]["content"]
        assert "world" in result[0]["content"]

    def test_sanitize_messages_skips_empty(self):
        from ollama_chain.common import sanitize_messages
        msgs = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "real"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 1


# ===================================================================
# 12. Chain Mode Registry
# ===================================================================

class TestChainModeRegistry:
    def test_all_modes_registered(self):
        from ollama_chain.chains import CHAINS
        expected = {
            "cascade", "auto", "route", "pipeline", "verify",
            "consensus", "search", "fast", "strong", "agent", "hack",
        }
        assert set(CHAINS.keys()) == expected

    def test_all_modes_are_callable(self):
        from ollama_chain.chains import CHAINS
        for name, fn in CHAINS.items():
            assert callable(fn), f"Chain mode {name} is not callable"

    def test_cli_only_modes_not_in_main_chains(self):
        from ollama_chain.chains import CHAINS, CLI_ONLY_MODES
        for mode in CLI_ONLY_MODES:
            assert mode not in CHAINS, f"CLI-only mode {mode} should not be in CHAINS"


# ===================================================================
# 13. Cross-Verification Prompt Patterns
# ===================================================================

class TestCrossVerificationPrompts:
    """Verify that review/verify prompts contain cross-verification language."""

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="search data")
    def test_cascade_review_mentions_cross_verify(self, _s, mock_ask):
        models = ["small:7b", "medium:14b", "large:32b"]
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        from ollama_chain.chains import chain_cascade
        chain_cascade("test", models, web_search=True, complexity="complex")
        review_prompt = mock_ask.call_args_list[1].args[0]
        assert "cross-verify" in review_prompt.lower() or "search results" in review_prompt.lower()

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="search data")
    def test_cascade_final_mentions_cross_verify(self, _s, mock_ask):
        models = ["small:7b", "medium:14b", "large:32b"]
        mock_ask.side_effect = ["draft", "reviewed", "final"]
        from ollama_chain.chains import chain_cascade
        chain_cascade("test", models, web_search=True, complexity="complex")
        final_prompt = mock_ask.call_args_list[2].args[0]
        assert "cross-verify" in final_prompt.lower() or "search results" in final_prompt.lower()

    @patch("ollama_chain.chains.ask")
    @patch("ollama_chain.chains._enrich_with_search", return_value="search data")
    def test_verify_mode_mentions_cross_verify(self, _s, mock_ask):
        models = ["small:7b", "large:32b"]
        mock_ask.side_effect = ["draft", "verified"]
        from ollama_chain.chains import chain_verify
        chain_verify("test", models, web_search=True)
        verify_prompt = mock_ask.call_args_list[1].args[0]
        assert (
            "cross-verify" in verify_prompt.lower()
            or "contradict" in verify_prompt.lower()
        )


# ===================================================================
# 14. Memory Module Quality
# ===================================================================

class TestMemoryQuality:
    def test_session_memory_fact_deduplication(self):
        from ollama_chain.memory import SessionMemory
        session = SessionMemory(session_id="test", goal="test")
        session.add_fact("OS: Fedora 41")
        session.add_fact("OS: Fedora 41")
        assert session.facts.count("OS: Fedora 41") == 1

    def test_session_memory_history_trimming(self):
        from ollama_chain.memory import SessionMemory, _MAX_HISTORY_ENTRIES
        session = SessionMemory(session_id="test", goal="test")
        for i in range(_MAX_HISTORY_ENTRIES + 50):
            session.add("user", f"message {i}")
        assert len(session.history) <= _MAX_HISTORY_ENTRIES

    def test_session_memory_clear(self):
        from ollama_chain.memory import SessionMemory
        session = SessionMemory(session_id="test", goal="test goal")
        session.add("user", "hello")
        session.add_fact("fact1")
        session.plan = [{"id": 1, "description": "step", "status": "pending"}]
        session.clear()
        assert session.history == []
        assert session.facts == []
        assert session.plan == []


# ===================================================================
# 15. Complexity Heuristic Quality
# ===================================================================

class TestComplexityHeuristic:
    def test_short_simple_question(self):
        complexity, _ = classify_complexity_heuristic("What is DNS?")
        assert complexity == "simple"

    def test_complex_multi_part_question(self):
        complexity, _ = classify_complexity_heuristic(
            "Explain the distributed consensus algorithm used in Kubernetes "
            "etcd cluster replication and how it handles network partitions "
            "with concurrent write operations?"
        )
        assert complexity == "complex"

    def test_returns_valid_complexity_levels(self):
        for q in ["What is X?", "How does Y work?", "Compare A and B in detail"]:
            complexity, confidence = classify_complexity_heuristic(q)
            assert complexity in ("simple", "moderate", "complex")
            assert 0.0 <= confidence <= 1.0
