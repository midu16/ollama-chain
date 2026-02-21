"""Unit tests for agent helper functions — no Ollama required."""

import pytest

from ollama_chain.agent import (
    _build_search_query,
    _extract_facts_from_output,
    _extract_quoted_strings,
    _is_safe_command,
    _parse_response,
    _validate_before_execution,
    monitor_and_replan,
)
from ollama_chain.memory import SessionMemory


class TestParseResponse:
    def test_tool_call(self):
        text = '<tool_call>\n{"name": "shell", "args": {"command": "ls"}}\n</tool_call>'
        r = _parse_response(text)
        assert r["type"] == "tool_call"
        assert r["name"] == "shell"
        assert r["args"] == {"command": "ls"}

    def test_final_answer(self):
        text = "<final_answer>\nThe answer is 42.\n</final_answer>"
        r = _parse_response(text)
        assert r["type"] == "final_answer"
        assert "42" in r["content"]

    def test_reasoning(self):
        text = "I think we should search next."
        r = _parse_response(text)
        assert r["type"] == "reasoning"

    def test_store_fact(self):
        text = (
            "<store_fact>\nOS: Fedora\n</store_fact>\n"
            '<tool_call>\n{"name":"shell","args":{"command":"ls"}}\n</tool_call>'
        )
        r = _parse_response(text)
        assert r["type"] == "tool_call"
        assert r["facts"] == ["OS: Fedora"]

    def test_multiple_facts(self):
        text = (
            "<store_fact>\nfact1\n</store_fact>\n"
            "<store_fact>\nfact2\n</store_fact>\n"
            "<final_answer>\ndone\n</final_answer>"
        )
        r = _parse_response(text)
        assert r["type"] == "final_answer"
        assert len(r["facts"]) == 2

    def test_malformed_tool_call(self):
        text = "<tool_call>\nnot json\n</tool_call>"
        r = _parse_response(text)
        assert r["type"] == "malformed_tool_call"


class TestExtractQuotedStrings:
    def test_single_quotes(self):
        assert _extract_quoted_strings("Run 'uname -a' now") == ["uname -a"]

    def test_double_quotes(self):
        assert _extract_quoted_strings('Run "cat /etc/os-release"') == [
            "cat /etc/os-release",
        ]

    def test_backticks(self):
        assert _extract_quoted_strings("Execute `lsb_release -a`") == [
            "lsb_release -a",
        ]

    def test_multiple(self):
        result = _extract_quoted_strings("Run 'cmd1' and 'cmd2' then `cmd3`")
        assert "cmd1" in result
        assert "cmd2" in result
        assert "cmd3" in result

    def test_short_strings_ignored(self):
        assert _extract_quoted_strings("Run 'x'") == []

    def test_empty(self):
        assert _extract_quoted_strings("no quotes here") == []


class TestIsSafeCommand:
    def test_safe_commands(self):
        assert _is_safe_command("uname -a")
        assert _is_safe_command("cat /etc/os-release")
        assert _is_safe_command("ls -la /tmp")
        assert _is_safe_command("df -h")

    def test_dangerous_commands(self):
        assert not _is_safe_command("rm -rf /")
        assert not _is_safe_command("mkfs.ext4 /dev/sda")
        assert not _is_safe_command("dd if=/dev/zero of=/dev/sda")
        assert not _is_safe_command("chmod -R 777 /")


class TestBuildSearchQuery:
    def test_strip_prefix(self):
        q = _build_search_query("Search for CVEs in Linux", [])
        assert q == "CVEs in Linux"

    def test_fact_substitution(self):
        facts = ["OS: Fedora Linux 43"]
        q = _build_search_query(
            "Search for CVEs for the identified OS version", facts,
        )
        assert "Fedora Linux 43" in q
        assert "identified" not in q.lower()

    def test_no_matching_facts(self):
        q = _build_search_query("Find kernel bugs", ["unrelated: data"])
        assert q == "kernel bugs"

    def test_no_prefix(self):
        q = _build_search_query("CVE-2024-1234 details", [])
        assert q == "CVE-2024-1234 details"

    def test_trailing_dot_stripped(self):
        q = _build_search_query("Search for something.", [])
        assert not q.endswith(".")


class TestExtractFacts:
    def test_os_release(self):
        output = (
            'NAME="Fedora Linux"\n'
            'VERSION="43 (Workstation Edition)"\n'
            "ID=fedora\n"
            "VERSION_ID=43\n"
            'PRETTY_NAME="Fedora Linux 43 (Workstation Edition)"\n'
        )
        facts = _extract_facts_from_output("shell", output)
        assert any("Fedora" in f for f in facts)
        assert any("43" in f for f in facts)
        assert any("fedora" in f for f in facts)

    def test_kernel_version(self):
        output = "Linux framework 6.18.12-200.fc43.x86_64 #1 SMP x86_64 GNU/Linux"
        facts = _extract_facts_from_output("shell", output)
        assert any("6.18.12" in f for f in facts)

    def test_non_shell_ignored(self):
        facts = _extract_facts_from_output("web_search", "PRETTY_NAME=Ubuntu")
        assert facts == []

    def test_no_match(self):
        facts = _extract_facts_from_output("shell", "just some random output")
        assert facts == []

    def test_combined(self):
        output = (
            'PRETTY_NAME="Ubuntu 24.04"\n'
            "ID=ubuntu\n"
            "VERSION_ID=24.04\n"
            "---\n"
            "Linux host 6.8.0-45-generic #45-Ubuntu SMP x86_64 GNU/Linux\n"
        )
        facts = _extract_facts_from_output("shell", output)
        os_facts = [f for f in facts if f.startswith("OS:")]
        kern_facts = [f for f in facts if f.startswith("Kernel:")]
        assert len(os_facts) >= 1
        assert len(kern_facts) == 1
        assert "6.8.0" in kern_facts[0]

    def test_list_dir(self):
        output = ".git\n.gitignore\nMakefile\nREADME.md\nollama_chain\npyproject.toml\nrequirements.txt\ntests"
        facts = _extract_facts_from_output("list_dir", output)
        assert any("Directory contents" in f for f in facts)
        assert any("Makefile" in f for f in facts)
        assert any("pyproject.toml" in f for f in facts)

    def test_list_dir_with_py_files(self):
        output = "__init__.py\ncli.py\nagent.py\nchains.py"
        facts = _extract_facts_from_output("list_dir", output)
        py_facts = [f for f in facts if f.startswith("Python files:")]
        assert len(py_facts) == 1
        assert "cli.py" in py_facts[0]
        assert "agent.py" in py_facts[0]

    def test_list_dir_empty(self):
        facts = _extract_facts_from_output("list_dir", "")
        assert facts == []


# ---------------------------------------------------------------------------
# Pre-execution validation
# ---------------------------------------------------------------------------

class TestValidateBeforeExecution:
    def _make_session(self, plan):
        session = SessionMemory(session_id="test", goal="test")
        session.plan = plan
        return session

    def test_valid_step_passes(self):
        plan = [
            {"id": 1, "description": "Run cmd", "tool": "shell",
             "depends_on": [], "status": "pending"},
        ]
        session = self._make_session(plan)
        assert _validate_before_execution(plan[0], session) is True

    def test_unknown_tool_downgraded(self):
        plan = [
            {"id": 1, "description": "Magic", "tool": "nonexistent",
             "depends_on": [], "status": "pending"},
        ]
        session = self._make_session(plan)
        assert _validate_before_execution(plan[0], session) is True
        assert plan[0]["tool"] == "none"

    def test_unmet_deps_rejected(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell",
             "depends_on": [], "status": "pending"},
            {"id": 2, "description": "B", "tool": "shell",
             "depends_on": [1], "status": "pending"},
        ]
        session = self._make_session(plan)
        assert _validate_before_execution(plan[1], session) is False

    def test_met_deps_pass(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell",
             "depends_on": [], "status": "completed"},
            {"id": 2, "description": "B", "tool": "shell",
             "depends_on": [1], "status": "pending"},
        ]
        session = self._make_session(plan)
        assert _validate_before_execution(plan[1], session) is True


# ---------------------------------------------------------------------------
# Monitor and replan
# ---------------------------------------------------------------------------

class TestMonitorAndReplan:
    def _make_session(self, plan, facts=None):
        session = SessionMemory(session_id="test", goal="test goal")
        session.plan = plan
        if facts:
            session.facts = list(facts)
        return session

    def test_no_replan_when_no_new_facts(self):
        plan = [
            {"id": 1, "description": "A", "tool": "shell",
             "depends_on": [], "status": "completed"},
            {"id": 2, "description": "B", "tool": "none",
             "depends_on": [1], "status": "pending"},
        ]
        session = self._make_session(plan)
        group = [plan[0]]

        did_replan, checkpoint = monitor_and_replan(
            "goal", session, group, "fast:7b", 0,
        )
        assert did_replan is False
        assert checkpoint == 0

    def test_no_replan_for_non_discovery_without_llm(self):
        plan = [
            {"id": 1, "description": "Run uname", "tool": "web_search",
             "depends_on": [], "status": "completed"},
            {"id": 2, "description": "B", "tool": "none",
             "depends_on": [1], "status": "pending"},
        ]
        session = self._make_session(plan, facts=["OS: Fedora"])
        group = [plan[0]]

        did_replan, checkpoint = monitor_and_replan(
            "goal", session, group, "fast:7b", 0,
        )
        assert did_replan is False
