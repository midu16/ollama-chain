"""Memory system for state persistence across agent interactions.

Provides two layers:
  - SessionMemory: working memory for the current agent run (plan, history, facts).
  - PersistentMemory: long-term storage on disk (~/.ollama_chain/) that survives
    across sessions, enabling the agent to recall facts and past goals.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

MEMORY_DIR = Path.home() / ".ollama_chain"


# ---------------------------------------------------------------------------
# Content types for multi-modal support
# ---------------------------------------------------------------------------

CONTENT_TEXT = "text"
CONTENT_TOOL_OUTPUT = "tool_output"
CONTENT_FACT = "fact"
CONTENT_JSON = "json_data"
CONTENT_IMAGE_REF = "image_ref"
CONTENT_BINARY_REF = "binary_ref"
CONTENT_ERROR = "error"


@dataclass
class MemoryEntry:
    role: str  # "user" | "assistant" | "tool" | "system"
    content: str
    content_type: str = CONTENT_TEXT
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionMemory:
    """Working memory for the current agent session."""

    session_id: str
    goal: str = ""
    plan: list[dict] = field(default_factory=list)
    history: list[MemoryEntry] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)

    # -- Core add methods --

    def add(self, role: str, content: str, **metadata):
        self.history.append(
            MemoryEntry(role=role, content=content, metadata=metadata)
        )

    def add_tool_output(
        self, tool_name: str, output: str, step_id: int, success: bool,
    ):
        self.history.append(MemoryEntry(
            role="user",
            content=f"Tool result ({tool_name}):\n{output}",
            content_type=CONTENT_TOOL_OUTPUT,
            metadata={"tool": tool_name, "step_id": step_id, "success": success},
            tags=["tool_result", tool_name],
        ))

    def add_fact(self, fact: str):
        if fact not in self.facts:
            self.facts.append(fact)
            self.history.append(MemoryEntry(
                role="assistant",
                content=fact,
                content_type=CONTENT_FACT,
                tags=["fact"],
            ))

    def add_error(self, error: str, step_id: int | None = None):
        self.history.append(MemoryEntry(
            role="user",
            content=error,
            content_type=CONTENT_ERROR,
            metadata={"step_id": step_id},
            tags=["error"],
        ))

    # -- Context retrieval (Gap 2 & 6: proper LLM-ready context) --

    def get_context_window(self, max_entries: int = 20) -> list[dict]:
        """Return recent history formatted for LLM chat consumption.

        Includes plan status and facts as a preamble, then recent history
        entries mapped to proper chat roles.
        """
        preamble = self.summarize()
        result: list[dict] = [{"role": "assistant", "content": preamble}]

        recent = self.history[-max_entries:]
        for e in recent:
            role = e.role if e.role in ("user", "assistant") else "assistant"
            result.append({"role": role, "content": e.content})
        return result

    def get_structured_context(
        self,
        current_step: dict | None = None,
        max_tool_results: int = 5,
        max_history: int = 8,
    ) -> str:
        """Build a prioritised, sectioned context string for the LLM.

        Addresses Gap 6 (shallow prompt engineering) by:
          - Injecting current plan status with step markers
          - Prioritising facts relevant to the current step
          - Including only the most recent / relevant tool outputs
          - Formatting clearly with section headers
        """
        sections: list[str] = []

        # --- Plan status ---
        if self.plan:
            plan_lines = ["Plan progress:"]
            for s in self.plan:
                status = s.get("status", "pending")
                marker = {
                    "completed": "[done]",
                    "in_progress": ">>",
                    "failed": "[FAIL]",
                }.get(status, "[ ]")
                plan_lines.append(f"  {marker} {s['id']}. {s['description']}")
            sections.append("\n".join(plan_lines))

        # --- Facts (prioritised by relevance to current step) ---
        if self.facts:
            if current_step:
                scored = _score_facts(self.facts, current_step.get("description", ""))
            else:
                scored = [(f, 0) for f in self.facts]
            scored.sort(key=lambda x: x[1], reverse=True)
            fact_lines = ["Known facts:"]
            for f, _ in scored[:15]:
                fact_lines.append(f"  - {f}")
            sections.append("\n".join(fact_lines))

        # --- Recent tool results ---
        recent_tools = self.tool_results[-max_tool_results:]
        if recent_tools:
            tool_lines = ["Recent tool outputs:"]
            for tr in recent_tools:
                status = "OK" if tr["success"] else "FAILED"
                output_preview = tr["output"][:200].replace("\n", " ")
                tool_lines.append(
                    f"  Step {tr['step']} ({tr['tool']}) [{status}]: "
                    f"{output_preview}"
                )
            sections.append("\n".join(tool_lines))

        # --- Errors (always show) ---
        errors = [
            e for e in self.history
            if e.content_type == CONTENT_ERROR
        ]
        if errors:
            err_lines = ["Errors encountered:"]
            for e in errors[-5:]:
                err_lines.append(f"  - {e.content[:150]}")
            sections.append("\n".join(err_lines))

        return "\n\n".join(sections)

    def get_relevant_history(
        self, keywords: list[str], max_entries: int = 10,
    ) -> list[MemoryEntry]:
        """Return history entries scored by keyword relevance."""
        if not keywords:
            return list(self.history[-max_entries:])

        scored: list[tuple[MemoryEntry, float]] = []
        kw_lower = [k.lower() for k in keywords]

        for entry in self.history:
            content_lower = entry.content.lower()
            score = sum(1 for kw in kw_lower if kw in content_lower)
            if entry.content_type == CONTENT_TOOL_OUTPUT:
                score += 0.5
            if entry.content_type == CONTENT_FACT:
                score += 0.3
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:max_entries]]

    # -- Summaries --

    def summarize(self) -> str:
        """Produce a text summary of current session state."""
        lines = [f"Goal: {self.goal}"]
        if self.plan:
            lines.append("\nPlan:")
            for step in self.plan:
                status = step.get("status", "pending")
                marker = {
                    "completed": "[done]",
                    "in_progress": "[running]",
                    "failed": "[FAILED]",
                }.get(status, "[ ]")
                deps = step.get("depends_on", [])
                dep_str = f" (after step {','.join(map(str, deps))})" if deps else ""
                lines.append(
                    f"  {marker} {step['id']}. {step['description']}{dep_str}"
                )
        if self.facts:
            lines.append("\nKnown facts:")
            for f in self.facts[-10:]:
                lines.append(f"  - {f}")
        return "\n".join(lines)

    def completed_step_count(self) -> int:
        return sum(1 for s in self.plan if s["status"] == "completed")

    def failed_step_count(self) -> int:
        return sum(1 for s in self.plan if s["status"] == "failed")

    def pending_steps(self) -> list[dict]:
        return [s for s in self.plan if s["status"] == "pending"]

    def clear(self):
        """Release all session data to free memory."""
        self.plan.clear()
        self.history.clear()
        self.facts.clear()
        self.tool_results.clear()
        self.goal = ""


# ---------------------------------------------------------------------------
# Relevance scoring helper
# ---------------------------------------------------------------------------

def _score_facts(facts: list[str], reference: str) -> list[tuple[str, float]]:
    """Score facts by keyword overlap with a reference string."""
    ref_words = set(reference.lower().split())
    scored = []
    for fact in facts:
        fact_words = set(fact.lower().split())
        overlap = len(ref_words & fact_words)
        scored.append((fact, overlap))
    return scored


# ---------------------------------------------------------------------------
# Persistent memory
# ---------------------------------------------------------------------------

class PersistentMemory:
    """Long-term memory stored on disk as JSON."""

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.facts_file = self.memory_dir / "facts.json"
        self.sessions_file = self.memory_dir / "sessions.json"
        self._facts: list[str] = self._load(self.facts_file, default=[])
        self._sessions: list[dict] = self._load(self.sessions_file, default=[])

    def _load(self, path: Path, default=None):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return default if default is not None else {}
        return default if default is not None else {}

    def _save(self, path: Path, data):
        path.write_text(json.dumps(data, indent=2, default=str))

    def store_fact(self, fact: str):
        if fact not in self._facts:
            self._facts.append(fact)
            self._save(self.facts_file, self._facts)

    def get_facts(self, limit: int = 20) -> list[str]:
        return self._facts[-limit:]

    def store_session_summary(self, session_id: str, goal: str, summary: str):
        self._sessions.append({
            "session_id": session_id,
            "goal": goal,
            "summary": summary,
            "timestamp": time.time(),
        })
        self._sessions = self._sessions[-50:]
        self._save(self.sessions_file, self._sessions)

    def get_recent_sessions(self, limit: int = 5) -> list[dict]:
        return self._sessions[-limit:]

    def get_relevant_context(self, query: str) -> str:
        """Return stored context that may be useful for a new session."""
        lines: list[str] = []
        facts = self.get_facts()
        if facts:
            lines.append("=== KNOWN FACTS FROM PREVIOUS SESSIONS ===")
            for f in facts:
                lines.append(f"  - {f}")

        sessions = self.get_recent_sessions()
        if sessions:
            lines.append("\n=== RECENT SESSION SUMMARIES ===")
            for s in sessions:
                lines.append(f"  [{s['session_id']}] Goal: {s['goal']}")
                lines.append(f"    {s['summary'][:200]}")

        return "\n".join(lines) if lines else ""

    def clear(self):
        """Wipe all persistent memory."""
        self._facts.clear()
        self._sessions.clear()
        self._save(self.facts_file, self._facts)
        self._save(self.sessions_file, self._sessions)
