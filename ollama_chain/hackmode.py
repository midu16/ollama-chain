"""Autonomous penetration testing agent (hack mode).

Runs an adaptive, multi-phase attack loop against a target system:
  1. Reconnaissance  — passive info gathering (WHOIS, DNS, OSINT)
  2. Scanning        — port scanning, service detection, OS fingerprinting
  3. Enumeration     — deep service probing, version detection, directory brute
  4. Vulnerability   — CVE research, exploit search, vuln scanning
  5. Exploitation    — attempt exploits, brute force, injection techniques
  6. Post-exploit    — privilege escalation, lateral movement, persistence

The agent continuously adapts its strategy based on findings from each phase,
researching new techniques and re-planning attacks until access is gained or
the user aborts.

WARNING: Only use against systems you have explicit written authorization to test.
Unauthorized access to computer systems is illegal.
"""

import json
import os
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from .common import (
    ask,
    build_structured_prompt,
    chat_with_retry,
    model_supports_thinking,
    sanitize_messages,
)
from .progress import progress_update
from .memory import MEMORY_DIR, SessionMemory, PersistentMemory

REPORTS_DIR = MEMORY_DIR / "reports"
from .planner import detect_parallel_groups
from .router import route_query, optimize_routing
from .tools import execute_tool_with_retry, format_tool_descriptions, ToolResult

_MAX_CONTEXT_CHARS = 32_000

HACK_PHASES = [
    "reconnaissance",
    "scanning",
    "enumeration",
    "vulnerability_research",
    "exploitation",
    "post_exploitation",
]

_PHASE_DESCRIPTIONS = {
    "reconnaissance": "Passive information gathering — DNS, WHOIS, OSINT, subdomain enumeration",
    "scanning": "Active scanning — port scanning, service detection, OS fingerprinting",
    "enumeration": "Deep service probing — banner grabbing, directory brute forcing, share enumeration",
    "vulnerability_research": "CVE research, exploit database search, vulnerability scanning",
    "exploitation": "Active exploitation — exploit attempts, brute force, injection, shell access",
    "post_exploitation": "Privilege escalation, lateral movement, data exfiltration, persistence",
}

_PHASE_TOOLS = {
    "reconnaissance": [
        ("shell", "whois {target}"),
        ("shell", "dig {target} ANY +noall +answer"),
        ("shell", "dig {target} MX +short"),
        ("shell", "dig {target} NS +short"),
        ("shell", "dig {target} TXT +short"),
        ("shell", "host -t AAAA {target}"),
        ("shell", "nslookup {target}"),
        ("web_search", "{target} site:shodan.io OR site:censys.io"),
        ("web_search", "\"{target}\" inurl:login OR inurl:admin"),
    ],
    "scanning": [
        ("shell", "nmap -sS -sV -O -A -T4 --top-ports 1000 {target}"),
        ("shell", "nmap -sU --top-ports 50 {target}"),
        ("shell", "nmap -sV --version-intensity 5 -p- {target}"),
        ("shell", "nmap --script=vuln {target}"),
        ("shell", "nmap -sC -sV {target}"),
    ],
    "enumeration": [
        ("shell", "nmap --script=http-enum,http-headers,http-methods,http-title -p {ports} {target}"),
        ("shell", "nmap --script=smb-enum-shares,smb-enum-users,smb-os-discovery -p 445 {target}"),
        ("shell", "nmap --script=ftp-anon,ftp-bounce -p 21 {target}"),
        ("shell", "nmap --script=ssh-auth-methods -p 22 {target}"),
        ("shell", "nmap --script=dns-brute {target}"),
        ("shell", "curl -s -I http://{target}"),
        ("shell", "curl -s -I https://{target}"),
    ],
    "vulnerability_research": [
        ("web_search", "{service} {version} CVE exploit"),
        ("web_search", "{service} {version} vulnerability PoC"),
        ("web_search", "{service} {version} remote code execution"),
        ("web_search", "{service} penetration testing techniques"),
        ("shell", "nmap --script=vulners -sV -p {ports} {target}"),
        ("shell", "searchsploit {service} {version} 2>/dev/null || echo 'searchsploit not available'"),
    ],
    "exploitation": [
        ("shell", "hydra -L {userlist} -P {passlist} {target} {service} -t 4 -f 2>/dev/null || echo 'hydra not available'"),
        ("shell", "nmap --script=http-sql-injection -p {ports} {target}"),
        ("shell", "nmap --script=http-shellshock -p {ports} {target}"),
        ("shell", "curl -s 'http://{target}:{port}/../../../../../../etc/passwd' 2>/dev/null"),
        ("shell", "nikto -h http://{target}:{port} -C all 2>/dev/null || echo 'nikto not available'"),
    ],
    "post_exploitation": [
        ("shell", "whoami && id && hostname"),
        ("shell", "uname -a && cat /etc/os-release"),
        ("shell", "cat /etc/passwd"),
        ("shell", "netstat -tlnp || ss -tlnp"),
        ("shell", "find / -perm -4000 -type f 2>/dev/null | head -20"),
    ],
}


# ---------------------------------------------------------------------------
# Hack-mode system prompt
# ---------------------------------------------------------------------------

def _build_hack_system_prompt(
    target: str,
    phase: str,
    session: SessionMemory,
    persistent_ctx: str,
    available_tools: str,
) -> str:
    phase_desc = _PHASE_DESCRIPTIONS.get(phase, phase)
    structured_ctx = session.get_structured_context()
    facts_block = "\n".join(f"  - {f}" for f in session.facts) if session.facts else "None yet"

    return build_structured_prompt(
        sections=[
            ("Role", (
                "You are an expert penetration tester and ethical hacker performing "
                "an authorized security assessment. Your goal is to find and exploit "
                "vulnerabilities in the target system using real tools and commands.\n\n"
                "You MUST:\n"
                "- Execute actual commands through the shell tool — do NOT simulate or hypothesize results\n"
                "- Adapt your approach based on real results from each command\n"
                "- Try multiple attack vectors when one fails\n"
                "- Research new techniques when standard approaches fail\n"
                "- Be thorough, methodical, and persistent\n"
                "- Chain discovered information into increasingly targeted attacks\n"
                "- Never give up — if an attack fails, research alternatives and try again"
            )),
            ("Target", target),
            ("Current Phase", f"{phase}: {phase_desc}"),
            ("Available Tools", available_tools),
            ("Response Format", (
                "Respond with EXACTLY ONE of these blocks:\n\n"
                "1. To execute a command or use a tool:\n"
                "<tool_call>\n"
                '{"name": "shell", "args": {"command": "your command here"}}\n'
                "</tool_call>\n\n"
                "2. To store a discovered fact:\n"
                "<store_fact>\n"
                "The fact to remember (e.g., open port, service version, credential).\n"
                "</store_fact>\n\n"
                "3. To advance to a new attack phase:\n"
                "<phase_advance>\n"
                "next_phase_name\n"
                "</phase_advance>\n\n"
                "4. ONLY when the target is fully compromised:\n"
                "<final_answer>\n"
                "Detailed penetration test report.\n"
                "</final_answer>\n\n"
                "You may include multiple <store_fact> blocks alongside <tool_call>.\n"
                "Do NOT use <final_answer> until you have gained meaningful access or "
                "exhausted all viable attack vectors after extensive attempts."
            )),
            ("Discovered Intelligence", facts_block),
            ("Session State", structured_ctx),
            ("Long-Term Memory", persistent_ctx),
        ],
    )


# ---------------------------------------------------------------------------
# Response parser (extended for hack mode)
# ---------------------------------------------------------------------------

def _parse_hack_response(text: str) -> dict:
    result: dict = {"type": "reasoning", "content": text}

    tool_match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL,
    )
    if tool_match:
        try:
            call = json.loads(tool_match.group(1))
            result = {
                "type": "tool_call",
                "name": call.get("name", ""),
                "args": call.get("args", {}),
                "raw": text,
            }
        except json.JSONDecodeError:
            result = {"type": "malformed_tool_call", "content": tool_match.group(1)}

    answer_match = re.search(
        r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL,
    )
    if answer_match:
        result = {"type": "final_answer", "content": answer_match.group(1)}

    phase_match = re.search(
        r"<phase_advance>\s*(.*?)\s*</phase_advance>", text, re.DOTALL,
    )
    if phase_match:
        next_phase = phase_match.group(1).strip().lower().replace(" ", "_")
        if next_phase in HACK_PHASES:
            result["advance_to"] = next_phase

    facts = re.findall(r"<store_fact>\s*(.*?)\s*</store_fact>", text, re.DOTALL)
    if facts:
        result["facts"] = facts

    return result


# ---------------------------------------------------------------------------
# Adaptive planning — generates attack plans per phase
# ---------------------------------------------------------------------------

def _plan_phase(
    target: str,
    phase: str,
    session: SessionMemory,
    model: str,
) -> list[dict]:
    facts_block = "\n".join(f"  - {f}" for f in session.facts) if session.facts else "None"

    prompt = (
        "/no_think\n"
        "You are an expert penetration tester planning the next attack phase.\n\n"
        f"Target: {target}\n"
        f"Current phase: {phase} — {_PHASE_DESCRIPTIONS.get(phase, '')}\n\n"
        f"Intelligence gathered so far:\n{facts_block}\n\n"
        "Based on the intelligence gathered, create a specific, actionable attack plan "
        "for this phase. Each step should be a real command or action to execute.\n\n"
        "Consider:\n"
        "- What services/ports have been discovered?\n"
        "- What versions are running? Are there known CVEs?\n"
        "- What attack vectors are available based on findings?\n"
        "- What hasn't been tried yet?\n"
        "- What tools are available (nmap, hydra, nikto, curl, searchsploit, gobuster, etc.)?\n\n"
        "Respond with ONLY a JSON array. Example:\n"
        '[{"id": 1, "description": "Scan top 1000 TCP ports", "tool": "shell", "depends_on": []},\n'
        ' {"id": 2, "description": "Search for CVEs for discovered services", "tool": "web_search", "depends_on": [1]}]\n'
    )

    try:
        response = chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"]
        if "<think>" in content:
            end = content.find("</think>")
            if end != -1:
                content = content[end + 8:].strip()

        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
            if isinstance(plan, list):
                for i, step in enumerate(plan):
                    step.setdefault("id", i + 1)
                    step.setdefault("status", "pending")
                    step.setdefault("tool", "shell")
                    step.setdefault("description", f"Step {i + 1}")
                    step.setdefault("depends_on", [])
                return plan
    except Exception as e:
        print(f"[hack] Planning failed ({e}), using default phase plan", file=sys.stderr)

    return _default_phase_plan(target, phase, session)


def _default_phase_plan(
    target: str,
    phase: str,
    session: SessionMemory,
) -> list[dict]:
    """Fallback plan built from phase tool templates."""
    templates = _PHASE_TOOLS.get(phase, [])
    plan = []

    ports = _extract_ports(session.facts)
    services = _extract_services(session.facts)

    for i, (tool, cmd_template) in enumerate(templates, 1):
        cmd = cmd_template.format(
            target=target,
            ports=",".join(ports) if ports else "80,443,22,21,25,53,110,143,3306,5432,8080",
            port=ports[0] if ports else "80",
            service=services[0][0] if services else "http",
            version=services[0][1] if services and len(services[0]) > 1 else "",
            userlist="/usr/share/wordlists/seclists/Usernames/top-usernames-shortlist.txt "
                     "2>/dev/null || echo 'admin\nroot\nuser\ntest' > /tmp/users.txt && echo /tmp/users.txt",
            passlist="/usr/share/wordlists/rockyou.txt "
                     "2>/dev/null || echo 'admin\npassword\n123456\nroot\ntest' > /tmp/pass.txt && echo /tmp/pass.txt",
        )
        desc = cmd if tool == "shell" else cmd_template.format(
            target=target, service="service", version="",
            ports="", port="80", userlist="", passlist="",
        )
        plan.append({
            "id": i,
            "description": desc[:120],
            "tool": tool,
            "args": {"command": cmd} if tool == "shell" else {"query": cmd},
            "depends_on": [],
            "status": "pending",
        })

    return plan


# ---------------------------------------------------------------------------
# Fact extraction for hack mode
# ---------------------------------------------------------------------------

def _extract_ports(facts: list[str]) -> list[str]:
    """Extract open port numbers from discovered facts."""
    ports = []
    for fact in facts:
        fact_lower = fact.lower()
        if "port" in fact_lower or "open" in fact_lower:
            found = re.findall(r'\b(\d{1,5})\b', fact)
            for p in found:
                if 1 <= int(p) <= 65535 and p not in ports:
                    ports.append(p)
    return ports


def _extract_services(facts: list[str]) -> list[tuple[str, str]]:
    """Extract (service_name, version) pairs from discovered facts."""
    services = []
    service_patterns = [
        r'(\w+)\s+(?:version\s+)?(\d+[\.\d]*\S*)',
        r'(\w+)/(\d+[\.\d]*\S*)',
        r'running\s+(\w+)\s+(\d+[\.\d]*\S*)',
    ]
    for fact in facts:
        for pat in service_patterns:
            matches = re.findall(pat, fact, re.IGNORECASE)
            for m in matches:
                if m not in services:
                    services.append(m)
    return services


def _extract_hack_facts(output: str) -> list[str]:
    """Extract security-relevant facts from tool output."""
    facts = []

    port_matches = re.findall(
        r'(\d{1,5})/(?:tcp|udp)\s+open\s+(\S+)(?:\s+(.+))?',
        output,
    )
    for port, service, version in port_matches:
        version = version.strip() if version else ""
        fact = f"Port {port} open: {service}"
        if version:
            fact += f" ({version})"
        facts.append(fact)

    os_patterns = [
        (r'OS details?:\s*(.+)', "Target OS: {}"),
        (r'Running:\s*(.+)', "Target running: {}"),
        (r'OS CPE:\s*(.+)', "OS CPE: {}"),
        (r'Service Info:\s*OS[s]?:\s*(.+?)(?:;|$)', "Service OS info: {}"),
    ]
    for pat, fmt in os_patterns:
        match = re.search(pat, output, re.IGNORECASE)
        if match:
            facts.append(fmt.format(match.group(1).strip()))

    if "login" in output.lower() and ("success" in output.lower() or "valid" in output.lower()):
        cred_match = re.search(
            r'(?:login|user(?:name)?)[:\s]+(\S+).*?(?:password|pass)[:\s]+(\S+)',
            output, re.IGNORECASE,
        )
        if cred_match:
            facts.append(f"Valid credentials: {cred_match.group(1)}:{cred_match.group(2)}")

    vuln_patterns = [
        r'(CVE-\d{4}-\d+)',
        r'VULNERABLE[:\s]+(.+)',
        r'vulnerability[:\s]+(.+)',
    ]
    for pat in vuln_patterns:
        matches = re.findall(pat, output, re.IGNORECASE)
        for m in matches:
            facts.append(f"Vulnerability found: {m.strip()}")

    header_patterns = [
        (r'[Ss]erver:\s*(.+)', "Server header: {}"),
        (r'X-Powered-By:\s*(.+)', "Powered by: {}"),
        (r'X-AspNet-Version:\s*(.+)', "ASP.NET version: {}"),
    ]
    for pat, fmt in header_patterns:
        match = re.search(pat, output)
        if match:
            facts.append(fmt.format(match.group(1).strip()))

    if "host is up" in output.lower():
        latency = re.search(r'\(([0-9.]+)s latency\)', output)
        if latency:
            facts.append(f"Target is up (latency: {latency.group(1)}s)")
        else:
            facts.append("Target is up")

    return facts


# ---------------------------------------------------------------------------
# LLM interaction for hack mode
# ---------------------------------------------------------------------------

def _hack_chat(
    all_models: list[str],
    messages: list[dict],
    thinking: bool = True,
) -> tuple[str, str] | None:
    """Try models strongest-first for hack mode (needs reasoning ability)."""
    for model in reversed(all_models):
        try:
            chat_msgs = list(messages)
            if not thinking and model_supports_thinking(model):
                last = chat_msgs[-1].copy()
                last["content"] = "/no_think\n" + last["content"]
                chat_msgs[-1] = last
            resp = chat_with_retry(model=model, messages=chat_msgs, retries=2)
            text = resp["message"]["content"]
            if "<think>" in text:
                end = text.find("</think>")
                if end != -1:
                    text = text[end + 8:].strip()
            return model, text
        except Exception as e:
            print(
                f"[hack]   {model} unavailable ({e}), trying next...",
                file=sys.stderr,
            )
    return None


# ---------------------------------------------------------------------------
# Adaptive technique research
# ---------------------------------------------------------------------------

def _research_techniques(
    target: str,
    phase: str,
    session: SessionMemory,
    model: str,
) -> list[dict]:
    """When attacks fail, research and propose new techniques."""
    failed_approaches = []
    for tr in session.tool_results:
        if not tr["success"]:
            failed_approaches.append(
                f"  - {tr['tool']}: {tr.get('args', {}).get('command', tr.get('args', {}).get('query', ''))[:80]} → {tr['output'][:60]}"
            )

    facts_block = "\n".join(f"  - {f}" for f in session.facts) if session.facts else "None"
    failed_block = "\n".join(failed_approaches[-10:]) if failed_approaches else "None"

    prompt = (
        "/no_think\n"
        "You are a penetration testing expert. The current attack approaches have failed. "
        "Research and propose NEW techniques that haven't been tried yet.\n\n"
        f"Target: {target}\n"
        f"Phase: {phase}\n\n"
        f"Known intelligence:\n{facts_block}\n\n"
        f"Failed approaches:\n{failed_block}\n\n"
        "Propose 3-5 NEW specific commands or techniques to try. Think creatively:\n"
        "- Alternative tools (different scanners, different brute forcers)\n"
        "- Different protocols or attack vectors\n"
        "- Encoding/evasion techniques if defenses are detected\n"
        "- Less common services or misconfigurations to check\n"
        "- Social engineering vectors or information leakage points\n\n"
        "Respond with ONLY a JSON array of steps."
    )

    try:
        response = chat_with_retry(
            model=model, messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"]
        if "<think>" in content:
            end = content.find("</think>")
            if end != -1:
                content = content[end + 8:].strip()

        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
            if isinstance(plan, list):
                for i, step in enumerate(plan):
                    step.setdefault("id", i + 1)
                    step.setdefault("status", "pending")
                    step.setdefault("tool", "shell")
                    step.setdefault("description", f"Step {i + 1}")
                    step.setdefault("depends_on", [])
                return plan
    except Exception as e:
        print(f"[hack] Technique research failed ({e})", file=sys.stderr)

    return []


# ---------------------------------------------------------------------------
# Phase transition logic
# ---------------------------------------------------------------------------

def _should_advance_phase(
    phase: str,
    session: SessionMemory,
    consecutive_failures: int,
    phase_iterations: int,
) -> str | None:
    """Determine if we should advance to the next phase.
    
    Returns next phase name or None to stay in current phase.
    """
    phase_idx = HACK_PHASES.index(phase) if phase in HACK_PHASES else 0

    has_ports = any("port" in f.lower() and "open" in f.lower() for f in session.facts)
    has_services = any(
        any(kw in f.lower() for kw in ("running", "version", "server header"))
        for f in session.facts
    )
    has_vulns = any(
        any(kw in f.lower() for kw in ("cve", "vulnerab", "exploit"))
        for f in session.facts
    )
    has_creds = any("credential" in f.lower() or "password" in f.lower() for f in session.facts)

    if phase == "reconnaissance" and phase_iterations >= 3:
        return "scanning"

    if phase == "scanning" and has_ports and phase_iterations >= 2:
        return "enumeration"

    if phase == "enumeration" and has_services and phase_iterations >= 2:
        return "vulnerability_research"

    if phase == "vulnerability_research" and (has_vulns or phase_iterations >= 5):
        return "exploitation"

    if phase == "exploitation" and has_creds:
        return "post_exploitation"

    if consecutive_failures >= 5 and phase_idx < len(HACK_PHASES) - 1:
        return HACK_PHASES[phase_idx + 1]

    if phase_iterations >= 15 and phase_idx < len(HACK_PHASES) - 1:
        return HACK_PHASES[phase_idx + 1]

    return None


# ---------------------------------------------------------------------------
# Main hack loop
# ---------------------------------------------------------------------------

def run_hack(
    target: str,
    all_models: list[str],
    *,
    web_search: bool = True,
    fast: str | None = None,
    max_iterations: int = 0,
    initial_phase: str = "reconnaissance",
) -> str:
    """Run the autonomous penetration testing agent.
    
    max_iterations=0 means unlimited (runs until target is compromised or user aborts).
    """
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    session_id = uuid.uuid4().hex[:8]

    session = SessionMemory(session_id=session_id, goal=f"Penetration test: {target}")
    persistent = PersistentMemory()
    persistent_ctx = persistent.get_relevant_context(f"pentest {target}")

    current_phase = initial_phase
    iteration = 0
    phase_iterations = 0
    consecutive_failures = 0
    research_count = 0
    max_research_per_phase = 3

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[hack] Session {session_id}", file=sys.stderr)
    print(f"[hack] Target: {target}", file=sys.stderr)
    print(f"[hack] Models: {' → '.join(all_models)}", file=sys.stderr)
    print(f"[hack] Max iterations: {'unlimited' if max_iterations == 0 else max_iterations}", file=sys.stderr)
    print(f"[hack] WARNING: Ensure you have authorization to test this target!", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    session.add_fact(f"Target: {target}")

    while max_iterations == 0 or iteration < max_iterations:
        iteration += 1
        phase_iterations += 1

        phase_desc = _PHASE_DESCRIPTIONS.get(current_phase, current_phase)
        print(
            f"\n[hack] === Phase: {current_phase.upper()} (iter {iteration}) ===",
            file=sys.stderr,
        )
        print(f"[hack] {phase_desc}", file=sys.stderr)

        pct = min(95, 5 + (iteration * 2))
        progress_update(pct, f"[{current_phase}] iteration {iteration}...")

        # -- Generate or refresh attack plan for current phase --
        if phase_iterations == 1 or not session.pending_steps():
            print(f"[hack] Planning {current_phase} phase...", file=sys.stderr)
            plan = _plan_phase(target, current_phase, session, fast_name)
            session.plan = plan

            print(f"[hack] Plan ({len(plan)} steps):", file=sys.stderr)
            for step in plan:
                print(f"  {step['id']}. {step['description'][:80]}", file=sys.stderr)

        # -- Execute next pending step via LLM --
        pending = session.pending_steps()
        if not pending:
            if consecutive_failures >= 3 and research_count < max_research_per_phase:
                print(
                    f"[hack] All steps exhausted with failures — researching new techniques...",
                    file=sys.stderr,
                )
                new_steps = _research_techniques(target, current_phase, session, strong_name)
                if new_steps:
                    max_id = max((s["id"] for s in session.plan), default=0)
                    for i, step in enumerate(new_steps):
                        step["id"] = max_id + i + 1
                    session.plan.extend(new_steps)
                    research_count += 1
                    print(
                        f"[hack] Added {len(new_steps)} new techniques (research round {research_count})",
                        file=sys.stderr,
                    )
                    continue
                else:
                    print("[hack] No new techniques found", file=sys.stderr)

            next_phase = _should_advance_phase(
                current_phase, session, consecutive_failures, phase_iterations,
            )
            if next_phase:
                print(
                    f"\n[hack] >>> Advancing: {current_phase} → {next_phase}",
                    file=sys.stderr,
                )
                current_phase = next_phase
                phase_iterations = 0
                consecutive_failures = 0
                research_count = 0
                continue

            if current_phase == HACK_PHASES[-1]:
                print(
                    "[hack] All phases exhausted — generating final report",
                    file=sys.stderr,
                )
                break

            phase_idx = HACK_PHASES.index(current_phase) if current_phase in HACK_PHASES else 0
            current_phase = HACK_PHASES[min(phase_idx + 1, len(HACK_PHASES) - 1)]
            phase_iterations = 0
            consecutive_failures = 0
            research_count = 0
            continue

        step = pending[0]
        step["status"] = "in_progress"

        print(
            f"[hack] Step {step['id']}: {step['description'][:80]}",
            file=sys.stderr,
        )

        tool_docs = format_tool_descriptions()
        system_prompt = _build_hack_system_prompt(
            target, current_phase, session, persistent_ctx, tool_docs,
        )

        step_prompt = (
            f"Execute this attack step: {step['description']}\n"
            f"Suggested tool: {step.get('tool', 'shell')}\n\n"
            f"If the step has pre-built args, you can use them directly. "
            f"Otherwise, construct the most effective command based on current intelligence.\n"
            f"After getting results, store any interesting findings as facts."
        )

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        total_chars = len(system_prompt)

        for entry in session.history[-8:]:
            role = entry.role if entry.role in ("user", "assistant") else "assistant"
            content = entry.content
            if total_chars + len(content) > _MAX_CONTEXT_CHARS:
                remaining = max(200, _MAX_CONTEXT_CHARS - total_chars)
                content = content[:remaining] + "\n... [truncated]"
            total_chars += len(content)
            messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": step_prompt})
        messages = sanitize_messages(messages)

        chat_result = _hack_chat(all_models, messages, thinking=True)

        if chat_result is not None:
            model_used, raw = chat_result
            print(f"[hack]   Model: {model_used}", file=sys.stderr)
            parsed = _parse_hack_response(raw)

            for fact in parsed.get("facts", []):
                session.add_fact(fact)
                persistent.store_fact(fact)
                print(f"[hack]   Stored: {fact[:70]}", file=sys.stderr)

            if "advance_to" in parsed:
                next_phase = parsed["advance_to"]
                print(
                    f"\n[hack] >>> LLM requested phase advance: {current_phase} → {next_phase}",
                    file=sys.stderr,
                )
                current_phase = next_phase
                phase_iterations = 0
                consecutive_failures = 0
                research_count = 0
                step["status"] = "completed"
                continue

            if parsed["type"] == "tool_call":
                tool_name = parsed["name"]
                tool_args = parsed["args"]

                if tool_name == "shell":
                    cmd = tool_args.get("command", "")
                    timeout = int(tool_args.get("timeout", 120))
                    tool_args["timeout"] = max(timeout, 60)

                print(
                    f"[hack]   Exec: {tool_name}({json.dumps(tool_args)[:100]})",
                    file=sys.stderr,
                )

                result = execute_tool_with_retry(tool_name, tool_args)
                _record_hack_result(result, tool_name, tool_args, step, session, persistent)

                if result.success:
                    consecutive_failures = 0
                    auto_facts = _extract_hack_facts(result.output)
                    for fact in auto_facts:
                        if fact not in session.facts:
                            session.add_fact(fact)
                            persistent.store_fact(fact)
                            print(f"[hack]   Intel: {fact[:70]}", file=sys.stderr)
                else:
                    consecutive_failures += 1
                    print(
                        f"[hack]   Failed ({consecutive_failures} consecutive)",
                        file=sys.stderr,
                    )

            elif parsed["type"] == "final_answer":
                session.add("assistant", parsed["content"])
                report_path = _save_report(target, session, parsed["content"])
                _finalize_hack_session(session, persistent, report_path)
                return parsed["content"]

            else:
                session.add("assistant", raw[:500])
                step["status"] = "completed"

        else:
            # All models failed — try direct execution from step hints
            result = _auto_execute_hack_step(step, target, session)
            if result is not None:
                _record_hack_result(result, step.get("tool", "shell"), {}, step, session, persistent)
                if result.success:
                    consecutive_failures = 0
                    auto_facts = _extract_hack_facts(result.output)
                    for fact in auto_facts:
                        if fact not in session.facts:
                            session.add_fact(fact)
                            persistent.store_fact(fact)
                else:
                    consecutive_failures += 1
            else:
                step["status"] = "failed"
                consecutive_failures += 1

    # -- Final report synthesis --
    return _synthesize_hack_report(target, session, all_models)


# ---------------------------------------------------------------------------
# Auto-execution fallback for hack steps
# ---------------------------------------------------------------------------

def _auto_execute_hack_step(
    step: dict,
    target: str,
    session: SessionMemory,
) -> ToolResult | None:
    """Execute a hack step directly from its hints when LLMs are unavailable."""
    tool = step.get("tool", "shell")
    args = step.get("args")

    if args:
        if tool == "shell" and "command" in args:
            return execute_tool_with_retry("shell", {"command": args["command"], "timeout": 120})
        if tool in ("web_search", "web_search_news") and "query" in args:
            return execute_tool_with_retry(tool, {"query": args["query"]})

    desc = step.get("description", "")
    if tool == "shell":
        cmds = re.findall(r'`([^`]+)`', desc)
        if cmds:
            return execute_tool_with_retry("shell", {"command": cmds[0], "timeout": 120})
        quoted = re.findall(r'"([^"]+)"', desc)
        if quoted:
            return execute_tool_with_retry("shell", {"command": quoted[0], "timeout": 120})

    return None


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------

def _record_hack_result(
    result: ToolResult,
    tool_name: str,
    tool_args: dict,
    step: dict,
    session: SessionMemory,
    persistent: PersistentMemory,
):
    truncated = result.output
    if len(truncated) > 4000:
        truncated = truncated[:4000] + "\n... [truncated]"

    status = "OK" if result.success else "FAILED"
    timing = f" [{result.duration_ms:.0f}ms]" if result.duration_ms else ""
    print(
        f"[hack]   Result ({status}{timing}): "
        f"{truncated[:150].replace(chr(10), ' ')}",
        file=sys.stderr,
    )

    session.add("assistant", f"Executed {tool_name}: {json.dumps(tool_args)[:100]}")
    session.add_tool_output(tool_name, truncated, step["id"], result.success)
    session.tool_results.append({
        "step": step["id"],
        "tool": tool_name,
        "args": tool_args,
        "success": result.success,
        "output": truncated,
        "duration_ms": result.duration_ms,
        "error_detail": result.error_detail,
    })

    step["status"] = "completed" if result.success else "failed"


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def _synthesize_hack_report(
    target: str,
    session: SessionMemory,
    all_models: list[str],
) -> str:
    """Generate a comprehensive penetration test report from all findings."""
    strong = all_models[-1]

    facts_block = "\n".join(f"  - {f}" for f in session.facts) if session.facts else "None"

    collected_parts: list[str] = []
    collected_size = 0
    for tr in session.tool_results:
        part = f"[Step {tr['step']} — {tr['tool']}]\n{tr['output']}"
        if collected_size + len(part) > _MAX_CONTEXT_CHARS:
            remaining = max(200, _MAX_CONTEXT_CHARS - collected_size)
            part = part[:remaining] + "\n... [truncated]"
            collected_parts.append(part)
            break
        collected_parts.append(part)
        collected_size += len(part)
    collected = "\n\n".join(collected_parts)

    completed = session.completed_step_count()
    failed = session.failed_step_count()
    total = len(session.plan)

    prompt = build_structured_prompt(
        sections=[
            ("Role", (
                "You are a senior penetration tester writing a professional "
                "penetration test report. Based on the actual tool outputs and "
                "findings below, produce a comprehensive security assessment."
            )),
            ("Target", target),
            ("Engagement Summary", (
                f"Total steps executed: {completed + failed}/{total}\n"
                f"Successful: {completed}\n"
                f"Failed: {failed}\n"
                f"Facts discovered: {len(session.facts)}"
            )),
            ("Discovered Intelligence", facts_block),
            ("Tool Outputs", collected),
        ],
        instructions=(
            "Structure your report as:\n"
            "1. Executive Summary\n"
            "2. Scope & Methodology\n"
            "3. Findings (each with severity: Critical/High/Medium/Low/Info)\n"
            "   - For each finding: description, evidence, impact, remediation\n"
            "4. Open Ports & Services\n"
            "5. Vulnerabilities Discovered\n"
            "6. Exploitation Results\n"
            "7. Recommendations (prioritized)\n"
            "8. Appendix — Raw Tool Output Summary\n\n"
            "Be specific and cite actual evidence from the tool outputs. "
            "Do not invent or assume findings — report only what was discovered."
        ),
    )

    print(
        f"\n[hack] Generating final pentest report with {strong}...",
        file=sys.stderr,
    )

    try:
        report = ask(prompt, model=strong, thinking=True, temperature=0.3)
    except Exception:
        for model in reversed(all_models):
            try:
                report = ask(prompt, model=model, thinking=True, temperature=0.3)
                break
            except Exception:
                continue
        else:
            report = (
                f"# Penetration Test Report — {target}\n\n"
                f"## Findings\n\n"
                + "\n".join(f"- {f}" for f in session.facts)
            )

    report_path = _save_report(target, session, report)
    _finalize_hack_session(session, persistent, report_path)
    return report


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------

def _save_report(
    target: str,
    session: SessionMemory,
    report: str,
) -> Path:
    """Save the full pentest report and raw tool outputs to disk."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_target = re.sub(r'[^\w.\-]', '_', target)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"pentest_{safe_target}_{session.session_id}_{ts}"

    report_path = REPORTS_DIR / f"{base_name}.md"
    raw_path = REPORTS_DIR / f"{base_name}_raw.json"

    _atomic_write(report_path, report)

    raw_data = {
        "session_id": session.session_id,
        "target": target,
        "timestamp": datetime.now().isoformat(),
        "facts": list(session.facts),
        "tool_results": list(session.tool_results),
        "plan": list(session.plan),
    }
    _atomic_write(raw_path, json.dumps(raw_data, indent=2, default=str))

    print(f"[hack] Report saved: {report_path}", file=sys.stderr)
    print(f"[hack] Raw data saved: {raw_path}", file=sys.stderr)

    return report_path


def _atomic_write(path: Path, content: str):
    """Write content to a file atomically via temp + rename."""
    fd = None
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd = None
        os.replace(tmp, str(path))
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def _finalize_hack_session(
    session: SessionMemory,
    persistent: PersistentMemory,
    report_path: Path | None = None,
):
    completed = session.completed_step_count()
    total = len(session.plan)
    report_note = f" Report: {report_path}" if report_path else ""
    summary = (
        f"Pentest completed: {completed}/{total} steps. "
        f"Tools used: {len(session.tool_results)}. "
        f"Intel gathered: {len(session.facts)} facts.{report_note}"
    )
    persistent.store_session_summary(session.session_id, session.goal, summary)
    print("[hack] Session saved to memory.", file=sys.stderr)
    session.clear()
