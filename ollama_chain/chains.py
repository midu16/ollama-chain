"""Chaining modes — orchestrate all local models with web search and source citation."""

import sys

import ollama as ollama_client

from .pcap import analyze_pcap, format_analysis
from .search import search_for_query

SOURCE_GUIDANCE = (
    "When making factual claims, reference authoritative public sources such as: "
    "official documentation (e.g. Red Hat, kernel.org, Mozilla MDN, Microsoft Docs), "
    "standards bodies (IEEE, IETF RFCs, ISO, W3C, NIST), "
    "peer-reviewed publications, and official project repositories. "
    "Format references as: [Source: <name>, <url or identifier>]. "
    "If you are uncertain about a claim and cannot back it with a source, say so explicitly."
)


def ask(prompt: str, model: str, thinking: bool = False) -> str:
    """Send a prompt to an Ollama model and return the response text."""
    if not thinking:
        prompt = "/no_think\n" + prompt
    response = ollama_client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response["message"]["content"]
    if "<think>" in content:
        end = content.find("</think>")
        if end != -1:
            content = content[end + len("</think>"):].strip()
    return content


def _enrich_with_search(query: str, fast: str, web_search: bool) -> str:
    """If web search is enabled, fetch results and return a context block."""
    if not web_search:
        return ""
    results = search_for_query(query, fast)
    if not results:
        return ""
    return (
        f"\n\n=== WEB SEARCH RESULTS ===\n"
        f"Use these as reference to improve accuracy. Cite sources when relevant.\n\n"
        f"{results}"
    )


def _inject_search_context(prompt: str, search_context: str) -> str:
    """Append search context to a prompt if available."""
    if search_context:
        return prompt + search_context
    return prompt


# ---------------------------------------------------------------------------
# CASCADE — default mode, chains through ALL models smallest → largest
# ---------------------------------------------------------------------------

def chain_cascade(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """
    Progressive refinement through every available model:
      1. Web search gathers context
      2. Smallest model produces initial draft
      3. Each subsequent model reviews, corrects, and refines
      4. Largest model produces the final authoritative answer
    """
    fast_name = fast or all_models[0]
    search_ctx = _enrich_with_search(query, fast_name, web_search)
    n = len(all_models)

    # --- Stage 1: first model drafts ---
    print(f"[cascade 1/{n}] Drafting with {all_models[0]}...", file=sys.stderr)
    draft_prompt = (
        f"{SOURCE_GUIDANCE}\n\n"
        f"Answer the following question thoroughly and accurately.\n\n"
        f"Question: {query}"
    )
    current_answer = ask(
        _inject_search_context(draft_prompt, search_ctx),
        model=all_models[0],
    )

    if n == 1:
        return current_answer

    # --- Stages 2..N-1: intermediate models review and refine ---
    for i, model in enumerate(all_models[1:-1], start=2):
        print(f"[cascade {i}/{n}] Reviewing with {model}...", file=sys.stderr)
        current_answer = ask(
            f"You are a reviewer improving the accuracy of an answer.\n"
            f"{SOURCE_GUIDANCE}\n\n"
            f"Original question: {query}\n\n"
            f"Current answer:\n{current_answer}\n\n"
            f"Instructions:\n"
            f"- Fix any factual errors\n"
            f"- Add missing important information\n"
            f"- Strengthen source references (add [Source: ...] citations)\n"
            f"- Remove unsupported speculation\n"
            f"- Improve clarity and structure\n"
            f"- Preserve what is already correct\n"
            f"Output ONLY the improved answer, not a commentary on the changes."
            + (f"\n{search_ctx}" if search_ctx else ""),
            model=model,
            thinking=True,
        )

    # --- Final stage: strongest model produces authoritative answer ---
    print(f"[cascade {n}/{n}] Final answer with {all_models[-1]}...", file=sys.stderr)
    return ask(
        f"You are the final reviewer producing the definitive answer.\n"
        f"{SOURCE_GUIDANCE}\n\n"
        f"Original question: {query}\n\n"
        f"Draft answer (refined by {n - 1} model(s)):\n{current_answer}\n\n"
        f"Instructions:\n"
        f"- Verify all factual claims and correct any remaining errors\n"
        f"- Ensure every key claim has a [Source: ...] reference to an authoritative source "
        f"(official docs, standards bodies like IEEE/IETF/ISO/NIST/W3C, "
        f"vendor docs like Red Hat/kernel.org/MDN, or peer-reviewed work)\n"
        f"- If a claim cannot be verified, mark it as unverified\n"
        f"- Produce a clean, well-structured final answer\n"
        f"- Do NOT include meta-commentary about the review process\n"
        f"Output ONLY the final authoritative answer."
        + (f"\n{search_ctx}" if search_ctx else ""),
        model=all_models[-1],
        thinking=True,
    )


# ---------------------------------------------------------------------------
# LEGACY MODES — still available, now with source guidance
# ---------------------------------------------------------------------------

def chain_route(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Fast model classifies complexity, routes to fast or strong."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    search_ctx = _enrich_with_search(query, fast_name, web_search)

    print(f"[route 1/2] Classifying with {fast_name}...", file=sys.stderr)
    verdict = ask(
        f"Rate the complexity of answering this query on a scale of 1-5. "
        f"Reply with ONLY a single digit.\n\nQuery: {query}",
        model=fast_name,
    )

    try:
        score = int("".join(c for c in verdict if c.isdigit())[:1])
    except (ValueError, IndexError):
        score = 5

    print(f"[route 1/2] Complexity: {score}/5", file=sys.stderr)

    enriched = _inject_search_context(
        f"{SOURCE_GUIDANCE}\n\n{query}", search_ctx,
    )

    if score <= 3:
        print(f"[route 2/2] Answering with {fast_name} (simple)...", file=sys.stderr)
        return ask(enriched, model=fast_name)
    else:
        print(f"[route 2/2] Answering with {strong_name} (complex)...", file=sys.stderr)
        return ask(enriched, model=strong_name, thinking=True)


def chain_pipeline(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Fast model extracts/classifies, strong model reasons with search context."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    search_ctx = _enrich_with_search(query, fast_name, web_search)

    print(f"[pipeline 1/3] Extracting key points with {fast_name}...", file=sys.stderr)
    key_points = ask(
        f"Extract the key points and core question from this. Be concise:\n\n{query}",
        model=fast_name,
    )

    print(f"[pipeline 2/3] Classifying domain with {fast_name}...", file=sys.stderr)
    domain = ask(
        f"What domain/field is this about? Reply in 1-3 words:\n\n{key_points}",
        model=fast_name,
    )

    print(f"[pipeline 3/3] Deep analysis with {strong_name} ({domain.strip()})...", file=sys.stderr)
    prompt = (
        f"You are an expert in {domain}.\n"
        f"{SOURCE_GUIDANCE}\n\n"
        f"Key points:\n{key_points}\n\n"
        f"Original query: {query}\n\n"
        f"Provide a thorough, well-structured answer."
    )
    return ask(_inject_search_context(prompt, search_ctx), model=strong_name, thinking=True)


def chain_verify(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Fast model drafts, strong model verifies with search context for fact-checking."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    search_ctx = _enrich_with_search(query, fast_name, web_search)

    print(f"[verify 1/2] Drafting with {fast_name}...", file=sys.stderr)
    draft = ask(f"{SOURCE_GUIDANCE}\n\n{query}", model=fast_name)

    print(f"[verify 2/2] Verifying with {strong_name}...", file=sys.stderr)
    prompt = (
        f"Another model answered the following question. "
        f"Verify the answer for correctness, fix any errors, and improve it. "
        f"Ensure all claims are backed by authoritative sources.\n"
        f"{SOURCE_GUIDANCE}\n\n"
        f"Question: {query}\n\n"
        f"Draft answer:\n{draft}"
    )
    return ask(_inject_search_context(prompt, search_ctx), model=strong_name, thinking=True)


def chain_consensus(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """All models answer independently, strongest merges the best parts."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    search_ctx = _enrich_with_search(query, fast_name, web_search)

    sourced_query = f"{SOURCE_GUIDANCE}\n\n{query}"
    answers = []
    for i, model in enumerate(all_models, 1):
        print(f"[consensus {i}/{len(all_models) + 1}] Answer from {model}...", file=sys.stderr)
        use_thinking = (model == strong_name)
        answers.append((model, ask(sourced_query, model=model, thinking=use_thinking)))

    answers_block = "\n\n".join(
        f"=== Answer from {name} ===\n{answer}"
        for name, answer in answers
    )

    print(
        f"[consensus {len(all_models) + 1}/{len(all_models) + 1}] "
        f"Merging with {strong_name}...",
        file=sys.stderr,
    )
    prompt = (
        f"{len(all_models)} models answered the same question independently. "
        f"Combine the best parts of all answers into a single, accurate, "
        f"well-structured response. Resolve contradictions by picking the "
        f"version best supported by authoritative sources.\n"
        f"{SOURCE_GUIDANCE}\n\n"
        f"Question: {query}\n\n"
        f"{answers_block}"
    )
    return ask(_inject_search_context(prompt, search_ctx), model=strong_name, thinking=True)


def chain_search(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Search-first mode: always searches, strongest model synthesizes."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    results = search_for_query(query, fast_name)

    if not results:
        print("[search] No results found, falling back to strong model...", file=sys.stderr)
        return ask(f"{SOURCE_GUIDANCE}\n\n{query}", model=strong_name, thinking=True)

    print(f"[search] Synthesizing answer with {strong_name}...", file=sys.stderr)
    return ask(
        f"Answer the following question using the web search results below. "
        f"Be accurate, cite sources by number and include their URLs. "
        f"Prioritize authoritative sources (official documentation, standards bodies "
        f"like IEEE/IETF/ISO/NIST/W3C, vendor docs like Red Hat/kernel.org/MDN). "
        f"Clearly state if the search results don't fully answer the question.\n"
        f"{SOURCE_GUIDANCE}\n\n"
        f"Question: {query}\n\n"
        f"=== WEB SEARCH RESULTS ===\n{results}",
        model=strong_name,
        thinking=True,
    )


def chain_fast(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Direct to fast model, optionally with search context."""
    fast_name = fast or all_models[0]
    search_ctx = _enrich_with_search(query, fast_name, web_search)
    print(f"[fast 1/1] {fast_name}...", file=sys.stderr)
    return ask(
        _inject_search_context(f"{SOURCE_GUIDANCE}\n\n{query}", search_ctx),
        model=fast_name,
    )


def chain_strong(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Direct to strong model, optionally with search context."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    search_ctx = _enrich_with_search(query, fast_name, web_search)
    print(f"[strong 1/1] {strong_name}...", file=sys.stderr)
    return ask(
        _inject_search_context(f"{SOURCE_GUIDANCE}\n\n{query}", search_ctx),
        model=strong_name,
        thinking=True,
    )


# ---------------------------------------------------------------------------
# PCAP — uses fast/strong from the full list
# ---------------------------------------------------------------------------

def chain_pcap(
    filepath: str, all_models: list[str],
    query: str | None = None, *, web_search: bool = True, fast: str | None = None,
) -> str:
    """
    Analyze a .pcap file through a multi-stage pipeline:
      1. Parse the pcap and produce structured stats (scapy)
      2. Fast model summarizes findings and flags anomalies
      3. Strong model produces a detailed expert report with references
    """
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]

    print(f"[pcap 1/4] Parsing {filepath} with scapy...", file=sys.stderr)
    analysis = analyze_pcap(filepath)
    report = format_analysis(analysis)
    print(
        f"[pcap 1/4] Parsed {analysis.total_packets} packets, "
        f"{len(analysis.errors)} errors detected",
        file=sys.stderr,
    )

    search_ctx = ""
    if web_search and analysis.errors:
        error_types = set()
        for e in analysis.errors[:10]:
            if "TCP RST" in e:
                error_types.add("TCP RST connection reset troubleshooting")
            elif "zero window" in e:
                error_types.add("TCP zero window buffer full remediation")
            elif "ICMP Destination Unreachable" in e:
                error_types.add("ICMP destination unreachable causes")
            elif "DNS error" in e:
                error_types.add("DNS NXDomain ServFail troubleshooting")
            elif "TTL" in e:
                error_types.add("IP TTL expired routing loop")
        if error_types:
            search_query = " ".join(list(error_types)[:2])
            search_ctx = _enrich_with_search(search_query, fast_name, True)

    print(f"[pcap 2/4] Summarizing with {fast_name}...", file=sys.stderr)
    summary = ask(
        f"You are a network analyst. Summarize this packet capture analysis. "
        f"Highlight the most important findings, anomalies, and potential issues. "
        f"Be concise but thorough.\n\n{report}",
        model=fast_name,
    )

    print(f"[pcap 3/4] Error analysis with {fast_name}...", file=sys.stderr)
    if analysis.errors or analysis.warnings:
        error_details = "\n".join(analysis.errors + analysis.warnings)
        error_section = ask(
            f"You are a network security analyst. Analyze these network errors and warnings. "
            f"Classify each by severity (critical/high/medium/low). "
            f"Explain what each error likely means and suggest remediation. "
            f"Reference relevant RFCs and standards where applicable.\n\n"
            f"Errors and warnings:\n{error_details}\n\n"
            f"Full capture context:\n{report}",
            model=fast_name,
        )
    else:
        error_section = "No errors or warnings were detected in the capture."

    user_context = ""
    if query:
        user_context = (
            f"\n\nThe user specifically asked: {query}\n"
            f"Make sure to address their question directly.\n"
        )

    print(f"[pcap 4/4] Expert report with {strong_name}...", file=sys.stderr)
    prompt = (
        f"You are a senior network engineer and security analyst. "
        f"Produce a comprehensive report for this packet capture.\n"
        f"{SOURCE_GUIDANCE}\n"
        f"Reference relevant IETF RFCs (e.g. RFC 793 for TCP, RFC 791 for IP, "
        f"RFC 1035 for DNS) when discussing protocol behavior and errors.\n\n"
        f"=== RAW ANALYSIS ===\n{report}\n\n"
        f"=== SUMMARY ===\n{summary}\n\n"
        f"=== ERROR ANALYSIS ===\n{error_section}\n"
        f"{user_context}\n"
        f"Structure your report with these sections:\n"
        f"1. Executive Summary\n"
        f"2. Traffic Overview & Workflow\n"
        f"3. Protocol Breakdown\n"
        f"4. Errors & Anomalies (with severity and remediation)\n"
        f"5. Security Observations\n"
        f"6. Recommendations\n"
    )
    return ask(_inject_search_context(prompt, search_ctx), model=strong_name, thinking=True)


CHAINS = {
    "cascade": chain_cascade,
    "route": chain_route,
    "pipeline": chain_pipeline,
    "verify": chain_verify,
    "consensus": chain_consensus,
    "search": chain_search,
    "fast": chain_fast,
    "strong": chain_strong,
}
