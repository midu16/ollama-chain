"""Chaining modes — orchestrate all local models with web search and source citation."""

import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from .common import SOURCE_GUIDANCE, ask, model_supports_thinking
from .pcap import analyze_pcap, format_analysis as format_pcap_analysis
from .k8s import analyze_cluster, format_analysis as format_k8s_analysis
from .progress import progress_update
from .router import build_fallback_chain, route_query
from .search import search_for_query

CLI_ONLY_MODES = frozenset({"pcap", "k8s"})

# Temperature presets for accuracy-critical stages
_TEMP_REVIEW = 0.4
_TEMP_FINAL = 0.3

_SEARCH_ENRICH_TIMEOUT = 45  # hard ceiling for _enrich_with_search


def _enrich_with_search(query: str, fast: str, web_search: bool) -> str:
    """If web search is enabled, fetch results and return a context block.

    Wrapped in a hard timeout so the answer pipeline is never blocked
    indefinitely by search failures.
    """
    if not web_search:
        return ""
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(search_for_query, query, fast)
            results = future.result(timeout=_SEARCH_ENRICH_TIMEOUT)
    except TimeoutError:
        print(
            f"[search] Search pipeline timed out after {_SEARCH_ENRICH_TIMEOUT}s, "
            f"proceeding without search results",
            file=sys.stderr,
        )
        return ""
    except Exception as e:
        print(
            f"[search] Search pipeline failed ({e}), proceeding without search results",
            file=sys.stderr,
        )
        return ""
    if not results:
        return ""
    return (
        f"\n\n=== SEARCH RESULTS (Web, GitHub, Stack Overflow, Docs) ===\n"
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
    complexity: str | None = None,
) -> str:
    """
    Progressive refinement through every available model:
      1. Web search gathers context
      2. Smallest model produces initial draft
      3. Each subsequent model reviews, corrects, and refines
      4. Largest model produces the final authoritative answer

    Thinking is engaged selectively based on *complexity*:
      - simple  → no thinking anywhere (speed priority)
      - moderate → thinking on final answer only
      - complex / None → thinking on review + final stages

    Error handling: if an intermediate model fails (all retries exhausted),
    it is skipped and the cascade continues.  If the draft or final model
    fails, the system falls back to the next available model.
    """
    fast_name = fast or all_models[0]
    n = len(all_models)
    _pct_step = 85.0 / max(n, 1)

    progress_update(2, "Searching web, GitHub, StackOverflow, Docs..." if web_search else "Preparing...")
    search_ctx = _enrich_with_search(query, fast_name, web_search)
    skipped: list[str] = []

    review_think = complexity not in ("simple", "moderate")
    final_think = complexity != "simple"

    # --- Stage 1: first model drafts (never thinks — speed matters) ---
    draft_prompt = (
        f"{SOURCE_GUIDANCE}\n\n"
        f"Answer the following question thoroughly and accurately.\n\n"
        f"Question: {query}"
    )
    enriched_draft = _inject_search_context(draft_prompt, search_ctx)

    current_answer: str | None = None
    for draft_model in all_models:
        progress_update(10, f"Drafting with {draft_model}...")
        print(f"[cascade 1/{n}] Drafting with {draft_model}...", file=sys.stderr)
        try:
            current_answer = ask(enriched_draft, model=draft_model)
            break
        except Exception as e:
            print(
                f"[cascade] {draft_model} failed during draft ({e}), "
                f"trying fallback...",
                file=sys.stderr,
            )
            skipped.append(draft_model)
    if current_answer is None:
        raise RuntimeError(
            "All models failed during cascade draft stage"
        )

    if n == 1 or len(skipped) == n - 1:
        if skipped:
            print(
                f"[cascade] Skipped models: {skipped}",
                file=sys.stderr,
            )
        return current_answer

    # --- Stages 2..N-1: intermediate models review and refine ---
    review_models = [
        m for m in all_models[1:-1] if m not in skipped
    ]
    think_label = " +think" if review_think else ""
    for i, model in enumerate(review_models, start=2):
        _idx = all_models.index(model) if model in all_models else i - 1
        progress_update(10 + _pct_step * _idx, f"Reviewing with {model}{think_label}...")
        print(
            f"[cascade {i}/{n}] Reviewing with {model}{think_label}...",
            file=sys.stderr,
        )
        try:
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
                thinking=review_think,
                temperature=_TEMP_REVIEW if review_think else None,
            )
        except Exception as e:
            print(
                f"[cascade {i}/{n}] {model} failed ({e}), skipping...",
                file=sys.stderr,
            )
            skipped.append(model)

    # --- Final stage: strongest model produces authoritative answer ---
    final_prompt = (
        f"You are the final reviewer producing the definitive answer.\n"
        f"{SOURCE_GUIDANCE}\n\n"
        f"Original question: {query}\n\n"
        f"Draft answer (refined by {n - 1 - len(skipped)} model(s)):\n"
        f"{current_answer}\n\n"
        f"Instructions:\n"
        f"- Verify all factual claims and correct any remaining errors\n"
        f"- Ensure every key claim has a [Source: ...] reference to an authoritative source "
        f"(official docs, standards bodies like IEEE/IETF/ISO/NIST/W3C, "
        f"vendor docs like Red Hat/kernel.org/MDN, or peer-reviewed work)\n"
        f"- If a claim cannot be verified, mark it as unverified\n"
        f"- Produce a clean, well-structured final answer\n"
        f"- Do NOT include meta-commentary about the review process\n"
        f"Output ONLY the final authoritative answer."
        + (f"\n{search_ctx}" if search_ctx else "")
    )

    final_models = [all_models[-1]] + build_fallback_chain(
        all_models, all_models[-1],
    )
    think_label = " +think" if final_think else ""
    for final_model in final_models:
        if final_model in skipped:
            continue
        progress_update(10 + _pct_step * (n - 1), f"Final answer with {final_model}{think_label}...")
        print(
            f"[cascade {n}/{n}] Final answer with {final_model}{think_label}...",
            file=sys.stderr,
        )
        try:
            result = ask(
                final_prompt, model=final_model,
                thinking=final_think,
                temperature=_TEMP_FINAL if final_think else None,
            )
            if skipped:
                print(
                    f"[cascade] Completed with skipped models: {skipped}",
                    file=sys.stderr,
                )
            return result
        except Exception as e:
            print(
                f"[cascade] {final_model} failed during final stage ({e}), "
                f"trying fallback...",
                file=sys.stderr,
            )
            skipped.append(final_model)

    if skipped:
        print(
            f"[cascade] All final-stage models failed; returning best draft. "
            f"Skipped: {skipped}",
            file=sys.stderr,
        )
    return current_answer


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

    progress_update(5, "Searching..." if web_search else "Preparing...")
    progress_update(20, f"Classifying complexity with {fast_name}...")
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
        progress_update(50, f"Answering with {fast_name} (simple)...")
        print(f"[route 2/2] Answering with {fast_name} (simple)...", file=sys.stderr)
        return ask(enriched, model=fast_name)
    else:
        progress_update(50, f"Answering with {strong_name} +think (complex)...")
        print(f"[route 2/2] Answering with {strong_name} +think (complex)...", file=sys.stderr)
        return ask(enriched, model=strong_name, thinking=True, temperature=_TEMP_FINAL)


def chain_pipeline(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Fast model extracts/classifies, strong model reasons with search context."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    search_ctx = _enrich_with_search(query, fast_name, web_search)

    progress_update(5, "Searching..." if web_search else "Preparing...")
    progress_update(15, f"Extracting key points with {fast_name}...")
    print(f"[pipeline 1/3] Extracting key points with {fast_name}...", file=sys.stderr)
    key_points = ask(
        f"Extract the key points and core question from this. Be concise:\n\n{query}",
        model=fast_name,
    )

    progress_update(35, f"Classifying domain with {fast_name}...")
    print(f"[pipeline 2/3] Classifying domain with {fast_name}...", file=sys.stderr)
    domain = ask(
        f"What domain/field is this about? Reply in 1-3 words:\n\n{key_points}",
        model=fast_name,
    )

    progress_update(55, f"Deep analysis with {strong_name} +think ({domain.strip()})...")
    print(f"[pipeline 3/3] Deep analysis with {strong_name} +think ({domain.strip()})...", file=sys.stderr)
    prompt = (
        f"You are an expert in {domain}.\n"
        f"{SOURCE_GUIDANCE}\n\n"
        f"Key points:\n{key_points}\n\n"
        f"Original query: {query}\n\n"
        f"Provide a thorough, well-structured answer."
    )
    return ask(
        _inject_search_context(prompt, search_ctx), model=strong_name,
        thinking=True, temperature=_TEMP_FINAL,
    )


def chain_verify(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Fast model drafts, strong model verifies with search context for fact-checking."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    search_ctx = _enrich_with_search(query, fast_name, web_search)

    progress_update(5, "Searching..." if web_search else "Preparing...")
    progress_update(20, f"Drafting with {fast_name}...")
    print(f"[verify 1/2] Drafting with {fast_name}...", file=sys.stderr)
    draft = ask(f"{SOURCE_GUIDANCE}\n\n{query}", model=fast_name)

    progress_update(50, f"Verifying with {strong_name} +think...")
    print(f"[verify 2/2] Verifying with {strong_name} +think...", file=sys.stderr)
    prompt = (
        f"Another model answered the following question. "
        f"Verify the answer for correctness, fix any errors, and improve it. "
        f"Ensure all claims are backed by authoritative sources.\n"
        f"{SOURCE_GUIDANCE}\n\n"
        f"Question: {query}\n\n"
        f"Draft answer:\n{draft}"
    )
    return ask(
        _inject_search_context(prompt, search_ctx), model=strong_name,
        thinking=True, temperature=_TEMP_REVIEW,
    )


def chain_consensus(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """All models answer independently, strongest merges the best parts."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    search_ctx = _enrich_with_search(query, fast_name, web_search)

    progress_update(5, "Searching..." if web_search else "Preparing...")
    sourced_query = f"{SOURCE_GUIDANCE}\n\n{query}"
    answers = []
    _n = len(all_models)
    for i, model in enumerate(all_models, 1):
        use_thinking = model_supports_thinking(model) and model != fast_name
        label = " +think" if use_thinking else ""
        progress_update(10 + (i - 1) / max(_n, 1) * 60, f"Answer from {model}{label}...")
        print(
            f"[consensus {i}/{_n + 1}] Answer from {model}{label}...",
            file=sys.stderr,
        )
        answers.append((model, ask(sourced_query, model=model, thinking=use_thinking)))

    answers_block = "\n\n".join(
        f"=== Answer from {name} ===\n{answer}"
        for name, answer in answers
    )

    progress_update(75, f"Merging consensus with {strong_name} +think...")
    print(
        f"[consensus {len(all_models) + 1}/{len(all_models) + 1}] "
        f"Merging with {strong_name} +think...",
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
    return ask(
        _inject_search_context(prompt, search_ctx), model=strong_name,
        thinking=True, temperature=_TEMP_FINAL,
    )


def chain_search(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Search-first mode: always searches, strongest model synthesizes."""
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]
    progress_update(5, "Searching web, GitHub, StackOverflow, Docs...")
    results = search_for_query(query, fast_name)

    if not results:
        progress_update(40, f"No results, answering with {strong_name} +think...")
        print("[search] No results found, falling back to strong model +think...", file=sys.stderr)
        return ask(
            f"{SOURCE_GUIDANCE}\n\n{query}", model=strong_name,
            thinking=True, temperature=_TEMP_FINAL,
        )

    progress_update(40, f"Synthesizing answer with {strong_name} +think...")
    print(f"[search] Synthesizing answer with {strong_name} +think...", file=sys.stderr)
    return ask(
        f"Answer the following question using the web search results below. "
        f"Be accurate, cite sources by number and include their URLs. "
        f"Prioritize authoritative sources (official documentation, standards bodies "
        f"like IEEE/IETF/ISO/NIST/W3C, vendor docs like Red Hat/kernel.org/MDN). "
        f"Clearly state if the search results don't fully answer the question.\n"
        f"{SOURCE_GUIDANCE}\n\n"
        f"Question: {query}\n\n"
        f"=== SEARCH RESULTS ===\n{results}",
        model=strong_name,
        thinking=True,
        temperature=_TEMP_FINAL,
    )


def chain_fast(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Direct to fast model, optionally with search context."""
    fast_name = fast or all_models[0]
    search_ctx = _enrich_with_search(query, fast_name, web_search)
    progress_update(20, f"Answering with {fast_name}...")
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
    progress_update(5, "Searching..." if web_search else "Preparing...")
    progress_update(20, f"Answering with {strong_name} +think...")
    print(f"[strong 1/1] {strong_name} +think...", file=sys.stderr)
    return ask(
        _inject_search_context(f"{SOURCE_GUIDANCE}\n\n{query}", search_ctx),
        model=strong_name,
        thinking=True,
        temperature=_TEMP_FINAL,
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

    progress_update(2, f"Parsing {filepath} with scapy...")
    print(f"[pcap 1/4] Parsing {filepath} with scapy...", file=sys.stderr)
    analysis = analyze_pcap(filepath)
    report = format_pcap_analysis(analysis)
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

    progress_update(25, f"Summarizing with {fast_name}...")
    print(f"[pcap 2/4] Summarizing with {fast_name}...", file=sys.stderr)
    summary = ask(
        f"You are a network analyst. Summarize this packet capture analysis. "
        f"Highlight the most important findings, anomalies, and potential issues. "
        f"Be concise but thorough.\n\n{report}",
        model=fast_name,
    )

    progress_update(45, f"Error analysis with {fast_name}...")
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

    progress_update(65, f"Expert report with {strong_name} +think...")
    print(f"[pcap 4/4] Expert report with {strong_name} +think...", file=sys.stderr)
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
    return ask(
        _inject_search_context(prompt, search_ctx), model=strong_name,
        thinking=True, temperature=_TEMP_FINAL,
    )


# ---------------------------------------------------------------------------
# K8S — Kubernetes / OpenShift cluster analysis (CLI only)
# ---------------------------------------------------------------------------

def chain_k8s(
    kubeconfig: str, all_models: list[str],
    query: str | None = None, *, web_search: bool = True, fast: str | None = None,
) -> str:
    """
    Analyze a Kubernetes / OpenShift cluster through a multi-stage pipeline:
      1. Gather comprehensive cluster state via oc/kubectl
      2. Fast model summarizes findings and flags issues
      3. Fast model performs focused issue analysis
      4. Strong model produces a detailed expert report with references
    """
    fast_name = fast or all_models[0]
    strong_name = all_models[-1]

    progress_update(2, f"Gathering cluster state from {kubeconfig}...")
    print(f"[k8s 1/4] Gathering cluster state from {kubeconfig}...", file=sys.stderr)
    analysis = analyze_cluster(kubeconfig)
    report = format_k8s_analysis(analysis)
    platform = "OpenShift" if analysis.is_openshift else "Kubernetes"
    print(
        f"[k8s 1/4] {platform} cluster — "
        f"{len(analysis.errors)} errors, {len(analysis.warnings)} warnings",
        file=sys.stderr,
    )

    search_ctx = ""
    if web_search and (analysis.errors or analysis.warnings):
        search_terms: list[str] = []
        for e in analysis.errors[:5]:
            if "DEGRADED" in e:
                op_name = e.split("'")[1] if "'" in e else ""
                search_terms.append(
                    f"OpenShift ClusterOperator {op_name} degraded troubleshooting"
                )
            elif "NOT AVAILABLE" in e:
                op_name = e.split("'")[1] if "'" in e else ""
                search_terms.append(
                    f"OpenShift ClusterOperator {op_name} unavailable fix"
                )
        for w in analysis.warnings[:5]:
            if "Unhealthy pod" in w:
                search_terms.append("Kubernetes pod CrashLoopBackOff troubleshooting")
                break
        if search_terms:
            search_query = " ".join(search_terms[:2])
            search_ctx = _enrich_with_search(search_query, fast_name, True)

    progress_update(25, f"Summarizing with {fast_name}...")
    print(f"[k8s 2/4] Summarizing with {fast_name}...", file=sys.stderr)
    summary = ask(
        f"You are a Kubernetes / OpenShift platform engineer. "
        f"Summarize this cluster state analysis. "
        f"Highlight the most important findings, health issues, and capacity concerns. "
        f"Be concise but thorough.\n\n{report}",
        model=fast_name,
    )

    progress_update(45, f"Issue analysis with {fast_name}...")
    print(f"[k8s 3/4] Issue analysis with {fast_name}...", file=sys.stderr)
    if analysis.errors or analysis.warnings:
        issue_details = "\n".join(analysis.errors + analysis.warnings)
        issue_section = ask(
            f"You are a Kubernetes / OpenShift SRE specialist. "
            f"Analyze these cluster issues. "
            f"Classify each by severity (critical/high/medium/low). "
            f"Explain root causes and suggest remediation steps. "
            f"Reference official Kubernetes or OpenShift documentation where applicable.\n\n"
            f"Issues:\n{issue_details}\n\n"
            f"Full cluster state:\n{report[:4000]}",
            model=fast_name,
        )
    else:
        issue_section = "No errors or warnings were detected in the cluster."

    user_context = ""
    if query:
        user_context = (
            f"\n\nThe user specifically asked: {query}\n"
            f"Make sure to address their question directly.\n"
        )

    progress_update(65, f"Expert report with {strong_name} +think...")
    print(f"[k8s 4/4] Expert report with {strong_name} +think...", file=sys.stderr)
    prompt = (
        f"You are a senior {platform} platform engineer and SRE specialist. "
        f"Produce a comprehensive report for this cluster.\n"
        f"{SOURCE_GUIDANCE}\n"
        f"Reference official documentation: "
        f"kubernetes.io, docs.openshift.com, Red Hat Knowledge Base, "
        f"and relevant upstream project docs.\n\n"
        f"=== RAW CLUSTER STATE ===\n{report}\n\n"
        f"=== SUMMARY ===\n{summary}\n\n"
        f"=== ISSUE ANALYSIS ===\n{issue_section}\n"
        f"{user_context}\n"
        f"Structure your report with these sections:\n"
        f"1. Executive Summary\n"
        f"2. Cluster Overview (platform, version, nodes, capacity)\n"
        f"3. Workload Health (pods, deployments, replicas)\n"
        f"4. Networking (services, routes/ingresses, DNS)\n"
        f"5. Storage (PVs, storage classes, utilization)\n"
        f"6. Issues & Anomalies (with severity and remediation)\n"
        f"7. Security Observations\n"
        f"8. Recommendations\n"
    )
    return ask(
        _inject_search_context(prompt, search_ctx), model=strong_name,
        thinking=True, temperature=_TEMP_FINAL,
    )


# ---------------------------------------------------------------------------
# AGENT — autonomous mode with planning, memory, tools, and control flow
# ---------------------------------------------------------------------------

def chain_agent(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
    max_iterations: int = 15,
) -> str:
    """
    Autonomous agent mode:
      1. Planner decomposes the goal into steps
      2. Agent executes steps using tools (shell, files, web search, python)
      3. Memory persists facts and session history across runs
      4. Control flow dynamically re-plans on failures
    """
    from .agent import run_agent

    return run_agent(
        query, all_models,
        web_search=web_search,
        fast=fast,
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# AUTO — router-driven mode selection
# ---------------------------------------------------------------------------

def chain_auto(
    query: str, all_models: list[str], *,
    web_search: bool = True, fast: str | None = None,
) -> str:
    """Auto-select the optimal chain mode based on query routing.

    The router classifies query complexity and picks the best strategy:
      simple   → direct to fast model (skip search)
      moderate → subset cascade (fast + strong)
      complex  → full cascade through all models
    """
    fast_name = fast or all_models[0]
    decision = route_query(
        query, all_models,
        fast_model=fast_name,
        web_search=web_search,
    )
    print(
        f"[auto] Routed: complexity={decision.complexity} "
        f"strategy={decision.strategy} models={len(decision.models)} "
        f"({decision.reasoning})",
        file=sys.stderr,
    )

    use_search = web_search and not decision.skip_search

    if decision.strategy == "direct_fast":
        return chain_fast(
            query, all_models, web_search=use_search, fast=fast,
        )
    if decision.strategy == "direct_strong":
        return chain_strong(
            query, all_models, web_search=use_search, fast=fast,
        )
    if decision.strategy == "subset_cascade":
        return chain_cascade(
            query, decision.models, web_search=use_search, fast=fast,
            complexity=decision.complexity,
        )
    return chain_cascade(
        query, all_models, web_search=use_search, fast=fast,
        complexity=decision.complexity,
    )


CHAINS = {
    "cascade": chain_cascade,
    "auto": chain_auto,
    "route": chain_route,
    "pipeline": chain_pipeline,
    "verify": chain_verify,
    "consensus": chain_consensus,
    "search": chain_search,
    "fast": chain_fast,
    "strong": chain_strong,
    "agent": chain_agent,
}
