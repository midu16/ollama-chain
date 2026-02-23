#!/usr/bin/env python3
"""CLI entry point for ollama-chain."""

import argparse
import gc
import sys

from .models import discover_models, ensure_memory_available, model_names, pick_models, list_models_table
from .chains import CHAINS, chain_pcap, chain_k8s
from .common import ensure_sources, unload_all_models
from .memory import PersistentMemory
from .metrics import evaluate_prompt, evaluate_mode_alignment, evaluate_response
from .optimizer import optimize_prompt, format_optimization_report


def detect_pcap_path(text: str) -> str | None:
    """If the query contains a path ending in .pcap/.pcapng/.cap, extract it."""
    for token in text.split():
        cleaned = token.strip("\"'(),;")
        if cleaned.lower().endswith((".pcap", ".pcapng", ".cap")):
            return cleaned
    return None


def main():
    parser = argparse.ArgumentParser(
        prog="ollama-chain",
        description=(
            "Chain ALL local Ollama models together with web search. "
            "Fully self-hosted, no API keys."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes:
  cascade     Chain ALL models smallest→largest, each refining the answer (default)
  auto        Router classifies complexity, picks the best strategy automatically
  consensus   All models answer independently, strongest merges the best parts
  route       Fast model scores complexity, routes to fast or strong
  pipeline    Fast model extracts + classifies, strong model reasons
  verify      Fast model drafts, strong model verifies and refines
  search      Search-first: always fetches web results, strong model synthesizes
  agent       Autonomous agent with planning, memory, tools, and dynamic control flow
  pcap        Analyze a .pcap file (auto-detected from query, or use --pcap)  [CLI only]
  k8s         Analyze a Kubernetes/OpenShift cluster (use --kubeconfig)       [CLI only]
  fast        Direct to smallest/fastest model
  strong      Direct to largest/strongest model

agent mode:
  The agent decomposes a goal into steps, uses tools (shell, file I/O,
  web search, python eval) to gather information, persists facts to
  long-term memory (~/.ollama_chain/), and re-plans dynamically on
  failures.  Use --max-iterations to control the execution budget.

web search (multi-source):
  All modes search multiple sources in parallel for context:
  DuckDuckGo (web), GitHub (repos + issues), Stack Overflow (Q&A),
  and trusted documentation sites (kubernetes.io, MDN, Red Hat, etc.).
  No API keys required.  Use --no-search to run fully offline.

source citations:
  All modes instruct models to cite authoritative sources (IETF RFCs,
  IEEE, ISO, NIST, W3C, Red Hat, kernel.org, MDN, official docs).
  Every answer includes a mandatory ## Sources section.  If the model
  omits it, a follow-up call appends one automatically.

k8s mode:
  The k8s mode gathers comprehensive cluster state via oc (preferred)
  or kubectl using the supplied --kubeconfig file.  Works with both
  vanilla Kubernetes and OpenShift clusters.  Collects nodes, pods,
  deployments, services, events, storage, operators, routes, etc.
  and produces an expert report.  Available only from the CLI.

prompt quality metrics:
  Use --metrics to display prompt quality scores alongside normal execution.
  Use --metrics-only to evaluate prompt quality without running any models.
  Metrics include: clarity, specificity, structure, actionability, context
  sufficiency, delimiter usage, chain-of-thought readiness, few-shot
  readiness, task decomposition, and mode alignment.

prompt optimization:
  Use --optimize to rewrite your prompt using LLM-driven optimization
  before execution. Applies techniques from the Prompt Engineering Guide:
  specificity, structured I/O, delimiters, task decomposition, few-shot
  examples, chain-of-thought, and ReAct patterns.
  Use --optimize-only to get the improved prompt without running it.

examples:
  %(prog)s "What is a binary search tree?"
  %(prog)s -m consensus "Compare REST vs GraphQL"
  %(prog)s -m search "latest Linux kernel release"
  %(prog)s -m agent "Find out what Linux kernel my machine is running and explain its key features"
  %(prog)s -m agent --max-iterations 20 "Research and summarize the CVEs from last week"
  %(prog)s --no-search "What is 2+2?"
  %(prog)s -m pcap --pcap capture.pcap
  %(prog)s "Analyze /tmp/dump.pcap for errors"
  %(prog)s -m k8s --kubeconfig ~/.kube/config "Why are pods crashing?"
  %(prog)s -m k8s --kubeconfig /tmp/ocp.kubeconfig "Show cluster health"
  %(prog)s --metrics "Explain the TCP three-way handshake step by step"
  %(prog)s --metrics-only "Compare REST vs GraphQL for microservice architectures"
  %(prog)s --optimize "explain docker"
  %(prog)s --optimize-only "tell me about networking stuff"
  %(prog)s --list-models
  %(prog)s --clear-memory
""",
    )
    parser.add_argument("query", nargs="*", help="Query to process")
    parser.add_argument(
        "--mode", "-m",
        choices=list(CHAINS.keys()) + ["pcap", "k8s"],
        default="cascade",
        help="Chaining mode (default: cascade)",
    )
    parser.add_argument(
        "--pcap", "-p",
        metavar="FILE",
        help="Path to .pcap/.pcapng file to analyze",
    )
    parser.add_argument(
        "--kubeconfig", "-k",
        metavar="FILE",
        help="Path to kubeconfig file for k8s mode",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable web search (fully offline mode)",
    )
    parser.add_argument(
        "--fast-model",
        metavar="MODEL",
        help="Override which model is used for search query generation",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        metavar="N",
        help="Max agent loop iterations (default: 15, agent mode only)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit",
    )
    parser.add_argument(
        "--clear-memory",
        action="store_true",
        help="Clear the agent's persistent memory and exit",
    )
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="Show stored facts and recent sessions, then exit",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Evaluate and display prompt quality metrics before execution",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Evaluate prompt quality metrics and exit (no model execution)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize the prompt using LLM before execution (shows before/after)",
    )
    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Optimize and display the improved prompt, then exit (no execution)",
    )

    args = parser.parse_args()

    # --- Memory management commands (no models needed) ---
    if args.clear_memory:
        PersistentMemory().clear()
        print("Agent memory cleared.")
        sys.exit(0)

    if args.show_memory:
        mem = PersistentMemory()
        facts = mem.get_facts()
        sessions = mem.get_recent_sessions()
        if not facts and not sessions:
            print("No stored memory yet.")
        else:
            if facts:
                print("=== Stored Facts ===")
                for f in facts:
                    print(f"  - {f}")
            if sessions:
                print("\n=== Recent Sessions ===")
                for s in sessions:
                    print(f"  [{s['session_id']}] {s['goal']}")
                    print(f"    {s['summary']}")
        sys.exit(0)

    models = discover_models()
    all_names = model_names(models)

    ensure_memory_available(all_names)

    if args.list_models:
        print(list_models_table(models))
        print(f"\n{len(models)} model(s) available.")
        print(f"Cascade order: {' → '.join(all_names)}")
        sys.exit(0)

    use_search = not args.no_search
    fast_override = args.fast_model
    search_label = "on" if use_search else "off"

    print(
        f"Models ({len(all_names)}): {' → '.join(all_names)} | Web search: {search_label}",
        file=sys.stderr,
    )

    query = " ".join(args.query) if args.query else ""

    # --- Prompt quality metrics ---
    if (args.metrics or args.metrics_only) and query:
        prompt_metrics = evaluate_prompt(query)
        mode_score, mode_explanation = evaluate_mode_alignment(query, args.mode)
        print(f"\n{'='*60}", file=sys.stderr)
        print(prompt_metrics.summary(), file=sys.stderr)
        bar_len = int(mode_score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"  {'Mode Alignment':<22} {bar} {mode_score * 100:5.1f}%  "
            f"{mode_explanation}",
            file=sys.stderr,
        )
        print(f"{'='*60}\n", file=sys.stderr)
        if args.metrics_only:
            sys.exit(0)

    # --- Prompt optimization ---
    if (args.optimize or args.optimize_only) and query:
        opt_model = fast_override or all_names[0]
        optimized, before, after = optimize_prompt(
            query, opt_model, mode=args.mode,
        )
        report = format_optimization_report(
            query, optimized, before, after, args.mode,
        )
        print(report, file=sys.stderr)
        if args.optimize_only:
            print(optimized)
            sys.exit(0)
        query = optimized

    pcap_path = args.pcap
    if not pcap_path and query:
        pcap_path = detect_pcap_path(query)

    kubeconfig = args.kubeconfig

    try:
        import time as _time
        _t0 = _time.monotonic()

        if args.mode == "pcap" or pcap_path:
            if not pcap_path:
                print("Error: pcap mode requires a .pcap file path (--pcap or in query)", file=sys.stderr)
                sys.exit(1)
            result = chain_pcap(
                pcap_path, all_names,
                query=query or None,
                web_search=use_search,
                fast=fast_override,
            )
        elif args.mode == "k8s" or kubeconfig:
            if not kubeconfig:
                print("Error: k8s mode requires --kubeconfig <file>", file=sys.stderr)
                sys.exit(1)
            result = chain_k8s(
                kubeconfig, all_names,
                query=query or None,
                web_search=use_search,
                fast=fast_override,
            )
        elif not query:
            parser.print_help()
            sys.exit(1)
        elif args.mode == "agent":
            chain_fn = CHAINS[args.mode]
            result = chain_fn(
                query, all_names,
                web_search=use_search,
                fast=fast_override,
                max_iterations=args.max_iterations,
            )
        else:
            chain_fn = CHAINS[args.mode]
            result = chain_fn(
                query, all_names,
                web_search=use_search,
                fast=fast_override,
            )

        elapsed_ms = (_time.monotonic() - _t0) * 1000

        effective_query = query or pcap_path or kubeconfig or ""
        result = ensure_sources(result, effective_query, all_names[-1])

        if args.metrics and query:
            resp_metrics = evaluate_response(query, result, elapsed_ms)
            print(f"\n{'='*60}", file=sys.stderr)
            print("Response Metrics:", file=sys.stderr)
            print(resp_metrics.summary(), file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

        print(result)
    finally:
        unload_all_models(all_names)
        gc.collect()
