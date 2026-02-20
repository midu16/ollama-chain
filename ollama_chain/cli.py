#!/usr/bin/env python3
"""CLI entry point for ollama-chain."""

import argparse
import sys

from .models import discover_models, model_names, pick_models, list_models_table
from .chains import CHAINS, chain_pcap


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
  consensus   All models answer independently, strongest merges the best parts
  route       Fast model scores complexity, routes to fast or strong
  pipeline    Fast model extracts + classifies, strong model reasons
  verify      Fast model drafts, strong model verifies and refines
  search      Search-first: always fetches web results, strong model synthesizes
  pcap        Analyze a .pcap file (auto-detected from query, or use --pcap)
  fast        Direct to smallest/fastest model
  strong      Direct to largest/strongest model

web search:
  All modes automatically search the web for context (via DuckDuckGo).
  Use --no-search to disable this and run fully offline.

source citations:
  All modes instruct models to cite authoritative sources (IETF RFCs,
  IEEE, ISO, NIST, W3C, Red Hat, kernel.org, MDN, official docs).

examples:
  %(prog)s "What is a binary search tree?"
  %(prog)s -m consensus "Compare REST vs GraphQL"
  %(prog)s -m search "latest Linux kernel release"
  %(prog)s --no-search "What is 2+2?"
  %(prog)s -m pcap --pcap capture.pcap
  %(prog)s "Analyze /tmp/dump.pcap for errors"
  %(prog)s --list-models
""",
    )
    parser.add_argument("query", nargs="*", help="Query to process")
    parser.add_argument(
        "--mode", "-m",
        choices=list(CHAINS.keys()) + ["pcap"],
        default="cascade",
        help="Chaining mode (default: cascade)",
    )
    parser.add_argument(
        "--pcap", "-p",
        metavar="FILE",
        help="Path to .pcap/.pcapng file to analyze",
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
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit",
    )

    args = parser.parse_args()

    models = discover_models()
    all_names = model_names(models)

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

    pcap_path = args.pcap
    if not pcap_path and query:
        pcap_path = detect_pcap_path(query)

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
        print(result)
        return

    if not query:
        parser.print_help()
        sys.exit(1)

    chain_fn = CHAINS[args.mode]
    result = chain_fn(
        query, all_names,
        web_search=use_search,
        fast=fast_override,
    )
    print(result)
