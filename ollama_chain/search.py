"""Multi-source search: DuckDuckGo, GitHub, Stack Overflow, trusted docs.

All providers are unauthenticated and require no API keys.  Results from
every source are tagged, deduplicated, and merged into a single context
block for LLM consumption.
"""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ddgs import DDGS

_HTTP_TIMEOUT = 8  # seconds for GitHub / Stack Overflow API calls

TRUSTED_DOCS_DOMAINS = (
    "kubernetes.io", "docs.openshift.com", "docs.redhat.com",
    "kernel.org", "developer.mozilla.org", "docs.python.org",
    "rfc-editor.org", "w3.org", "docs.docker.com",
    "learn.microsoft.com", "wiki.archlinux.org",
    "man7.org", "nginx.org", "postgresql.org",
    "go.dev", "rust-lang.org", "cppreference.com",
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search DuckDuckGo and return top results."""
    try:
        ddgs = DDGS()
        raw = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"[search] Warning: web search failed — {e}", file=sys.stderr)
        return []

    results = []
    for r in raw:
        results.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("href", ""),
            snippet=r.get("body", ""),
            source="web",
        ))
    return results


def web_search_news(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search DuckDuckGo news for recent/time-sensitive queries."""
    try:
        ddgs = DDGS()
        raw = list(ddgs.news(query, max_results=max_results))
    except Exception as e:
        print(f"[search] Warning: news search failed — {e}", file=sys.stderr)
        return []

    results = []
    for r in raw:
        results.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("body", ""),
            source="news",
        ))
    return results


# ---------------------------------------------------------------------------
# GitHub search (unauthenticated REST API — 10 req/min)
# ---------------------------------------------------------------------------

def github_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search GitHub repositories via the public REST API."""
    params = urllib.parse.urlencode({
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": max_results,
    })
    url = f"https://api.github.com/search/repositories?{params}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "ollama-chain",
    })
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[search] Warning: GitHub search failed — {e}", file=sys.stderr)
        return []

    results: list[SearchResult] = []
    for item in data.get("items", [])[:max_results]:
        stars = item.get("stargazers_count", 0)
        lang = item.get("language") or ""
        topics = ", ".join(item.get("topics", [])[:5])
        desc = item.get("description") or "(no description)"
        meta_parts = [f"★{stars}"]
        if lang:
            meta_parts.append(lang)
        if topics:
            meta_parts.append(topics)
        snippet = f"{desc}  [{' | '.join(meta_parts)}]"
        results.append(SearchResult(
            title=item.get("full_name", ""),
            url=item.get("html_url", ""),
            snippet=snippet,
            source="github",
        ))
    return results


def github_search_issues(
    query: str, max_results: int = 5,
) -> list[SearchResult]:
    """Search GitHub issues and discussions for troubleshooting context."""
    params = urllib.parse.urlencode({
        "q": query,
        "sort": "reactions",
        "order": "desc",
        "per_page": max_results,
    })
    url = f"https://api.github.com/search/issues?{params}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "ollama-chain",
    })
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[search] Warning: GitHub issue search failed — {e}", file=sys.stderr)
        return []

    results: list[SearchResult] = []
    for item in data.get("items", [])[:max_results]:
        state = item.get("state", "")
        comments = item.get("comments", 0)
        labels = ", ".join(l.get("name", "") for l in item.get("labels", [])[:3])
        snippet = f"{item.get('body', '')[:200]}  [state: {state}, comments: {comments}"
        if labels:
            snippet += f", labels: {labels}"
        snippet += "]"
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("html_url", ""),
            snippet=snippet,
            source="github-issues",
        ))
    return results


# ---------------------------------------------------------------------------
# Stack Overflow search (Stack Exchange API — unauthenticated)
# ---------------------------------------------------------------------------

def stackoverflow_search(
    query: str, max_results: int = 5,
) -> list[SearchResult]:
    """Search Stack Overflow via the Stack Exchange API."""
    params = urllib.parse.urlencode({
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow",
        "pagesize": max_results,
        "filter": "!nNPvSNdWme",
    })
    url = f"https://api.stackexchange.com/2.3/search/advanced?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "ollama-chain"})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            raw = resp.read()
            import gzip
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
            data = json.loads(raw.decode())
    except Exception as e:
        print(f"[search] Warning: Stack Overflow search failed — {e}", file=sys.stderr)
        return []

    results: list[SearchResult] = []
    for item in data.get("items", [])[:max_results]:
        score = item.get("score", 0)
        answers = item.get("answer_count", 0)
        accepted = " ✓" if item.get("is_answered") else ""
        tags = ", ".join(item.get("tags", [])[:5])
        snippet = f"Score: {score} | Answers: {answers}{accepted} | Tags: {tags}"
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=snippet,
            source="stackoverflow",
        ))
    return results


# ---------------------------------------------------------------------------
# Trusted documentation search (DuckDuckGo site-scoped)
# ---------------------------------------------------------------------------

def docs_search(
    query: str,
    domains: tuple[str, ...] = TRUSTED_DOCS_DOMAINS,
    max_results: int = 5,
) -> list[SearchResult]:
    """Search trusted documentation sites via DuckDuckGo site: operators."""
    site_clause = " OR ".join(f"site:{d}" for d in domains[:8])
    scoped_query = f"{query} ({site_clause})"
    try:
        ddgs = DDGS()
        raw = list(ddgs.text(scoped_query, max_results=max_results))
    except Exception as e:
        print(f"[search] Warning: docs search failed — {e}", file=sys.stderr)
        return []

    results: list[SearchResult] = []
    for r in raw:
        results.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("href", ""),
            snippet=r.get("body", ""),
            source="docs",
        ))
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_SOURCE_LABELS = {
    "web": "Web",
    "news": "News",
    "github": "GitHub",
    "github-issues": "GitHub Issues",
    "stackoverflow": "Stack Overflow",
    "docs": "Official Docs",
}


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results into a text block for LLM context injection."""
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results, 1):
        label = _SOURCE_LABELS.get(r.source, r.source)
        lines.append(f"[{i}] [{label}] {r.title}")
        lines.append(f"    {r.url}")
        lines.append(f"    {r.snippet}")
        lines.append("")
    return "\n".join(lines)


def generate_search_queries(query: str, fast_model: str) -> list[str]:
    """Use the fast model to generate optimal search queries from the user's question."""
    from .common import chat_with_retry

    response = chat_with_retry(
        model=fast_model,
        messages=[{"role": "user", "content": (
            "/no_think\n"
            "Generate 1-3 concise web search queries that would help answer "
            "this question accurately. Return ONLY the queries, one per line, "
            "no numbering, no explanation.\n\n"
            f"Question: {query}"
        )}],
    )
    content = response["message"]["content"]
    if "<think>" in content:
        end = content.find("</think>")
        if end != -1:
            content = content[end + len("</think>"):].strip()

    queries = [
        line.strip().strip('"').strip("'").lstrip("0123456789.-) ")
        for line in content.strip().splitlines()
        if line.strip() and len(line.strip()) > 5
    ]
    return queries[:3]


def search_for_query(query: str, fast_model: str, max_results: int = 5) -> str:
    """
    Full multi-source search pipeline:
      1. Fast model generates search queries
      2. All providers searched in parallel:
         - DuckDuckGo (general web)
         - GitHub (repositories + issues)
         - Stack Overflow (Q&A)
         - Trusted documentation sites (site-scoped DDG)
      3. Results aggregated, deduplicated, and formatted
    """
    print(f"[search] Generating search queries with {fast_model}...", file=sys.stderr)
    search_queries = generate_search_queries(query, fast_model)

    if not search_queries:
        search_queries = [query]

    primary_query = search_queries[0]
    print(f"[search] Searching: {search_queries}", file=sys.stderr)

    seen_urls: set[str] = set()
    all_results: list[SearchResult] = []

    def _collect(results: list[SearchResult]) -> None:
        for r in results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}

        for sq in search_queries:
            futures[pool.submit(web_search, sq, max_results)] = f"web:{sq}"

        futures[pool.submit(github_search, primary_query, 3)] = "github"
        futures[pool.submit(github_search_issues, primary_query, 3)] = "github-issues"
        futures[pool.submit(stackoverflow_search, primary_query, 3)] = "stackoverflow"
        futures[pool.submit(docs_search, primary_query)] = "docs"

        for future in as_completed(futures):
            label = futures[future]
            try:
                results = future.result()
                _collect(results)
            except Exception as e:
                print(
                    f"[search] {label} failed: {e}",
                    file=sys.stderr,
                )

    all_results = all_results[:max_results * 4]

    source_counts: dict[str, int] = {}
    for r in all_results:
        source_counts[r.source] = source_counts.get(r.source, 0) + 1
    summary = ", ".join(
        f"{_SOURCE_LABELS.get(s, s)}: {c}" for s, c in sorted(source_counts.items())
    )
    print(f"[search] Found {len(all_results)} results ({summary})", file=sys.stderr)

    return format_search_results(all_results)
