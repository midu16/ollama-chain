"""Web search via DuckDuckGo — no API keys required."""

import sys
from dataclasses import dataclass

from ddgs import DDGS


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


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
        ))
    return results


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results into a text block for LLM context injection."""
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    {r.url}")
        lines.append(f"    {r.snippet}")
        lines.append("")
    return "\n".join(lines)


def generate_search_queries(query: str, fast_model: str) -> list[str]:
    """Use the fast model to generate optimal search queries from the user's question."""
    import ollama as ollama_client

    response = ollama_client.chat(
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
    Full search pipeline:
      1. Fast model generates search queries
      2. DuckDuckGo executes them
      3. Results are deduplicated and formatted
    """
    print(f"[search] Generating search queries with {fast_model}...", file=sys.stderr)
    search_queries = generate_search_queries(query, fast_model)

    if not search_queries:
        search_queries = [query]

    print(f"[search] Searching: {search_queries}", file=sys.stderr)

    seen_urls = set()
    all_results = []
    for sq in search_queries:
        results = web_search(sq, max_results=max_results)
        for r in results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)

    all_results = all_results[:max_results * 2]
    print(f"[search] Found {len(all_results)} results", file=sys.stderr)
    return format_search_results(all_results)
