"""Unit tests for search.py — SearchResult, formatting, providers (mocked)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ollama_chain.search import (
    SearchResult,
    TRUSTED_DOCS_DOMAINS,
    _SOURCE_LABELS,
    format_search_results,
    generate_search_queries,
    search_for_query,
    web_search_news,
    github_search_issues,
)


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_default_source(self):
        r = SearchResult(title="t", url="u", snippet="s")
        assert r.source == "web"

    def test_custom_source(self):
        r = SearchResult(title="t", url="u", snippet="s", source="github")
        assert r.source == "github"


# ---------------------------------------------------------------------------
# format_search_results
# ---------------------------------------------------------------------------

class TestFormatSearchResults:
    def test_empty_list(self):
        assert format_search_results([]) == ""

    def test_single_result(self):
        results = [SearchResult("Title", "https://x.com", "Snippet", "web")]
        output = format_search_results(results)
        assert "[1]" in output
        assert "[Web]" in output
        assert "Title" in output
        assert "https://x.com" in output
        assert "Snippet" in output

    def test_multiple_sources_labelled(self):
        results = [
            SearchResult("A", "u1", "s1", "github"),
            SearchResult("B", "u2", "s2", "stackoverflow"),
            SearchResult("C", "u3", "s3", "docs"),
        ]
        output = format_search_results(results)
        assert "[GitHub]" in output
        assert "[Stack Overflow]" in output
        assert "[Official Docs]" in output

    def test_numbering(self):
        results = [
            SearchResult(f"R{i}", f"u{i}", f"s{i}") for i in range(3)
        ]
        output = format_search_results(results)
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output

    def test_unknown_source_uses_raw(self):
        results = [SearchResult("T", "U", "S", "custom-src")]
        output = format_search_results(results)
        assert "[custom-src]" in output


# ---------------------------------------------------------------------------
# Source labels
# ---------------------------------------------------------------------------

class TestSourceLabels:
    def test_all_labels_defined(self):
        expected = {"web", "news", "github", "github-issues", "stackoverflow", "docs"}
        assert expected == set(_SOURCE_LABELS.keys())


# ---------------------------------------------------------------------------
# TRUSTED_DOCS_DOMAINS
# ---------------------------------------------------------------------------

class TestTrustedDomains:
    def test_known_domains_present(self):
        assert "kubernetes.io" in TRUSTED_DOCS_DOMAINS
        assert "docs.openshift.com" in TRUSTED_DOCS_DOMAINS
        assert "developer.mozilla.org" in TRUSTED_DOCS_DOMAINS

    def test_at_least_ten_domains(self):
        assert len(TRUSTED_DOCS_DOMAINS) >= 10


# ---------------------------------------------------------------------------
# web_search (mocked)
# ---------------------------------------------------------------------------

class TestWebSearch:
    @patch("ollama_chain.search.DDGS")
    def test_returns_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "T1", "href": "http://a.com", "body": "B1"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        from ollama_chain.search import web_search
        results = web_search("test")
        assert len(results) == 1
        assert results[0].source == "web"
        assert results[0].title == "T1"

    @patch("ollama_chain.search.DDGS")
    def test_handles_exception(self, mock_ddgs_cls):
        mock_ddgs_cls.side_effect = Exception("network error")

        from ollama_chain.search import web_search
        results = web_search("test")
        assert results == []


# ---------------------------------------------------------------------------
# github_search (mocked)
# ---------------------------------------------------------------------------

class TestGithubSearch:
    @patch("ollama_chain.search.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen):
        data = {
            "items": [{
                "full_name": "user/repo",
                "html_url": "https://github.com/user/repo",
                "description": "A repo",
                "stargazers_count": 100,
                "language": "Python",
                "topics": ["ml"],
            }]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        from ollama_chain.search import github_search
        results = github_search("test")
        assert len(results) == 1
        assert results[0].source == "github"
        assert "★100" in results[0].snippet

    @patch("ollama_chain.search.urllib.request.urlopen")
    def test_handles_exception(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("timeout")

        from ollama_chain.search import github_search
        assert github_search("test") == []


# ---------------------------------------------------------------------------
# stackoverflow_search (mocked)
# ---------------------------------------------------------------------------

class TestStackoverflowSearch:
    @patch("ollama_chain.search.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen):
        data = {
            "items": [{
                "title": "How to X?",
                "link": "https://stackoverflow.com/q/123",
                "score": 42,
                "answer_count": 3,
                "is_answered": True,
                "tags": ["python", "linux"],
            }]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        from ollama_chain.search import stackoverflow_search
        results = stackoverflow_search("test")
        assert len(results) == 1
        assert results[0].source == "stackoverflow"
        assert "42" in results[0].snippet


# ---------------------------------------------------------------------------
# docs_search (mocked)
# ---------------------------------------------------------------------------

class TestDocsSearch:
    @patch("ollama_chain.search.DDGS")
    def test_returns_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "K8s Docs", "href": "https://kubernetes.io/docs/x", "body": "Info"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        from ollama_chain.search import docs_search
        results = docs_search("kubernetes pods")
        assert len(results) == 1
        assert results[0].source == "docs"

        scoped = mock_ddgs.text.call_args[0][0]
        assert "site:" in scoped


# ---------------------------------------------------------------------------
# web_search_news (mocked)
# ---------------------------------------------------------------------------

class TestWebSearchNews:
    @patch("ollama_chain.search.DDGS")
    def test_returns_news_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.news.return_value = [
            {"title": "Breaking", "url": "http://n.com", "body": "News body"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        results = web_search_news("test")
        assert len(results) == 1
        assert results[0].source == "news"
        assert results[0].title == "Breaking"

    @patch("ollama_chain.search.DDGS")
    def test_handles_exception(self, mock_ddgs_cls):
        mock_ddgs_cls.side_effect = Exception("fail")
        results = web_search_news("test")
        assert results == []

    @patch("ollama_chain.search.DDGS")
    def test_empty_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.news.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs
        results = web_search_news("test")
        assert results == []


# ---------------------------------------------------------------------------
# github_search_issues (mocked)
# ---------------------------------------------------------------------------

class TestGithubSearchIssues:
    @patch("ollama_chain.search.urllib.request.urlopen")
    def test_returns_issues(self, mock_urlopen):
        data = {
            "items": [{
                "title": "Bug: crash on start",
                "html_url": "https://github.com/user/repo/issues/1",
                "state": "open",
                "comments": 5,
                "labels": [{"name": "bug"}],
                "body": "Crash description here",
            }]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        results = github_search_issues("crash bug")
        assert len(results) == 1
        assert results[0].source == "github-issues"
        assert "open" in results[0].snippet
        assert "bug" in results[0].snippet

    @patch("ollama_chain.search.urllib.request.urlopen")
    def test_handles_exception(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("timeout")
        assert github_search_issues("test") == []


# ---------------------------------------------------------------------------
# generate_search_queries (mocked LLM)
# ---------------------------------------------------------------------------

class TestGenerateSearchQueries:
    @patch("ollama_chain.common.chat_with_retry")
    def test_generates_queries(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": "OpenShift latest version\nOCP release notes 2025\n"}
        }
        queries = generate_search_queries("What is the latest OpenShift release?", "fast:7b")
        assert len(queries) >= 1
        assert any("OpenShift" in q for q in queries)

    @patch("ollama_chain.common.chat_with_retry")
    def test_strips_thinking_tags(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": "<think>thinking...</think>\nactual query\n"}
        }
        queries = generate_search_queries("test", "fast:7b")
        assert len(queries) == 1
        assert queries[0] == "actual query"

    @patch("ollama_chain.common.chat_with_retry")
    def test_filters_short_lines(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": "1.\n2.\ngood search query here\n"}
        }
        queries = generate_search_queries("test", "fast:7b")
        assert all(len(q) > 5 for q in queries)

    @patch("ollama_chain.common.chat_with_retry")
    def test_caps_at_three(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": "query one\nquery two\nquery three\nquery four\nquery five\n"}
        }
        queries = generate_search_queries("test", "fast:7b")
        assert len(queries) <= 3


# ---------------------------------------------------------------------------
# search_for_query (integration-level, mocked providers)
# ---------------------------------------------------------------------------

class TestSearchForQuery:
    @patch("ollama_chain.search.docs_search", return_value=[])
    @patch("ollama_chain.search.stackoverflow_search", return_value=[])
    @patch("ollama_chain.search.github_search_issues", return_value=[])
    @patch("ollama_chain.search.github_search", return_value=[])
    @patch("ollama_chain.search.web_search", return_value=[
        SearchResult("T1", "http://a.com", "S1", "web"),
    ])
    @patch("ollama_chain.search.generate_search_queries", return_value=["test query"])
    def test_returns_formatted_results(self, _gq, _ws, _gh, _ghi, _so, _docs):
        result = search_for_query("test", "fast:7b")
        assert "T1" in result
        assert "http://a.com" in result

    @patch("ollama_chain.search.docs_search", return_value=[])
    @patch("ollama_chain.search.stackoverflow_search", return_value=[])
    @patch("ollama_chain.search.github_search_issues", return_value=[])
    @patch("ollama_chain.search.github_search", return_value=[])
    @patch("ollama_chain.search.web_search", return_value=[])
    @patch("ollama_chain.search.generate_search_queries", return_value=["test"])
    def test_returns_empty_when_no_results(self, _gq, _ws, _gh, _ghi, _so, _docs):
        result = search_for_query("test", "fast:7b")
        assert result == ""

    @patch("ollama_chain.search.docs_search", return_value=[
        SearchResult("Doc", "http://docs.example.com", "doc info", "docs"),
    ])
    @patch("ollama_chain.search.stackoverflow_search", return_value=[
        SearchResult("SO", "http://so.com/q", "so info", "stackoverflow"),
    ])
    @patch("ollama_chain.search.github_search_issues", return_value=[])
    @patch("ollama_chain.search.github_search", return_value=[
        SearchResult("Repo", "http://gh.com/r", "gh info", "github"),
    ])
    @patch("ollama_chain.search.web_search", return_value=[
        SearchResult("Web", "http://w.com", "web info", "web"),
    ])
    @patch("ollama_chain.search.generate_search_queries", return_value=["test"])
    def test_deduplicates_urls(self, _gq, _ws, _gh, _ghi, _so, _docs):
        result = search_for_query("test", "fast:7b")
        assert result.count("http://w.com") == 1

    @patch("ollama_chain.search.generate_search_queries", side_effect=Exception("fail"))
    @patch("ollama_chain.search.docs_search", return_value=[])
    @patch("ollama_chain.search.stackoverflow_search", return_value=[])
    @patch("ollama_chain.search.github_search_issues", return_value=[])
    @patch("ollama_chain.search.github_search", return_value=[])
    @patch("ollama_chain.search.web_search", return_value=[
        SearchResult("Fallback", "http://f.com", "fb", "web"),
    ])
    def test_fallback_to_original_query_on_generation_failure(self, _ws, _gh, _ghi, _so, _docs, _gq):
        result = search_for_query("original query", "fast:7b")
        assert "Fallback" in result
