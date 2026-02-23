"""Unit tests for search.py — SearchResult, formatting, providers (mocked)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ollama_chain.search import (
    SearchResult,
    TRUSTED_DOCS_DOMAINS,
    _SOURCE_LABELS,
    format_search_results,
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
