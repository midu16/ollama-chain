"""Unit tests for cli.py — pcap path detection, argument parsing, fallbacks."""

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from ollama_chain.cli import detect_pcap_path, main


# ---------------------------------------------------------------------------
# detect_pcap_path
# ---------------------------------------------------------------------------

class TestDetectPcapPath:
    def test_pcap_extension(self):
        assert detect_pcap_path("Analyze /tmp/capture.pcap") == "/tmp/capture.pcap"

    def test_pcapng_extension(self):
        assert detect_pcap_path("Look at dump.pcapng") == "dump.pcapng"

    def test_cap_extension(self):
        assert detect_pcap_path("Process data.cap") == "data.cap"

    def test_no_pcap_path(self):
        assert detect_pcap_path("What is TCP?") is None

    def test_path_with_quotes(self):
        assert detect_pcap_path('Analyze "/tmp/my.pcap"') == "/tmp/my.pcap"

    def test_multiple_tokens_first_wins(self):
        result = detect_pcap_path("compare a.pcap b.pcap")
        assert result == "a.pcap"

    def test_empty_string(self):
        assert detect_pcap_path("") is None

    def test_case_insensitive_extension(self):
        assert detect_pcap_path("file.PCAP") == "file.PCAP"

    def test_path_with_directories(self):
        assert detect_pcap_path("check /var/log/dump.pcapng for errors") == "/var/log/dump.pcapng"


# ---------------------------------------------------------------------------
# Argparse structure validation
# ---------------------------------------------------------------------------

class TestArgparseStructure:
    def _parser(self):
        """Build the parser the same way main() does, without running models."""
        from ollama_chain.chains import CHAINS
        parser = argparse.ArgumentParser(prog="ollama-chain")
        parser.add_argument("query", nargs="*")
        parser.add_argument("--mode", "-m", choices=list(CHAINS.keys()) + ["pcap", "k8s"], default="cascade")
        parser.add_argument("--target", "-t")
        parser.add_argument("--pcap", "-p")
        parser.add_argument("--kubeconfig", "-k")
        parser.add_argument("--no-search", action="store_true")
        parser.add_argument("--fast-model")
        parser.add_argument("--max-iterations", type=int, default=None)
        parser.add_argument("--list-models", action="store_true")
        parser.add_argument("--clear-memory", action="store_true")
        parser.add_argument("--show-memory", action="store_true")
        parser.add_argument("--metrics", action="store_true")
        parser.add_argument("--metrics-only", action="store_true")
        parser.add_argument("--optimize", action="store_true")
        parser.add_argument("--optimize-only", action="store_true")
        parser.add_argument("--verbose", "-v", action="store_true")
        return parser

    def test_default_mode_is_cascade(self):
        parser = self._parser()
        args = parser.parse_args(["hello"])
        assert args.mode == "cascade"

    def test_mode_flag_works(self):
        parser = self._parser()
        args = parser.parse_args(["-m", "search", "test query"])
        assert args.mode == "search"

    def test_all_modes_accepted(self):
        from ollama_chain.chains import CHAINS
        parser = self._parser()
        for mode in list(CHAINS.keys()) + ["pcap", "k8s"]:
            args = parser.parse_args(["-m", mode, "q"])
            assert args.mode == mode

    def test_no_search_flag(self):
        parser = self._parser()
        args = parser.parse_args(["--no-search", "test"])
        assert args.no_search is True

    def test_verbose_flag(self):
        parser = self._parser()
        args = parser.parse_args(["-v", "test"])
        assert args.verbose is True

    def test_max_iterations(self):
        parser = self._parser()
        args = parser.parse_args(["--max-iterations", "25", "test"])
        assert args.max_iterations == 25

    def test_pcap_flag(self):
        parser = self._parser()
        args = parser.parse_args(["-p", "capture.pcap"])
        assert args.pcap == "capture.pcap"

    def test_kubeconfig_flag(self):
        parser = self._parser()
        args = parser.parse_args(["-k", "/tmp/kubeconfig"])
        assert args.kubeconfig == "/tmp/kubeconfig"

    def test_target_flag(self):
        parser = self._parser()
        args = parser.parse_args(["-t", "192.168.1.1", "scan"])
        assert args.target == "192.168.1.1"

    def test_fast_model_flag(self):
        parser = self._parser()
        args = parser.parse_args(["--fast-model", "qwen3:1.7b", "test"])
        assert args.fast_model == "qwen3:1.7b"

    def test_memory_flags(self):
        parser = self._parser()
        assert parser.parse_args(["--clear-memory"]).clear_memory is True
        assert parser.parse_args(["--show-memory"]).show_memory is True

    def test_metrics_flags(self):
        parser = self._parser()
        assert parser.parse_args(["--metrics", "test"]).metrics is True
        assert parser.parse_args(["--metrics-only", "test"]).metrics_only is True

    def test_optimize_flags(self):
        parser = self._parser()
        assert parser.parse_args(["--optimize", "test"]).optimize is True
        assert parser.parse_args(["--optimize-only", "test"]).optimize_only is True

    def test_query_joining(self):
        parser = self._parser()
        args = parser.parse_args(["hello", "world", "foo"])
        assert " ".join(args.query) == "hello world foo"


# ---------------------------------------------------------------------------
# clear-memory / show-memory (no Ollama needed)
# ---------------------------------------------------------------------------

class TestMemoryCommands:
    @patch("ollama_chain.cli.PersistentMemory")
    def test_clear_memory_exits(self, mock_mem_cls):
        mock_mem = MagicMock()
        mock_mem_cls.return_value = mock_mem
        with pytest.raises(SystemExit) as exc_info:
            sys.argv = ["ollama-chain", "--clear-memory"]
            main()
        assert exc_info.value.code == 0
        mock_mem.clear.assert_called_once()

    @patch("ollama_chain.cli.PersistentMemory")
    def test_show_memory_empty(self, mock_mem_cls):
        mock_mem = MagicMock()
        mock_mem.get_facts.return_value = []
        mock_mem.get_recent_sessions.return_value = []
        mock_mem_cls.return_value = mock_mem
        with pytest.raises(SystemExit) as exc_info:
            sys.argv = ["ollama-chain", "--show-memory"]
            main()
        assert exc_info.value.code == 0
