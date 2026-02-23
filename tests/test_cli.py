"""Unit tests for cli.py — pcap path detection, argument structure."""

import pytest

from ollama_chain.cli import detect_pcap_path


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
