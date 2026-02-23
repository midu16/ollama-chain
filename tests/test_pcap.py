"""Unit tests for pcap.py — PcapAnalysis, format_analysis, port classification."""

import pytest

from ollama_chain.pcap import (
    PcapAnalysis,
    _classify_tcp_port,
    _classify_udp_port,
    format_analysis,
)


# ---------------------------------------------------------------------------
# PcapAnalysis dataclass
# ---------------------------------------------------------------------------

class TestPcapAnalysis:
    def test_defaults(self):
        a = PcapAnalysis(filepath="/tmp/test.pcap")
        assert a.filepath == "/tmp/test.pcap"
        assert a.total_packets == 0
        assert a.duration_seconds == 0.0
        assert a.protocols == {}
        assert a.errors == []
        assert a.warnings == []
        assert a.conversations == []
        assert a.dns_queries == []
        assert a.top_talkers == []


# ---------------------------------------------------------------------------
# _classify_tcp_port
# ---------------------------------------------------------------------------

class TestClassifyTcpPort:
    @pytest.mark.parametrize("port,expected", [
        (80, "HTTP"),
        (443, "HTTPS/TLS"),
        (22, "SSH"),
        (21, "FTP"),
        (25, "SMTP"),
        (53, "DNS/TCP"),
        (3306, "MySQL"),
        (5432, "PostgreSQL"),
        (6379, "Redis"),
        (8080, "HTTP-Alt"),
        (3389, "RDP"),
        (27017, "MongoDB"),
    ])
    def test_known_ports_by_dport(self, port, expected):
        assert _classify_tcp_port(12345, port) == expected

    @pytest.mark.parametrize("port,expected", [
        (80, "HTTP"),
        (443, "HTTPS/TLS"),
    ])
    def test_known_ports_by_sport(self, port, expected):
        assert _classify_tcp_port(port, 54321) == expected

    def test_unknown_port(self):
        assert _classify_tcp_port(12345, 54321) is None

    def test_dport_takes_precedence(self):
        result = _classify_tcp_port(80, 443)
        assert result == "HTTPS/TLS"


# ---------------------------------------------------------------------------
# _classify_udp_port
# ---------------------------------------------------------------------------

class TestClassifyUdpPort:
    @pytest.mark.parametrize("port,expected", [
        (53, "DNS"),
        (67, "DHCP-Server"),
        (68, "DHCP-Client"),
        (123, "NTP"),
        (161, "SNMP"),
        (443, "QUIC"),
        (514, "Syslog"),
        (5353, "mDNS"),
        (51820, "WireGuard"),
    ])
    def test_known_ports(self, port, expected):
        assert _classify_udp_port(12345, port) == expected

    def test_unknown_port(self):
        assert _classify_udp_port(12345, 54321) is None


# ---------------------------------------------------------------------------
# format_analysis
# ---------------------------------------------------------------------------

class TestFormatAnalysis:
    def test_basic_report(self):
        a = PcapAnalysis(
            filepath="/tmp/test.pcap",
            total_packets=100,
            duration_seconds=5.5,
            packet_sizes={"min": 54, "max": 1514, "avg": 256.3, "total_bytes": 25630},
        )
        report = format_analysis(a)
        assert "test.pcap" in report
        assert "100" in report
        assert "5.5" in report
        assert "54" in report

    def test_protocol_distribution(self):
        a = PcapAnalysis(
            filepath="/f",
            total_packets=50,
            protocols={"TCP": 30, "UDP": 15, "ICMP": 5},
        )
        report = format_analysis(a)
        assert "PROTOCOL DISTRIBUTION" in report
        assert "TCP" in report
        assert "UDP" in report
        assert "ICMP" in report

    def test_errors_section(self):
        a = PcapAnalysis(filepath="/f")
        a.errors = [
            "Packet #1: TCP RST — 1.2.3.4:80 → 5.6.7.8:12345",
            "Packet #5: ICMP Destination Unreachable",
        ]
        report = format_analysis(a)
        assert "ERRORS DETECTED" in report
        assert "TCP RST" in report
        assert "ICMP" in report

    def test_warnings_section(self):
        a = PcapAnalysis(filepath="/f")
        a.warnings = ["10 TCP connections without teardown"]
        report = format_analysis(a)
        assert "WARNINGS" in report

    def test_no_issues(self):
        a = PcapAnalysis(filepath="/f")
        report = format_analysis(a)
        assert "No errors or warnings detected" in report

    def test_dns_queries(self):
        a = PcapAnalysis(filepath="/f")
        a.dns_queries = [{"query": "example.com", "count": 5}]
        report = format_analysis(a)
        assert "DNS QUERIES" in report
        assert "example.com" in report

    def test_top_talkers(self):
        a = PcapAnalysis(filepath="/f")
        a.top_talkers = [{"ip": "192.168.1.1", "packets": 42}]
        report = format_analysis(a)
        assert "TOP TALKERS" in report
        assert "192.168.1.1" in report

    def test_conversations(self):
        a = PcapAnalysis(filepath="/f")
        a.conversations = [{"src": "10.0.0.1", "dst": "10.0.0.2", "packets": 100}]
        report = format_analysis(a)
        assert "TOP CONVERSATIONS" in report
        assert "10.0.0.1" in report

    def test_tcp_flags(self):
        a = PcapAnalysis(filepath="/f")
        a.tcp_flags_summary = {"S": 10, "SA": 8, "A": 50, "F": 5}
        report = format_analysis(a)
        assert "TCP FLAGS" in report
