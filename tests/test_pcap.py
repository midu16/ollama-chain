"""Unit tests for pcap.py — PcapAnalysis, format_analysis, port classification, deep analysis helpers."""

import pytest

from ollama_chain.pcap import (
    PcapAnalysis,
    _check_dns_tunneling,
    _classify_ip,
    _classify_tcp_port,
    _classify_udp_port,
    _dns_qtype_name,
    _extract_http_request,
    _extract_http_response,
    _extract_tls_sni,
    _extract_tls_version,
    _infer_os,
    _percentile,
    _shannon_entropy,
    format_analysis,
)
from collections import Counter
import struct


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

    def test_deep_analysis_defaults(self):
        a = PcapAnalysis(filepath="/tmp/test.pcap")
        assert a.tcp_streams == []
        assert a.retransmissions == 0
        assert a.rtt_estimates == []
        assert a.tls_versions == []
        assert a.http_requests == []
        assert a.port_scan_suspects == []
        assert a.bandwidth_timeline == []
        assert a.mac_addresses == []
        assert a.vlan_ids == []
        assert a.ip_classification == {}
        assert a.dns_answers == []
        assert a.payload_entropy_summary == {}
        assert a.tunnel_protocols == []
        assert a.ttl_analysis == {}
        assert a.tcp_window_stats == {}

    def test_extended_analysis_defaults(self):
        a = PcapAnalysis(filepath="/tmp/test.pcap")
        assert a.tcp_handshakes == {}
        assert a.tls_sni_hosts == []
        assert a.http_responses == []
        assert a.dns_record_types == {}
        assert a.dns_latency == []
        assert a.dns_tunneling_suspects == []
        assert a.arp_anomalies == []
        assert a.tcp_options_summary == {}
        assert a.os_fingerprints == []
        assert a.packet_size_distribution == {}
        assert a.icmp_latency == []
        assert a.inter_arrival_time == {}
        assert a.duplicate_acks == 0
        assert a.security_indicators == []


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

    def test_ip_classification_section(self):
        a = PcapAnalysis(filepath="/f")
        a.ip_classification = {"private (RFC1918)": 40, "public": 10}
        report = format_analysis(a)
        assert "IP ADDRESS CLASSIFICATION" in report
        assert "private (RFC1918)" in report

    def test_retransmissions_section(self):
        a = PcapAnalysis(filepath="/f", retransmissions=12)
        report = format_analysis(a)
        assert "TCP RETRANSMISSIONS: 12" in report

    def test_tcp_streams_section(self):
        a = PcapAnalysis(filepath="/f")
        a.tcp_streams = [{
            "src": "10.0.0.1", "dst": "10.0.0.2",
            "sport": 12345, "dport": 80,
            "bytes": 50000, "duration": 2.5, "throughput_kbps": 160.0,
        }]
        report = format_analysis(a)
        assert "TOP TCP STREAMS" in report
        assert "10.0.0.1" in report
        assert "160.0 kbps" in report

    def test_tcp_window_stats_section(self):
        a = PcapAnalysis(filepath="/f")
        a.tcp_window_stats = {"min": 0, "max": 65535, "avg": 32000.5, "zero_count": 2}
        report = format_analysis(a)
        assert "TCP WINDOW SIZE STATS" in report
        assert "zero_window_count=2" in report

    def test_rtt_estimates_section(self):
        a = PcapAnalysis(filepath="/f")
        a.rtt_estimates = [{"src": "1.1.1.1", "dst": "2.2.2.2", "sport": 1234, "dport": 443, "rtt_ms": 15.3}]
        report = format_analysis(a)
        assert "RTT ESTIMATES" in report
        assert "15.3ms" in report

    def test_tls_versions_section(self):
        a = PcapAnalysis(filepath="/f")
        a.tls_versions = [{"version": "TLS 1.2", "count": 20}]
        report = format_analysis(a)
        assert "TLS/SSL VERSIONS" in report
        assert "TLS 1.2" in report

    def test_http_requests_section(self):
        a = PcapAnalysis(filepath="/f")
        a.http_requests = [{"method": "GET", "uri": "/index.html", "src": "10.0.0.1", "dst": "10.0.0.2", "pkt": 5}]
        report = format_analysis(a)
        assert "HTTP REQUESTS" in report
        assert "GET" in report
        assert "/index.html" in report

    def test_port_scan_section(self):
        a = PcapAnalysis(filepath="/f")
        a.port_scan_suspects = [{"ip": "192.168.1.100", "unique_dst_ports": 50}]
        report = format_analysis(a)
        assert "PORT SCAN SUSPECTS" in report
        assert "192.168.1.100" in report

    def test_dns_answers_section(self):
        a = PcapAnalysis(filepath="/f")
        a.dns_answers = [{"query": "example.com", "answer": "93.184.216.34", "type": 1}]
        report = format_analysis(a)
        assert "DNS ANSWERS" in report
        assert "93.184.216.34" in report

    def test_mac_addresses_section(self):
        a = PcapAnalysis(filepath="/f")
        a.mac_addresses = [{"mac": "aa:bb:cc:dd:ee:ff", "packets": 100}]
        report = format_analysis(a)
        assert "MAC ADDRESSES" in report
        assert "aa:bb:cc:dd:ee:ff" in report

    def test_vlan_ids_section(self):
        a = PcapAnalysis(filepath="/f")
        a.vlan_ids = [10, 20, 100]
        report = format_analysis(a)
        assert "VLAN IDs DETECTED" in report
        assert "10" in report

    def test_tunnel_protocols_section(self):
        a = PcapAnalysis(filepath="/f")
        a.tunnel_protocols = [{"protocol": "GRE", "count": 5}]
        report = format_analysis(a)
        assert "TUNNEL PROTOCOLS" in report
        assert "GRE" in report

    def test_ttl_analysis_section(self):
        a = PcapAnalysis(filepath="/f")
        a.ttl_analysis = {
            "most_common": [{"ttl": 64, "count": 100}],
            "unique_ttls": 3,
            "avg_ttl": 60.5,
        }
        report = format_analysis(a)
        assert "TTL ANALYSIS" in report
        assert "TTL=64" in report

    def test_payload_entropy_section(self):
        a = PcapAnalysis(filepath="/f")
        a.payload_entropy_summary = {
            "avg_entropy": 5.2, "max_entropy": 7.9,
            "high_entropy_payloads": 10, "total_sampled": 50,
        }
        report = format_analysis(a)
        assert "PAYLOAD ENTROPY" in report
        assert "5.2" in report

    def test_bandwidth_timeline_section(self):
        a = PcapAnalysis(filepath="/f")
        a.bandwidth_timeline = [
            {"second": 0, "packets": 10, "bytes": 5000},
            {"second": 1, "packets": 20, "bytes": 15000},
        ]
        report = format_analysis(a)
        assert "BANDWIDTH TIMELINE" in report
        assert "Peak" in report


# ---------------------------------------------------------------------------
# _classify_ip
# ---------------------------------------------------------------------------

class TestClassifyIp:
    def test_private_rfc1918(self):
        assert _classify_ip("192.168.1.1") == "private (RFC1918)"
        assert _classify_ip("10.0.0.1") == "private (RFC1918)"
        assert _classify_ip("172.16.0.1") == "private (RFC1918)"

    def test_public(self):
        assert _classify_ip("8.8.8.8") == "public"
        assert _classify_ip("1.1.1.1") == "public"

    def test_loopback(self):
        assert _classify_ip("127.0.0.1") == "loopback"

    def test_multicast(self):
        assert _classify_ip("224.0.0.1") == "multicast"

    def test_link_local(self):
        assert _classify_ip("169.254.1.1") == "link-local"

    def test_invalid(self):
        assert _classify_ip("not_an_ip") == "invalid"


# ---------------------------------------------------------------------------
# _shannon_entropy
# ---------------------------------------------------------------------------

class TestShannonEntropy:
    def test_zero_entropy(self):
        assert _shannon_entropy(b"\x00" * 100) == 0.0

    def test_max_entropy(self):
        data = bytes(range(256))
        entropy = _shannon_entropy(data)
        assert 7.9 < entropy <= 8.0

    def test_empty(self):
        assert _shannon_entropy(b"") == 0.0

    def test_moderate_entropy(self):
        data = b"AAABBBCCC"
        entropy = _shannon_entropy(data)
        assert 1.0 < entropy < 2.0


# ---------------------------------------------------------------------------
# _extract_tls_version
# ---------------------------------------------------------------------------

class TestExtractTlsVersion:
    def test_tls12_record(self):
        data = bytes([22, 3, 3, 0, 5, 0, 0, 0, 0, 0, 0])
        counter = Counter()
        _extract_tls_version(data, counter)
        assert counter["TLS 1.2"] == 1

    def test_tls13_record(self):
        data = bytes([22, 3, 4, 0, 5, 0, 0, 0, 0, 0, 0])
        counter = Counter()
        _extract_tls_version(data, counter)
        assert counter["TLS 1.3"] == 1

    def test_client_hello_detected(self):
        data = bytes([22, 3, 1, 0, 10, 1, 0, 0, 6, 3, 3, 0, 0, 0, 0])
        counter = Counter()
        _extract_tls_version(data, counter)
        assert counter["TLS 1.0"] == 1
        assert counter["ClientHello TLS 1.2"] == 1

    def test_non_tls_ignored(self):
        data = bytes([0, 3, 3, 0, 5, 0, 0, 0, 0, 0, 0])
        counter = Counter()
        _extract_tls_version(data, counter)
        assert len(counter) == 0

    def test_too_short(self):
        data = bytes([22, 3])
        counter = Counter()
        _extract_tls_version(data, counter)
        assert len(counter) == 0


# ---------------------------------------------------------------------------
# _extract_http_request
# ---------------------------------------------------------------------------

class TestExtractHttpRequest:
    def test_get_request(self):
        data = b"GET /api/v1/users HTTP/1.1\r\nHost: example.com\r\n"
        results = []
        _extract_http_request(data, results, "10.0.0.1", "10.0.0.2", 0)
        assert len(results) == 1
        assert results[0]["method"] == "GET"
        assert results[0]["uri"] == "/api/v1/users"

    def test_post_request(self):
        data = b"POST /submit HTTP/1.1\r\nContent-Type: text/html\r\n"
        results = []
        _extract_http_request(data, results, "10.0.0.1", "10.0.0.2", 5)
        assert len(results) == 1
        assert results[0]["method"] == "POST"
        assert results[0]["pkt"] == 6

    def test_non_http_ignored(self):
        data = b"\x00\x01\x02\x03binary data"
        results = []
        _extract_http_request(data, results, "10.0.0.1", "10.0.0.2", 0)
        assert len(results) == 0

    def test_all_methods(self):
        for method in (b"PUT", b"DELETE", b"HEAD", b"PATCH", b"OPTIONS", b"CONNECT"):
            data = method + b" /path HTTP/1.1\r\n"
            results = []
            _extract_http_request(data, results, None, None, 0)
            assert len(results) == 1
            assert results[0]["method"] == method.decode()


# ---------------------------------------------------------------------------
# _extract_tls_sni
# ---------------------------------------------------------------------------

def _build_client_hello(sni_hostname: str = "example.com") -> bytes:
    """Build a minimal TLS ClientHello with an SNI extension."""
    sni_bytes = sni_hostname.encode("ascii")
    sni_ext_data = (
        struct.pack("!H", len(sni_bytes) + 3) +  # server name list length
        b"\x00" +                                  # host name type
        struct.pack("!H", len(sni_bytes)) +        # host name length
        sni_bytes
    )
    sni_ext = struct.pack("!HH", 0, len(sni_ext_data)) + sni_ext_data
    extensions = struct.pack("!H", len(sni_ext)) + sni_ext
    cipher_suites = struct.pack("!H", 2) + struct.pack("!H", 0x1301)
    compression = b"\x01\x00"
    client_hello_body = (
        b"\x03\x03" +            # version TLS 1.2
        b"\x00" * 32 +           # random
        b"\x00" +                # session ID length = 0
        cipher_suites +
        compression +
        extensions
    )
    handshake = b"\x01" + struct.pack("!I", len(client_hello_body))[1:] + client_hello_body
    record = b"\x16\x03\x01" + struct.pack("!H", len(handshake)) + handshake
    return record


class TestExtractTlsSni:
    def test_sni_extraction(self):
        data = _build_client_hello("www.example.com")
        result = _extract_tls_sni(data)
        assert result == "www.example.com"

    def test_sni_different_host(self):
        data = _build_client_hello("api.github.com")
        assert _extract_tls_sni(data) == "api.github.com"

    def test_non_tls_returns_none(self):
        assert _extract_tls_sni(b"\x00" * 100) is None

    def test_too_short_returns_none(self):
        assert _extract_tls_sni(b"\x16\x03\x01") is None

    def test_not_client_hello_returns_none(self):
        data = b"\x16\x03\x03\x00\x05\x02\x00\x00\x00\x00\x00"
        assert _extract_tls_sni(data) is None


# ---------------------------------------------------------------------------
# _extract_http_response
# ---------------------------------------------------------------------------

class TestExtractHttpResponse:
    def test_200_ok(self):
        data = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
        results = []
        _extract_http_response(data, results, "10.0.0.1", "10.0.0.2", 0)
        assert len(results) == 1
        assert results[0]["status_code"] == 200
        assert results[0]["reason"] == "OK"
        assert results[0]["version"] == "HTTP/1.1"

    def test_404_not_found(self):
        data = b"HTTP/1.1 404 Not Found\r\n"
        results = []
        _extract_http_response(data, results, None, None, 5)
        assert len(results) == 1
        assert results[0]["status_code"] == 404
        assert results[0]["pkt"] == 6

    def test_500_server_error(self):
        data = b"HTTP/1.0 500 Internal Server Error\r\n"
        results = []
        _extract_http_response(data, results, "1.1.1.1", "2.2.2.2", 0)
        assert results[0]["status_code"] == 500

    def test_non_http_ignored(self):
        data = b"NOT HTTP DATA"
        results = []
        _extract_http_response(data, results, None, None, 0)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# _dns_qtype_name
# ---------------------------------------------------------------------------

class TestDnsQtypeName:
    @pytest.mark.parametrize("qtype,expected", [
        (1, "A"), (28, "AAAA"), (5, "CNAME"), (15, "MX"),
        (16, "TXT"), (2, "NS"), (6, "SOA"), (33, "SRV"),
        (12, "PTR"), (255, "ANY"), (252, "AXFR"), (65, "HTTPS"),
    ])
    def test_known_types(self, qtype, expected):
        assert _dns_qtype_name(qtype) == expected

    def test_unknown_type(self):
        assert _dns_qtype_name(9999) == "TYPE9999"


# ---------------------------------------------------------------------------
# _check_dns_tunneling
# ---------------------------------------------------------------------------

class TestCheckDnsTunneling:
    def test_normal_domain_returns_none(self):
        assert _check_dns_tunneling("www.google.com") is None

    def test_short_subdomain_returns_none(self):
        assert _check_dns_tunneling("a.b.com") is None

    def test_high_entropy_long_subdomain(self):
        qname = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2.evil.com"
        result = _check_dns_tunneling(qname)
        assert result is not None
        assert result["domain"] == "evil.com"

    def test_hex_encoded_labels(self):
        hex_sub = "4a6f686e446f6520536d697468" * 2
        qname = f"{hex_sub}.exfil.net"
        result = _check_dns_tunneling(qname)
        assert result is not None
        assert "hex_encoded" in result

    def test_very_long_query(self):
        long_sub = "a" * 60 + "." + "b" * 60
        qname = f"{long_sub}.tunnel.org"
        result = _check_dns_tunneling(qname)
        assert result is not None
        assert result.get("long_query") or result.get("long_label")


# ---------------------------------------------------------------------------
# _infer_os
# ---------------------------------------------------------------------------

class TestInferOs:
    def test_linux_ttl64_wscale7(self):
        result = _infer_os(64, 29200, 1460, 7)
        assert "Linux" in result

    def test_linux_ttl_hop_decremented(self):
        result = _infer_os(58, 29200, 1460, 7)
        assert "Linux" in result

    def test_macos_ttl64_wscale6(self):
        result = _infer_os(64, 65535, 1460, 6)
        assert "macOS" in result or "iOS" in result

    def test_windows_ttl128_wscale8(self):
        result = _infer_os(128, 65535, 1460, 8)
        assert "Windows" in result

    def test_windows_xp(self):
        result = _infer_os(128, 8192, 1460, None)
        assert "Windows" in result and "XP" in result

    def test_network_equipment_ttl255(self):
        result = _infer_os(255, 16384, 1460, None)
        assert "Network equipment" in result or "Solaris" in result

    def test_unknown_ttl(self):
        result = _infer_os(30, 1024, None, None)
        assert "Legacy" in result or "TTL=32" in result


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_median_odd(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_median_even(self):
        assert _percentile([1, 2, 3, 4], 50) == 2.5

    def test_p0(self):
        assert _percentile([10, 20, 30], 0) == 10.0

    def test_p100(self):
        assert _percentile([10, 20, 30], 100) == 30.0

    def test_p25(self):
        result = _percentile([1, 2, 3, 4, 5, 6, 7, 8], 25)
        assert 2.0 <= result <= 3.0

    def test_p95(self):
        data = list(range(1, 101))
        result = _percentile(data, 95)
        assert 95.0 <= result <= 96.0

    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([42], 50) == 42.0


# ---------------------------------------------------------------------------
# format_analysis — extended sections
# ---------------------------------------------------------------------------

class TestFormatAnalysisExtended:
    def test_tcp_handshakes_section(self):
        a = PcapAnalysis(filepath="/f")
        a.tcp_handshakes = {
            "established": 10, "syn_sent_no_reply": 3,
            "syn_received_incomplete": 1, "reset": 2, "total_attempted": 16,
        }
        report = format_analysis(a)
        assert "HANDSHAKE ANALYSIS" in report
        assert "Established" in report
        assert "SYN no reply" in report

    def test_duplicate_acks_section(self):
        a = PcapAnalysis(filepath="/f", duplicate_acks=42)
        report = format_analysis(a)
        assert "DUPLICATE ACKs: 42" in report

    def test_tcp_options_section(self):
        a = PcapAnalysis(filepath="/f")
        a.tcp_options_summary = {
            "syn_packets_analyzed": 5, "mss_most_common": 1460,
            "mss_min": 1360, "mss_max": 1460,
            "wscale_most_common": 7, "wscale_min": 6, "wscale_max": 8,
            "sack_capable": 4, "timestamp_capable": 3,
        }
        report = format_analysis(a)
        assert "TCP OPTIONS" in report
        assert "MSS" in report
        assert "1460" in report
        assert "SACK capable" in report

    def test_os_fingerprints_section(self):
        a = PcapAnalysis(filepath="/f")
        a.os_fingerprints = [{"os": "Linux 3.x-6.x", "syn_count": 12}]
        report = format_analysis(a)
        assert "OS FINGERPRINTS" in report
        assert "Linux" in report

    def test_tls_sni_section(self):
        a = PcapAnalysis(filepath="/f")
        a.tls_sni_hosts = [{"host": "www.example.com", "count": 5}]
        report = format_analysis(a)
        assert "TLS SNI HOSTNAMES" in report
        assert "www.example.com" in report

    def test_http_responses_section(self):
        a = PcapAnalysis(filepath="/f")
        a.http_responses = [
            {"version": "HTTP/1.1", "status_code": 200, "reason": "OK",
             "src": "1.1.1.1", "dst": "2.2.2.2", "pkt": 1},
            {"version": "HTTP/1.1", "status_code": 200, "reason": "OK",
             "src": "1.1.1.1", "dst": "2.2.2.2", "pkt": 3},
            {"version": "HTTP/1.1", "status_code": 404, "reason": "Not Found",
             "src": "1.1.1.1", "dst": "2.2.2.2", "pkt": 5},
        ]
        report = format_analysis(a)
        assert "HTTP RESPONSES" in report
        assert "200" in report
        assert "404" in report

    def test_dns_record_types_section(self):
        a = PcapAnalysis(filepath="/f")
        a.dns_record_types = {"A": 50, "AAAA": 20, "CNAME": 5}
        report = format_analysis(a)
        assert "DNS RECORD TYPES" in report
        assert "AAAA" in report

    def test_dns_latency_section(self):
        a = PcapAnalysis(filepath="/f")
        a.dns_latency = [{"query": "example.com", "latency_ms": 15.3}]
        report = format_analysis(a)
        assert "DNS QUERY LATENCY" in report
        assert "15.3" in report

    def test_dns_tunneling_section(self):
        a = PcapAnalysis(filepath="/f")
        a.dns_tunneling_suspects = [{
            "domain": "evil.com", "qname": "abc123.evil.com",
            "high_entropy": 4.5,
        }]
        report = format_analysis(a)
        assert "DNS TUNNELING SUSPECTS" in report
        assert "evil.com" in report

    def test_arp_anomalies_section(self):
        a = PcapAnalysis(filepath="/f")
        a.arp_anomalies = [{
            "type": "ip_mac_conflict", "ip": "10.0.0.1",
            "macs": ["aa:bb:cc:dd:ee:01", "aa:bb:cc:dd:ee:02"],
            "detail": "IP 10.0.0.1 seen with 2 MACs",
        }]
        report = format_analysis(a)
        assert "ARP ANOMALIES" in report
        assert "ip_mac_conflict" in report

    def test_icmp_latency_section(self):
        a = PcapAnalysis(filepath="/f")
        a.icmp_latency = [{"src": "10.0.0.1", "dst": "10.0.0.2", "seq": 1, "rtt_ms": 3.5}]
        report = format_analysis(a)
        assert "ICMP ECHO LATENCY" in report
        assert "3.5" in report

    def test_packet_size_distribution_section(self):
        a = PcapAnalysis(filepath="/f")
        a.packet_size_distribution = {
            "p25": 64.0, "p50": 256.0, "p75": 1024.0,
            "p95": 1460.0, "p99": 1514.0, "stddev": 450.2,
        }
        report = format_analysis(a)
        assert "PACKET SIZE DISTRIBUTION" in report
        assert "p50=256B" in report
        assert "p95=1460B" in report

    def test_inter_arrival_time_section(self):
        a = PcapAnalysis(filepath="/f")
        a.inter_arrival_time = {
            "mean_ms": 1.5, "stddev_ms": 0.8, "min_ms": 0.01,
            "max_ms": 50.0, "p50_ms": 1.0, "p99_ms": 10.0,
            "jitter_mean_ms": 0.5, "burst_packets": 100, "burst_pct": 25.0,
        }
        report = format_analysis(a)
        assert "INTER-ARRIVAL TIME" in report
        assert "jitter" in report
        assert "Burst" in report

    def test_security_indicators_section(self):
        a = PcapAnalysis(filepath="/f")
        a.security_indicators = [{
            "severity": "high", "category": "deprecated_tls",
            "detail": "Deprecated TLS/SSL versions: SSL 3.0",
        }, {
            "severity": "medium", "category": "cleartext_protocols",
            "detail": "Cleartext protocols in use: HTTP(50)",
        }]
        report = format_analysis(a)
        assert "SECURITY ASSESSMENT" in report
        assert "HIGH" in report
        assert "MEDIUM" in report
        assert "deprecated_tls" in report
