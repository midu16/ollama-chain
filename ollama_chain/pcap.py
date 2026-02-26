"""Parse .pcap/.pcapng files and extract structured analysis for LLM consumption."""

import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from scapy.all import (
    rdpcap,
    IP,
    IPv6,
    TCP,
    UDP,
    ICMP,
    DNS,
    ARP,
    Ether,
    Raw,
    conf,
)

conf.verb = 0


@dataclass
class PcapAnalysis:
    filepath: str
    total_packets: int = 0
    duration_seconds: float = 0.0
    protocols: dict = field(default_factory=dict)
    conversations: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    dns_queries: list = field(default_factory=list)
    tcp_flags_summary: dict = field(default_factory=dict)
    packet_sizes: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    top_talkers: list = field(default_factory=list)


def analyze_pcap(filepath: str, max_packets: int = 50000) -> PcapAnalysis:
    """Read a pcap file and return structured analysis."""
    if not os.path.isfile(filepath):
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in (".pcap", ".pcapng", ".cap"):
        print(f"Warning: unexpected extension '{ext}', attempting to read anyway", file=sys.stderr)

    try:
        packets = rdpcap(filepath, count=max_packets)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Error reading pcap: {e}", file=sys.stderr)
        sys.exit(1)

    if packets is None:
        packets = []

    analysis = PcapAnalysis(filepath=filepath, total_packets=len(packets))

    if not packets:
        analysis.warnings.append("Capture file is empty — no packets found.")
        return analysis

    timestamps = [float(p.time) for p in packets if p.time]
    if len(timestamps) >= 2:
        analysis.duration_seconds = round(timestamps[-1] - timestamps[0], 3)

    protocol_counter = Counter()
    tcp_flags_counter = Counter()
    ip_counter = Counter()
    conversation_counter = Counter()
    dns_queries = []
    errors = []
    warnings = []

    syn_seen = set()
    fin_seen = set()

    size_min = float('inf')
    size_max = 0
    size_sum = 0
    size_count = 0

    for i, pkt in enumerate(packets):
        size = len(pkt)
        size_sum += size
        size_count += 1
        if size < size_min:
            size_min = size
        if size > size_max:
            size_max = size

        if pkt.haslayer(ARP):
            protocol_counter["ARP"] += 1
            if pkt[ARP].op == 1:
                protocol_counter["ARP Request"] += 1
            elif pkt[ARP].op == 2:
                protocol_counter["ARP Reply"] += 1

        src_ip, dst_ip = None, None
        if pkt.haslayer(IP):
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            protocol_counter["IPv4"] += 1

            if pkt[IP].ttl == 0:
                errors.append(f"Packet #{i+1}: TTL=0 (expired) {src_ip} → {dst_ip}")
            if pkt[IP].ttl == 1:
                warnings.append(f"Packet #{i+1}: TTL=1 (traceroute/expiring) {src_ip} → {dst_ip}")
            if pkt[IP].flags.MF or pkt[IP].frag > 0:
                protocol_counter["IP Fragmented"] += 1

        elif pkt.haslayer(IPv6):
            src_ip = pkt[IPv6].src
            dst_ip = pkt[IPv6].dst
            protocol_counter["IPv6"] += 1

        if src_ip and dst_ip:
            ip_counter[src_ip] += 1
            ip_counter[dst_ip] += 1
            conv_key = tuple(sorted([src_ip, dst_ip]))
            conversation_counter[conv_key] += 1

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            sport, dport = tcp.sport, tcp.dport
            protocol_counter["TCP"] += 1

            flags = str(tcp.flags)
            tcp_flags_counter[flags] += 1

            stream_key = (src_ip, dst_ip, sport, dport) if src_ip else None

            if "S" in flags and "A" not in flags:
                protocol_counter["TCP SYN"] += 1
                if stream_key:
                    syn_seen.add(stream_key)

            if "R" in flags:
                protocol_counter["TCP RST"] += 1
                errors.append(
                    f"Packet #{i+1}: TCP RST — {src_ip}:{sport} → {dst_ip}:{dport} "
                    f"(connection refused or reset)"
                )

            if "F" in flags:
                protocol_counter["TCP FIN"] += 1
                if stream_key:
                    fin_seen.add(stream_key)

            if tcp.window == 0:
                errors.append(
                    f"Packet #{i+1}: TCP zero window — {src_ip}:{sport} → {dst_ip}:{dport} "
                    f"(receiver buffer full)"
                )
                protocol_counter["TCP Zero Window"] += 1

            well_known = _classify_tcp_port(sport, dport)
            if well_known:
                protocol_counter[well_known] += 1

        elif pkt.haslayer(UDP):
            protocol_counter["UDP"] += 1
            udp = pkt[UDP]
            well_known = _classify_udp_port(udp.sport, udp.dport)
            if well_known:
                protocol_counter[well_known] += 1

        elif pkt.haslayer(ICMP):
            protocol_counter["ICMP"] += 1
            icmp = pkt[ICMP]
            if icmp.type == 3:
                errors.append(
                    f"Packet #{i+1}: ICMP Destination Unreachable "
                    f"(code={icmp.code}) {src_ip} → {dst_ip}"
                )
                protocol_counter["ICMP Dest Unreachable"] += 1
            elif icmp.type == 11:
                errors.append(
                    f"Packet #{i+1}: ICMP Time Exceeded {src_ip} → {dst_ip}"
                )
                protocol_counter["ICMP Time Exceeded"] += 1
            elif icmp.type == 5:
                warnings.append(
                    f"Packet #{i+1}: ICMP Redirect {src_ip} → {dst_ip}"
                )

        if pkt.haslayer(DNS):
            protocol_counter["DNS"] += 1
            dns = pkt[DNS]
            if dns.qr == 0 and dns.qd:
                qname = dns.qd.qname.decode(errors="ignore").rstrip(".")
                dns_queries.append(qname)
            if dns.qr == 1 and dns.rcode != 0:
                rcode_map = {1: "FormErr", 2: "ServFail", 3: "NXDomain", 4: "NotImp", 5: "Refused"}
                rcode_name = rcode_map.get(dns.rcode, f"code={dns.rcode}")
                qname = dns.qd.qname.decode(errors="ignore").rstrip(".") if dns.qd else "?"
                errors.append(f"Packet #{i+1}: DNS error {rcode_name} for '{qname}'")

    half_open = len(syn_seen) - len(syn_seen & fin_seen)
    if half_open > 10:
        warnings.append(
            f"{half_open} TCP connections initiated (SYN) without proper teardown — "
            f"possible scan, aborted connections, or capture ended mid-session"
        )

    analysis.protocols = dict(protocol_counter.most_common())
    analysis.tcp_flags_summary = dict(tcp_flags_counter.most_common(15))

    analysis.top_talkers = [
        {"ip": ip, "packets": count}
        for ip, count in ip_counter.most_common(15)
    ]

    analysis.conversations = [
        {"src": pair[0], "dst": pair[1], "packets": count}
        for pair, count in conversation_counter.most_common(20)
    ]

    dns_counts = Counter(dns_queries)
    analysis.dns_queries = [
        {"query": q, "count": c}
        for q, c in dns_counts.most_common(25)
    ]

    if size_count:
        analysis.packet_sizes = {
            "min": size_min,
            "max": size_max,
            "avg": round(size_sum / size_count, 1),
            "total_bytes": size_sum,
        }

    analysis.errors = errors[:100]
    analysis.warnings = warnings[:50]

    return analysis


def format_analysis(a: PcapAnalysis) -> str:
    """Format PcapAnalysis into a structured text report for LLM consumption."""
    lines = []
    lines.append(f"=== PCAP ANALYSIS: {a.filepath} ===\n")
    lines.append(f"Total packets: {a.total_packets}")
    lines.append(f"Capture duration: {a.duration_seconds}s")

    if a.packet_sizes:
        ps = a.packet_sizes
        total_mb = ps['total_bytes'] / (1024 * 1024)
        lines.append(f"Total data: {total_mb:.2f} MB")
        lines.append(f"Packet sizes: min={ps['min']}B, max={ps['max']}B, avg={ps['avg']}B")

    if a.protocols:
        lines.append("\n--- PROTOCOL DISTRIBUTION ---")
        for proto, count in a.protocols.items():
            pct = (count / a.total_packets) * 100 if a.total_packets else 0
            lines.append(f"  {proto:<25} {count:>8}  ({pct:.1f}%)")

    if a.tcp_flags_summary:
        lines.append("\n--- TCP FLAGS ---")
        for flags, count in a.tcp_flags_summary.items():
            lines.append(f"  {flags:<10} {count:>8}")

    if a.top_talkers:
        lines.append("\n--- TOP TALKERS (by packet count) ---")
        for t in a.top_talkers:
            lines.append(f"  {t['ip']:<40} {t['packets']:>8} packets")

    if a.conversations:
        lines.append("\n--- TOP CONVERSATIONS ---")
        for c in a.conversations:
            lines.append(f"  {c['src']:<40} <> {c['dst']:<40} {c['packets']:>6} packets")

    if a.dns_queries:
        lines.append("\n--- DNS QUERIES ---")
        for d in a.dns_queries:
            lines.append(f"  {d['query']:<50} {d['count']:>5}x")

    if a.errors:
        lines.append(f"\n--- ERRORS DETECTED ({len(a.errors)}) ---")
        for e in a.errors:
            lines.append(f"  [ERR] {e}")

    if a.warnings:
        lines.append(f"\n--- WARNINGS ({len(a.warnings)}) ---")
        for w in a.warnings:
            lines.append(f"  [WARN] {w}")

    if not a.errors and not a.warnings:
        lines.append("\n--- No errors or warnings detected ---")

    return "\n".join(lines)


def _classify_tcp_port(sport: int, dport: int) -> str | None:
    known = {
        20: "FTP-Data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
        53: "DNS/TCP", 80: "HTTP", 110: "POP3", 143: "IMAP",
        443: "HTTPS/TLS", 465: "SMTPS", 587: "SMTP-Submission",
        993: "IMAPS", 995: "POP3S", 3306: "MySQL", 3389: "RDP",
        5432: "PostgreSQL", 5900: "VNC", 6379: "Redis", 8080: "HTTP-Alt",
        8443: "HTTPS-Alt", 27017: "MongoDB",
    }
    return known.get(dport) or known.get(sport)


def _classify_udp_port(sport: int, dport: int) -> str | None:
    known = {
        53: "DNS", 67: "DHCP-Server", 68: "DHCP-Client",
        123: "NTP", 161: "SNMP", 162: "SNMP-Trap",
        443: "QUIC", 500: "IKE/IPsec", 514: "Syslog",
        1194: "OpenVPN", 5353: "mDNS", 51820: "WireGuard",
    }
    return known.get(dport) or known.get(sport)
