"""Parse .pcap/.pcapng files and extract structured analysis for LLM consumption."""

import ipaddress
import math
import os
import struct
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
    Dot1Q,
    GRE,
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
    # --- deep analysis fields ---
    tcp_streams: list = field(default_factory=list)
    retransmissions: int = 0
    rtt_estimates: list = field(default_factory=list)
    tls_versions: list = field(default_factory=list)
    http_requests: list = field(default_factory=list)
    port_scan_suspects: list = field(default_factory=list)
    bandwidth_timeline: list = field(default_factory=list)
    mac_addresses: list = field(default_factory=list)
    vlan_ids: list = field(default_factory=list)
    ip_classification: dict = field(default_factory=dict)
    dns_answers: list = field(default_factory=list)
    payload_entropy_summary: dict = field(default_factory=dict)
    tunnel_protocols: list = field(default_factory=list)
    ttl_analysis: dict = field(default_factory=dict)
    tcp_window_stats: dict = field(default_factory=dict)
    # --- extended deep analysis fields ---
    tcp_handshakes: dict = field(default_factory=dict)
    tls_sni_hosts: list = field(default_factory=list)
    http_responses: list = field(default_factory=list)
    dns_record_types: dict = field(default_factory=dict)
    dns_latency: list = field(default_factory=list)
    dns_tunneling_suspects: list = field(default_factory=list)
    arp_anomalies: list = field(default_factory=list)
    tcp_options_summary: dict = field(default_factory=dict)
    os_fingerprints: list = field(default_factory=list)
    packet_size_distribution: dict = field(default_factory=dict)
    icmp_latency: list = field(default_factory=list)
    inter_arrival_time: dict = field(default_factory=dict)
    duplicate_acks: int = 0
    security_indicators: list = field(default_factory=list)


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

    # deep analysis collectors
    tcp_seq_tracker: dict[tuple, list] = defaultdict(list)
    syn_timestamps: dict[tuple, float] = {}
    synack_timestamps: dict[tuple, float] = {}
    stream_bytes: dict[tuple, int] = defaultdict(int)
    stream_timestamps: dict[tuple, list] = defaultdict(list)
    tcp_windows: list[int] = []
    mac_counter = Counter()
    vlan_set: set[int] = set()
    ip_class_counter = Counter()
    tls_versions_seen = Counter()
    http_reqs: list[dict] = []
    src_dst_ports: dict[str, set] = defaultdict(set)
    time_buckets: dict[int, dict] = defaultdict(lambda: {"packets": 0, "bytes": 0})
    dns_answer_list: list[dict] = []
    payload_sizes: list[int] = []
    payload_entropy_vals: list[float] = []
    tunnel_counter = Counter()
    ttl_values: list[int] = []

    # extended deep analysis collectors
    all_pkt_sizes: list[int] = []
    handshake_state: dict[tuple, str] = {}
    mss_values: list[int] = []
    wscale_values: list[int] = []
    sack_capable_count = 0
    ts_capable_count = 0
    syn_fingerprints: list[dict] = []
    ack_seen_set: set[tuple] = set()
    dup_ack_count = 0
    icmp_echo_tracker: dict[tuple, float] = {}
    icmp_latency_list: list[dict] = []
    dns_qtype_counter = Counter()
    dns_query_times: dict[int, tuple] = {}
    dns_latency_list: list[dict] = []
    dns_tunnel_indicators: list[dict] = []
    arp_ip_to_macs: dict[str, set] = defaultdict(set)
    gratuitous_arp_count = 0
    sni_counter = Counter()
    http_resp_list: list[dict] = []

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

            if dns.qr == 1 and dns.ancount and dns.ancount > 0:
                try:
                    qname = dns.qd.qname.decode(errors="ignore").rstrip(".") if dns.qd else "?"
                    for rr_idx in range(dns.ancount):
                        rr = dns.an[rr_idx] if dns.an else None
                        if rr and hasattr(rr, "rdata"):
                            rdata = rr.rdata
                            if isinstance(rdata, bytes):
                                rdata = rdata.decode(errors="ignore")
                            dns_answer_list.append({"query": qname, "answer": str(rdata), "type": rr.type})
                            break
                except Exception:
                    pass

        # --- deep per-packet analysis (additive, no existing logic altered) ---

        pkt_time = float(pkt.time) if pkt.time else None

        if pkt_time is not None and timestamps:
            bucket = int(pkt_time - timestamps[0])
            time_buckets[bucket]["packets"] += 1
            time_buckets[bucket]["bytes"] += size

        if pkt.haslayer(Ether):
            eth = pkt[Ether]
            mac_counter[eth.src] += 1
            mac_counter[eth.dst] += 1

        if pkt.haslayer(Dot1Q):
            vlan_set.add(pkt[Dot1Q].vlan)
            protocol_counter["802.1Q VLAN"] += 1

        if pkt.haslayer(GRE):
            tunnel_counter["GRE"] += 1

        if src_ip:
            ip_class_counter[_classify_ip(src_ip)] += 1
        if dst_ip:
            ip_class_counter[_classify_ip(dst_ip)] += 1

        if pkt.haslayer(IP):
            ttl_values.append(pkt[IP].ttl)

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            tcp_windows.append(tcp.window)
            payload_len = len(tcp.payload) if tcp.payload else 0
            if src_ip and dst_ip:
                fwd_key = (src_ip, dst_ip, tcp.sport, tcp.dport)
                rev_key = (dst_ip, src_ip, tcp.dport, tcp.sport)

                stream_bytes[fwd_key] += payload_len
                if pkt_time is not None:
                    stream_timestamps[fwd_key].append(pkt_time)

                tcp_seq_tracker[fwd_key].append((i, tcp.seq, payload_len, pkt_time))

                flags = str(tcp.flags)
                if "S" in flags and "A" not in flags and pkt_time is not None:
                    syn_timestamps[fwd_key] = pkt_time
                if "S" in flags and "A" in flags and pkt_time is not None:
                    synack_timestamps[fwd_key] = pkt_time

                src_dst_ports[src_ip].add(tcp.dport)

        if pkt.haslayer(Raw):
            raw_data = bytes(pkt[Raw].load)
            pl = len(raw_data)
            if pl > 0:
                payload_sizes.append(pl)
                if pl >= 16:
                    payload_entropy_vals.append(_shannon_entropy(raw_data[:256]))

                _extract_tls_version(raw_data, tls_versions_seen)
                _extract_http_request(raw_data, http_reqs, src_ip, dst_ip, i)

        # --- extended per-packet analysis ---

        all_pkt_sizes.append(size)

        if pkt.haslayer(TCP) and src_ip and dst_ip:
            tcp_e = pkt[TCP]
            flags_e = str(tcp_e.flags)
            fwd = (src_ip, dst_ip, tcp_e.sport, tcp_e.dport)
            rev = (dst_ip, src_ip, tcp_e.dport, tcp_e.sport)
            payload_e = len(tcp_e.payload) if tcp_e.payload else 0

            if "S" in flags_e and "A" not in flags_e:
                if fwd not in handshake_state:
                    handshake_state[fwd] = "SYN_SENT"
                syn_fp = {
                    "src": src_ip, "dst": dst_ip,
                    "ttl": pkt[IP].ttl if pkt.haslayer(IP) else None,
                    "window": tcp_e.window,
                }
                for opt_name, opt_val in tcp_e.options:
                    if opt_name == "MSS":
                        mss_values.append(opt_val)
                        syn_fp["mss"] = opt_val
                    elif opt_name == "WScale":
                        wscale_values.append(opt_val)
                        syn_fp["wscale"] = opt_val
                    elif opt_name == "SAckOK":
                        sack_capable_count += 1
                        syn_fp["sack"] = True
                    elif opt_name == "Timestamp":
                        ts_capable_count += 1
                syn_fingerprints.append(syn_fp)
            elif "S" in flags_e and "A" in flags_e:
                if rev in handshake_state and handshake_state[rev] == "SYN_SENT":
                    handshake_state[rev] = "SYN_RECEIVED"
            elif "R" in flags_e:
                for k in (fwd, rev):
                    if k in handshake_state and handshake_state[k] in ("SYN_SENT", "SYN_RECEIVED"):
                        handshake_state[k] = "RESET"
            elif "A" in flags_e and "S" not in flags_e and "F" not in flags_e and "R" not in flags_e:
                if fwd in handshake_state and handshake_state[fwd] == "SYN_RECEIVED":
                    handshake_state[fwd] = "ESTABLISHED"

            if "A" in flags_e and payload_e == 0 and "S" not in flags_e and "F" not in flags_e and "R" not in flags_e:
                ack_key = (src_ip, dst_ip, tcp_e.sport, tcp_e.dport, tcp_e.ack)
                if ack_key in ack_seen_set:
                    dup_ack_count += 1
                else:
                    ack_seen_set.add(ack_key)

        if pkt.haslayer(ICMP) and src_ip and dst_ip:
            icmp_e = pkt[ICMP]
            icmp_id = getattr(icmp_e, "id", 0)
            icmp_seq = getattr(icmp_e, "seq", 0)
            if icmp_e.type == 8:
                if pkt_time is not None:
                    icmp_echo_tracker[(src_ip, dst_ip, icmp_id, icmp_seq)] = pkt_time
            elif icmp_e.type == 0:
                req_key = (dst_ip, src_ip, icmp_id, icmp_seq)
                req_time = icmp_echo_tracker.get(req_key)
                if req_time is not None and pkt_time is not None and pkt_time > req_time:
                    icmp_latency_list.append({
                        "src": dst_ip, "dst": src_ip,
                        "seq": icmp_seq, "rtt_ms": round((pkt_time - req_time) * 1000, 2),
                    })

        if pkt.haslayer(DNS):
            dns_e = pkt[DNS]
            if dns_e.qr == 0 and dns_e.qd:
                qtype = dns_e.qd.qtype
                dns_qtype_counter[_dns_qtype_name(qtype)] += 1
                qname_e = dns_e.qd.qname.decode(errors="ignore").rstrip(".")
                tun = _check_dns_tunneling(qname_e)
                if tun:
                    dns_tunnel_indicators.append(tun)
                if pkt_time is not None:
                    dns_query_times[dns_e.id] = (pkt_time, qname_e)
            elif dns_e.qr == 1:
                qinfo = dns_query_times.get(dns_e.id)
                if qinfo and pkt_time is not None:
                    lat = round((pkt_time - qinfo[0]) * 1000, 2)
                    if lat >= 0:
                        dns_latency_list.append({"query": qinfo[1], "latency_ms": lat})

        if pkt.haslayer(ARP):
            arp_e = pkt[ARP]
            if arp_e.op in (1, 2):
                s_ip, s_mac = arp_e.psrc, arp_e.hwsrc
                if s_ip and s_mac and s_ip != "0.0.0.0":
                    arp_ip_to_macs[s_ip].add(s_mac)
                if arp_e.op == 2 and arp_e.psrc == arp_e.pdst:
                    gratuitous_arp_count += 1

        if pkt.haslayer(Raw):
            raw_e = bytes(pkt[Raw].load)
            if len(raw_e) >= 44 and raw_e[0] == 22 and len(raw_e) > 5 and raw_e[5] == 1:
                sni = _extract_tls_sni(raw_e)
                if sni:
                    sni_counter[sni] += 1
            if len(raw_e) > 12 and raw_e[:5] == b"HTTP/":
                _extract_http_response(raw_e, http_resp_list, src_ip, dst_ip, i)

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

    # --- deep analysis aggregation (all additive) ---

    retransmission_count = 0
    for key, entries in tcp_seq_tracker.items():
        seen_seqs: dict[int, int] = {}
        for _idx, seq, plen, _t in entries:
            if plen == 0:
                continue
            if seq in seen_seqs:
                retransmission_count += 1
            else:
                seen_seqs[seq] = plen
    analysis.retransmissions = retransmission_count
    if retransmission_count > 0:
        warnings.append(
            f"{retransmission_count} probable TCP retransmissions detected "
            f"(duplicate sequence numbers with payload)"
        )

    rtt_list = []
    for fwd_key, syn_t in syn_timestamps.items():
        rev_key = (fwd_key[2], fwd_key[3], fwd_key[0], fwd_key[1])
        sa_key = (fwd_key[1], fwd_key[0], fwd_key[3], fwd_key[2])
        sa_time = synack_timestamps.get(sa_key)
        if sa_time is not None and sa_time > syn_t:
            rtt_ms = round((sa_time - syn_t) * 1000, 2)
            rtt_list.append({
                "src": fwd_key[0], "dst": fwd_key[1],
                "sport": fwd_key[2], "dport": fwd_key[3],
                "rtt_ms": rtt_ms,
            })
    rtt_list.sort(key=lambda x: x["rtt_ms"], reverse=True)
    analysis.rtt_estimates = rtt_list[:20]

    merged_streams: dict[tuple, dict] = {}
    for key, nbytes in stream_bytes.items():
        norm_key = tuple(sorted([(key[0], key[2]), (key[1], key[3])]))
        if norm_key not in merged_streams:
            merged_streams[norm_key] = {
                "src": key[0], "dst": key[1],
                "sport": key[2], "dport": key[3],
                "bytes": 0, "duration": 0.0,
            }
        merged_streams[norm_key]["bytes"] += nbytes
        ts = stream_timestamps.get(key, [])
        if len(ts) >= 2:
            d = ts[-1] - ts[0]
            if d > merged_streams[norm_key]["duration"]:
                merged_streams[norm_key]["duration"] = round(d, 3)
    sorted_streams = sorted(merged_streams.values(), key=lambda s: s["bytes"], reverse=True)
    for s in sorted_streams:
        if s["duration"] > 0:
            s["throughput_kbps"] = round((s["bytes"] * 8) / (s["duration"] * 1000), 2)
        else:
            s["throughput_kbps"] = 0.0
    analysis.tcp_streams = sorted_streams[:25]

    if tcp_windows:
        analysis.tcp_window_stats = {
            "min": min(tcp_windows),
            "max": max(tcp_windows),
            "avg": round(sum(tcp_windows) / len(tcp_windows), 1),
            "zero_count": tcp_windows.count(0),
        }

    analysis.tls_versions = [
        {"version": v, "count": c}
        for v, c in tls_versions_seen.most_common(10)
    ]

    analysis.http_requests = http_reqs[:50]

    scan_suspects = []
    for src, ports in src_dst_ports.items():
        if len(ports) >= 15:
            scan_suspects.append({"ip": src, "unique_dst_ports": len(ports)})
    scan_suspects.sort(key=lambda x: x["unique_dst_ports"], reverse=True)
    analysis.port_scan_suspects = scan_suspects[:10]
    for s in scan_suspects[:3]:
        warnings.append(
            f"Possible port scan from {s['ip']}: "
            f"{s['unique_dst_ports']} unique destination ports contacted"
        )

    if time_buckets:
        sorted_buckets = sorted(time_buckets.items())
        analysis.bandwidth_timeline = [
            {"second": sec, "packets": d["packets"], "bytes": d["bytes"]}
            for sec, d in sorted_buckets
        ]

    analysis.mac_addresses = [
        {"mac": mac, "packets": cnt}
        for mac, cnt in mac_counter.most_common(20)
    ]

    analysis.vlan_ids = sorted(vlan_set)

    analysis.ip_classification = dict(ip_class_counter.most_common())

    dns_ans_counter: dict[str, dict] = {}
    for entry in dns_answer_list:
        k = f"{entry['query']}→{entry['answer']}"
        if k not in dns_ans_counter:
            dns_ans_counter[k] = entry
        else:
            dns_ans_counter[k]["count"] = dns_ans_counter[k].get("count", 1) + 1
    analysis.dns_answers = list(dns_ans_counter.values())[:30]

    if payload_entropy_vals:
        avg_ent = sum(payload_entropy_vals) / len(payload_entropy_vals)
        high_entropy = sum(1 for e in payload_entropy_vals if e > 7.0)
        analysis.payload_entropy_summary = {
            "avg_entropy": round(avg_ent, 3),
            "max_entropy": round(max(payload_entropy_vals), 3),
            "high_entropy_payloads": high_entropy,
            "total_sampled": len(payload_entropy_vals),
        }
        if high_entropy > len(payload_entropy_vals) * 0.5:
            warnings.append(
                f"{high_entropy}/{len(payload_entropy_vals)} sampled payloads have high entropy (>7.0) "
                f"— likely encrypted or compressed traffic"
            )

    if tunnel_counter:
        analysis.tunnel_protocols = [
            {"protocol": p, "count": c}
            for p, c in tunnel_counter.most_common()
        ]

    if ttl_values:
        ttl_counter = Counter(ttl_values)
        analysis.ttl_analysis = {
            "most_common": [
                {"ttl": t, "count": c}
                for t, c in ttl_counter.most_common(5)
            ],
            "unique_ttls": len(ttl_counter),
            "avg_ttl": round(sum(ttl_values) / len(ttl_values), 1),
        }

    # --- extended deep analysis aggregation ---

    hs_counts = Counter(handshake_state.values())
    if hs_counts:
        analysis.tcp_handshakes = {
            "established": hs_counts.get("ESTABLISHED", 0),
            "syn_sent_no_reply": hs_counts.get("SYN_SENT", 0),
            "syn_received_incomplete": hs_counts.get("SYN_RECEIVED", 0),
            "reset": hs_counts.get("RESET", 0),
            "total_attempted": len(handshake_state),
        }
        no_reply = hs_counts.get("SYN_SENT", 0)
        if no_reply > 5:
            warnings.append(
                f"{no_reply} TCP SYNs received no SYN-ACK — "
                f"host unreachable, filtered, or capture is one-sided"
            )

    analysis.tls_sni_hosts = [
        {"host": h, "count": c}
        for h, c in sni_counter.most_common(30)
    ]

    analysis.http_responses = http_resp_list[:50]

    if dns_qtype_counter:
        analysis.dns_record_types = dict(dns_qtype_counter.most_common())

    dns_latency_list.sort(key=lambda x: x["latency_ms"], reverse=True)
    analysis.dns_latency = dns_latency_list[:30]

    seen_tunneling: set[str] = set()
    deduped_tunnel: list[dict] = []
    for ind in dns_tunnel_indicators:
        key = ind.get("domain", "")
        if key not in seen_tunneling:
            seen_tunneling.add(key)
            deduped_tunnel.append(ind)
    analysis.dns_tunneling_suspects = deduped_tunnel[:15]
    if deduped_tunnel:
        warnings.append(
            f"{len(deduped_tunnel)} domain(s) show DNS tunneling indicators "
            f"(high entropy labels, unusually long subdomains)"
        )

    arp_anom: list[dict] = []
    for ip_addr, macs in arp_ip_to_macs.items():
        if len(macs) > 1:
            arp_anom.append({
                "type": "ip_mac_conflict",
                "ip": ip_addr,
                "macs": sorted(macs),
                "detail": f"IP {ip_addr} seen with {len(macs)} different MACs — possible ARP spoofing",
            })
    if gratuitous_arp_count > 0:
        arp_anom.append({
            "type": "gratuitous_arp",
            "count": gratuitous_arp_count,
            "detail": f"{gratuitous_arp_count} gratuitous ARP replies (sender=target) — "
                      f"legitimate failover or spoofing",
        })
    analysis.arp_anomalies = arp_anom[:20]
    for a_item in arp_anom:
        if a_item["type"] == "ip_mac_conflict":
            warnings.append(f"[ARP] {a_item['detail']}")

    syn_count = len(syn_fingerprints)
    if syn_count > 0 or mss_values or wscale_values:
        analysis.tcp_options_summary = {
            "syn_packets_analyzed": syn_count,
            "mss_min": min(mss_values) if mss_values else None,
            "mss_max": max(mss_values) if mss_values else None,
            "mss_most_common": Counter(mss_values).most_common(1)[0][0] if mss_values else None,
            "wscale_min": min(wscale_values) if wscale_values else None,
            "wscale_max": max(wscale_values) if wscale_values else None,
            "wscale_most_common": Counter(wscale_values).most_common(1)[0][0] if wscale_values else None,
            "sack_capable": sack_capable_count,
            "timestamp_capable": ts_capable_count,
        }

    os_guesses: Counter = Counter()
    for fp in syn_fingerprints:
        ttl_fp = fp.get("ttl")
        win_fp = fp.get("window")
        mss_fp = fp.get("mss")
        wscale_fp = fp.get("wscale")
        if ttl_fp is not None and win_fp is not None:
            guess = _infer_os(ttl_fp, win_fp, mss_fp, wscale_fp)
            os_guesses[guess] += 1
    if os_guesses:
        analysis.os_fingerprints = [
            {"os": name, "syn_count": cnt}
            for name, cnt in os_guesses.most_common(10)
        ]

    if all_pkt_sizes:
        sorted_sizes = sorted(all_pkt_sizes)
        analysis.packet_size_distribution = {
            "p25": _percentile(sorted_sizes, 25),
            "p50": _percentile(sorted_sizes, 50),
            "p75": _percentile(sorted_sizes, 75),
            "p95": _percentile(sorted_sizes, 95),
            "p99": _percentile(sorted_sizes, 99),
            "stddev": round(
                (sum((s - size_sum / size_count) ** 2 for s in all_pkt_sizes) / len(all_pkt_sizes)) ** 0.5, 1
            ) if len(all_pkt_sizes) > 1 else 0.0,
        }

    analysis.icmp_latency = icmp_latency_list[:20]

    if len(timestamps) >= 3:
        iats = [timestamps[j + 1] - timestamps[j] for j in range(len(timestamps) - 1)]
        iat_mean = sum(iats) / len(iats)
        iat_var = sum((x - iat_mean) ** 2 for x in iats) / len(iats)
        iat_stddev = iat_var ** 0.5
        sorted_iats = sorted(iats)
        jitter_vals = [abs(iats[j + 1] - iats[j]) for j in range(len(iats) - 1)]
        jitter_mean = sum(jitter_vals) / len(jitter_vals) if jitter_vals else 0.0
        burst_threshold = 0.001
        burst_count = sum(1 for x in iats if x < burst_threshold)
        analysis.inter_arrival_time = {
            "mean_ms": round(iat_mean * 1000, 3),
            "stddev_ms": round(iat_stddev * 1000, 3),
            "min_ms": round(sorted_iats[0] * 1000, 3),
            "max_ms": round(sorted_iats[-1] * 1000, 3),
            "p50_ms": round(_percentile(sorted_iats, 50) * 1000, 3),
            "p99_ms": round(_percentile(sorted_iats, 99) * 1000, 3),
            "jitter_mean_ms": round(jitter_mean * 1000, 3),
            "burst_packets": burst_count,
            "burst_pct": round(burst_count / len(iats) * 100, 1) if iats else 0.0,
        }

    analysis.duplicate_acks = dup_ack_count
    if dup_ack_count > 20:
        warnings.append(
            f"{dup_ack_count} duplicate ACKs detected — indicates packet loss triggering fast retransmit"
        )

    sec_indicators: list[dict] = []
    cleartext_protos = []
    for proto_name in ("HTTP", "FTP", "FTP-Data", "Telnet", "SMTP", "POP3", "IMAP"):
        cnt = protocol_counter.get(proto_name, 0)
        if cnt > 0:
            cleartext_protos.append(f"{proto_name}({cnt})")
    if cleartext_protos:
        sec_indicators.append({
            "severity": "medium",
            "category": "cleartext_protocols",
            "detail": f"Cleartext protocols in use: {', '.join(cleartext_protos)}"
        })

    deprecated_tls = []
    for tv in analysis.tls_versions:
        if tv["version"] in ("SSL 3.0", "TLS 1.0", "TLS 1.1",
                              "ClientHello SSL 3.0", "ClientHello TLS 1.0", "ClientHello TLS 1.1"):
            deprecated_tls.append(tv["version"])
    if deprecated_tls:
        sec_indicators.append({
            "severity": "high",
            "category": "deprecated_tls",
            "detail": f"Deprecated TLS/SSL versions: {', '.join(deprecated_tls)}"
        })

    if analysis.arp_anomalies:
        conflicts = [a_i for a_i in analysis.arp_anomalies if a_i["type"] == "ip_mac_conflict"]
        if conflicts:
            sec_indicators.append({
                "severity": "high",
                "category": "arp_spoofing",
                "detail": f"{len(conflicts)} IP(s) associated with multiple MACs"
            })

    if analysis.dns_tunneling_suspects:
        sec_indicators.append({
            "severity": "high",
            "category": "dns_tunneling",
            "detail": f"{len(analysis.dns_tunneling_suspects)} domain(s) with tunneling indicators"
        })

    if analysis.port_scan_suspects:
        sec_indicators.append({
            "severity": "medium",
            "category": "port_scanning",
            "detail": f"{len(analysis.port_scan_suspects)} source(s) contacted 15+ unique ports"
        })

    rst_count = protocol_counter.get("TCP RST", 0)
    if rst_count > 50:
        sec_indicators.append({
            "severity": "medium",
            "category": "rst_storm",
            "detail": f"{rst_count} TCP RSTs — possible scan response, firewall rejection, or DoS"
        })

    if analysis.retransmissions > 100:
        sec_indicators.append({
            "severity": "low",
            "category": "network_degradation",
            "detail": f"{analysis.retransmissions} retransmissions — significant packet loss or congestion"
        })

    analysis.security_indicators = sec_indicators

    analysis.warnings = warnings[:100]

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

    # --- deep analysis sections ---

    if a.ip_classification:
        lines.append("\n--- IP ADDRESS CLASSIFICATION ---")
        for cls, count in a.ip_classification.items():
            lines.append(f"  {cls:<25} {count:>8}")

    if a.retransmissions:
        lines.append(f"\n--- TCP RETRANSMISSIONS: {a.retransmissions} ---")

    if a.tcp_streams:
        lines.append("\n--- TOP TCP STREAMS (by bytes transferred) ---")
        for s in a.tcp_streams[:15]:
            label = f"{s['src']}:{s['sport']} → {s['dst']}:{s['dport']}"
            bw = f"{s['throughput_kbps']} kbps" if s.get("throughput_kbps") else ""
            lines.append(f"  {label:<55} {s['bytes']:>10}B  {s['duration']:>8.3f}s  {bw}")

    if a.tcp_window_stats:
        ws = a.tcp_window_stats
        lines.append("\n--- TCP WINDOW SIZE STATS ---")
        lines.append(f"  min={ws['min']}  max={ws['max']}  avg={ws['avg']}  zero_window_count={ws['zero_count']}")

    if a.rtt_estimates:
        lines.append("\n--- TCP RTT ESTIMATES (SYN→SYN-ACK) ---")
        for r in a.rtt_estimates[:10]:
            lines.append(
                f"  {r['src']}:{r['sport']} → {r['dst']}:{r['dport']}  "
                f"RTT={r['rtt_ms']}ms"
            )

    if a.tls_versions:
        lines.append("\n--- TLS/SSL VERSIONS DETECTED ---")
        for t in a.tls_versions:
            lines.append(f"  {t['version']:<20} {t['count']:>6}x")

    if a.http_requests:
        lines.append(f"\n--- HTTP REQUESTS ({len(a.http_requests)}) ---")
        for h in a.http_requests[:20]:
            src_label = f"{h.get('src', '?')} → {h.get('dst', '?')}"
            lines.append(f"  [{h.get('pkt', '?')}] {h['method']:<8} {h['uri']:<60} {src_label}")

    if a.port_scan_suspects:
        lines.append("\n--- PORT SCAN SUSPECTS ---")
        for s in a.port_scan_suspects:
            lines.append(f"  {s['ip']:<40} {s['unique_dst_ports']:>6} unique dst ports")

    if a.dns_answers:
        lines.append("\n--- DNS ANSWERS ---")
        for d in a.dns_answers[:20]:
            lines.append(f"  {d.get('query', '?'):<40} → {d.get('answer', '?')}")

    if a.mac_addresses:
        lines.append("\n--- MAC ADDRESSES ---")
        for m in a.mac_addresses[:10]:
            lines.append(f"  {m['mac']:<20} {m['packets']:>8} packets")

    if a.vlan_ids:
        lines.append(f"\n--- VLAN IDs DETECTED: {a.vlan_ids} ---")

    if a.tunnel_protocols:
        lines.append("\n--- TUNNEL PROTOCOLS ---")
        for t in a.tunnel_protocols:
            lines.append(f"  {t['protocol']:<15} {t['count']:>6}x")

    if a.ttl_analysis:
        ta = a.ttl_analysis
        lines.append("\n--- TTL ANALYSIS ---")
        lines.append(f"  Unique TTL values: {ta['unique_ttls']}  avg={ta['avg_ttl']}")
        for t in ta.get("most_common", []):
            lines.append(f"  TTL={t['ttl']:<5} {t['count']:>8} packets")

    if a.payload_entropy_summary:
        pe = a.payload_entropy_summary
        lines.append("\n--- PAYLOAD ENTROPY ANALYSIS ---")
        lines.append(f"  Sampled: {pe['total_sampled']} payloads")
        lines.append(f"  Avg entropy: {pe['avg_entropy']} bits/byte  (max: {pe['max_entropy']})")
        lines.append(f"  High entropy (>7.0): {pe['high_entropy_payloads']}")

    if a.bandwidth_timeline and len(a.bandwidth_timeline) > 1:
        bw = a.bandwidth_timeline
        peak = max(bw, key=lambda x: x["bytes"])
        lines.append("\n--- BANDWIDTH TIMELINE (peak) ---")
        lines.append(
            f"  Peak at t={peak['second']}s: "
            f"{peak['packets']} pkts, {peak['bytes']} bytes "
            f"({peak['bytes'] * 8 / 1000:.1f} kbps)"
        )
        total_secs = len(bw)
        total_bytes = sum(b["bytes"] for b in bw)
        if total_secs > 0:
            lines.append(
                f"  Average: {total_bytes // total_secs} bytes/s "
                f"({total_bytes * 8 / total_secs / 1000:.1f} kbps) "
                f"over {total_secs}s"
            )

    # --- extended deep analysis sections ---

    if a.tcp_handshakes:
        hs = a.tcp_handshakes
        lines.append("\n--- TCP HANDSHAKE ANALYSIS ---")
        lines.append(f"  Total attempted:     {hs.get('total_attempted', 0)}")
        lines.append(f"  Established (3WHS):  {hs.get('established', 0)}")
        lines.append(f"  SYN no reply:        {hs.get('syn_sent_no_reply', 0)}")
        lines.append(f"  SYN-ACK incomplete:  {hs.get('syn_received_incomplete', 0)}")
        lines.append(f"  Reset before estab:  {hs.get('reset', 0)}")

    if a.duplicate_acks:
        lines.append(f"\n--- DUPLICATE ACKs: {a.duplicate_acks} ---")

    if a.tcp_options_summary:
        to = a.tcp_options_summary
        lines.append("\n--- TCP OPTIONS (from SYN packets) ---")
        lines.append(f"  SYN packets analyzed: {to.get('syn_packets_analyzed', 0)}")
        if to.get("mss_most_common") is not None:
            lines.append(f"  MSS: most_common={to['mss_most_common']}  "
                         f"min={to.get('mss_min')}  max={to.get('mss_max')}")
        if to.get("wscale_most_common") is not None:
            lines.append(f"  Window Scale: most_common={to['wscale_most_common']}  "
                         f"min={to.get('wscale_min')}  max={to.get('wscale_max')}")
        lines.append(f"  SACK capable: {to.get('sack_capable', 0)}  "
                     f"TCP Timestamps: {to.get('timestamp_capable', 0)}")

    if a.os_fingerprints:
        lines.append("\n--- OS FINGERPRINTS (passive, from SYN characteristics) ---")
        for fp in a.os_fingerprints:
            lines.append(f"  {fp['os']:<40} {fp['syn_count']:>6} SYN(s)")

    if a.tls_sni_hosts:
        lines.append("\n--- TLS SNI HOSTNAMES ---")
        for s in a.tls_sni_hosts[:20]:
            lines.append(f"  {s['host']:<50} {s['count']:>5}x")

    if a.http_responses:
        lines.append(f"\n--- HTTP RESPONSES ({len(a.http_responses)}) ---")
        status_counter: dict[int, int] = {}
        for hr in a.http_responses:
            sc = hr.get("status_code", 0)
            status_counter[sc] = status_counter.get(sc, 0) + 1
        for sc in sorted(status_counter, key=status_counter.get, reverse=True):
            lines.append(f"  HTTP {sc:<5} {status_counter[sc]:>6}x")

    if a.dns_record_types:
        lines.append("\n--- DNS RECORD TYPES ---")
        for rtype, cnt in a.dns_record_types.items():
            lines.append(f"  {rtype:<10} {cnt:>8}")

    if a.dns_latency:
        lines.append("\n--- DNS QUERY LATENCY (query→response) ---")
        lats = [d["latency_ms"] for d in a.dns_latency]
        if lats:
            lines.append(f"  min={min(lats):.2f}ms  max={max(lats):.2f}ms  "
                         f"avg={sum(lats)/len(lats):.2f}ms  samples={len(lats)}")
        for d in a.dns_latency[:10]:
            lines.append(f"  {d['query']:<40} {d['latency_ms']:>8.2f}ms")

    if a.dns_tunneling_suspects:
        lines.append(f"\n--- DNS TUNNELING SUSPECTS ({len(a.dns_tunneling_suspects)}) ---")
        for dt in a.dns_tunneling_suspects[:10]:
            flags = []
            if "high_entropy" in dt:
                flags.append(f"entropy={dt['high_entropy']}")
            if "long_label" in dt:
                flags.append(f"label_len={dt['long_label']}")
            if "hex_encoded" in dt:
                flags.append("hex-encoded")
            if "long_query" in dt:
                flags.append(f"query_len={dt['long_query']}")
            lines.append(f"  {dt.get('domain', '?'):<30} [{', '.join(flags)}]")

    if a.arp_anomalies:
        lines.append(f"\n--- ARP ANOMALIES ({len(a.arp_anomalies)}) ---")
        for aa in a.arp_anomalies:
            lines.append(f"  [{aa['type']}] {aa['detail']}")

    if a.icmp_latency:
        lines.append("\n--- ICMP ECHO LATENCY (ping RTT) ---")
        icmp_lats = [x["rtt_ms"] for x in a.icmp_latency]
        if icmp_lats:
            lines.append(f"  min={min(icmp_lats):.2f}ms  max={max(icmp_lats):.2f}ms  "
                         f"avg={sum(icmp_lats)/len(icmp_lats):.2f}ms  samples={len(icmp_lats)}")
        for il in a.icmp_latency[:10]:
            lines.append(f"  {il['src']} → {il['dst']}  seq={il['seq']}  RTT={il['rtt_ms']}ms")

    if a.packet_size_distribution:
        pd = a.packet_size_distribution
        lines.append("\n--- PACKET SIZE DISTRIBUTION ---")
        lines.append(f"  p25={pd['p25']:.0f}B  p50={pd['p50']:.0f}B  p75={pd['p75']:.0f}B  "
                     f"p95={pd['p95']:.0f}B  p99={pd['p99']:.0f}B  stddev={pd['stddev']}B")

    if a.inter_arrival_time:
        iat = a.inter_arrival_time
        lines.append("\n--- INTER-ARRIVAL TIME ANALYSIS ---")
        lines.append(f"  mean={iat['mean_ms']}ms  stddev={iat['stddev_ms']}ms  "
                     f"min={iat['min_ms']}ms  max={iat['max_ms']}ms")
        lines.append(f"  p50={iat['p50_ms']}ms  p99={iat['p99_ms']}ms  "
                     f"jitter(mean)={iat['jitter_mean_ms']}ms")
        lines.append(f"  Burst packets (<1ms IAT): {iat['burst_packets']} ({iat['burst_pct']}%)")

    if a.security_indicators:
        lines.append(f"\n--- SECURITY ASSESSMENT ({len(a.security_indicators)} findings) ---")
        for si in sorted(a.security_indicators, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["severity"], 3)):
            lines.append(f"  [{si['severity'].upper():<6}] [{si['category']}] {si['detail']}")

    return "\n".join(lines)


def _classify_ip(addr: str) -> str:
    """Classify an IP address into network categories."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return "invalid"
    if ip.is_loopback:
        return "loopback"
    if ip.is_multicast:
        return "multicast"
    if ip.is_link_local:
        return "link-local"
    if ip.is_private:
        return "private (RFC1918)"
    if ip.is_reserved:
        return "reserved"
    return "public"


def _shannon_entropy(data: bytes) -> float:
    """Calculate Shannon entropy (bits per byte) for a byte sequence."""
    if not data:
        return 0.0
    freq = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


_TLS_VERSION_MAP = {
    (3, 0): "SSL 3.0",
    (3, 1): "TLS 1.0",
    (3, 2): "TLS 1.1",
    (3, 3): "TLS 1.2",
    (3, 4): "TLS 1.3",
}


def _extract_tls_version(data: bytes, counter: Counter) -> None:
    """Detect TLS record or ClientHello and record the version."""
    if len(data) < 5:
        return
    content_type = data[0]
    if content_type not in (20, 21, 22, 23):
        return
    major, minor = data[1], data[2]
    version = _TLS_VERSION_MAP.get((major, minor))
    if version:
        counter[version] += 1
    if content_type == 22 and len(data) >= 11:
        hs_type = data[5]
        if hs_type == 1:
            ch_major, ch_minor = data[9], data[10]
            ch_version = _TLS_VERSION_MAP.get((ch_major, ch_minor))
            if ch_version:
                counter[f"ClientHello {ch_version}"] += 1


_HTTP_METHODS = (b"GET ", b"POST ", b"PUT ", b"DELETE ", b"HEAD ", b"PATCH ", b"OPTIONS ", b"CONNECT ")


def _extract_http_request(data: bytes, results: list, src_ip: str | None, dst_ip: str | None, pkt_idx: int) -> None:
    """Extract HTTP request method and URI from raw payload."""
    for method_prefix in _HTTP_METHODS:
        if data[:len(method_prefix)] == method_prefix:
            try:
                first_line = data.split(b"\r\n", 1)[0].decode(errors="ignore")
                parts = first_line.split(" ", 2)
                if len(parts) >= 2:
                    results.append({
                        "method": parts[0],
                        "uri": parts[1],
                        "src": src_ip or "?",
                        "dst": dst_ip or "?",
                        "pkt": pkt_idx + 1,
                    })
            except Exception:
                pass
            return


def _extract_tls_sni(data: bytes) -> str | None:
    """Extract Server Name Indication from a TLS ClientHello payload.

    Parses the variable-length fields (session ID, cipher suites,
    compression methods) to reach the extensions block, then scans
    for extension type 0x0000 (server_name).
    """
    if len(data) < 44 or data[0] != 22 or data[5] != 1:
        return None
    offset = 43  # record(5) + handshake(4) + version(2) + random(32)
    if offset >= len(data):
        return None
    session_id_len = data[offset]
    offset += 1 + session_id_len
    if offset + 2 > len(data):
        return None
    cs_len = struct.unpack("!H", data[offset:offset + 2])[0]
    offset += 2 + cs_len
    if offset + 1 > len(data):
        return None
    comp_len = data[offset]
    offset += 1 + comp_len
    if offset + 2 > len(data):
        return None
    ext_total = struct.unpack("!H", data[offset:offset + 2])[0]
    offset += 2
    ext_end = offset + ext_total
    while offset + 4 <= ext_end and offset + 4 <= len(data):
        ext_type = struct.unpack("!H", data[offset:offset + 2])[0]
        ext_len = struct.unpack("!H", data[offset + 2:offset + 4])[0]
        offset += 4
        if ext_type == 0 and ext_len >= 5 and offset + 5 <= len(data):
            sn_type = data[offset + 2]
            sn_len = struct.unpack("!H", data[offset + 3:offset + 5])[0]
            if sn_type == 0 and offset + 5 + sn_len <= len(data):
                try:
                    return data[offset + 5:offset + 5 + sn_len].decode("ascii")
                except (UnicodeDecodeError, ValueError):
                    return None
            return None
        offset += ext_len
    return None


def _extract_http_response(data: bytes, results: list, src_ip: str | None,
                           dst_ip: str | None, pkt_idx: int) -> None:
    """Extract HTTP response status code and reason from raw payload."""
    try:
        first_line = data.split(b"\r\n", 1)[0].decode(errors="ignore")
        parts = first_line.split(" ", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            results.append({
                "version": parts[0],
                "status_code": int(parts[1]),
                "reason": parts[2] if len(parts) >= 3 else "",
                "src": src_ip or "?",
                "dst": dst_ip or "?",
                "pkt": pkt_idx + 1,
            })
    except Exception:
        pass


_DNS_QTYPE_MAP = {
    1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 12: "PTR",
    15: "MX", 16: "TXT", 28: "AAAA", 33: "SRV", 35: "NAPTR",
    43: "DS", 46: "RRSIG", 47: "NSEC", 48: "DNSKEY",
    52: "TLSA", 65: "HTTPS", 99: "SPF", 255: "ANY", 252: "AXFR",
    257: "CAA",
}


def _dns_qtype_name(qtype: int) -> str:
    """Map a DNS query type number to its mnemonic name."""
    return _DNS_QTYPE_MAP.get(qtype, f"TYPE{qtype}")


def _check_dns_tunneling(qname: str) -> dict | None:
    """Detect DNS tunneling indicators from a query name.

    Flags queries with unusually long labels, high Shannon entropy
    in the subdomain portion, or hex-like encoding patterns.
    """
    parts = qname.split(".")
    if len(parts) < 2:
        return None
    max_label_len = max(len(p) for p in parts)
    subdomain = ".".join(parts[:-2]) if len(parts) > 2 else parts[0]
    if len(subdomain) < 10:
        return None
    entropy = _shannon_entropy(subdomain.encode("ascii", errors="ignore"))
    indicators: dict = {}
    if max_label_len > 52:
        indicators["long_label"] = max_label_len
    if len(qname) > 100:
        indicators["long_query"] = len(qname)
    if entropy > 3.5 and len(subdomain) > 20:
        indicators["high_entropy"] = round(entropy, 2)
    hex_chars = sum(1 for c in subdomain if c in "0123456789abcdef")
    if len(subdomain) > 20 and hex_chars > len(subdomain) * 0.7:
        indicators["hex_encoded"] = True
    if indicators:
        indicators["domain"] = ".".join(parts[-2:]) if len(parts) >= 2 else qname
        indicators["qname"] = qname
        return indicators
    return None


def _infer_os(ttl: int, window: int, mss: int | None, wscale: int | None) -> str:
    """Heuristic OS identification from TCP SYN packet characteristics.

    Uses initial TTL, TCP window size, MSS, and window scale factor.
    Based on p0f-style passive fingerprinting methodology.
    """
    if ttl <= 32:
        initial_ttl = 32
    elif ttl <= 64:
        initial_ttl = 64
    elif ttl <= 128:
        initial_ttl = 128
    else:
        initial_ttl = 255

    if initial_ttl == 64:
        if wscale == 7:
            if window in (29200, 26883, 65535, 28960):
                return "Linux 3.x-6.x"
            return "Linux"
        if wscale == 6 and window == 65535:
            return "macOS / iOS"
        if mss and mss == 1460 and wscale is not None:
            return "Linux"
        if wscale == 9:
            return "Linux (recent kernel)"
        return "Unix-like (TTL=64)"
    elif initial_ttl == 128:
        if wscale == 8:
            return "Windows 10/11 / Server 2016+"
        if window == 8192:
            return "Windows XP / Server 2003"
        if window == 65535 and wscale is None:
            return "Windows 7 / Vista"
        if wscale is not None:
            return "Windows 8+ / Server 2012+"
        return "Windows (TTL=128)"
    elif initial_ttl == 255:
        return "Network equipment / Solaris (TTL=255)"
    elif initial_ttl == 32:
        return "Legacy OS (TTL=32)"
    return f"Unknown (TTL={ttl})"


def _percentile(sorted_data: list, p: float) -> float:
    """Compute the p-th percentile (0-100) using linear interpolation."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return float(sorted_data[-1])
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


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
