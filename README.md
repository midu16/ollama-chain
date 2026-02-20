# ollama-chain

Chain **all** local Ollama models together with web search for maximum answer accuracy. Every available model participates — smallest drafts, each larger model refines, largest delivers the final answer with authoritative source citations. Fully self-hosted, no API keys.

## How It Works

1. **Model discovery** — ollama-chain queries Ollama at startup and sorts all available models by parameter count
2. **Web search** — DuckDuckGo fetches live context for the query (no API key needed)
3. **Cascade** — the query flows through every model from smallest to largest:
   - Smallest model drafts the initial answer with search context
   - Each intermediate model reviews, corrects errors, and strengthens citations
   - Largest model produces the final authoritative answer
4. **Source citation** — all models are instructed to reference authoritative public sources (IETF RFCs, IEEE, ISO, NIST, W3C, Red Hat, kernel.org, MDN, official documentation)

```
User query
    │
    ├── DuckDuckGo web search (context gathering)
    │
    ├── Model 1 (smallest) ──→ initial draft + sources
    ├── Model 2 ──→ review, correct, refine
    ├── ...
    └── Model N (largest) ──→ final authoritative answer with citations
```

## Requirements

- [Ollama](https://ollama.com) running locally (systemd service or standalone)
- Python 3.11+
- At least one pulled Ollama model (more models = more refinement stages)
- Internet connection for web search (optional — use `--no-search` for offline mode)
- `tcpdump` or `tshark` for generating .pcap files (optional)

## Installation

```bash
git clone https://github.com/midu16/ollama-chain.git
cd ollama-chain
pip install -r requirements.txt
```

Verify Ollama is running:

```bash
systemctl status ollama.service
ollama list
```

Pull models (more models = deeper cascade):

```bash
ollama pull qwen3:14b
ollama pull qwen3:32b
```

## Usage

```bash
cd ollama-chain

# Default: cascade through ALL models with web search + source citations
python -m ollama_chain "How does TCP congestion control work?"

# All models answer independently, strongest merges best parts
python -m ollama_chain -m consensus "Compare REST vs GraphQL"

# Search-first: fetch web results, strong model synthesizes
python -m ollama_chain -m search "latest Linux kernel release date"

# Other modes
python -m ollama_chain -m route "What is a binary search tree?"
python -m ollama_chain -m pipeline "Explain the CAP theorem"
python -m ollama_chain -m verify "How does DNS resolution work?"

# Direct model access
python -m ollama_chain -m fast "Quick: what port does HTTPS use?"
python -m ollama_chain -m strong "Derive the time complexity of Dijkstra's algorithm"

# Fully offline (no web search)
python -m ollama_chain --no-search "What is a linked list?"

# List models and cascade order
python -m ollama_chain --list-models
```

## Chaining Modes

| Mode | Pipeline | Best For |
|---|---|---|
| `cascade` | ALL models chain smallest→largest, each refining (default) | Maximum accuracy |
| `consensus` | All models answer independently, strongest merges | Reducing model bias |
| `route` | Fast classifies complexity (1-5), routes to fast or strong | Quick simple queries |
| `pipeline` | Fast extracts + classifies domain, strong reasons | Long/complex input |
| `verify` | Fast drafts, strong verifies and corrects | Fact-checking |
| `search` | Search first, strong synthesizes from results | Time-sensitive queries |
| `pcap` | Scapy parses → models analyze → expert report | Network analysis |
| `fast` | Direct to smallest model | Speed |
| `strong` | Direct to largest model | Single-model quality |

## Source Citations

All modes instruct models to cite authoritative sources using this format:

```
[Source: IETF RFC 793, https://www.rfc-editor.org/rfc/rfc793]
[Source: Red Hat Documentation, https://docs.redhat.com/...]
[Source: IEEE 802.11, https://standards.ieee.org/...]
```

Prioritized source categories:
- **Standards bodies** — IETF RFCs, IEEE, ISO, NIST, W3C
- **Official documentation** — Red Hat, kernel.org, Mozilla MDN, Microsoft Docs
- **Vendor/project docs** — PostgreSQL docs, Nginx docs, etc.
- **Peer-reviewed work** — academic papers, published research

Models are instructed to mark claims as unverified when no authoritative source can be referenced.

## Web Search

All modes include web search by default (via DuckDuckGo, no API key):

1. Fast model generates 1-3 optimized search queries
2. DuckDuckGo executes them, results are deduplicated
3. Results are injected as context alongside source-citation instructions
4. Models cross-reference their knowledge against live search results

Use `--no-search` for fully offline operation.

## PCAP Analysis

Analyze `.pcap` and `.pcapng` network capture files:

1. **Scapy** parses packets and extracts structured statistics
2. **Web search** finds troubleshooting context for detected errors
3. **Fast model** summarizes findings and classifies errors by severity
4. **Strong model** produces expert report with RFC references

### What It Detects

- Protocol distribution (IPv4/IPv6, TCP, UDP, ICMP, DNS, ARP, HTTP, HTTPS, SSH, etc.)
- TCP errors (RST, zero window, incomplete handshakes, SYN floods)
- ICMP errors (destination unreachable, time exceeded, redirects)
- DNS errors (NXDomain, ServFail, refused queries)
- IP issues (TTL expired, fragmentation)
- Traffic patterns (top talkers, conversations, packet sizes)
- Security observations (port scans, unusual patterns, unencrypted protocols)

### PCAP Usage

```bash
sudo tcpdump -i any -w /tmp/capture.pcap -c 1000

python -m ollama_chain "Analyze /tmp/capture.pcap for errors"
python -m ollama_chain -m pcap --pcap /tmp/capture.pcap
python -m ollama_chain -m pcap --pcap /tmp/capture.pcap "Why are there so many RST packets?"
python -m ollama_chain --no-search -m pcap --pcap /tmp/capture.pcap
```

## Auto Model Discovery

All models are used by default, sorted by parameter count:

```
$ python -m ollama_chain --list-models
#    MODEL                     PARAMS     QUANT      FAMILY     SIZE
---------------------------------------------------------------------
1    qwen3:14b                 14.8      B Q4_K_M    qwen3      8.6 GB
2    qwen3:32b                 32.8      B Q4_K_M    qwen3      18.6 GB

2 model(s) available.
Cascade order: qwen3:14b → qwen3:32b
```

Adding more models deepens the cascade. For example, pulling a 7B model adds a fast initial drafter:

```bash
ollama pull qwen3:8b
# Cascade becomes: qwen3:8b → qwen3:14b → qwen3:32b
```
## Demo output


```Markdown
$ python -m ollama_chain -m cascade "What port does SSH use?"
Models (3): qwen3:14b → qwen3:32b → qwen3-coder-next:latest | Web search: on
[search] Generating search queries with qwen3:14b...
[search] Searching: ['What port does SSH use', 'default port for SSH', 'SSH port number']
Impersonate 'safari_17.2.1' does not exist, using 'random'
[search] Found 10 results
[cascade 1/3] Drafting with qwen3:14b...
[cascade 2/3] Reviewing with qwen3:32b...
[cascade 3/3] Final answer with qwen3-coder-next:latest...
SSH (Secure Shell) uses **port 22** by default for its connections. This port was officially assigned to SSH by the Internet Assigned Numbers Authority (IANA) in 1995 [Source: IANA, [Service Name and Transport Protocol Port Number Registry](https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml)].

The OpenSSH reference implementation explicitly documents that the default port is 22 and that it can be changed via the `Port` directive in `sshd_config` [Source: OpenSSH, [sshd_config man page](https://man.openbsd.org/sshd_config#Port)]. Clients connect using the `-p` option to specify a non-default port, e.g., `ssh -p 2222 user@host` [Source: OpenSSH, [ssh man page](https://man.openbsd.org/ssh#p)].

Although the IETF RFCs that define the SSH protocol (e.g., RFC 4251) specify protocol behavior, they do not assign port numbers—this is the responsibility of IANA [Source: IETF, [RFC 4251: The Secure Shell (SSH) Protocol Architecture](https://datatracker.ietf.org/doc/html/rfc4251)].

While port 22 remains the IANA-registered and universally recognized default, system administrators sometimes configure SSH to use alternative ports (e.g., 2222, 3022) to reduce exposure to automated attacks [Source: OpenSSH, sshd_config man page]. However, such changes do not alter the official standard assignment.

No other port is officially designated for SSH by IANA.
```

## Project Structure

```
ollama-chain/
├── ollama_chain/
│   ├── __init__.py
│   ├── __main__.py      # python -m ollama_chain entry point
│   ├── cli.py           # CLI argument parsing and dispatch
│   ├── models.py        # Ollama model discovery and ordering
│   ├── chains.py        # Cascade + chaining mode implementations
│   ├── search.py        # DuckDuckGo web search (no API key)
│   └── pcap.py          # PCAP parsing and analysis with scapy
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT
