# ollama-chain

Chain **all** local Ollama models together with web search for maximum answer accuracy. Every available model participates — smallest drafts, each larger model refines, largest delivers the final answer with authoritative source citations. Includes an **autonomous agent** with planning, persistent memory, tool use, and dynamic control flow. Fully self-hosted, no API keys.

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

# Auto mode: router classifies complexity and picks the best strategy
python -m ollama_chain -m auto "What port does SSH use?"
python -m ollama_chain -m auto "Derive the time complexity of Dijkstra's algorithm"

# All models answer independently, strongest merges best parts
python -m ollama_chain -m consensus "Compare REST vs GraphQL"

# Search-first: fetch web results, strong model synthesizes
python -m ollama_chain -m search "latest Linux kernel release date"

# Autonomous agent: plans, uses tools, remembers across sessions
python -m ollama_chain -m agent "Find out what Linux kernel my machine runs and list recent CVEs"

# Agent with custom iteration budget
python -m ollama_chain -m agent --max-iterations 20 "Research and summarize recent CVEs of my current system"

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

# Memory management
python -m ollama_chain --show-memory
python -m ollama_chain --clear-memory
```

## Chaining Modes

| Mode | Pipeline | Best For |
|---|---|---|
| `cascade` | ALL models chain smallest→largest, each refining (default) | Maximum accuracy |
| `auto` | Router classifies complexity, picks optimal strategy automatically | Adaptive routing |
| `consensus` | All models answer independently, strongest merges | Reducing model bias |
| `route` | Fast classifies complexity (1-5), routes to fast or strong | Quick simple queries |
| `pipeline` | Fast extracts + classifies domain, strong reasons | Long/complex input |
| `verify` | Fast drafts, strong verifies and corrects | Fact-checking |
| `search` | Search first, strong synthesizes from results | Time-sensitive queries |
| `agent` | Autonomous agent with planning, memory, tools, and control flow | Multi-step tasks |
| `pcap` | Scapy parses → models analyze → expert report | Network analysis |
| `fast` | Direct to smallest model | Speed |
| `strong` | Direct to largest model | Single-model quality |

## Autonomous Agent

The `agent` mode turns ollama-chain into a fully autonomous system that can decompose goals, interact with the environment, and persist knowledge across sessions.

### Architecture

```
User goal
    │
    ├── 1. PLANNING ── fast model decomposes goal into ordered steps
    │       each step tagged with a tool: shell, web_search, read_file, ...
    │       plan validated + auto-fixed (dangling deps, unknown tools, cycles)
    │
    ├── 1b. ROUTING OPTIMIZATION ── assign preferred models per step
    │       data-gathering steps → fast model
    │       reasoning/synthesis steps → strong model
    │       deprioritise previously failed models
    │
    ├── 2. EXECUTION LOOP (ReAct-style)
    │       │
    │       ├── Pre-execution validation (tool availability, dependency checks)
    │       │
    │       ├── Try preferred models first, then fallback chain
    │       │   Model outputs <tool_call>, <final_answer>, or <store_fact>
    │       │
    │       ├── If all models fail → auto-execute tool from plan hints
    │       │   (no LLM needed — runs shell commands, web searches, etc.)
    │       │
    │       ├── Extract facts from tool output → store in memory
    │       │
    │       └── Monitor & replan (discovery triggers, new facts, failures)
    │
    ├── 3. SYNTHESIS ── strongest available model produces final answer
    │       from all collected tool results and discovered facts
    │
    └── 4. MEMORY ── session summary + facts persisted to ~/.ollama_chain/
```

### Agent-Native Enhancements (v0.6.0)

The agent pipeline includes several layers that make it resilient and adaptive:

- **Plan validation & auto-repair** — after the LLM generates a plan, `validate_and_fix_plan()` repairs dangling dependency references, replaces unknown tool names with `"none"`, fills missing descriptions, and detects circular dependency cycles
- **Plan-level routing optimization** — `optimize_routing()` assigns preferred models to each step based on tool type (data-gathering uses the fast model, reasoning uses the strong model) and deprioritises models that have previously failed
- **Pre-execution step validation** — before running each step, `_validate_before_execution()` checks tool availability and dependency satisfaction; steps with unknown tools are downgraded to reasoning, steps with unmet deps are skipped
- **Unified monitoring & replanning** — `monitor_and_replan()` encapsulates three replan triggers (discovery completion, new fact evaluation, accumulated failures) into a single composable function, re-running routing optimization after each replan

### Planning

The fast model breaks a high-level goal into concrete, ordered steps. Each step is tagged with which tool to use:

```
[agent] Plan (4 steps):
  1. Execute 'cat /etc/os-release' and 'uname -a' to identify the OS  [shell]
  2. Search the web for recent CVEs related to Fedora Linux 43         [web_search]
  3. Analyze the search results to extract CVE details                  [none]
  4. Compile a summary with severity ratings and mitigation steps       [none]
```

When execution reveals that the original plan needs adjustment (e.g. a step fails or new information changes the approach), the planner is invoked again to produce a revised plan while preserving completed steps.

### Memory

Two layers of memory provide state persistence:

- **Session memory** (working) — current goal, plan, conversation history, tool results, and discovered facts. Lives for the duration of a single agent run.
- **Persistent memory** (long-term) — key facts and session summaries stored as JSON at `~/.ollama_chain/`. Survives across runs so the agent can recall prior discoveries.

```bash
# View what the agent remembers
python -m ollama_chain --show-memory

=== Stored Facts ===
  - OS: Fedora Linux 43 (Workstation Edition)
  - Kernel: 6.18.12-200.fc43.x86_64

=== Recent Sessions ===
  [e3977c25] Research and summarize recent CVEs of my current system
    Completed 4/4 steps. Tools used: 2. Facts learned: 4.

# Wipe all stored memory
python -m ollama_chain --clear-memory
```

Facts discovered in one session are automatically available as context in subsequent sessions, so the agent does not repeat work it has already done.

### Tool Use

The agent interacts with the environment through a registry of tools:

| Tool | Description |
|---|---|
| `shell` | Execute shell commands (system info, processes, packages, network) |
| `read_file` | Read file contents (configs, logs, source code) |
| `write_file` | Write content to a file |
| `append_file` | Append content to an existing file |
| `list_dir` | List directory contents |
| `web_search` | Search the web via DuckDuckGo |
| `web_search_news` | Search recent news articles |
| `python_eval` | Evaluate Python expressions (math, string ops, list comprehensions) |

When the LLM is available, it chooses the tool and constructs the arguments. When all models are unavailable (e.g. transient Ollama errors), the agent **auto-executes tools directly from plan hints** — extracting quoted commands from step descriptions, applying heuristics for common tasks (OS detection, package listing), and building search queries from discovered facts. This means the agent can complete data-gathering steps even during model outages.

A basic safety check rejects obviously destructive shell commands (`rm -rf`, `mkfs`, `dd`, etc.).

### Autonomous Control Flow

The agent makes dynamic decisions at every iteration:

- **Model fallback chain** — tries the strongest model first, then falls back through progressively smaller models on transient errors (EOF, timeout, connection reset), with automatic retry and exponential backoff at each level.
- **Auto-execution** — when no model can respond, tool steps are executed directly from the plan. Reasoning-only steps are skipped gracefully (the final synthesis handles analysis).
- **Fact extraction** — tool output is automatically parsed for structured data (e.g. `/etc/os-release` fields, kernel version from `uname -a`) and stored in both session and persistent memory.
- **Re-planning** — every few steps, the agent checks for failures and invokes the planner to produce a revised plan incorporating observations from execution so far.
- **Iteration budget** — `--max-iterations` (default 15) provides a hard cap to prevent runaway loops.

### Agent Usage

```bash
# System investigation
python -m ollama_chain -m agent "What OS am I running? List the top 10 largest installed packages."

# Research task
python -m ollama_chain -m agent "Research and summarize recent CVEs of my current system"

# File exploration
python -m ollama_chain -m agent "Read my nginx config and suggest security improvements"

# With higher iteration budget
python -m ollama_chain -m agent --max-iterations 25 "Audit my system for open ports and explain each service"

# Fully offline (disables web_search tools)
python -m ollama_chain -m agent --no-search "Analyze my /var/log/syslog for errors in the last day"
```

### Agent Example Output

```
$ python -m ollama_chain -m agent "Identify my OS and find recent CVEs"

============================================================
[agent] Session e3977c25
[agent] Goal: Identify my OS and find recent CVEs
============================================================
[agent] Planning with qwen3:14b...
[agent] Plan (4 steps):
  1. Execute 'cat /etc/os-release' and 'uname -a'     [shell]
  2. Search for recent CVEs related to the identified OS [web_search]
  3. Analyze the search results                          [none]
  4. Compile a summary with severity and mitigation      [none]
[agent 1/15] Step 1: Execute 'cat /etc/os-release' and 'uname -a'
[agent]   Model: qwen3:14b
[agent]   Tool: shell({"command": "cat /etc/os-release && uname -a"})
[agent]   Result (OK): NAME="Fedora Linux" VERSION="43 (Workstation...
[agent]   Extracted fact: OS: Fedora Linux 43 (Workstation Edition)
[agent]   Extracted fact: Kernel: 6.18.12-200.fc43.x86_64
[agent 2/15] Step 2: Search for recent CVEs related to Fedora Linux 43
[agent]   Model: qwen3:14b
[agent]   Tool: web_search({"query": "recent CVEs Fedora Linux 43 2025"})
[agent]   Result (OK): [1] Fedora 43 Security Advisories...
[agent 3/15] Step 3: Analyze the search results
[agent]   Model: qwen3:32b
[agent]   (reasoning step — analysis completed)
[agent 4/15] Step 4: Compile a summary
[agent]   Model: qwen3:32b
[agent]   <final_answer> produced

============================================================
[agent] Step summary:
  [  OK] 1. Execute 'cat /etc/os-release' and 'uname -a'
  [  OK] 2. Search for recent CVEs related to Fedora Linux 43
  [  OK] 3. Analyze the search results
  [  OK] 4. Compile a summary with severity and mitigation
[agent] Facts discovered (4):
  - OS: Fedora Linux 43 (Workstation Edition)
  - Kernel: 6.18.12-200.fc43.x86_64
  - OS ID: fedora
  - OS Version ID: 43
============================================================
[agent] Session saved to memory.

Your system is running **Fedora Linux 43 (Workstation Edition)** with
kernel **6.18.12-200.fc43.x86_64**. ...
```

## Router Integration

The `auto` mode uses `router.py` to dynamically classify query complexity and select the optimal execution strategy — without requiring the user to manually choose a chain mode.

### How It Works

1. **Complexity classification** — the router analyzes the query using heuristics (word count, technical term density, multi-part indicators) and optionally a fast LLM call to classify it as `simple`, `moderate`, or `complex`
2. **Strategy selection** — based on complexity, the router picks the best chain strategy:

| Complexity | Strategy | Models Used | Search |
|---|---|---|---|
| `simple` | `direct_fast` | Fast model only | Skipped |
| `moderate` | `subset_cascade` | Fast + strong (skip intermediates) | On |
| `complex` | `full_cascade` | All models, smallest→largest | On |

3. **Fallback chains** — when a model fails, `build_fallback_chain()` provides an ordered list of alternative models (preferring larger models first)
4. **Step-level routing** — in agent mode, `select_models_for_step()` picks the right model subset per plan step (e.g., fast model for data-gathering, strong model for reasoning)

```
User query
    │
    ├── Router classifies complexity
    │     simple? ──→ direct_fast (fast model, skip search)
    │     moderate? ──→ subset_cascade (fast + strong)
    │     complex? ──→ full_cascade (all models)
    │
    └── Selected strategy executes with fallback on errors
```

### Router in Agent Mode

When the agent starts a session, the router classifies the goal's complexity and passes it to the planner as a `complexity_hint`. This affects:

- **Plan granularity** — simple goals get 1-3 coarse steps; complex goals get fine-grained decomposition with parallelism hints
- **Model selection per step** — data-gathering steps (shell, read_file) use the fast model for simple queries; reasoning steps always prefer the strongest model
- **Fallback ordering** — when a model is unavailable, the router provides an ordered fallback chain rather than a simple reverse iteration

## Cascading Error Handling

All chain modes now handle model failures gracefully, ensuring the cascade completes even when individual models are unavailable.

### Error Recovery Strategy

```
Model 1 (draft)  ──→ fails? try Model 2, then Model 3, ...
Model 2 (review) ──→ fails? skip and continue with current answer
Model 3 (review) ──→ fails? skip and continue
Model N (final)  ──→ fails? try Model N-1, then Model N-2, ...
                      all fail? return best draft available
```

**Draft stage**: if the first model fails, the cascade tries each subsequent model until one produces a draft. If all models fail, a `RuntimeError` is raised.

**Review stages**: if an intermediate model fails, it is skipped and the cascade continues with the current answer. The skipped model is logged for observability.

**Final stage**: if the strongest model fails, the cascade tries each remaining model in reverse order (largest first). If all fail, the best available draft is returned rather than raising an error.

### Observability

Skipped models are logged to stderr:
```
[cascade 2/3] medium:14b failed (connection reset), skipping...
[cascade] Completed with skipped models: ['medium:14b']
```

## Validation

The `validation.py` module provides functions to validate plan steps and model sequences before execution:

- `validate_step()` — checks that a step's tool exists, description is present, and dependencies are met
- `validate_plan()` — validates an entire plan for duplicate IDs, dangling dependencies, and invalid tools
- `validate_model_sequence()` — checks for empty or duplicate model lists

## Resilience

All Ollama calls across every mode (cascade, agent, search, planning) use automatic retry with exponential backoff on transient errors:

- **Retries** — up to 3 attempts per model with 2s/4s/8s delays
- **Model fallback** (agent mode) — if the strongest model is unavailable, the agent tries progressively smaller models before falling back to auto-execution
- **`keep_alive`** — models are kept loaded for 10 minutes to avoid repeated load/unload cycles during multi-step operations

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
# Cascade becomes: qwen3:14b → qwen3:32b → deepseek-r1:70b → mistral-large:123b
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
│   ├── common.py        # Shared utilities (ask, retry, message sanitization)
│   ├── models.py        # Ollama model discovery and ordering
│   ├── router.py        # Query routing, complexity classification, fallback chains
│   ├── chains.py        # Cascade + chaining mode implementations (with error handling)
│   ├── agent.py         # Autonomous agent (planning, tools, control flow)
│   ├── planner.py       # LLM-driven goal decomposition and re-planning
│   ├── memory.py        # Session + persistent memory (~/.ollama_chain/)
│   ├── tools.py         # Tool registry (shell, files, search, python_eval)
│   ├── validation.py    # Plan and model sequence validation
│   ├── search.py        # DuckDuckGo web search (no API key)
│   └── pcap.py          # PCAP parsing and analysis with scapy
├── tests/
│   ├── test_agent.py    # Agent helper tests
│   ├── test_cascade_errors.py  # Cascade error handling + auto mode tests
│   ├── test_memory.py   # Memory system tests
│   ├── test_planner.py  # Planner tests
│   ├── test_router.py   # Router classification, routing, fallback tests
│   ├── test_tools.py    # Tool registry tests
│   └── test_validation.py  # Validation tests
├── architecture.excalidraw  # Architecture diagram
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT
