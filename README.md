<p align="center">
  <img src="ollama-chain-vscode-extension/resources/icon.svg" alt="ollama-chain logo" width="128">
</p>

# ollama-chain

Chain **all** local Ollama models together with **multi-source search** (DuckDuckGo, GitHub, Stack Overflow, trusted documentation sites) for maximum answer accuracy. Every available model participates — smallest drafts, each larger model refines, largest delivers the final answer with authoritative source citations. Includes an **autonomous agent** with planning, persistent memory, tool use, and dynamic control flow. **Kubernetes / OpenShift cluster analysis** via `oc`/`kubectl`. **PCAP network analysis** via Scapy. Exposes an **HTTP API server** with a memory-aware scheduler for safe multi-client access, plus a **VS Code extension** for IDE-integrated chat. Fully self-hosted, no API keys.

## How It Works

1. **Model discovery** — ollama-chain queries Ollama at startup and sorts all available models by parameter count
2. **Multi-source search** — DuckDuckGo, GitHub, Stack Overflow, and trusted documentation sites are searched in parallel (no API keys needed)
3. **Cascade** — the query flows through every model from smallest to largest:
   - Smallest model drafts the initial answer with search context
   - Each intermediate model reviews, corrects errors, and strengthens citations
   - Largest model produces the final authoritative answer
4. **Source citation** — all models are instructed to reference authoritative public sources (IETF RFCs, IEEE, ISO, NIST, W3C, Red Hat, kernel.org, MDN, official documentation)

```
User query
    │
    ├── Multi-source search (parallel):
    │     ├── DuckDuckGo (general web)
    │     ├── GitHub (repos + issues)
    │     ├── Stack Overflow (Q&A)
    │     └── Trusted docs (site-scoped)
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
- `tcpdump` or `tshark` for generating .pcap files (optional, pcap mode)
- `oc` (OpenShift CLI) or `kubectl` for Kubernetes/OpenShift cluster analysis (optional, k8s mode)

## Installation

```bash
git clone https://github.com/midu16/ollama-chain.git
cd ollama-chain
pip install .
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

# Kubernetes / OpenShift cluster analysis (CLI only)
python -m ollama_chain -m k8s --kubeconfig ~/.kube/config "What is the cluster health?"
python -m ollama_chain -m k8s --kubeconfig /tmp/ocp.kubeconfig "Why are pods crashing?"

# Fully offline (no web search)
python -m ollama_chain --no-search "What is a linked list?"

# List models and cascade order
python -m ollama_chain --list-models

# Memory management
python -m ollama_chain --show-memory
python -m ollama_chain --clear-memory
```

## Chaining Modes

| Mode | Pipeline | Best For | Availability |
|---|---|---|---|
| `cascade` | ALL models chain smallest→largest, each refining (default) | Maximum accuracy | CLI, API, Extension |
| `auto` | Router classifies complexity, picks optimal strategy automatically | Adaptive routing | CLI, API, Extension |
| `consensus` | All models answer independently, strongest merges | Reducing model bias | CLI, API, Extension |
| `route` | Fast classifies complexity (1-5), routes to fast or strong | Quick simple queries | CLI, API, Extension |
| `pipeline` | Fast extracts + classifies domain, strong reasons | Long/complex input | CLI, API, Extension |
| `verify` | Fast drafts, strong verifies and corrects | Fact-checking | CLI, API, Extension |
| `search` | Search first, strong synthesizes from results | Time-sensitive queries | CLI, API, Extension |
| `agent` | Autonomous agent with planning, memory, tools, and control flow | Multi-step tasks | CLI, API, Extension |
| `pcap` | Scapy parses → models analyze → expert report | Network analysis | **CLI only** |
| `k8s` | oc/kubectl gathers cluster state → models produce expert report | Cluster analysis | **CLI only** |
| `fast` | Direct to smallest model | Speed | CLI, API, Extension |
| `strong` | Direct to largest model | Single-model quality | CLI, API, Extension |

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
| `github_search` | Search GitHub repositories and issues |
| `stackoverflow_search` | Search Stack Overflow Q&A |
| `docs_search` | Search trusted documentation sites (kubernetes.io, MDN, Red Hat, etc.) |
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

## Selective Thinking & Temperature Control

ollama-chain dynamically applies chain-of-thought ("thinking") to models that support it, based on query complexity and chain stage. This maximizes accuracy on hard questions without wasting cycles on trivial ones.

### How It Works

1. **Model capability detection** — `model_supports_thinking()` queries `ollama.show()` for `capabilities: ["thinking"]`. Results are cached per-model. Falls back to name-based heuristics (qwen3, deepseek-r1, qwq) if the API call fails.
2. **Selective activation** — thinking is applied per-stage based on complexity:

| Complexity | Draft stage | Review stages | Final answer |
|---|---|---|---|
| `simple` | No thinking | No thinking | No thinking |
| `moderate` | No thinking | No thinking | **Thinking ON** |
| `complex` | No thinking | **Thinking ON** | **Thinking ON** |

3. **`/no_think` directive** — for models that support thinking, `/no_think` is prepended to prompts where thinking is intentionally disabled. Models without thinking capability receive the prompt unmodified.
4. **`<think>` tag stripping** — any `<think>...</think>` blocks in model responses are automatically stripped, so only the clean answer is returned.

### Temperature Control

Low temperatures are used in accuracy-critical stages to encourage deterministic, factual responses:

| Stage | Temperature | Purpose |
|---|---|---|
| Draft / data-gathering | default (model default) | Creative initial exploration |
| Review / verification | `0.4` | Focused error correction |
| Final answer / synthesis | `0.3` | Maximum factual precision |

### Per-Mode Thinking

| Mode | Thinking stages |
|---|---|
| `cascade` | Complexity-driven (see table above) |
| `auto` | Propagates complexity to cascade |
| `route` | Final answer for complex queries |
| `pipeline` | Final reasoning step |
| `verify` | Verification + final |
| `consensus` | Individual answers (if capable) + final merge |
| `search` | Final synthesis |
| `strong` | Always on |
| `agent` | Reasoning steps (non-simple) + final synthesis |
| `pcap` | Expert report |
| `k8s` | Expert report |

## Resilience

All Ollama calls across every mode (cascade, agent, search, planning) use automatic retry with exponential backoff on transient errors:

- **Retries** — up to 3 attempts per model with 2s/4s/8s delays
- **Model fallback** (agent mode) — if the strongest model is unavailable, the agent tries progressively smaller models before falling back to auto-execution
- **`keep_alive`** — models are kept loaded for 15 minutes (`_DEFAULT_KEEP_ALIVE = "15m"`) to avoid repeated load/unload cycles during multi-step operations, especially important for CPU-bound inference

## Source Citations

Every answer — regardless of mode — includes a **## Sources** section at the end listing all referenced sources. This is enforced at two levels:

1. **Prompt guidance** — all modes instruct models to cite sources inline and include a mandatory `## Sources` section
2. **Post-processing** — if the model omits the Sources section, a lightweight follow-up LLM call automatically appends one

Inline citation format:
```
[Source: IETF RFC 793, https://www.rfc-editor.org/rfc/rfc793]
[Source: Red Hat Documentation, https://docs.redhat.com/...]
[Source: IEEE 802.11, https://standards.ieee.org/...]
```

Prioritized source categories:
- **Standards bodies** — IETF RFCs, IEEE, ISO, NIST, W3C
- **Official documentation** — Red Hat, kernel.org, Mozilla MDN, Microsoft Docs
- **Vendor/project docs** — PostgreSQL docs, Nginx docs, kubernetes.io, docs.openshift.com, etc.
- **Peer-reviewed work** — academic papers, published research

Models are instructed to mark claims as unverified when no authoritative source can be referenced.

## Web Search (Multi-Source)

All modes include web search by default. Results are gathered from **multiple trusted sources in parallel** — no API keys required:

| Source | Provider | What it searches |
|---|---|---|
| **Web** | DuckDuckGo | General web results |
| **GitHub** | GitHub REST API | Repositories (by stars) and issues/discussions |
| **Stack Overflow** | Stack Exchange API | Programming Q&A with votes and answer counts |
| **Official Docs** | DuckDuckGo (site-scoped) | Trusted documentation sites only |
| **News** | DuckDuckGo News | Recent/time-sensitive articles |

### Search Pipeline

1. Fast model generates 1-3 optimized search queries
2. All providers searched **in parallel** (ThreadPoolExecutor):
   - DuckDuckGo web search (general)
   - GitHub repository search
   - GitHub issue/discussion search
   - Stack Overflow Q&A search
   - Trusted documentation site search (site-scoped DuckDuckGo)
3. Results aggregated, deduplicated by URL, and tagged by source
4. Models cross-reference their knowledge against multi-source results

### Trusted Documentation Domains

The docs search provider scopes queries to these authoritative sites:

`kubernetes.io` · `docs.openshift.com` · `docs.redhat.com` · `kernel.org` ·
`developer.mozilla.org` (MDN) · `docs.python.org` · `rfc-editor.org` · `w3.org` ·
`docs.docker.com` · `learn.microsoft.com` · `wiki.archlinux.org` · `man7.org` ·
`nginx.org` · `postgresql.org` · `go.dev` · `rust-lang.org` · `cppreference.com`

### Search Tool Fallbacks

When a search provider fails, the agent automatically tries alternatives:

| Tool | Fallbacks |
|---|---|
| `web_search` | `web_search_news` → `docs_search` |
| `github_search` | `web_search` |
| `stackoverflow_search` | `web_search` |
| `docs_search` | `web_search` |

### Example Search Output

```
[search] Generating search queries with qwen3:14b...
[search] Searching: ['TCP congestion control algorithms', 'TCP CUBIC vs BBR']
[search] Found 18 results (GitHub: 3, Official Docs: 4, Stack Overflow: 3, Web: 8)
```

Each result is tagged with its source for LLM context:
```
[1] [Web] TCP Congestion Control - Wikipedia
    https://en.wikipedia.org/wiki/TCP_congestion_control
    TCP uses a number of mechanisms to achieve high performance...

[2] [GitHub] google/bbr
    https://github.com/google/bbr
    BBR congestion control  [★3200 | C]

[3] [Stack Overflow] What is the difference between TCP CUBIC and BBR?
    https://stackoverflow.com/questions/...
    Score: 45 | Answers: 3 ✓ | Tags: tcp, networking, congestion-control

[4] [Official Docs] TCP Congestion Control - kernel.org
    https://www.kernel.org/doc/html/latest/networking/...
    The Linux kernel supports multiple TCP congestion control algorithms...
```

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

## Kubernetes / OpenShift Analysis (CLI only)

Analyze Kubernetes and OpenShift clusters using a kubeconfig file:

1. **oc/kubectl** gathers comprehensive cluster state (nodes, pods, deployments, services, events, storage, operators)
2. **Web search** finds troubleshooting context for detected issues
3. **Fast model** summarizes findings and classifies issues by severity
4. **Strong model** produces expert report with official documentation references

### What It Collects

- Cluster version and platform (vanilla K8s vs OpenShift)
- Nodes (status, roles, capacity, labels)
- Pods (all namespaces, unhealthy pod detection)
- Deployments and replica status
- Services and networking (Ingresses, Routes for OpenShift)
- Warning events (sorted by timestamp)
- Storage (PersistentVolumes, StorageClasses)
- Resource usage (CPU/memory via metrics-server, if available)
- OpenShift-specific: ClusterOperators, Machines, MachineConfigPools, installed operators (CSV), infrastructure config

### Issue Detection

- Degraded or unavailable ClusterOperators (OpenShift)
- Unhealthy pods (CrashLoopBackOff, Pending, Failed)
- High warning event volume
- Capacity and resource utilization concerns

### K8s Usage

```bash
# Basic cluster health report
python -m ollama_chain -m k8s --kubeconfig ~/.kube/config

# Ask specific questions about the cluster
python -m ollama_chain -m k8s --kubeconfig ~/.kube/config "Why are pods crashing in namespace foo?"
python -m ollama_chain -m k8s --kubeconfig /tmp/ocp.kubeconfig "Show cluster operator status"
python -m ollama_chain -m k8s --kubeconfig ~/.kube/config "Is the cluster healthy for production?"

# Without web search
python -m ollama_chain --no-search -m k8s --kubeconfig ~/.kube/config "Summarize the cluster"
```

> **Note:** The k8s mode uses `oc` (preferred) or `kubectl`, whichever is available in PATH. It performs **read-only** operations — no cluster state is modified.

## API Server

The API server (v0.7) provides an HTTP interface to ollama-chain with a **memory-aware scheduler** that queues prompts and prevents OOM when multiple clients send requests concurrently. Each prompt is executed as an isolated subprocess, and results are streamed back via Server-Sent Events (SSE).

### Starting the Server

```bash
# Default: listen on 127.0.0.1:8585, max 1 concurrent chain
ollama-chain-server

# Custom host/port
ollama-chain-server --host 0.0.0.0 --port 9090

# Allow 2 concurrent chains (requires sufficient RAM/VRAM)
ollama-chain-server --max-concurrent 2

# Increase per-job timeout to 15 minutes (default: 600s / 10 min)
ollama-chain-server --job-timeout 900

# Custom log directory
ollama-chain-server --log-dir /var/log/ollama-chain
```

### Logging

All server activity is logged to `.logs/ollama-chain-server.log` at **DEBUG** level for troubleshooting. A summary INFO stream is also printed to stderr.

| Component | What is logged |
|---|---|
| **HTTP requests** | Method, path, remote IP, status code, latency (ms) |
| **Prompt submission** | Mode, web_search flag, prompt preview, job ID |
| **Job lifecycle** | Queued, running, completed, failed, cancelled transitions |
| **Subprocess** | Command args, PID, stderr progress lines, return code |
| **Memory gating** | Available memory ratio, wait events |
| **Errors** | Full tracebacks for unhandled exceptions |

```bash
# View live logs
tail -f .logs/ollama-chain-server.log

# Search for errors
grep ERROR .logs/ollama-chain-server.log

# Filter by job ID
grep "abc123def456" .logs/ollama-chain-server.log
```

Log format: `YYYY-MM-DD HH:MM:SS  LEVEL     module  message`

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/prompt` | Submit a prompt — returns `{job_id, status, position}` |
| `GET` | `/api/prompt/{job_id}/stream` | SSE stream of progress events + final result |
| `GET` | `/api/prompt/{job_id}` | Poll job status and result |
| `DELETE` | `/api/prompt/{job_id}` | Cancel a queued or running job |
| `GET` | `/api/models` | List available Ollama models |
| `GET` | `/api/health` | Health check with queue stats |

### Submitting a Prompt

```bash
# Submit (timeout defaults to server's --job-timeout, or specify per-request)
curl -X POST http://localhost:8585/api/prompt \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is TCP?", "mode": "cascade", "web_search": true, "timeout": 600}'

# Response: {"job_id": "a1b2c3d4e5f6", "status": "queued", "position": 0}

# Stream results (SSE — includes periodic keepalive comments)
curl -N http://localhost:8585/api/prompt/a1b2c3d4e5f6/stream

# Cancel
curl -X DELETE http://localhost:8585/api/prompt/a1b2c3d4e5f6
```

### SSE Event Types

| Event | Data | When |
|---|---|---|
| `queued` | `{position}` | Job is waiting in the queue |
| `progress` | `{line}` | Chain execution progress (model stages, search) |
| `complete` | `{result}` | Final answer text |
| `timed_out` | `{error, partial_result}` | Job exceeded its timeout; partial result may be included |
| `error` | `{error}` | Chain execution failed |
| `cancelled` | `{}` | Job was cancelled |
| `: keepalive` | (SSE comment) | Heartbeat sent every 15s to keep the connection alive |

### Timeout & Keepalive

Each job has a configurable timeout (default: 600s / 10 minutes). The server enforces it at the subprocess level — when a chain exceeds its limit, the subprocess is killed and any partial output produced so far is returned alongside a `timed_out` event.

**Per-request timeout**: clients can set a `timeout` field (in seconds) in the `POST /api/prompt` body to override the server default (capped at 3600s):

```bash
curl -X POST http://localhost:8585/api/prompt \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Complex analysis...", "mode": "cascade", "timeout": 900}'
```

**SSE heartbeat**: the server sends `: keepalive <elapsed>s` comments every 15 seconds on idle SSE streams. This prevents proxies, load balancers, and client socket timeouts from dropping long-running connections. The VS Code extension uses these heartbeats to reset its idle timer, and falls back to polling if no data arrives within 2 minutes.

### Memory-Aware Scheduling

The scheduler checks `/proc/meminfo` before dequeuing each job. If available system memory drops below 10% of total, the scheduler pauses and waits for resources to free up. Combined with the default `max_concurrent=1`, this prevents concurrent chain executions from exhausting system memory when Ollama loads large models.

## VS Code Extension

The VS Code extension (v0.5) provides an IDE-integrated chat UI for ollama-chain. It connects exclusively to the API server for prompt execution — the API server must be running.

### Features

- Webview chat panel with markdown rendering and code block actions (copy, insert at cursor)
- Mode selector for all chain modes (cascade, auto, agent, etc.)
- Real-time progress display via SSE streaming from the API server
- Cancel support for running/queued jobs
- Session preservation — the webview retains context even when hidden, with SSE reconnection and polling fallback to keep sessions alive until the answer arrives
- Queue position indicator when multiple prompts are pending
- Elapsed time indicator during active sessions

### Setup

1. Start the API server: `ollama-chain-server`
2. Open VS Code, find "Ollama Chain" in the activity bar (llama + bee icon)
3. Configure `ollamaChain.apiUrl` if the server runs on a non-default address

### Configuration

| Setting | Default | Description |
|---|---|---|
| `ollamaChain.mode` | `cascade` | Default chain mode |
| `ollamaChain.webSearch` | `true` | Enable web search context |
| `ollamaChain.maxIterations` | `15` | Agent mode iteration budget |
| `ollamaChain.apiUrl` | `http://localhost:8585` | API server URL |
| `ollamaChain.timeout` | `600` | Per-job timeout in seconds (sent to server, which kills the subprocess after this duration) |

### Connection Flow

The extension communicates exclusively with the ollama-chain API server:

1. Checks API availability via `GET /api/health`
2. Submits prompts via `POST /api/prompt` (including the configured `timeout`)
3. Streams progress and results via SSE (`GET /api/prompt/{job_id}/stream`)
4. Resets an idle timer on every received chunk (including server heartbeats) — idle connections trigger SSE reconnection after 2 minutes of silence, not a hard error
5. Falls back to polling (`GET /api/prompt/{job_id}`) if SSE reconnection is exhausted (up to 600 poll attempts at 2s intervals = 20 minutes)
6. Handles `timed_out` events gracefully — displays partial results when available

If the API server is not running, the extension displays an error with instructions to start it.

## Makefile

All common tasks are available via `make`:

```bash
make help            # Show all targets with descriptions
make install         # Install runtime deps + editable package
make install-dev     # Install runtime + dev deps (pytest, pytest-asyncio, pytest-aiohttp)
make test            # Run full test suite (538 tests)
make test-quick      # Run tests without verbose output
make lint            # Syntax-check all Python source files
make server          # Start the API server (default: 127.0.0.1:8585)

# Run individual modes
make run-cascade QUERY="your question"
make run-auto QUERY="your question"
make run-agent AGENT_GOAL="your goal" MAX_ITER=20
make run-pcap PCAP_FILE=/tmp/capture.pcap
make run-k8s KUBECONFIG=~/.kube/config

# Utilities
make list-models
make show-memory
make clear-memory
```

| Variable | Default | Description |
|---|---|---|
| `QUERY` | `"What is TCP congestion control..."` | Query for chaining modes |
| `AGENT_GOAL` | `"Identify my OS..."` | Goal for agent mode |
| `MAX_ITER` | `15` | Agent max iterations |
| `PCAP_FILE` | `capture.pcap` | PCAP file for pcap mode |
| `KUBECONFIG` | `~/.kube/config` | Kubeconfig file for k8s mode |

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
[search] Found 16 results (GitHub: 3, Official Docs: 3, Stack Overflow: 3, Web: 7)
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
│   ├── server.py        # HTTP API server (aiohttp + CORS + SSE streaming)
│   ├── scheduler.py     # Memory-aware async job scheduler (subprocess isolation)
│   ├── common.py        # Shared utilities (ask, retry, message sanitization)
│   ├── models.py        # Ollama model discovery and ordering
│   ├── router.py        # Query routing, complexity classification, fallback chains
│   ├── chains.py        # Cascade + chaining mode implementations (with error handling)
│   ├── agent.py         # Autonomous agent (planning, tools, control flow)
│   ├── planner.py       # LLM-driven goal decomposition and re-planning
│   ├── memory.py        # Session + persistent memory (~/.ollama_chain/)
│   ├── tools.py         # Tool registry (shell, files, multi-source search, python_eval)
│   ├── validation.py    # Plan and model sequence validation
│   ├── search.py        # Multi-source search (DuckDuckGo, GitHub, Stack Overflow, docs)
│   ├── metrics.py       # Prompt quality metrics (9 heuristic dimensions)
│   ├── optimizer.py     # LLM-driven prompt rewriting
│   ├── pcap.py          # PCAP parsing and analysis with scapy (CLI only)
│   └── k8s.py           # Kubernetes/OpenShift cluster analysis (CLI only)
├── ollama-chain-vscode-extension/
│   ├── src/
│   │   ├── extension.ts          # Extension activation + command registration
│   │   ├── ChatViewProvider.ts   # Webview chat panel (API-only)
│   │   └── OllamaChainRunner.ts  # API client (SSE + polling fallback)
│   ├── resources/
│   │   ├── icon.svg              # Extension icon (llama + bee)
│   │   └── webview.js            # Chat UI logic
│   ├── package.json              # Extension manifest + configuration schema
│   └── tsconfig.json
├── tests/                      # 538 tests (18 files)
│   ├── test_agent.py           # Agent helper tests
│   ├── test_cascade_errors.py  # Cascade error handling + auto mode tests
│   ├── test_chains_modes.py    # Chain modes: selective thinking + temperature
│   ├── test_cli.py             # CLI argument parsing, pcap path detection
│   ├── test_common.py          # ask(), chat_with_retry(), model_supports_thinking
│   ├── test_k8s.py             # K8s/OCP analysis, format_analysis, issue detection
│   ├── test_memory.py          # Memory system tests
│   ├── test_metrics.py         # Prompt quality metrics
│   ├── test_models.py          # Model discovery, pick_models, memory eviction
│   ├── test_optimizer.py       # Prompt rewriting
│   ├── test_pcap.py            # PCAP analysis, port classification, format_analysis
│   ├── test_planner.py         # Planner tests
│   ├── test_router.py          # Router classification, routing, fallback tests
│   ├── test_scheduler.py       # Job lifecycle, queue positions, cancellation
│   ├── test_search.py          # Multi-source search (DDG, GitHub, SO, docs)
│   ├── test_server.py          # API endpoints, CORS, SSE, CLI-only rejection
│   ├── test_tools.py           # Tool registry tests
│   └── test_validation.py      # Validation tests
├── .logs/                      # Server logs (auto-created, git-ignored)
│   └── ollama-chain-server.log # DEBUG-level server log
├── architecture.excalidraw  # Architecture diagram
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT
