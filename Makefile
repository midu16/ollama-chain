# ollama-chain Makefile
# Build, test, and run all chaining modes from a single place.

PYTHON     ?= python3
PIP        ?= pip
QUERY      ?= "What is TCP congestion control and how does it work?"
AGENT_GOAL ?= "Identify my OS and kernel version, then find relevant security advisories"
MAX_ITER   ?= 15
PCAP_FILE  ?= capture.pcap
PACKAGE    := ollama_chain

# ───────────────────────────────── help ─────────────────────────────────

.PHONY: help
help: ## Show this help
	@echo "ollama-chain — chain all local Ollama models together"
	@echo ""
	@echo "Usage:  make <target> [QUERY=\"your question\"] [AGENT_GOAL=\"your goal\"]"
	@echo ""
	@echo "Build & Install:"
	@grep -E '^(install|install-dev|build|clean)\b' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  %-18s %s\n", $$1, $$NF}'
	@echo ""
	@echo "Testing & Quality:"
	@grep -E '^(test|lint)\b' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  %-18s %s\n", $$1, $$NF}'
	@echo ""
	@echo "Run Modes:"
	@grep -E '^run-' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  %-18s %s\n", $$1, $$NF}'
	@echo ""
	@echo "Utilities:"
	@grep -E '^(list-models|show-memory|clear-memory)\b' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  %-18s %s\n", $$1, $$NF}'
	@echo ""
	@echo "Variables:"
	@echo '  QUERY        Query for chaining modes (default: "$(QUERY)")'
	@echo '  AGENT_GOAL   Goal for agent mode  (default: "$(AGENT_GOAL)")'
	@echo '  MAX_ITER     Agent max iterations (default: $(MAX_ITER))'
	@echo '  PCAP_FILE    PCAP file for pcap mode (default: $(PCAP_FILE))'

# ──────────────────────────── build & install ───────────────────────────

.PHONY: install
install: ## Install runtime dependencies and the package (editable)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-dev
install-dev: ## Install runtime + dev dependencies (pytest, build)
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

.PHONY: build
build: ## Build distributable wheel and sdist
	$(PIP) install --quiet build
	$(PYTHON) -m build

.PHONY: clean
clean: ## Remove build artefacts, caches, eggs
	rm -rf build/ dist/ *.egg-info $(PACKAGE).egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

# ─────────────────────────── testing & lint ─────────────────────────────

.PHONY: test
test: ## Run the test suite with pytest
	$(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: test-quick
test-quick: ## Run tests without verbose output
	$(PYTHON) -m pytest tests/ -q

.PHONY: lint
lint: ## Syntax-check all Python source files
	@echo "Checking syntax..."
	@$(PYTHON) -m py_compile $(PACKAGE)/__init__.py
	@$(PYTHON) -m py_compile $(PACKAGE)/__main__.py
	@$(PYTHON) -m py_compile $(PACKAGE)/cli.py
	@$(PYTHON) -m py_compile $(PACKAGE)/chains.py
	@$(PYTHON) -m py_compile $(PACKAGE)/common.py
	@$(PYTHON) -m py_compile $(PACKAGE)/models.py
	@$(PYTHON) -m py_compile $(PACKAGE)/search.py
	@$(PYTHON) -m py_compile $(PACKAGE)/pcap.py
	@$(PYTHON) -m py_compile $(PACKAGE)/memory.py
	@$(PYTHON) -m py_compile $(PACKAGE)/tools.py
	@$(PYTHON) -m py_compile $(PACKAGE)/planner.py
	@$(PYTHON) -m py_compile $(PACKAGE)/agent.py
	@echo "All files OK."

# ─────────────────────────── run: all modes ─────────────────────────────

.PHONY: run-cascade
run-cascade: ## Run cascade mode (default — all models smallest→largest)
	$(PYTHON) -m $(PACKAGE) -m cascade $(QUERY)

.PHONY: run-consensus
run-consensus: ## Run consensus mode (all models answer, strongest merges)
	$(PYTHON) -m $(PACKAGE) -m consensus $(QUERY)

.PHONY: run-route
run-route: ## Run route mode (fast model scores complexity, routes)
	$(PYTHON) -m $(PACKAGE) -m route $(QUERY)

.PHONY: run-pipeline
run-pipeline: ## Run pipeline mode (extract→classify→reason)
	$(PYTHON) -m $(PACKAGE) -m pipeline $(QUERY)

.PHONY: run-verify
run-verify: ## Run verify mode (fast drafts, strong verifies)
	$(PYTHON) -m $(PACKAGE) -m verify $(QUERY)

.PHONY: run-search
run-search: ## Run search mode (web search first, strong synthesizes)
	$(PYTHON) -m $(PACKAGE) -m search $(QUERY)

.PHONY: run-fast
run-fast: ## Run fast mode (direct to smallest model)
	$(PYTHON) -m $(PACKAGE) -m fast $(QUERY)

.PHONY: run-strong
run-strong: ## Run strong mode (direct to largest model)
	$(PYTHON) -m $(PACKAGE) -m strong $(QUERY)

.PHONY: run-agent
run-agent: ## Run agent mode (autonomous planning + tools + memory)
	$(PYTHON) -m $(PACKAGE) -m agent --max-iterations $(MAX_ITER) $(AGENT_GOAL)

.PHONY: run-pcap
run-pcap: ## Run pcap mode (analyze a .pcap file)
	$(PYTHON) -m $(PACKAGE) -m pcap --pcap $(PCAP_FILE)

.PHONY: run-all
run-all: ## Run ALL chaining modes sequentially with the same query
	@echo "════════════════════════════════════════════════════════════"
	@echo " Running all modes with: $(QUERY)"
	@echo "════════════════════════════════════════════════════════════"
	@for mode in cascade consensus route pipeline verify search fast strong; do \
		echo ""; \
		echo "──────────── Mode: $$mode ────────────"; \
		$(PYTHON) -m $(PACKAGE) -m $$mode $(QUERY); \
		echo ""; \
	done
	@echo ""
	@echo "──────────── Mode: agent ────────────"
	$(PYTHON) -m $(PACKAGE) -m agent --max-iterations $(MAX_ITER) $(AGENT_GOAL)
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo " All modes completed."
	@echo "════════════════════════════════════════════════════════════"

# ──────────────────────────── utilities ─────────────────────────────────

.PHONY: list-models
list-models: ## List available Ollama models
	$(PYTHON) -m $(PACKAGE) --list-models

.PHONY: show-memory
show-memory: ## Show the agent's persistent memory
	$(PYTHON) -m $(PACKAGE) --show-memory

.PHONY: clear-memory
clear-memory: ## Clear the agent's persistent memory
	$(PYTHON) -m $(PACKAGE) --clear-memory
