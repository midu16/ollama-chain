# ollama-chain Makefile
# Build, test, install, serve, and run all chaining modes from a single place.

PYTHON     ?= python3
PIP        ?= pip
QUERY      ?= "What is TCP congestion control and how does it work?"
AGENT_GOAL ?= "Identify my OS and kernel version, then find relevant security advisories"
MAX_ITER   ?= 15
PCAP_FILE  ?= capture.pcap
KUBECONFIG ?= ~/.kube/config
HOST       ?= 127.0.0.1
PORT       ?= 8585
PACKAGE    := ollama_chain
SRC_FILES  := $(wildcard $(PACKAGE)/*.py)
TEST_FILES := $(wildcard tests/test_*.py)

# ───────────────────────────────── help ─────────────────────────────────

.PHONY: help
help: ## Show this help
	@echo "ollama-chain — chain all local Ollama models together"
	@echo ""
	@echo "Usage:  make <target> [QUERY=\"your question\"] [AGENT_GOAL=\"your goal\"]"
	@echo ""
	@echo "Build & Install:"
	@grep -E '^(install|install-dev|build|dist|clean|uninstall)\b' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  %-20s %s\n", $$1, $$NF}'
	@echo ""
	@echo "Testing & Quality:"
	@grep -E '^(test|lint|typecheck|coverage|check|ci)\b' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  %-20s %s\n", $$1, $$NF}'
	@echo ""
	@echo "Run Modes:"
	@grep -E '^run-' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  %-20s %s\n", $$1, $$NF}'
	@echo ""
	@echo "Server:"
	@grep -E '^server' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  %-20s %s\n", $$1, $$NF}'
	@echo ""
	@echo "Utilities:"
	@grep -E '^(list-models|show-memory|clear-memory|show-reports|version)\b' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  %-20s %s\n", $$1, $$NF}'
	@echo ""
	@echo "Variables:"
	@echo '  QUERY        Query for chaining modes (default: "$(QUERY)")'
	@echo '  AGENT_GOAL   Goal for agent mode  (default: "$(AGENT_GOAL)")'
	@echo '  MAX_ITER     Agent max iterations (default: $(MAX_ITER))'
	@echo '  PCAP_FILE    PCAP file for pcap mode (default: $(PCAP_FILE))'
	@echo '  KUBECONFIG   Kubeconfig file for k8s mode (default: $(KUBECONFIG))'
	@echo '  HOST         Server bind address (default: $(HOST))'
	@echo '  PORT         Server port (default: $(PORT))'

# ──────────────────────────── build & install ───────────────────────────

.PHONY: install
install: ## Install runtime dependencies and the package (editable)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-dev
install-dev: ## Install runtime + dev dependencies (pytest, build, lint)
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	$(PIP) install pytest-asyncio pytest-aiohttp pytest-cov ruff

.PHONY: uninstall
uninstall: ## Uninstall the package
	$(PIP) uninstall -y ollama-chain

.PHONY: build
build: clean ## Build distributable wheel and sdist
	$(PIP) install --quiet build
	$(PYTHON) -m build

.PHONY: dist
dist: build ## Alias for build

.PHONY: clean
clean: ## Remove build artefacts, caches, eggs, coverage data
	rm -rf build/ dist/ *.egg-info $(PACKAGE).egg-info .pytest_cache
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@echo "Clean."

# ─────────────────────────── testing & lint ─────────────────────────────

.PHONY: test
test: ## Run the full test suite (737 tests)
	$(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: test-quick
test-quick: ## Run tests without verbose output (fast CI check)
	$(PYTHON) -m pytest tests/ -q

.PHONY: test-unit
test-unit: ## Run only unit tests (no integration / server tests)
	$(PYTHON) -m pytest tests/ -v --tb=short \
		--ignore=tests/test_server.py \
		--ignore=tests/test_scheduler.py

.PHONY: test-quality
test-quality: ## Run code quality and accuracy tests only
	$(PYTHON) -m pytest tests/test_code_quality.py -v --tb=short

.PHONY: test-chains
test-chains: ## Run chain mode tests (cascade, route, pipeline, etc.)
	$(PYTHON) -m pytest tests/test_chains_modes.py tests/test_cascade_errors.py -v --tb=short

.PHONY: test-agent
test-agent: ## Run agent and planner tests
	$(PYTHON) -m pytest tests/test_agent.py tests/test_planner.py -v --tb=short

.PHONY: test-tools
test-tools: ## Run tool registry and execution tests
	$(PYTHON) -m pytest tests/test_tools.py -v --tb=short

.PHONY: test-search
test-search: ## Run search provider tests
	$(PYTHON) -m pytest tests/test_search.py -v --tb=short

.PHONY: test-router
test-router: ## Run router and complexity classification tests
	$(PYTHON) -m pytest tests/test_router.py -v --tb=short

.PHONY: test-memory
test-memory: ## Run memory system tests
	$(PYTHON) -m pytest tests/test_memory.py -v --tb=short

.PHONY: test-server
test-server: ## Run API server tests
	$(PYTHON) -m pytest tests/test_server.py tests/test_scheduler.py -v --tb=short

.PHONY: test-cli
test-cli: ## Run CLI argument and integration tests
	$(PYTHON) -m pytest tests/test_cli.py -v --tb=short

.PHONY: test-validation
test-validation: ## Run validation and plan repair tests
	$(PYTHON) -m pytest tests/test_validation.py -v --tb=short

.PHONY: test-metrics
test-metrics: ## Run prompt quality metrics and optimizer tests
	$(PYTHON) -m pytest tests/test_metrics.py tests/test_optimizer.py -v --tb=short

.PHONY: test-progress
test-progress: ## Run progress bar tests
	$(PYTHON) -m pytest tests/test_progress.py -v --tb=short

.PHONY: test-file
test-file: ## Run a single test file: make test-file F=tests/test_tools.py
	$(PYTHON) -m pytest $(F) -v --tb=short

.PHONY: test-match
test-match: ## Run tests matching a keyword: make test-match K=time_sensitive
	$(PYTHON) -m pytest tests/ -v --tb=short -k "$(K)"

.PHONY: test-failed
test-failed: ## Re-run only previously failed tests
	$(PYTHON) -m pytest tests/ --lf -v --tb=short

.PHONY: coverage
coverage: ## Run tests with coverage report (HTML + terminal)
	$(PYTHON) -m pytest tests/ --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html --tb=short
	@echo ""
	@echo "HTML report: htmlcov/index.html"

.PHONY: coverage-xml
coverage-xml: ## Run tests with coverage XML output (for CI)
	$(PYTHON) -m pytest tests/ --cov=$(PACKAGE) --cov-report=xml --tb=short

.PHONY: lint
lint: ## Syntax-check all Python source files
	@echo "Checking syntax of $(words $(SRC_FILES)) source files..."
	@for f in $(SRC_FILES); do \
		$(PYTHON) -m py_compile "$$f" || exit 1; \
	done
	@echo "Checking syntax of $(words $(TEST_FILES)) test files..."
	@for f in $(TEST_FILES); do \
		$(PYTHON) -m py_compile "$$f" || exit 1; \
	done
	@echo "All files OK."

.PHONY: lint-ruff
lint-ruff: ## Run ruff linter (install with: pip install ruff)
	ruff check $(PACKAGE)/ tests/

.PHONY: typecheck
typecheck: ## Run type checking with mypy (install with: pip install mypy)
	mypy $(PACKAGE)/ --ignore-missing-imports --no-error-summary || true

.PHONY: check
check: lint test ## Run lint + full test suite (pre-commit gate)

.PHONY: ci
ci: lint test-quick ## Lightweight CI check (lint + fast tests)

# ─────────────────────────── run: all modes ─────────────────────────────

.PHONY: run-cascade
run-cascade: ## Run cascade mode (default — all models smallest→largest)
	$(PYTHON) -m $(PACKAGE) -m cascade $(QUERY)

.PHONY: run-auto
run-auto: ## Run auto mode (router classifies complexity, picks best strategy)
	$(PYTHON) -m $(PACKAGE) -m auto $(QUERY)

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
run-pcap: ## Run pcap mode (analyze a .pcap file — CLI only)
	$(PYTHON) -m $(PACKAGE) -m pcap --pcap $(PCAP_FILE)

.PHONY: run-k8s
run-k8s: ## Run k8s mode (analyze a Kubernetes/OpenShift cluster — CLI only)
	$(PYTHON) -m $(PACKAGE) -m k8s --kubeconfig $(KUBECONFIG)

.PHONY: run-hack
run-hack: ## Run hack mode (penetration testing — requires --target)
	@echo "Usage: make run-hack TARGET=192.168.1.100"
	@test -n "$(TARGET)" || (echo "Error: TARGET is required" && exit 1)
	$(PYTHON) -m $(PACKAGE) -m hack --target $(TARGET) --max-iterations $(MAX_ITER)

.PHONY: run-all
run-all: ## Run ALL chaining modes sequentially with the same query
	@echo "════════════════════════════════════════════════════════════"
	@echo " Running all modes with: $(QUERY)"
	@echo "════════════════════════════════════════════════════════════"
	@for mode in cascade auto consensus route pipeline verify search fast strong; do \
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

.PHONY: run-offline
run-offline: ## Run cascade mode fully offline (no web search)
	$(PYTHON) -m $(PACKAGE) -m cascade --no-search $(QUERY)

.PHONY: run-metrics
run-metrics: ## Evaluate prompt quality metrics without running models
	$(PYTHON) -m $(PACKAGE) --metrics-only $(QUERY)

.PHONY: run-optimize
run-optimize: ## Optimize prompt and display result without running models
	$(PYTHON) -m $(PACKAGE) --optimize-only $(QUERY)

# ──────────────────────────── server ──────────────────────────────────

.PHONY: server
server: ## Start the API server (default: 127.0.0.1:8585)
	ollama-chain-server --host $(HOST) --port $(PORT)

.PHONY: server-dev
server-dev: ## Start the API server with verbose logging and extended timeout
	ollama-chain-server --host $(HOST) --port $(PORT) --job-timeout 900 --log-dir .logs

.PHONY: server-public
server-public: ## Start server on all interfaces (0.0.0.0)
	ollama-chain-server --host 0.0.0.0 --port $(PORT)

.PHONY: server-health
server-health: ## Check API server health
	@curl -sf http://$(HOST):$(PORT)/api/health | python3 -m json.tool 2>/dev/null || \
		echo "Server not running on $(HOST):$(PORT)"

.PHONY: server-models
server-models: ## List models via the API server
	@curl -sf http://$(HOST):$(PORT)/api/models | python3 -m json.tool 2>/dev/null || \
		echo "Server not running on $(HOST):$(PORT)"

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

.PHONY: show-reports
show-reports: ## List saved penetration test reports
	$(PYTHON) -m $(PACKAGE) --show-reports

.PHONY: version
version: ## Show package version
	@$(PYTHON) -c "from ollama_chain import __version__; print(f'ollama-chain v{__version__}')"

.PHONY: test-count
test-count: ## Count total tests across all test files
	@$(PYTHON) -m pytest tests/ --co -q 2>&1 | tail -1

.PHONY: test-list
test-list: ## List all test files with counts
	@for f in tests/test_*.py; do \
		line=$$($(PYTHON) -m pytest "$$f" --co -q 2>&1 | tail -1); \
		count=$$(echo "$$line" | sed 's/[^0-9].*//' ); \
		printf "  %-30s %3s tests\n" "$$(basename $$f)" "$$count"; \
	done
	@echo ""
	@$(PYTHON) -m pytest tests/ --co -q 2>&1 | tail -1 | sed 's/^/  Total: /'
