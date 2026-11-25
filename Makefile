# ==========================================================
# sentiment-analyzer - Makefile
# ----------------------------------------------------------
# Unified interface for development, testing, and Docker
# workflows. Run `make help` to see available targets.
# ==========================================================

.DEFAULT_GOAL := help

# ----------------------------------------------------------
# Variables
# ----------------------------------------------------------

SOURCE       := sentiment_analyzer
VENV_PYTHON  := .venv/bin/python

# Advisories accepted for `make audit`. Each needs a reason and a re-check
# trigger; see docs/todos/01_general_audit.md (SEC-10).
#   chromadb CVE-2026-45829/45830/45831/45833: Chroma *HTTP server* only
#   (/api/v2 collection endpoints, server auth/RBAC). We embed Chroma
#   in-process (PersistentClient) and never run or expose its server.
#   No fixed release as of 2026-11-25 (latest 1.5.9). Re-check on upgrade.
AUDIT_IGNORES := \
	GHSA-f4j7-r4q5-qw2c PYSEC-2026-311 \
	GHSA-36p7-vc44-83pf PYSEC-2026-3814 \
	GHSA-2wm9-hf6c-p5cr PYSEC-2026-3813 \
	GHSA-xph7-9rjv-w5fr PYSEC-2026-3815

# ----------------------------------------------------------
# Help
# ----------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "  sentiment-analyzer"
	@echo "  ===================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ----------------------------------------------------------
# Environment Setup
# ----------------------------------------------------------

.PHONY: venv
venv: ## Create a virtual environment (Python version from .python-version)
	uv venv --python-preference only-managed

.PHONY: install
install: ## Install runtime dependencies incl. the ml + frontend extras
	uv sync --locked --all-extras --no-dev

.PHONY: install-dev
install-dev: ## Install everything: extras + dev + test tooling
	uv sync --locked --all-extras --group dev

.PHONY: install-test
install-test: ## Install extras + test dependencies
	uv sync --locked --all-extras --group test

.PHONY: update
update: ## Upgrade all dependency versions in the lockfile
	uv lock --upgrade

# ----------------------------------------------------------
# Code Quality (target dirs configured in pyproject.toml)
# ----------------------------------------------------------

.PHONY: format
format: ## Auto-format code
	uv run ruff format

.PHONY: format-check
format-check: ## Check formatting without changes
	uv run ruff format --check

.PHONY: lint
lint: ## Run linter and type checker
	uv run ruff check
	uv run mypy

.PHONY: fix
fix: ## Auto-fix lint issues
	uv run ruff check --fix

# ----------------------------------------------------------
# Testing
# ----------------------------------------------------------

.PHONY: test
test: ## Run test suite with coverage (fails under the pyproject threshold)
	uv run pytest

.PHONY: audit
audit: ## Scan locked dependencies for known vulnerabilities (pip-audit)
	@# torch is locked as a local "+cpu" build from the PyTorch index; audit the
	@# matching upstream PyPI release instead (same source, same advisories).
	uv export --frozen --all-extras --no-dev --no-hashes --no-emit-project --quiet \
		| sed -E 's/^(torch==[^+ ]+)\+cpu/\1/' > .audit-requirements.txt
	uvx pip-audit --disable-pip --no-deps --progress-spinner off \
		$(addprefix --ignore-vuln ,$(AUDIT_IGNORES)) -r .audit-requirements.txt; \
		status=$$?; rm -f .audit-requirements.txt; exit $$status

# ----------------------------------------------------------
# Running
# ----------------------------------------------------------

.PHONY: run
run: ## Start the FastAPI dev server on localhost:8000
	uv run uvicorn $(SOURCE).main:app --reload --host 127.0.0.1 --port 8000

.PHONY: frontend
frontend: ## Start the Streamlit frontend on port 8501 (ASA_API_URL / ASA_API_KEY)
	uv run streamlit run frontend/app.py

.PHONY: rag-bootstrap
rag-bootstrap: ## Seed the RAG vector store with starter documents (stop the API first)
	uv run python -m $(SOURCE).scripts.bootstrap_rag_store

# ----------------------------------------------------------
# Docker (managed via docker-compose.yml)
# ----------------------------------------------------------

.PHONY: docker-up
docker-up: ## Build and start the API (:8000) and UI (:8501) containers
	docker compose up --build -d

.PHONY: docker-down
docker-down: ## Stop and remove containers
	docker compose down

.PHONY: docker-logs
docker-logs: ## Tail container logs
	docker compose logs -f
