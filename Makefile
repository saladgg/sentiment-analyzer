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
venv: ## Create a virtual environment with Python 3.14
	uv venv --python 3.14 --python-preference only-managed

.PHONY: install
install: ## Install production dependencies
	uv sync

.PHONY: install-dev
install-dev: ## Install dev + test dependencies
	uv sync --group dev

.PHONY: install-test
install-test: ## Install test dependencies only
	uv sync --group test

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
test: ## Run test suite with coverage
	uv run pytest --cov=$(SOURCE) --cov-report=term-missing

# ----------------------------------------------------------
# Running
# ----------------------------------------------------------

.PHONY: run
run: ## Start the FastAPI dev server on port 8000
	uv run uvicorn $(SOURCE).main:app --reload --host 0.0.0.0 --port 8000

# ----------------------------------------------------------
# Docker (managed via docker-compose.yml)
# ----------------------------------------------------------

.PHONY: docker-up
docker-up: ## Build and start containers
	docker compose up --build -d

.PHONY: docker-down
docker-down: ## Stop and remove containers
	docker compose down

.PHONY: docker-logs
docker-logs: ## Tail container logs
	docker compose logs -f
