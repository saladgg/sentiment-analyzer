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
IMAGE_NAME   := sentiment-analyzer
IMAGE_TAG    := latest

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
# Code Quality
# ----------------------------------------------------------

.PHONY: format
format: ## Auto-format code with ruff
	uv run ruff format $(SOURCE) tests

.PHONY: format-check
format-check: ## Check formatting without modifying files
	uv run ruff format --check $(SOURCE) tests

.PHONY: lint
lint: ## Run linter and type checker
	uv run ruff check $(SOURCE) tests
	uv run mypy $(SOURCE)

.PHONY: fix
fix: ## Auto-fix lint issues
	uv run ruff check $(SOURCE) tests --fix

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
# Docker
# ----------------------------------------------------------

.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: docker-run
docker-run: ## Run the container on port 8000
	docker run --rm -p 8000:8000 --env-file .env $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: docker-up
docker-up: ## Start services with docker compose
	docker compose up --build -d

.PHONY: docker-down
docker-down: ## Stop docker compose services
	docker compose down

.PHONY: docker-logs
docker-logs: ## Tail docker compose logs
	docker compose logs -f

# ----------------------------------------------------------
# CI / Aggregate Targets
# ----------------------------------------------------------

.PHONY: check
check: format-check lint test ## Run all CI checks (format, lint, test)

.PHONY: all
all: format lint test ## Format, lint, and test
