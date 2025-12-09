# MatrixMind MCP Server - Development Makefile
# Usage: make <target>
#
# Requires: uv (https://docs.astral.sh/uv/)

.PHONY: help install dev lint fmt typecheck test test-cov test-smoke clean build run pre-commit release version-bump

# Default target
help:
	@echo "MatrixMind MCP Server - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install dependencies"
	@echo "  make dev          Install with dev dependencies"
	@echo "  make pre-commit   Install pre-commit hooks"
	@echo ""
	@echo "Quality:"
	@echo "  make lint         Run ruff linter"
	@echo "  make fmt          Format code with ruff"
	@echo "  make typecheck    Run mypy type checker"
	@echo "  make check        Run all checks (lint + typecheck)"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run all tests"
	@echo "  make test-cov     Run tests with coverage"
	@echo "  make test-smoke   Run smoke tests only"
	@echo ""
	@echo "Build & Run:"
	@echo "  make build        Build package"
	@echo "  make run          Run MCP server"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Release:"
	@echo "  make release V=X.Y.Z   Create and push release tag"
	@echo "  make version-bump V=X.Y.Z  Update version in pyproject.toml"

# =============================================================================
# Setup
# =============================================================================

install:
	uv sync

dev:
	uv sync --all-extras

pre-commit:
	uv run pre-commit install
	@echo "Pre-commit hooks installed!"

# =============================================================================
# Code Quality
# =============================================================================

lint:
	uv run ruff check src/ tests/ examples/

fmt:
	uv run ruff format src/ tests/ examples/
	uv run ruff check --fix src/ tests/ examples/

typecheck:
	uv run mypy src/ --ignore-missing-imports

check: lint typecheck
	@echo "All checks passed!"

# =============================================================================
# Testing
# =============================================================================

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-smoke:
	uv run pytest tests/test_smoke.py -v

test-e2e:
	uv run pytest tests/test_e2e_mcp.py -v

# =============================================================================
# Build & Run
# =============================================================================

build:
	uv build

run:
	uv run matrixmind-mcp

# Run with stdio transport (for MCP clients)
run-stdio:
	uv run python -m src.server

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts"

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t matrixmind-mcp .

docker-run:
	docker run --rm -it \
		-e OPENAI_API_KEY \
		-e EMBEDDING_MODEL \
		matrixmind-mcp

# =============================================================================
# Release
# =============================================================================

# Update version in pyproject.toml
# Usage: make version-bump V=0.2.0
version-bump:
ifndef V
	$(error VERSION is not set. Usage: make version-bump V=X.Y.Z)
endif
	@echo "Bumping version to $(V)..."
	@sed -i 's/^version = ".*"/version = "$(V)"/' pyproject.toml
	@echo "Version updated to $(V) in pyproject.toml"
	@grep 'version = ' pyproject.toml | head -1

# Create a release (bump version, commit, tag, push)
# Usage: make release V=0.2.0
release: check test
ifndef V
	$(error VERSION is not set. Usage: make release V=X.Y.Z)
endif
	@echo "Creating release v$(V)..."
	@# Update version
	@sed -i 's/^version = ".*"/version = "$(V)"/' pyproject.toml
	@# Commit version bump
	git add pyproject.toml
	git commit -m "chore: bump version to $(V)"
	@# Create and push tag
	git tag -a "v$(V)" -m "Release v$(V)"
	@echo ""
	@echo "Release v$(V) created locally."
	@echo ""
	@echo "To publish, run:"
	@echo "  git push origin main"
	@echo "  git push origin v$(V)"
	@echo ""
	@echo "This will trigger the GitHub Actions release workflow."
