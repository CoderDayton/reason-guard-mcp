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
	@# Validate version format (semver)
	@echo "$(V)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$$' || (echo "Error: Invalid semver format. Use X.Y.Z or X.Y.Z-suffix"; exit 1)
	@# Update version in pyproject.toml
	@sed -i 's/^version = ".*"/version = "$(V)"/' pyproject.toml
	@# Move [Unreleased] to new version in CHANGELOG.md
	@sed -i 's/## \[Unreleased\]/## [Unreleased]\n\n## [$(V)] - $(shell date +%Y-%m-%d)/' CHANGELOG.md
	@# Commit version bump and changelog
	git add pyproject.toml CHANGELOG.md
	git commit -m "chore: release v$(V)"
	@# Create annotated tag
	git tag -a "v$(V)" -m "Release v$(V)"
	@# Push commit and tag
	git push origin main
	git push origin "v$(V)"
	@echo ""
	@echo "✓ Release v$(V) published!"
	@echo "  → GitHub Actions will now build and publish to PyPI"
	@echo "  → View progress: https://github.com/$$(git remote get-url origin | sed 's/.*github.com[:/]//' | sed 's/.git$$//')/actions"

# Dry-run release (validates without pushing)
# Usage: make release-dry V=0.2.0
release-dry: check test
ifndef V
	$(error VERSION is not set. Usage: make release-dry V=X.Y.Z)
endif
	@echo "Dry-run release v$(V)..."
	@echo "$(V)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$$' || (echo "Error: Invalid semver format"; exit 1)
	@echo "✓ Version format valid"
	@echo "✓ Lint passed"
	@echo "✓ Tests passed"
	@echo ""
	@echo "Ready to release. Run: make release V=$(V)"
