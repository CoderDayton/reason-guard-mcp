---
name: pre-commit-python-stack
description: Generate pre-commit configuration for Python projects with ruff, mypy, bandit, gitleaks, and file hygiene hooks. Includes pyproject.toml tool configs.
license: MIT
---

# Pre-commit Python Stack

## Overview

Generate a production-ready pre-commit configuration for Python projects. This skill creates both `.pre-commit-config.yaml` and the corresponding `pyproject.toml` tool configurations for a complete quality gate setup.

## When to Use

- Setting up a new Python project
- Adding pre-commit hooks to an existing project
- Updating or standardizing quality gates
- User asks for "linting", "formatting", "pre-commit", or "code quality" setup

## Files to Generate

### 1. `.pre-commit-config.yaml`

```yaml
# Pre-commit hooks for Python project
# Install: pre-commit install
# Run all: pre-commit run --all-files

repos:
  # Ruff - Fast Python linter and formatter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.2
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
        types_or: [python, pyi]
      - id: ruff-format
        types_or: [python, pyi]

  # General file hygiene
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
        exclude: ^docs/
      - id: end-of-file-fixer
        exclude: ^docs/
      - id: check-yaml
        args: [--unsafe]  # Allow custom tags
      - id: check-toml
      - id: check-json
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-merge-conflict
      - id: debug-statements
      - id: detect-private-key

  # Type checking with mypy
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports, --no-error-summary]
        additional_dependencies:
          - types-PyYAML
          - pydantic>=2.0
        pass_filenames: false
        entry: mypy src/

  # Security checks
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.10
    hooks:
      - id: bandit
        args: [-c, pyproject.toml, -r, src/]
        additional_dependencies: ["bandit[toml]"]

  # Secret detection with gitleaks
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks

# CI configuration
ci:
  autofix_commit_msg: "style: auto-fix from pre-commit hooks"
  autofix_prs: true
  autoupdate_branch: main
  autoupdate_commit_msg: "chore: update pre-commit hooks"
  autoupdate_schedule: weekly
  skip: [mypy]  # Skip mypy in CI (run separately with full deps)
```

### 2. `pyproject.toml` Tool Sections

Add these sections to `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "D", "UP", "B", "C4", "SIM"]
ignore = [
    "D100", "D104", "D107", "D203", "D213",
    "D401",  # First line imperative mood
]

[tool.bandit]
exclude_dirs = ["tests", "examples"]
skips = ["B101"]  # Allow assert for contracts
```

### 3. Dev Dependencies

Add to `[project.optional-dependencies]` or `[dependency-groups]`:

```toml
dev = [
    "mypy>=1.5.0",
    "ruff>=0.1.0",
    "pre-commit>=3.4.0",
    "types-pyyaml>=6.0",  # Add type stubs as needed
]
```

## Hook Reference

| Hook | Purpose | Auto-fix |
|------|---------|----------|
| `ruff` | Linting (flake8 replacement) | Yes |
| `ruff-format` | Formatting (black replacement) | Yes |
| `trailing-whitespace` | Remove trailing whitespace | Yes |
| `end-of-file-fixer` | Ensure newline at EOF | Yes |
| `check-yaml/toml/json` | Validate config files | No |
| `check-added-large-files` | Prevent large file commits | No |
| `check-merge-conflict` | Detect conflict markers | No |
| `debug-statements` | Find leftover breakpoints | No |
| `detect-private-key` | Block private key commits | No |
| `mypy` | Type checking | No |
| `bandit` | Security vulnerability scan | No |
| `gitleaks` | Secret detection | No |

## Ruff Rule Categories

| Code | Category | Description |
|------|----------|-------------|
| `E` | pycodestyle errors | PEP 8 violations |
| `F` | Pyflakes | Logical errors |
| `W` | pycodestyle warnings | Style warnings |
| `I` | isort | Import sorting |
| `N` | pep8-naming | Naming conventions |
| `D` | pydocstyle | Docstring conventions |
| `UP` | pyupgrade | Python version upgrades |
| `B` | flake8-bugbear | Bug-prone patterns |
| `C4` | flake8-comprehensions | Comprehension improvements |
| `SIM` | flake8-simplify | Code simplification |

## Customization Points

### Adjust for Source Directory

If using `src/` layout:
```yaml
entry: mypy src/
args: [-c, pyproject.toml, -r, src/]
```

If using flat layout:
```yaml
entry: mypy .
args: [-c, pyproject.toml, -r, .]
```

### Add Type Stubs

```yaml
additional_dependencies:
  - types-PyYAML
  - types-requests
  - types-redis
  - pydantic>=2.0
```

### Skip Rules

```toml
[tool.ruff.lint]
ignore = [
    "D100",  # Missing module docstring
    "D104",  # Missing package docstring
    "D107",  # Missing __init__ docstring
    "D203",  # Blank line before class docstring
    "D213",  # Multi-line summary second line
]
```

### Exclude Paths

```toml
[tool.bandit]
exclude_dirs = ["tests", "examples", "scripts"]
```

## Installation Commands

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files (first time)
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

## Troubleshooting

**mypy fails with missing imports**:
- Add type stubs to `additional_dependencies`
- Or add `ignore_missing_imports = true` to `[tool.mypy]`

**ruff conflicts with existing formatter**:
- Remove black/isort if present - ruff replaces both
- Remove flake8 - ruff is a drop-in replacement

**gitleaks false positives**:
- Create `.gitleaksignore` with patterns to ignore
- Or use inline `# gitleaks:allow` comments

**Large file blocked**:
- Increase `--maxkb` limit or use Git LFS
- Or exclude specific paths in the hook config
