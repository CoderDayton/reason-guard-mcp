# Contributing to MatrixMind MCP

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/coderdayton/matrixmind-mcp.git
cd matrixmind-mcp

# Install dependencies (including dev)
make dev

# Install pre-commit hooks
make pre-commit

# Run tests to verify setup
make test
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Test additions/changes
- `chore/` - Maintenance tasks

### 2. Make Changes

Write your code following these guidelines:

- **Type hints**: All public functions must have type annotations
- **Docstrings**: Use Google-style docstrings for public APIs
- **Tests**: Add tests for new functionality
- **Line length**: 100 characters max

### 3. Run Quality Checks

```bash
# Run all checks (required before PR)
make check

# Individual checks
make lint       # Ruff linter
make typecheck  # Mypy type checker
make fmt        # Auto-format code

# Run tests
make test           # All tests
make test-smoke     # Quick smoke tests
make test-cov       # With coverage report
```

### 4. Commit Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code restructuring |
| `test` | Adding/updating tests |
| `chore` | Maintenance, deps, CI |
| `perf` | Performance improvement |

**Examples:**
```bash
# Feature
git commit -m "feat(compress): add adaptive compression ratio"

# Bug fix
git commit -m "fix(mot): handle empty context gracefully"

# With body explaining why
git commit -m "refactor(encoder): switch to lazy model loading

Reduces startup time from 8s to <1s by deferring model
download until first use. Addresses #42."
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub. The PR template will guide you through:
- Describing the change
- Linking related issues
- Confirming tests pass

## Code Style

### Python Style

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

def compress_context(
    context: str,
    question: str,
    ratio: float = 0.3,
) -> CompressionResult:
    """Compress context using semantic similarity.

    Args:
        context: The text to compress.
        question: Query to optimize compression for.
        ratio: Target compression ratio (0.0-1.0).

    Returns:
        CompressionResult with compressed text and metrics.

    Raises:
        CompressionException: If compression fails.
    """
    if not context:
        raise CompressionException("Context cannot be empty")
    ...
```

### Import Order

Imports are sorted automatically by ruff:

1. Standard library
2. Third-party packages
3. Local imports

### Error Handling

- Use custom exceptions from `src/utils/errors.py`
- Always provide meaningful error messages
- Log errors with context using `loguru`

```python
from src.utils.errors import CompressionException

try:
    result = process(data)
except SomeError as e:
    logger.error(f"Processing failed: {e}", exc_info=True)
    raise CompressionException(f"Failed to process: {e}") from e
```

## Testing

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── test_smoke.py        # Quick validation (<5s)
├── test_integration.py  # Pipeline tests
├── test_performance.py  # Load/stress tests
└── test_e2e_mcp.py      # End-to-end MCP tests
```

### Writing Tests

```python
import pytest
from src.tools.compress import ContextAwareCompressionTool

class TestCompressionTool:
    """Tests for compression tool."""

    def test_compress_reduces_tokens(self, mock_encoder):
        """Compression should reduce token count."""
        tool = ContextAwareCompressionTool(encoder=mock_encoder)
        result = tool.compress("long text...", "question", ratio=0.3)

        assert result.compressed_tokens < result.original_tokens
        assert result.compression_ratio <= 0.3

    def test_compress_empty_raises(self, mock_encoder):
        """Empty context should raise CompressionException."""
        tool = ContextAwareCompressionTool(encoder=mock_encoder)

        with pytest.raises(CompressionException, match="cannot be empty"):
            tool.compress("", "question")
```

### Test Markers

```python
@pytest.mark.slow        # Long-running tests
@pytest.mark.stress      # Stress/load tests
@pytest.mark.benchmark   # Performance benchmarks
```

Run specific markers:
```bash
pytest -m "not slow"     # Skip slow tests
pytest -m stress         # Only stress tests
```

## Release Process

Releases are automated via GitHub Actions when a version tag is pushed.

### Creating a Release

```bash
# 1. Ensure you're on main and up to date
git checkout main
git pull origin main

# 2. Run checks and tests
make check
make test

# 3. Create release (bumps version, commits, tags)
make release V=0.2.0

# 4. Push to trigger CI/CD
git push origin main
git push origin v0.2.0
```

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.2.0): New features, backward compatible
- **PATCH** (0.1.1): Bug fixes, backward compatible

Pre-release versions: `0.2.0-alpha`, `0.2.0-beta.1`, `0.2.0-rc.1`

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/coderdayton/matrixmind-mcp/discussions)
- **Bugs**: Open an [Issue](https://github.com/coderdayton/matrixmind-mcp/issues)
- **Security**: See [SECURITY.md](SECURITY.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
