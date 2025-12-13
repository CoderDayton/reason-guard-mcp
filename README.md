# MatrixMind MCP Server

[![CI](https://github.com/coderdayton/matrixmind-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/matrixmind-mcp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/coderdayton/matrixmind-mcp/graph/badge.svg)](https://codecov.io/gh/coderdayton/matrixmind-mcp)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastMCP 2.0](https://img.shields.io/badge/FastMCP-2.0-green.svg)](https://gofastmcp.com)

**Advanced reasoning state management for LLMs.** A unified reasoning engine with auto-mode selection, blind spot detection, domain-aware validation, and learning from feedback—all via MCP.

## Quick Start

```bash
# Install and run
uvx matrixmind-mcp

# Or with uv
uv tool install matrixmind-mcp
matrixmind-mcp
```

```python
# Start a reasoning session (auto-selects optimal mode)
think(action="start", problem="What is 15% of 200?")
# → {"session_id": "a1b2c3d4-...", "mode": "chain", "domain": "math"}

# Add reasoning steps
think(action="continue", session_id="...", thought="15% = 0.15")
think(action="continue", session_id="...", thought="0.15 × 200 = 30")

# Complete with answer
think(action="finish", session_id="...", answer="30", confidence=0.99)
```

## Architecture

MatrixMind tools are **state managers**, not LLM wrappers. The calling LLM does all reasoning; tools track, validate, and optimize the process:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Your LLM (Claude, GPT, etc.)                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Reasoning Process                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │               MatrixMind MCP (Unified Reasoner)                │  │
│  │                                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │  │
│  │  │ Auto-Mode    │  │ Thought      │  │ Domain Handlers      │  │  │
│  │  │ Selection    │  │ Graph        │  │ (math/code/logic)    │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │  │
│  │                                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │  │
│  │  │ Blind Spot   │  │ CISC         │  │ RLVR Learning        │  │  │
│  │  │ Detection    │  │ Confidence   │  │ (Weight Store)       │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Auto-Mode Selection** | Automatically chooses chain, matrix, or hybrid based on problem complexity |
| **Domain Detection** | Specialized handlers for math, code, logic, and factual problems |
| **Blind Spot Detection** | Identifies gaps in reasoning paths |
| **CISC Confidence** | Calibrated confidence scoring with multiple methods |
| **Thought Graph** | Sparse graph tracking relationships between reasoning steps |
| **RLVR Learning** | Records feedback to improve suggestion quality over time |
| **Semantic Compression** | Context-aware prompt compression preserving key information |

## Tools

### Primary Tool: `think`

A unified reasoning interface with 12 actions:

| Action | Purpose | Key Parameters |
|--------|---------|----------------|
| `start` | Initialize session | `problem`, `mode` (optional) |
| `continue` | Add reasoning step | `thought`, `row`/`col` (matrix) |
| `branch` | Create alternative path | `from_step`, `thought` |
| `revise` | Update previous thought | `step_index`, `thought` |
| `synthesize` | Summarize matrix column | `col`, `synthesis` |
| `verify` | Check a claim | `claim`, `status`, `evidence` |
| `finish` | Complete with answer | `answer`, `confidence` |
| `resolve` | Handle contradictions | `resolution`, `chosen_path` |
| `analyze` | Get quality metrics | — |
| `suggest` | Get next action hint | — |
| `feedback` | Record suggestion outcome | `outcome` |
| `auto` | Execute suggested action | — |

### Auxiliary Tools

| Tool | Purpose |
|------|---------|
| `compress` | Semantic context compression (stateless) |
| `status` | Server health and session stats |

## IDE Integration

<details open>
<summary><strong>VS Code / Cursor</strong></summary>

```json
{
  "servers": {
    "matrixmind": {
      "command": "uvx",
      "args": ["matrixmind-mcp"]
    }
  }
}
```
</details>

<details>
<summary><strong>Claude Desktop</strong></summary>

Add to `~/.config/claude/claude_desktop_config.json` (Linux/Mac) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "matrixmind": {
      "command": "uvx",
      "args": ["matrixmind-mcp"]
    }
  }
}
```
</details>

<details>
<summary><strong>HTTP/SSE Mode</strong></summary>

```bash
export SERVER_TRANSPORT=http  # or "sse"
export SERVER_PORT=8000
uvx matrixmind-mcp
# Connect to http://127.0.0.1:8000
```
</details>

## Usage Examples

### Auto-Mode Selection

```python
# MatrixMind auto-detects optimal mode based on problem complexity
think(action="start", problem="Calculate compound interest on $1000 at 5% for 3 years")
# → { "session_id": "...", "mode": "chain", "domain": "math" }

think(action="start", problem="Compare REST vs GraphQL vs gRPC for our microservices")
# → { "session_id": "...", "mode": "matrix", "domain": "general" }
```

### Chain Reasoning

```python
think(action="start", mode="chain", problem="Make 24 using 3, 4, 5, 6")
# → { "session_id": "abc123" }

think(action="continue", session_id="abc123", thought="Try (6-3) × (5+4) = 27. Too high.")
think(action="continue", session_id="abc123", thought="Try 6 × 4 = 24, use 5-3=2...")
think(action="continue", session_id="abc123", thought="6 × 4 × (5-3) / 2 = 24!")

think(action="finish", session_id="abc123", answer="6 × 4 × (5-3) / 2 = 24", confidence=0.95)
```

### Matrix of Thought

```python
think(action="start", mode="matrix",
      problem="Should we migrate to microservices?",
      rows=3, cols=2,
      strategies=["technical", "business", "operational"])
# → { "session_id": "xyz789" }

# Fill cells
think(action="continue", session_id="xyz789", row=0, col=0,
      thought="Technical pros: scalability, independent deployments")
think(action="continue", session_id="xyz789", row=0, col=1,
      thought="Technical cons: distributed complexity, debugging")

# Synthesize
think(action="synthesize", session_id="xyz789", col=0,
      synthesis="Pros outweigh cons for our scale")

think(action="finish", session_id="xyz789",
      answer="Yes, migrate with phased approach", confidence=0.8)
```

### Suggestion-Guided Reasoning

```python
think(action="start", problem="Complex optimization problem")

# Get AI suggestion for next action
think(action="suggest", session_id="abc")
# → { "suggested_action": "continue", "reasoning": "Add first reasoning step" }

# Auto-execute the suggestion
think(action="auto", session_id="abc")

# Provide feedback to improve future suggestions
think(action="feedback", session_id="abc", outcome="accepted")
```

### Verification Mode

```python
think(action="start", mode="verify",
      answer="Einstein published special relativity in 1905.",
      context="Albert Einstein published his theory of special relativity in 1905...")

think(action="verify", session_id="v123",
      claim="Published in 1905",
      status="supported",
      evidence="Context confirms 1905 publication")

think(action="finish", session_id="v123")
# → { "verified": true, "summary": { "supported": 1 } }
```

## Configuration

### Server Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SERVER_TRANSPORT` | Transport: `stdio`, `http`, or `sse` | `stdio` |
| `SERVER_HOST` | Bind address for http/sse | `127.0.0.1` |
| `SERVER_PORT` | Port for http/sse | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Model Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence embedding model | `Snowflake/snowflake-arctic-embed-xs` |
| `EMBEDDING_CACHE_DIR` | Model cache directory | `~/.cache/matrixmind-mcp/models/` |

### Rate Limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `RATE_LIMIT_MAX_SESSIONS` | Max new sessions per window | `100` |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window | `60` |
| `MAX_TOTAL_SESSIONS` | Max concurrent sessions | `500` |
| `SESSION_MAX_AGE_MINUTES` | Session TTL | `30` |

### Input Limits (Security)

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_PROBLEM_SIZE` | Max problem text size | `50000` |
| `MAX_THOUGHT_SIZE` | Max thought text size | `10000` |
| `MAX_CONTEXT_SIZE` | Max context text size | `100000` |
| `MAX_ALTERNATIVES` | Max MPPA alternatives | `10` |
| `MAX_THOUGHTS_PER_SESSION` | Max thoughts per session | `1000` |

### Storage (Optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_RAG` | Enable vector store for RAG | `false` |
| `VECTOR_DB_PATH` | Vector DB path (or `:memory:`) | `:memory:` |
| `MATRIXMIND_ALLOWED_DB_DIRS` | Colon-separated allowed DB directories | `~/.matrixmind:~/.local/share/matrixmind:/tmp:$CWD` |

## Security

MatrixMind includes several security hardening measures:

- **Input validation**: Size limits on all text inputs to prevent resource exhaustion
- **Path traversal protection**: Database paths validated against allowlist
- **Session limits**: Per-session thought limits to prevent memory exhaustion
- **Localhost binding**: HTTP/SSE defaults to `127.0.0.1` (not `0.0.0.0`)
- **Generic error messages**: Session errors don't leak internal state
- **Full UUIDs**: Cryptographically strong session identifiers

For production deployments with network exposure, add authentication middleware and configure `MATRIXMIND_ALLOWED_DB_DIRS` appropriately.

## Development

```bash
# Clone and install
git clone https://github.com/coderdayton/matrixmind-mcp.git
cd matrixmind-mcp
uv sync --dev

# Run tests
make test          # All tests
make test-smoke    # Quick validation
make lint          # Linting

# Run locally
python -m src.server
```

<details>
<summary><strong>Docker</strong></summary>

```bash
docker build -t matrixmind-mcp .
docker run -p 8000:8000 -e SERVER_TRANSPORT=http matrixmind-mcp

# Or with docker-compose
docker compose up -d
```
</details>

## Research

This implementation synthesizes techniques from:

- [Chain of Thought Empowers Transformers](https://arxiv.org/abs/2402.12875) (ICLR 2024)
- [Let Me Think! Long Chain-of-Thought](https://arxiv.org/abs/2505.21825) (2025)
- [Matrix of Thought: Re-evaluating Complex Reasoning](https://arxiv.org/abs/2509.03918) (2025)
- [Prompt Compression with Context-Aware Sentence Encoding](https://arxiv.org/abs/2409.01227) (AAAI 2025)
- [Multi-Path Plan Aggregation (MPPA)](https://arxiv.org/abs/2510.11620) (2025)
- [Forward-Backward Reasoning (FOBAR)](https://arxiv.org/abs/2308.07758) (ACL 2024)

## License

MIT — see [LICENSE](LICENSE)

---

**[Contributing](CONTRIBUTING.md)** · **[Security](SECURITY.md)** · **[Issues](https://github.com/coderdayton/matrixmind-mcp/issues)** · Built with [FastMCP](https://gofastmcp.com)
