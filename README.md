# Reason Guard MCP

[![CI](https://github.com/coderdayton/reason-guard-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/reason-guard-mcp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/coderdayton/reason-guard-mcp/graph/badge.svg)](https://codecov.io/gh/coderdayton/reason-guard-mcp)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Dual-paradigm reasoning state manager for LLMs.** Track, validate, and optionally *enforce* structured reasoning via MCP.

## Two Paradigms

Reason Guard offers two ways to manage reasoning:

| Paradigm | Behavior | Best For |
|----------|----------|----------|
| **Guidance** | Suggestions, warnings, rewards. Bad steps recorded with feedback. | Open-ended analysis, debugging, brainstorming |
| **Enforcement** | REJECTS invalid steps. Must fix and retry. | Proofs, paradoxes, logical arguments |

**Start with `paradigm_hint`** — it analyzes your problem and recommends which to use.

## Quick Start

```bash
uvx reason-guard
```

```python
# 1. Get recommendation
paradigm_hint(problem="Prove the Monty Hall solution")
# → { "recommendation": "enforcement", "confidence": 0.85, "trap_detected": true }

# 2a. GUIDANCE MODE - flexible with feedback
think(action="start", problem="What is 15% of 200?")
think(action="continue", session_id="...", thought="15% = 0.15, so 0.15 × 200 = 30")
think(action="finish", session_id="...", answer="30", confidence=0.99)

# 2b. ENFORCEMENT MODE - strict step validation
initialize_reasoning(problem="Prove Monty Hall", complexity="high")
submit_step(session_id="...", step_type="premise", content="3 doors, 1 prize", confidence=0.95)
# → ACCEPTED or REJECTED
```

## Tools (9 total)

### Paradigm Selection

| Tool | Purpose |
|------|---------|
| `paradigm_hint` | Analyze problem, get paradigm recommendation |

### Guidance Mode (4 tools)

| Tool | Purpose |
|------|---------|
| `think` | Unified reasoning: start, continue, branch, revise, verify, finish |
| `compress` | Semantic context compression |
| `status` | Server/session health |

### Enforcement Mode (4 tools)

| Tool | Purpose |
|------|---------|
| `initialize_reasoning` | Start enforced session with step bounds |
| `submit_step` | Submit step (premise → hypothesis → verification → synthesis) |
| `create_branch` | Required when confidence is low |
| `verify_claims` | Verify claims before synthesis |
| `router_status` | Enforcement session state |

## Guidance Mode

The LLM reasons freely; server provides feedback.

```python
# Start session (auto-selects chain/matrix/hybrid)
think(action="start", problem="Calculate compound interest on $1000 at 5% for 3 years")
# → { "session_id": "abc123", "mode": "chain", "domain": "math" }

# Add steps - receive guidance, blind spot warnings, rewards
think(action="continue", session_id="abc123", thought="Year 1: 1000 × 1.05 = 1050")
think(action="continue", session_id="abc123", thought="Year 2: 1050 × 1.05 = 1102.50")
think(action="continue", session_id="abc123", thought="Year 3: 1102.50 × 1.05 = 1157.63")

# Complete
think(action="finish", session_id="abc123", answer="$1157.63", confidence=0.95)
```

### Think Actions

| Action | Purpose | Parameters |
|--------|---------|------------|
| `start` | Begin session | `problem`, `mode` (optional) |
| `continue` | Add step | `thought`, `row`/`col` (matrix) |
| `branch` | Alternative path | `from_step`, `thought` |
| `revise` | Update step | `step_index`, `thought` |
| `synthesize` | Summarize column | `col`, `synthesis` |
| `verify` | Check claim | `claim`, `status`, `evidence` |
| `finish` | Complete | `answer`, `confidence` |
| `analyze` | Get metrics | — |
| `suggest` | Get hint | — |
| `feedback` | Record outcome | `outcome` |
| `auto` | Execute suggestion | — |

## Enforcement Mode

Server REJECTS invalid steps. Forces disciplined reasoning.

```python
# Initialize with complexity bounds
initialize_reasoning(problem="Prove the Monty Hall solution", complexity="high")
# → { "session_id": "xyz789", "min_steps": 6, "max_steps": 12, "trap_detected": true }

# Submit steps - must follow state machine
submit_step(session_id="xyz789", step_type="premise",
            content="There are 3 doors, one with a prize", confidence=0.95)
# → { "status": "ACCEPTED", "step_number": 1 }

submit_step(session_id="xyz789", step_type="hypothesis",
            content="Switching wins 2/3 of the time", confidence=0.5)
# → { "status": "BRANCH_REQUIRED", "reason": "confidence < 0.75" }

# Must branch when confidence is low
create_branch(session_id="xyz789",
              alternatives=["Switching wins 2/3", "Staying wins 1/2", "Equal odds"])
# → { "status": "ACCEPTED" }

# Verify before synthesis
submit_step(session_id="xyz789", step_type="verification",
            content="Enumerating all cases shows switching wins 2/3", confidence=0.9)
# → { "status": "ACCEPTED" }

submit_step(session_id="xyz789", step_type="synthesis",
            content="Switching provably wins 2/3", confidence=0.95)
# → { "status": "ACCEPTED", "session_complete": true }
```

### Enforcement Rules

| Rule | Description |
|------|-------------|
| **A** | Cannot synthesize until `min_steps` reached |
| **B** | Low confidence (< 0.75) requires `create_branch` |
| **C** | Must verify before synthesis |
| **D** | Must follow state machine: premise → hypothesis → verification → synthesis |
| **E** | Must synthesize at `max_steps` |

### Step Types

| Type | Purpose |
|------|---------|
| `premise` | Establish facts |
| `hypothesis` | Propose claims |
| `verification` | Test claims |
| `synthesis` | Draw conclusions |

## When to Use Which

| Problem Type | Paradigm | Why |
|--------------|----------|-----|
| Open-ended analysis | Guidance | Flexibility, exploration |
| Math proofs | Enforcement | Prevents jumping to conclusions |
| Paradoxes (Monty Hall) | Enforcement | Traps require discipline |
| Code debugging | Guidance | Iterative, exploratory |
| Logical arguments | Enforcement | Forces verification |
| Brainstorming | Guidance | No strict structure |

## IDE Integration

<details open>
<summary><strong>VS Code / Cursor</strong></summary>

```json
{
  "servers": {
    "reason-guard": {
      "command": "uvx",
      "args": ["reason-guard"]
    }
  }
}
```
</details>

<details>
<summary><strong>Claude Desktop</strong></summary>

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "reason-guard": {
      "command": "uvx",
      "args": ["reason-guard"]
    }
  }
}
```
</details>

<details>
<summary><strong>HTTP/SSE Mode</strong></summary>

```bash
export SERVER_TRANSPORT=http
export SERVER_PORT=8000
uvx reason-guard
```
</details>

## Benchmark Results

| Metric | Baseline | Reason Guard | Improvement |
|--------|----------|--------------|-------------|
| **Overall Accuracy** | 50.0% | **83.3%** | **+33.3%** |
| Math (GSM8K) | 38.9% | 72.2% | +33.3% |
| Logic (LogiQA) | 66.7% | 100.0% | +33.3% |

See **[BENCHMARKS.md](BENCHMARKS.md)** for methodology.

## Configuration

Reason Guard is configured via environment variables. Below are the most common settings. For the complete reference including security, rate limiting, and observability options, see **[ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)**.

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_TRANSPORT` | `stdio` | `stdio`, `http`, or `sse` |
| `SERVER_HOST` | `127.0.0.1` | Bind address (use `0.0.0.0` for Docker) |
| `SERVER_PORT` | `8000` | Port for http/sse |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `text` | `text` or `json` |

### Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_PROBLEM_SIZE` | `50000` | Max problem text length |
| `MAX_THOUGHT_SIZE` | `10000` | Max thought text length |
| `MAX_THOUGHTS_PER_SESSION` | `1000` | Steps per session |
| `MAX_TOTAL_SESSIONS` | `500` | Concurrent sessions |

### Embedding Model

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `Snowflake/snowflake-arctic-embed-xs` | HuggingFace model for compression |
| `EMBEDDING_CACHE_DIR` | `~/.cache/reason-guard-mcp/models/` | Model cache directory |

See **[ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)** for:
- Authentication & API keys
- Rate limiting configuration
- Metrics & observability
- Session management
- Security settings

## Development

```bash
git clone https://github.com/coderdayton/reason-guard-mcp.git
cd reason-guard-mcp
uv sync --dev

make test       # All tests (698)
make test-smoke # Quick validation
make lint       # Linting
```

## Research

Based on:
- [Chain of Thought Empowers Transformers](https://arxiv.org/abs/2402.12875) (ICLR 2024)
- [Matrix of Thought](https://arxiv.org/abs/2509.03918) (2025)
- [Prompt Compression with Context-Aware Encoding](https://arxiv.org/abs/2409.01227) (AAAI 2025)

## License

MIT — see [LICENSE](LICENSE)

---

**[Contributing](CONTRIBUTING.md)** · **[Security](SECURITY.md)** · **[Changelog](CHANGELOG.md)**
