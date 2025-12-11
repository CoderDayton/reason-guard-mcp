# MatrixMind MCP Server

[![CI](https://github.com/coderdayton/matrixmind-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/matrixmind-mcp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/coderdayton/matrixmind-mcp/graph/badge.svg)](https://codecov.io/gh/coderdayton/matrixmind-mcp)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastMCP 2.0](https://img.shields.io/badge/FastMCP-2.0-green.svg)](https://gofastmcp.com)

**Advanced reasoning state management for LLMs.** Like `sequential-thinking` but with Matrix-of-Thought, long-chain reasoning, semantic compression, and fact verification—all via MCP.

## Architecture

MatrixMind tools are **state managers**, not LLM wrappers. The calling LLM does all reasoning; tools track and organize the process:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your LLM (Claude, GPT, etc.)             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Reasoning Process                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MatrixMind MCP (State Managers)             │   │
│  │  • chain_start → chain_add_step → chain_finalize        │   │
│  │  • matrix_start → matrix_set_cell → matrix_finalize     │   │
│  │  • verify_start → verify_add_claim → verify_finalize    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Tools

| Tool Family | Purpose | API |
|-------------|---------|-----|
| **Chain Reasoning** | Step-by-step reasoning with branching/revision | `chain_start` → `chain_add_step` → `chain_finalize` |
| **Matrix of Thought** | Multi-perspective reasoning grid | `matrix_start` → `matrix_set_cell` → `matrix_synthesize` → `matrix_finalize` |
| **Verification** | Claim-level fact checking | `verify_start` → `verify_add_claim` → `verify_claim` → `verify_finalize` |
| **Compression** | Semantic context compression | `compress_prompt` (stateless) |
| **Strategy** | Auto-select optimal approach | `recommend_reasoning_strategy` (stateless) |

## IDE Integration

<details open>
<summary><strong>VS Code</strong></summary>

Add to `.vscode/settings.json`:

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
<summary><strong>Cursor</strong></summary>

Add to `.cursor/mcp.json`:

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
<summary><strong>HTTP Mode</strong></summary>

```bash
export SERVER_TRANSPORT=http
export SERVER_PORT=8000
uvx matrixmind-mcp
# Connect to http://localhost:8000
```
</details>

## Usage Examples

### Long Chain Reasoning

For step-by-step problems requiring sequential logic:

```python
# 1. Start a chain
chain_start(problem="Make 24 using 3, 4, 5, 6", expected_steps=10)
# Returns: { "session_id": "abc123", "status": "started", ... }

# 2. Add reasoning steps (LLM provides the reasoning)
chain_add_step(session_id="abc123", thought="I can try (6-3) × (5+4) = 3 × 9 = 27. Too high.")
chain_add_step(session_id="abc123", thought="Let me try 6 × (5 - 3 + 4) = 6 × 6 = 36. Too high.")
chain_add_step(session_id="abc123", thought="Try (6 - 4) × (5 + 3) = 2 × 8 = 16. Too low.")
chain_add_step(session_id="abc123", thought="Try (5 - 3) × 6 + 4 = 2 × 6 + 4 = 16. Still wrong.")
chain_add_step(session_id="abc123", thought="Try (3 + 5) × (6 - 4) = 8 × 2 = 16. Not 24.")
chain_add_step(session_id="abc123", thought="Try 6 / (1 - 3/4)... wait, I don't have 1.")
chain_add_step(session_id="abc123", thought="(6 × 4) × (5 - 3) / 4 = 24 × 2 / 4 = 12. Hmm.")
chain_add_step(session_id="abc123", thought="(5 × 3 - 6) × 4 = (15 - 6) × 4 = 9 × 4 = 36.")
chain_add_step(session_id="abc123", thought="(6 - 5 + 3) × 4 = 4 × 4 = 16.")
chain_add_step(session_id="abc123", thought="6 × 4 = 24, and 5 - 3 = 2. So 6 × 4 × (5-3)/2 = 24!")

# 3. Finalize with answer
chain_finalize(session_id="abc123", answer="6 × 4 × (5-3) / 2 = 24", confidence=0.95)
```

### Matrix of Thought

For problems benefiting from multiple perspectives:

```python
# 1. Start matrix (3 perspectives × 2 criteria)
matrix_start(
    question="Should we migrate to microservices?",
    rows=3, cols=2,
    strategies=["technical", "business", "operational"]
)

# 2. Fill cells with analysis
matrix_set_cell(session_id="xyz", row=0, col=0, thought="Technical pros: scalability, independent deployments")
matrix_set_cell(session_id="xyz", row=0, col=1, thought="Technical cons: distributed complexity, debugging")
matrix_set_cell(session_id="xyz", row=1, col=0, thought="Business pros: faster feature delivery")
matrix_set_cell(session_id="xyz", row=1, col=1, thought="Business cons: higher initial cost")
# ... fill remaining cells

# 3. Synthesize columns
matrix_synthesize(session_id="xyz", col=0, synthesis="Pros outweigh cons for our scale")

# 4. Finalize
matrix_finalize(session_id="xyz", answer="Yes, migrate with phased approach", confidence=0.8)
```

### Fact Verification

For checking claims against context:

```python
# 1. Start verification
verify_start(
    answer="Einstein published special relativity in 1905.",
    context="Albert Einstein published his theory of special relativity in 1905..."
)

# 2. Add claims to verify
verify_add_claim(session_id="v123", content="Einstein published special relativity in 1905")

# 3. Verify each claim
verify_claim(session_id="v123", claim_id=0, status="supported",
             evidence="Context confirms 1905 publication", confidence=0.98)

# 4. Get verification result
verify_finalize(session_id="v123")
# Returns: { "verified": true, "summary": { "supported": 1, "contradicted": 0 } }
```

## Tool Reference

<details>
<summary><strong>Chain Tools</strong></summary>

### `chain_start`
Initialize a reasoning chain.

| Parameter | Type | Description |
|-----------|------|-------------|
| `problem` | string | Problem statement |
| `expected_steps` | int | Expected number of steps |
| `metadata` | object | Optional metadata |

### `chain_add_step`
Add a reasoning step.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session from chain_start |
| `thought` | string | Reasoning content |
| `branch_from` | int | Optional: branch from step N |
| `revises` | int | Optional: revise step N |

### `chain_get`
Get current chain state.

### `chain_finalize`
Complete the chain with final answer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session ID |
| `answer` | string | Final answer |
| `confidence` | float | 0.0-1.0 confidence |

</details>

<details>
<summary><strong>Matrix Tools</strong></summary>

### `matrix_start`
Initialize a reasoning matrix.

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | string | Question to analyze |
| `rows` | int | Number of perspectives |
| `cols` | int | Number of criteria |
| `strategies` | list | Optional: named strategies |

### `matrix_set_cell`
Fill a matrix cell.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session ID |
| `row` | int | Row index |
| `col` | int | Column index |
| `thought` | string | Analysis content |

### `matrix_synthesize`
Synthesize a column.

### `matrix_finalize`
Complete with final answer.

</details>

<details>
<summary><strong>Verify Tools</strong></summary>

### `verify_start`
Initialize verification session.

| Parameter | Type | Description |
|-----------|------|-------------|
| `answer` | string | Answer to verify |
| `context` | string | Context to verify against |

### `verify_add_claim`
Add a claim to verify.

### `verify_claim`
Verify a specific claim.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session ID |
| `claim_id` | int | Claim index |
| `status` | string | "supported" or "contradicted" |
| `evidence` | string | Evidence for verdict |
| `confidence` | float | 0.0-1.0 confidence |

### `verify_finalize`
Get verification summary.

</details>

<details>
<summary><strong>Stateless Tools</strong></summary>

### `compress_prompt`
Compress context using semantic filtering.

| Parameter | Type | Description |
|-----------|------|-------------|
| `context` | string | Text to compress |
| `question` | string | Question for relevance |
| `compression_ratio` | float | Target ratio (0.3 = 3× compression) |

### `recommend_reasoning_strategy`
Get recommendation for optimal approach.

| Parameter | Type | Description |
|-----------|------|-------------|
| `problem` | string | Problem description |
| `token_budget` | int | Available token budget |

</details>

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence embedding model | `Snowflake/snowflake-arctic-embed-xs` |
| `EMBEDDING_CACHE_DIR` | Model cache directory | `~/.cache/matrixmind-mcp/models/` |
| `SERVER_TRANSPORT` | `stdio`, `http`, or `sse` | `stdio` |
| `SERVER_HOST` | Host for http/sse | `localhost` |
| `SERVER_PORT` | Port for http/sse | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Development

```bash
# Clone and install
git clone https://github.com/coderdayton/matrixmind-mcp.git
cd matrixmind-mcp
uv sync --dev

# Run tests
make test          # All tests
make test-smoke    # Quick validation

# Run locally
python -m src.server
```

<details>
<summary><strong>Docker</strong></summary>

```bash
docker build -t matrixmind-mcp .
docker run -p 8000:8000 matrixmind-mcp

# Or with docker-compose
docker compose up -d
```
</details>

## Research

This implementation synthesizes techniques from:

- [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875) (ICLR 2024)
- [Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](https://arxiv.org/abs/2505.21825) (2025)
- [Matrix of Thought: Re-evaluating Complex Reasoning](https://arxiv.org/abs/2509.03918) (2025)
- [Prompt Compression with Context-Aware Sentence Encoding](https://arxiv.org/abs/2409.01227) (AAAI 2025)
- [Enhancing Long Chain-of-Thought Reasoning through Multi-Path Plan Aggregation](https://arxiv.org/abs/2510.11620) (2025) — MPPA
- [Forward-Backward Reasoning in Large Language Models](https://arxiv.org/abs/2308.07758) (ACL 2024) — FOBAR

## Benchmark Results

### Performance Metrics

Run `python examples/benchmark.py --full` to reproduce.

| Operation | p50 | p95 | p99 | Throughput |
|-----------|-----|-----|-----|------------|
| Status (baseline) | 4.2ms | 4.9ms | 5.1ms | 298 ops/s |
| Compress (435 tokens) | 5.4ms | 6.1ms | 6.4ms | — |
| Chain workflow (3 ops) | 9.7ms | 11.3ms | 11.6ms | — |
| Matrix workflow (2x2, 8 ops) | 26.4ms | 31.2ms | 36.2ms | — |
| Verify workflow (4 ops) | 14.4ms | 15.8ms | 16.0ms | — |

### Compression Efficiency

| Context Size | Tokens | Latency | ms/token |
|--------------|--------|---------|----------|
| Small | 100 | 6.6ms | 0.066 |
| Medium | 1,000 | 5.8ms | 0.006 |
| Large | 5,000 | 9.2ms | 0.002 |

**Key metrics:** 48% token reduction, 100% key information preserved.

### Concurrency Scaling

| Concurrent Sessions | Total Time | Throughput | Errors |
|---------------------|------------|------------|--------|
| 1 | 4.2ms | 237 ops/s | 0 |
| 5 | 13.5ms | 371 ops/s | 0 |
| 10 | 30.5ms | 328 ops/s | 0 |
| 20 | 52.7ms | 380 ops/s | 0 |

### Strategy Comparison

| Strategy | Win Rate | Avg Coverage | Avg Time | Best For |
|----------|----------|--------------|----------|----------|
| **Matrix of Thought** | **83%** | 0.73 | 8ms | All problem types |
| Long Chain + MPPA | 14% | 0.72 | 5ms | Math, Logic |
| Baseline | 1% | 0.34 | 1ms | — |

| Problem Type | MoT Wins | Long Chain | Baseline |
|--------------|----------|------------|----------|
| Math (25) | **24** | 1 | 0 |
| Logic (25) | **16** | 7 | 0 |
| Multi-hop (25) | **24** | 0 | 1 |
| Analysis (25) | **19** | 6 | 0 |

## License

MIT — see [LICENSE](LICENSE)

---

**[Contributing](CONTRIBUTING.md)** · **[Issues](https://github.com/coderdayton/matrixmind-mcp/issues)** · Built with [FastMCP](https://gofastmcp.com)
