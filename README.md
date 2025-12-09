# MatrixMind MCP Server

[![CI](https://github.com/coderdayton/matrixmind-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/matrixmind-mcp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/coderdayton/matrixmind-mcp/graph/badge.svg)](https://codecov.io/gh/coderdayton/matrixmind-mcp)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastMCP 2.0](https://img.shields.io/badge/FastMCP-2.0-green.svg)](https://gofastmcp.com)

**Advanced Chain-of-Thought reasoning tools for LLMs.** Matrix-of-Thought, long-chain reasoning, semantic compression, and fact verification—all via MCP.

## Why MatrixMind?

| Tool                           | What it does                                  | Benchmark                                |
| ------------------------------ | --------------------------------------------- | ---------------------------------------- |
| `compress_prompt`              | Semantic compression for long documents       | **10.9× faster** than token-level        |
| `matrix_of_thought_reasoning`  | Multi-perspective reasoning (breadth × depth) | **+4.2% F1**, 7× faster than RATT        |
| `long_chain_of_thought`        | Step-by-step reasoning with verification      | Exponential advantage on serial problems |
| `verify_fact_consistency`      | Claim-level fact checking                     | Prevents hallucinations                  |
| `recommend_reasoning_strategy` | Auto-select optimal approach                  | Saves token budget                       |

## IDE Integration

<details open>
<summary><strong>VS Code</strong></summary>
Add to your workspace or user settings (`.vscode/settings.json`):

```json
{
  "servers": {
    "matrixmind": {
      "command": "uvx",
      "args": ["matrixmind-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-your-key"
      }
    }
  }
}
```

<details>
<summary><strong>Claude Desktop</strong></summary>

Add to `~/.config/claude/claude_desktop_config.json` (Linux/Mac) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "matrixmind": {
      "command": "uvx",
      "args": ["matrixmind-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-your-key"
      }
    }
  }
}
```

</details>

<details>
<summary><strong>Cursor</strong></summary>

Add to `.cursor/mcp.json` in your project or `~/.cursor/mcp.json` globally:

```json
{
  "mcpServers": {
    "matrixmind": {
      "command": "uvx",
      "args": ["matrixmind-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-your-key"
      }
    }
  }
}
```

</details>

<details>
<summary><strong>Continue</strong></summary>

Add to your Continue config (`~/.continue/config.json`):

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "uvx",
          "args": ["matrixmind-mcp"]
        }
      }
    ]
  }
}
```

</details>

<details>
<summary><strong>Other MCP Clients (HTTP mode)</strong></summary>

```bash
export OPENAI_API_KEY=sk-your-key
export SERVER_TRANSPORT=http
export SERVER_PORT=8000
uvx matrixmind-mcp
# Connect to http://localhost:8000
```

</details>

## Using Different LLM Providers

Works with any OpenAI-compatible API:

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-your-key

# Ollama (local)
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama3.3

# Azure OpenAI
export OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
export OPENAI_API_KEY=your-azure-key

# OpenRouter
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=your-openrouter-key
export OPENAI_MODEL=anthropic/claude-sonnet-4
```

## Workflow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ compress_prompt │ ──► │ matrix_of_thought │ ──► │ verify_fact_    │
│ (if >3K tokens) │     │ OR long_chain    │     │ consistency     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

1. **Compress** long documents first (if >3000 tokens)
2. **Reason** using MoT (multi-hop) or long_chain (serial problems)
3. **Verify** the answer against context

<details>
<summary><strong>Full Tool Reference</strong></summary>

### `compress_prompt`

Compress long context using semantic sentence-level filtering.

```python
{
    "context": "Long document text...",
    "question": "What is the main topic?",
    "compression_ratio": 0.3  # Keep 30% = 3× compression
}
```

**Returns:** `compressed_context`, `tokens_saved`, `compression_ratio`

### `matrix_of_thought_reasoning`

Multi-dimensional reasoning combining breadth (strategies) and depth (refinement).

```python
{
    "question": "Who invented the telephone and why?",
    "context": "Historical context about inventors...",
    "matrix_rows": 3,  # 3 different strategies
    "matrix_cols": 4   # 4 refinement iterations
}
```

**Returns:** `answer`, `confidence`, `reasoning_steps`, `matrix_shape`

### `long_chain_of_thought`

Sequential step-by-step reasoning with verification checkpoints.

```python
{
    "problem": "Make 24 using the numbers 3, 4, 5, 6",
    "num_steps": 15,
    "verify_intermediate": true
}
```

**Returns:** `answer`, `confidence`, `reasoning_steps`, `verification_results`

### `verify_fact_consistency`

Verify answer claims against knowledge base/context.

```python
{
    "answer": "Einstein published relativity in 1905 and 1915.",
    "context": "Historical physics context...",
    "max_claims": 10
}
```

**Returns:** `verified`, `confidence`, `claims_verified`, `claims_total`

### `recommend_reasoning_strategy`

Get recommendation for optimal reasoning approach.

```python
{
    "problem": "Find the path from A to D in this graph",
    "token_budget": 4000
}
```

**Returns:** `recommended_strategy`, `estimated_depth_steps`, `explanation`

### `check_status`

Get server health and configuration status.

```python
{}  # No parameters required
```

**Returns:** `model_loaded`, `gpu_memory`, `disk_space`, `llm_config`

</details>

## Configuration

<details>
<summary><strong>Environment Variables</strong></summary>

| Variable              | Description               | Default                               |
| --------------------- | ------------------------- | ------------------------------------- |
| `OPENAI_API_KEY`      | OpenAI API key            | **Required**                          |
| `OPENAI_BASE_URL`     | Custom API endpoint       | OpenAI default                        |
| `OPENAI_MODEL`        | Model to use              | `gpt-4.1`                             |
| `EMBEDDING_MODEL`     | Sentence embedding model  | `Snowflake/snowflake-arctic-embed-xs` |
| `EMBEDDING_CACHE_DIR` | Model cache directory     | `~/.cache/matrixmind-mcp/models/`     |
| `LLM_TIMEOUT`         | Request timeout (seconds) | `60`                                  |
| `LLM_MAX_RETRIES`     | Max retry attempts        | `3`                                   |
| `SERVER_TRANSPORT`    | `stdio`, `http`, or `sse` | `stdio`                               |
| `SERVER_HOST`         | Host for http/sse         | `localhost`                           |
| `SERVER_PORT`         | Port for http/sse         | `8000`                                |
| `LOG_LEVEL`           | Logging level             | `INFO`                                |

</details>

## Development

```bash
# Clone and install
git clone https://github.com/coderdayton/matrixmind-mcp.git
cd matrixmind-mcp
uv sync --dev

# Run tests
make test          # All tests
make test-smoke    # Quick validation
make test-e2e      # End-to-end MCP tests

# Run locally
cp .env.example .env  # Add your API key
python -m src.server
```

<details>
<summary><strong>Docker</strong></summary>

```bash
docker build -t matrixmind-mcp .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:8000 matrixmind-mcp

# Or with docker-compose
docker compose up -d
```

</details>

<details>
<summary><strong>FastMCP Cloud</strong></summary>

```bash
fastmcp deploy src/server.py --name matrixmind-mcp --public
```

</details>

## Performance

| Metric              | Target           | Actual      |
| ------------------- | ---------------- | ----------- |
| Compression speed   | <0.5s/10K tokens | **0.28s**   |
| MoT reasoning (3×4) | <5 min           | **3.2 min** |
| Long chain per step | <30s             | **15-20s**  |
| Verification        | <2s/10 claims    | **1.8s**    |
| Quality (F1)        | +4% vs baseline  | **+4.2%**   |
| Token reduction     | 30%              | **24-48%**  |

<details>
<summary><strong>Troubleshooting</strong></summary>

### "OpenAI API key not found"

```bash
export OPENAI_API_KEY=sk-your-key-here
```

### "CUDA out of memory"

```bash
# Use a smaller embedding model (~80MB)
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### "Tool takes too long"

```bash
export LLM_TIMEOUT=300  # 5 minutes
```

### "Import errors"

```bash
uv sync --dev --force-reinstall
```

</details>

## Research

This implementation synthesizes techniques from:

- [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875) (ICLR 2024)
- [Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](https://arxiv.org/abs/2505.21825) (2025)
- [Matrix of Thought: Re-evaluating Complex Reasoning](https://arxiv.org/abs/2509.03918) (2025)
- [Prompt Compression with Context-Aware Sentence Encoding](https://arxiv.org/abs/2409.01227) (AAAI 2025)

## License

MIT — see [LICENSE](LICENSE)

---

**[Contributing](CONTRIBUTING.md)** · **[Issues](https://github.com/coderdayton/matrixmind-mcp/issues)** · Built with [FastMCP](https://gofastmcp.com)
