# Environment Variables Reference

Complete reference for all configurable environment variables in Reason Guard MCP.

## Table of Contents

- [Server Configuration](#server-configuration)
- [Embedding Model](#embedding-model)
- [Session Management](#session-management)
- [Rate Limiting](#rate-limiting)
- [Authentication & Security](#authentication--security)
- [Observability & Metrics](#observability--metrics)
- [Logging](#logging)
- [Database & Storage](#database--storage)
- [Input Limits](#input-limits)

---

## Server Configuration

### `SERVER_NAME`

| Property | Value |
|----------|-------|
| **Default** | `Reason-Guard-MCP` |
| **Type** | String |

Server name shown to MCP clients during capability negotiation.

**Use Case:** Identify server instance in multi-server setups or logs.

```bash
SERVER_NAME=MyReasonGuard-Prod
```

---

### `SERVER_TRANSPORT`

| Property | Value |
|----------|-------|
| **Default** | `stdio` |
| **Type** | `stdio` \| `http` \| `sse` |

Transport protocol for MCP communication.

| Mode | Use Case |
|------|----------|
| `stdio` | Local IDE integration (Claude Desktop, Cursor, VS Code) |
| `http` | REST API, Docker deployments, load balancing |
| `sse` | Server-Sent Events for streaming responses |

**Recommended:** Use `stdio` for desktop IDEs, `http` for production deployments.

```bash
# Local development
SERVER_TRANSPORT=stdio

# Production deployment
SERVER_TRANSPORT=http
```

---

### `SERVER_HOST`

| Property | Value |
|----------|-------|
| **Default** | `127.0.0.1` |
| **Type** | IP address or hostname |

Bind address for HTTP/SSE transport.

| Value | Use Case |
|-------|----------|
| `127.0.0.1` | Local only (secure default) |
| `0.0.0.0` | Accept connections from any interface (Docker, Kubernetes) |
| `10.0.0.5` | Bind to specific network interface |

**Security Warning:** Only use `0.0.0.0` with authentication enabled or behind a reverse proxy.

```bash
# Docker container
SERVER_HOST=0.0.0.0

# Specific interface
SERVER_HOST=192.168.1.100
```

---

### `SERVER_PORT`

| Property | Value |
|----------|-------|
| **Default** | `8000` |
| **Type** | Integer (1-65535) |

Port for HTTP/SSE transport. Ignored in `stdio` mode.

**Recommended:** Use ports > 1024 to avoid requiring root privileges.

```bash
SERVER_PORT=8080
```

---

## Embedding Model

### `EMBEDDING_MODEL`

| Property | Value |
|----------|-------|
| **Default** | `Snowflake/snowflake-arctic-embed-xs` |
| **Type** | HuggingFace model identifier |

Sentence embedding model for semantic compression and RAG features.

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| `Snowflake/snowflake-arctic-embed-xs` | ~90MB | Good | Fast | Default, resource-constrained |
| `Snowflake/snowflake-arctic-embed-m` | ~440MB | Better | Medium | Balanced |
| `sentence-transformers/all-MiniLM-L6-v2` | ~80MB | Good | Fast | Classic, well-tested |
| `sentence-transformers/all-mpnet-base-v2` | ~420MB | High | Slower | Quality-focused |
| `BAAI/bge-small-en-v1.5` | ~130MB | Good | Fast | Good balance |
| `BAAI/bge-m3` | ~2.2GB | Excellent | Slow | Maximum quality |

**Recommended:**
- Development/CI: `Snowflake/snowflake-arctic-embed-xs`
- Production: `sentence-transformers/all-mpnet-base-v2` or `BAAI/bge-small-en-v1.5`

```bash
# Lightweight (CI/testing)
EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-xs

# Production quality
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

---

### `EMBEDDING_MODEL_REVISION`

| Property | Value |
|----------|-------|
| **Default** | `main` |
| **Type** | Git revision (branch, tag, or commit hash) |

HuggingFace model revision for supply chain security. Pin to a specific commit hash for reproducible builds.

**Security Note:** Using `main` accepts the latest version, which could change. For production, pin to a commit hash.

```bash
# Accept latest (development)
EMBEDDING_MODEL_REVISION=main

# Pinned for security (production)
EMBEDDING_MODEL_REVISION=a1b2c3d4e5f6...
```

---

### `EMBEDDING_CACHE_DIR`

| Property | Value |
|----------|-------|
| **Default** | `~/.cache/reason-guard-mcp/models/` |
| **Type** | Directory path |

Directory for caching downloaded embedding models. Models are downloaded once and reused.

**Use Case:** Share model cache across containers, use faster storage, or control disk usage.

```bash
# Shared volume in Docker
EMBEDDING_CACHE_DIR=/models/cache

# Fast SSD storage
EMBEDDING_CACHE_DIR=/nvme/model-cache
```

---

### `ENABLE_RAG`

| Property | Value |
|----------|-------|
| **Default** | `false` |
| **Type** | Boolean (`true`/`false`) |

Enable Retrieval-Augmented Generation features. When enabled, the server maintains a vector store for semantic search over reasoning history.

**Use Case:** Enable for long-running sessions where past reasoning should inform current steps.

```bash
ENABLE_RAG=true
```

---

## Session Management

### `SESSION_MAX_AGE_MINUTES`

| Property | Value |
|----------|-------|
| **Default** | `60` |
| **Type** | Integer (minutes) |

Maximum session lifetime. Sessions older than this are automatically cleaned up.

**Use Case:** Prevent memory leaks from abandoned sessions.

```bash
# Short sessions (interactive use)
SESSION_MAX_AGE_MINUTES=30

# Long sessions (batch processing)
SESSION_MAX_AGE_MINUTES=240
```

---

### `CLEANUP_INTERVAL_SECONDS`

| Property | Value |
|----------|-------|
| **Default** | `300` (5 minutes) |
| **Type** | Integer (seconds) |

Interval between session cleanup runs. Lower values free memory faster but use more CPU.

```bash
# Aggressive cleanup (memory-constrained)
CLEANUP_INTERVAL_SECONDS=60

# Relaxed cleanup (plenty of memory)
CLEANUP_INTERVAL_SECONDS=600
```

---

### `MAX_TOTAL_SESSIONS`

| Property | Value |
|----------|-------|
| **Default** | `500` |
| **Type** | Integer |

Maximum concurrent sessions across all users. New session requests are rejected when limit is reached.

**Recommended:** Set based on available memory. Each session uses ~1-5KB base, plus thought storage.

```bash
# Development
MAX_TOTAL_SESSIONS=100

# Production (8GB+ RAM)
MAX_TOTAL_SESSIONS=1000
```

---

### `MAX_THOUGHTS_PER_SESSION`

| Property | Value |
|----------|-------|
| **Default** | `1000` |
| **Type** | Integer |

Maximum reasoning steps per session. Prevents runaway sessions from consuming excessive memory.

```bash
# Simple problems
MAX_THOUGHTS_PER_SESSION=100

# Complex multi-step reasoning
MAX_THOUGHTS_PER_SESSION=2000
```

---

### `MAX_ALTERNATIVES`

| Property | Value |
|----------|-------|
| **Default** | `10` |
| **Type** | Integer |

Maximum alternative branches in enforcement mode's `create_branch` tool.

```bash
MAX_ALTERNATIVES=5
```

---

## Rate Limiting

### `RATE_LIMIT_MAX_SESSIONS`

| Property | Value |
|----------|-------|
| **Default** | `100` |
| **Type** | Integer |

Maximum new sessions allowed per time window (see `RATE_LIMIT_WINDOW_SECONDS`).

**Use Case:** Prevent session creation spam.

```bash
# Strict limit
RATE_LIMIT_MAX_SESSIONS=20

# Generous limit
RATE_LIMIT_MAX_SESSIONS=500
```

---

### `RATE_LIMIT_WINDOW_SECONDS`

| Property | Value |
|----------|-------|
| **Default** | `60` |
| **Type** | Integer (seconds) |

Time window for session rate limiting.

```bash
# 100 sessions per minute (default)
RATE_LIMIT_WINDOW_SECONDS=60

# 50 sessions per 30 seconds
RATE_LIMIT_MAX_SESSIONS=50
RATE_LIMIT_WINDOW_SECONDS=30
```

---

### `IP_RATE_LIMIT_ENABLED`

| Property | Value |
|----------|-------|
| **Default** | `false` |
| **Type** | Boolean |

Enable per-IP rate limiting. Requires HTTP/SSE transport.

```bash
IP_RATE_LIMIT_ENABLED=true
```

---

### `IP_RATE_LIMIT_MAX_REQUESTS`

| Property | Value |
|----------|-------|
| **Default** | `100` |
| **Type** | Integer |

Maximum requests per IP per time window.

```bash
IP_RATE_LIMIT_MAX_REQUESTS=50
```

---

### `IP_RATE_LIMIT_WINDOW_SECONDS`

| Property | Value |
|----------|-------|
| **Default** | `60` |
| **Type** | Integer (seconds) |

Time window for IP rate limiting.

```bash
IP_RATE_LIMIT_WINDOW_SECONDS=60
```

---

## Authentication & Security

### `AUTH_ENABLED`

| Property | Value |
|----------|-------|
| **Default** | `false` |
| **Type** | Boolean |

Enable API key authentication. When enabled, requests must include a valid API key.

**Security:** Always enable in production with HTTP/SSE transport.

```bash
AUTH_ENABLED=true
```

---

### `AUTH_BYPASS_LOCALHOST`

| Property | Value |
|----------|-------|
| **Default** | `true` |
| **Type** | Boolean |

Allow unauthenticated requests from localhost (127.0.0.1, ::1). Useful for local development while auth is enabled.

**Security Warning:** Disable in production or container environments.

```bash
# Development
AUTH_BYPASS_LOCALHOST=true

# Production
AUTH_BYPASS_LOCALHOST=false
```

---

### `REASONGUARD_API_KEYS`

| Property | Value |
|----------|-------|
| **Default** | (none) |
| **Type** | Comma-separated strings |

API keys for authentication. Use strong, randomly generated keys.

```bash
# Single key
REASONGUARD_API_KEYS=sk-your-secret-key-here

# Multiple keys (for rotation)
REASONGUARD_API_KEYS=sk-key1-xxx,sk-key2-yyy
```

---

### `REASONGUARD_API_KEYS_FILE`

| Property | Value |
|----------|-------|
| **Default** | (none) |
| **Type** | File path |

Path to file containing API keys (one per line). Preferred over environment variable for sensitive data.

```bash
REASONGUARD_API_KEYS_FILE=/run/secrets/api_keys
```

---

### `SESSION_SIGNING_KEY`

| Property | Value |
|----------|-------|
| **Default** | (auto-generated) |
| **Type** | Base64-encoded 32-byte key |

Key for signing session tokens. Auto-generated if not set, but sessions won't survive restarts.

**Production:** Set explicitly for session persistence across restarts.

```bash
# Generate: python -c "import secrets; print(secrets.token_urlsafe(32))"
SESSION_SIGNING_KEY=your-32-byte-base64-key
```

---

### `TRUSTED_PROXIES`

| Property | Value |
|----------|-------|
| **Default** | (none) |
| **Type** | Comma-separated CIDR ranges |

IP ranges to trust for X-Forwarded-For headers. Required when behind a load balancer.

```bash
# Single proxy
TRUSTED_PROXIES=10.0.0.1/32

# AWS ALB ranges
TRUSTED_PROXIES=10.0.0.0/8,172.16.0.0/12
```

---

### `REASONGUARD_ALLOWED_DB_DIRS`

| Property | Value |
|----------|-------|
| **Default** | (current directory) |
| **Type** | Comma-separated directory paths |

Directories where SQLite databases can be created. Security measure to prevent arbitrary file writes.

```bash
REASONGUARD_ALLOWED_DB_DIRS=/data/reason-guard,/tmp
```

---

## Observability & Metrics

### `REASONGUARD_METRICS_ENABLED`

| Property | Value |
|----------|-------|
| **Default** | `false` |
| **Type** | Boolean |

Enable SQLite-based metrics collection for observability.

```bash
REASONGUARD_METRICS_ENABLED=true
```

---

### `REASONGUARD_METRICS_DB`

| Property | Value |
|----------|-------|
| **Default** | `:memory:` |
| **Type** | File path or `:memory:` |

Path to metrics SQLite database. Use `:memory:` for ephemeral metrics or a file path for persistence.

```bash
# Ephemeral (development)
REASONGUARD_METRICS_DB=:memory:

# Persistent (production)
REASONGUARD_METRICS_DB=/data/metrics/reason-guard-metrics.db
```

---

### `REASONGUARD_METRICS_RETENTION_HOURS`

| Property | Value |
|----------|-------|
| **Default** | `24` |
| **Type** | Integer (hours) |

How long to retain metrics data before automatic cleanup.

```bash
# Short retention (high volume)
REASONGUARD_METRICS_RETENTION_HOURS=6

# Long retention (analysis)
REASONGUARD_METRICS_RETENTION_HOURS=168  # 7 days
```

---

### `REASONGUARD_METRICS_ROTATION_MB`

| Property | Value |
|----------|-------|
| **Default** | `100` |
| **Type** | Integer (megabytes) |

Database size threshold that triggers rotation. When exceeded, current DB is archived and a new one starts.

```bash
REASONGUARD_METRICS_ROTATION_MB=50
```

---

### `REASONGUARD_METRICS_ROTATION_KEEP`

| Property | Value |
|----------|-------|
| **Default** | `3` |
| **Type** | Integer |

Number of rotated database archives to keep.

```bash
REASONGUARD_METRICS_ROTATION_KEEP=5
```

---

### `METRICS_HOST`

| Property | Value |
|----------|-------|
| **Default** | `0.0.0.0` |
| **Type** | IP address |

Bind address for Prometheus metrics endpoint (if enabled separately).

```bash
METRICS_HOST=127.0.0.1
```

---

### `METRICS_PORT`

| Property | Value |
|----------|-------|
| **Default** | `9090` |
| **Type** | Integer |

Port for Prometheus metrics endpoint.

```bash
METRICS_PORT=9091
```

---

## Logging

### `LOG_LEVEL`

| Property | Value |
|----------|-------|
| **Default** | `INFO` |
| **Type** | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

Minimum log level to output.

| Level | Use Case |
|-------|----------|
| `DEBUG` | Development, troubleshooting |
| `INFO` | Normal operation |
| `WARNING` | Production (less noise) |
| `ERROR` | Minimal logging |

```bash
# Development
LOG_LEVEL=DEBUG

# Production
LOG_LEVEL=WARNING
```

---

### `LOG_FORMAT`

| Property | Value |
|----------|-------|
| **Default** | `text` |
| **Type** | `text` \| `json` |

Log output format.

| Format | Use Case |
|--------|----------|
| `text` | Human-readable, development |
| `json` | Machine-parseable, log aggregators (Datadog, Splunk, ELK) |

```bash
# Development
LOG_FORMAT=text

# Production with log aggregation
LOG_FORMAT=json
```

---

### `LOG_FILE`

| Property | Value |
|----------|-------|
| **Default** | (none, stdout only) |
| **Type** | File path |

Path to log file. Logs are written to both stdout and this file.

```bash
LOG_FILE=/var/log/reason-guard/server.log
```

---

## Database & Storage

### `ATOMIC_ROUTER_DB`

| Property | Value |
|----------|-------|
| **Default** | `:memory:` |
| **Type** | File path or `:memory:` |

SQLite database for enforcement mode session persistence.

```bash
# Ephemeral (default)
ATOMIC_ROUTER_DB=:memory:

# Persistent
ATOMIC_ROUTER_DB=/data/atomic-router.db
```

---

### `VECTOR_DB_PATH`

| Property | Value |
|----------|-------|
| **Default** | `:memory:` |
| **Type** | File path or `:memory:` |

Path for vector store database (when RAG is enabled).

```bash
VECTOR_DB_PATH=/data/vectors.db
```

---

## Input Limits

### `MAX_PROBLEM_SIZE`

| Property | Value |
|----------|-------|
| **Default** | `50000` |
| **Type** | Integer (characters) |

Maximum length of problem text in `think(action="start")` or `initialize_reasoning()`.

```bash
MAX_PROBLEM_SIZE=100000
```

---

### `MAX_THOUGHT_SIZE`

| Property | Value |
|----------|-------|
| **Default** | `10000` |
| **Type** | Integer (characters) |

Maximum length of individual thought/step content.

```bash
MAX_THOUGHT_SIZE=20000
```

---

### `MAX_CONTEXT_SIZE`

| Property | Value |
|----------|-------|
| **Default** | `100000` |
| **Type** | Integer (characters) |

Maximum context size for compression operations.

```bash
MAX_CONTEXT_SIZE=200000
```

---

## Example Configurations

### Development (Local)

```bash
# .env
SERVER_TRANSPORT=stdio
LOG_LEVEL=DEBUG
LOG_FORMAT=text
EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-xs
MAX_TOTAL_SESSIONS=50
```

### Production (Docker/Kubernetes)

```bash
# .env
SERVER_TRANSPORT=http
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=WARNING
LOG_FORMAT=json

# Security
AUTH_ENABLED=true
AUTH_BYPASS_LOCALHOST=false
REASONGUARD_API_KEYS_FILE=/run/secrets/api_keys
SESSION_SIGNING_KEY=${SESSION_SIGNING_KEY}
TRUSTED_PROXIES=10.0.0.0/8

# Performance
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_CACHE_DIR=/models/cache
MAX_TOTAL_SESSIONS=1000
MAX_THOUGHTS_PER_SESSION=500

# Observability
REASONGUARD_METRICS_ENABLED=true
REASONGUARD_METRICS_DB=/data/metrics.db
REASONGUARD_METRICS_RETENTION_HOURS=168
```

### CI/Testing

```bash
# .env
SERVER_TRANSPORT=stdio
LOG_LEVEL=WARNING
EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-xs
MAX_TOTAL_SESSIONS=10
SESSION_MAX_AGE_MINUTES=5
REASONGUARD_METRICS_ENABLED=false
```

---

## See Also

- [README.md](README.md) - Quick start guide
- [SECURITY.md](SECURITY.md) - Security best practices
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup
