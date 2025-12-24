# Changelog

All notable changes to Reason Guard MCP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Project renamed**: MatrixMind MCP → Reason Guard MCP
  - Package: `reason-guard-mcp`
  - CLI command: `reason-guard` (was `matrixmind-mcp`)
  - Metrics CLI: `reason-guard-metrics` (was `matrixmind-metrics`)
  - Env vars: `REASONGUARD_*` prefix (was `MATRIXMIND_*`)

### Added
- **Dual-paradigm architecture**: Guidance mode (suggestions) vs Enforcement mode (rejects invalid steps)
- **9 MCP tools**:
  - `paradigm_hint` - Analyze problem, recommend paradigm
  - `think` - Unified guidance mode reasoning (start, continue, branch, verify, finish)
  - `compress` - Semantic context compression
  - `status` - Server/session health
  - `initialize_reasoning` - Start enforcement mode session
  - `submit_step` - Submit step with validation (premise → hypothesis → verification → synthesis)
  - `create_branch` - Branch when confidence is low
  - `verify_claims` - Verify claims before synthesis
  - `router_status` - Enforcement session state
- **SQLite observability** (zero dependencies):
  - Session tracking with duration, steps, outcomes
  - Tool call metrics (latency, errors)
  - Automatic cleanup with configurable retention
  - Database rotation when size exceeds threshold
  - Thread-safe with WAL mode
- **Correlation ID logging**: `session_id` context propagation in logs
- **Enforcement rules**:
  - Rule A: Cannot synthesize until min_steps reached
  - Rule B: Low confidence requires branching
  - Rule C: Must verify before synthesis
  - Rule D: State machine enforcement (premise → hypothesis → verification → synthesis)
  - Rule E: Must synthesize at max_steps
- Comprehensive test suite (698 tests)
- MCP protocol compliance with FastMCP 2.0

### Configuration
- `REASONGUARD_METRICS_ENABLED` - Enable SQLite metrics (default: false)
- `REASONGUARD_METRICS_DB` - Metrics database path (default: :memory:)
- `REASONGUARD_METRICS_RETENTION_HOURS` - Data retention (default: 24)
- `REASONGUARD_METRICS_ROTATION_MB` - Rotate DB when exceeds size (default: 100)
- `REASONGUARD_METRICS_ROTATION_KEEP` - Archives to keep (default: 3)

---

<!-- Releases will be auto-appended below this line -->
