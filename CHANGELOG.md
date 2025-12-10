# Changelog

All notable changes to MatrixMind MCP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of MatrixMind MCP server
- **State Manager Tools** for structured reasoning workflows:
  - Long Chain of Thought (`chain_start`, `chain_add_step`, `chain_get`, `chain_finalize`)
  - Matrix of Thought (`matrix_start`, `matrix_set_cell`, `matrix_synthesize`, `matrix_get`, `matrix_finalize`)
  - Fact Verification (`verify_start`, `verify_add_claim`, `verify_claim`, `verify_get`, `verify_finalize`)
- **Context Compression** (`compress_prompt`) for efficient token usage
- **Strategy Recommendation** (`recommend_reasoning_strategy`) for selecting optimal reasoning approach
- **Server Status** (`check_status`) for monitoring active sessions
- Comprehensive test suite (332 tests)
- MCP protocol compliance with FastMCP

---

<!-- Releases will be auto-appended below this line -->
