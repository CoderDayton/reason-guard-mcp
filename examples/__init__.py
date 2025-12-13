"""MatrixMind MCP Examples Package.

This package contains example scripts demonstrating MatrixMind MCP usage:

- llm_client: Shared LLM client utilities for Ollama/OpenAI-compatible APIs
- benchmark: Performance benchmarking for MCP server operations
- reasoning_comparison: Compare reasoning strategies (baseline, chain, MoT, hybrid)
- basic_usage: Simple usage examples
- multi_hop_qa: Multi-hop question answering demo
- constraint_solving: Constraint satisfaction problem demo

Usage from project root:
    from examples.llm_client import LLMClient, init_llm_client
    from examples.reasoning_comparison import BENCHMARK_PROBLEMS
"""

from examples.llm_client import (
    LLMClient,
    add_llm_args,
    close_llm_client,
    get_default_llm_model,
    get_default_llm_url,
    get_llm_answer,
    get_llm_client,
    init_llm_client,
    is_llm_enabled,
)

__all__ = [
    "LLMClient",
    "add_llm_args",
    "close_llm_client",
    "get_default_llm_model",
    "get_default_llm_url",
    "get_llm_answer",
    "get_llm_client",
    "init_llm_client",
    "is_llm_enabled",
]
