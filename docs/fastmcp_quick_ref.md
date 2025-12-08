# FastMCP 2.0 Quick Reference & Best Practices

## Modern FastMCP Patterns for Enhanced CoT MCP

---

## **CORE FastMCP 2.0 CONCEPTS**

### Tool Definition Pattern (Pythonic & Simple)

```python
from fastmcp import FastMCP, Context

mcp = FastMCP("My Server")

@mcp.tool
def my_tool(param1: str, param2: int = 10, ctx: Context = None) -> str:
    """Tool documentation (shown to LLM).
    
    Tool docstrings are critical - they're sent to the LLM as:
    - Description of what tool does
    - Input parameter meanings
    - Expected output format
    - Use cases and examples
    """
    # Log to client
    await ctx.info(f"Processing {param1}")
    
    # Return string (MCP converts to proper format)
    return f"Result: {param1}"
```

**Key FastMCP advantages:**
- ✅ Type hints auto-generate schemas
- ✅ Docstrings become LLM-visible descriptions
- ✅ Context injection (logging, sampling, resource access)
- ✅ Async/sync both supported
- ✅ Automatic error handling

### Error Handling Pattern

```python
from fastmcp import FastMCP, Context
import json

mcp = FastMCP("My Server")

@mcp.tool
def safe_tool(input_data: str, ctx: Context = None) -> str:
    """Safe tool with proper error handling."""
    try:
        # Input validation
        if not input_data or len(input_data) > 10000:
            raise ValueError("Input must be 1-10000 characters")
        
        # Log progress
        await ctx.info(f"Processing {len(input_data)} chars")
        
        # Do work
        result = process(input_data)
        
        # Return success (FastMCP auto-formats)
        return json.dumps({"status": "success", "result": result})
    
    except ValueError as e:
        # Return error as JSON (LLM expects structured response)
        return json.dumps({"status": "error", "message": str(e)})
    
    except Exception as e:
        # Log unexpected errors
        await ctx.error(f"Unexpected error: {e}")
        return json.dumps({"status": "error", "message": "Unexpected server error"})
```

**Error handling best practices:**
- Always return JSON (even for errors)
- Use try-except for every external call
- Log errors with ctx.error()
- Return user-friendly messages
- Include error codes/types for debugging

---

## **OPTIMIZED TOOL PATTERNS FOR OUR SYSTEM**

### Pattern 1: Compression Tool (I/O Optimization)

```python
from fastmcp import FastMCP, Context
from typing import Dict, Any
import json

@mcp.tool
def compress_prompt(
    context: str,
    question: str,
    compression_ratio: float = 0.3,
    ctx: Context = None
) -> str:
    """Compress long context for faster reasoning.
    
    ❗ CRITICAL FOR LLM:
    - Returns JSON with compression_ratio, tokens_saved, sample_output
    - LLM uses this to decide: compress? how much? what question?
    
    Args:
        context: Long text (can be 5-20K+ tokens)
        question: Query to determine relevance
        compression_ratio: 0.1-1.0 (default 0.3 = 3× compression)
    
    Returns JSON:
        {
            "compressed_context": "...",
            "compression_ratio": 0.3,
            "tokens_saved": 2500,
            "sentences_kept": 8,
            "sentences_removed": 12,
            "top_relevant": ["Sentence 1...", "Sentence 2..."]
        }
    
    Examples:
        Long document + "What is X?" → Compressed doc focusing on X
        Wikipedia page + "Who did Y?" → Extract paragraphs about Y
    """
    try:
        # Validate inputs
        if not context or not question:
            return json.dumps({"error": "Context and question required"})
        
        if not 0.1 <= compression_ratio <= 1.0:
            return json.dumps({"error": "Ratio must be 0.1-1.0"})
        
        await ctx.info(f"Compressing {len(context)} chars by {compression_ratio:.0%}")
        
        # Call compression tool
        result = compression_tool.compress(
            context, question, compression_ratio
        )
        
        # Return metrics LLM cares about
        return json.dumps({
            "compressed_context": result.compressed_context,
            "compression_ratio": round(result.compression_ratio, 3),
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "tokens_saved": result.original_tokens - result.compressed_tokens,
            "sentences_kept": result.sentences_kept,
            "sentences_removed": result.sentences_removed,
            "top_relevant": [
                {"sentence": s[:60] + "...", "relevance": round(sc, 2)}
                for s, sc in result.relevance_scores[:3]
            ]
        })
    
    except Exception as e:
        await ctx.error(f"Compression failed: {e}")
        return json.dumps({"error": str(e), "status": "failed"})
```

### Pattern 2: Reasoning Tool (Multi-Step Logic)

```python
@mcp.tool
def matrix_of_thought_reasoning(
    question: str,
    context: str,
    matrix_rows: int = 3,
    matrix_cols: int = 4,
    ctx: Context = None
) -> str:
    """Complex reasoning with multiple perspectives.
    
    ❗ CRITICAL FOR LLM:
    - Returns JSON with: answer, confidence, reasoning_steps, trace
    - LLM uses confidence to decide: accept? verify? try different approach?
    
    Args:
        question: Problem to solve
        context: Background info
        matrix_rows: 2-5, breadth (num strategies)
        matrix_cols: 2-5, depth (num refinements)
    
    Returns JSON:
        {
            "answer": "...",
            "confidence": 0.82,
            "reasoning_steps": ["Step 1: ...", "Step 2: ...", ...],
            "matrix_shape": [3, 4],
            "total_thoughts_generated": 12,
            "num_refinements": 4
        }
    
    When to use:
        - Multi-hop reasoning (requires breadth)
        - Complex constraints (requires depth)
        - Uncertainty needed (matrix explores multiple paths)
    
    Examples:
        Q: "Who won and why?" → Needs multiple strategy views
        Q: "What are implications?" → Needs iterative refinement
    """
    try:
        # Validate
        if not question or not context:
            return json.dumps({"error": "Question and context required"})
        
        if not (2 <= matrix_rows <= 5) or not (2 <= matrix_cols <= 5):
            return json.dumps({"error": "Matrix size must be 2-5"})
        
        await ctx.info(f"MoT: {matrix_rows}×{matrix_cols} matrix reasoning")
        
        # Run reasoning
        result = mot_tool.reason(
            question, context, matrix_rows, matrix_cols
        )
        
        # Return with confidence for LLM decision-making
        return json.dumps({
            "answer": result.answer,
            "confidence": round(result.confidence, 3),
            "is_confident": result.confidence > 0.75,
            "reasoning_steps": result.reasoning_steps[:5],
            "tokens_used": result.tokens_used,
            "matrix_shape": [matrix_rows, matrix_cols],
            "num_thoughts": len(result.reasoning_steps)
        })
    
    except Exception as e:
        await ctx.error(f"MoT reasoning failed: {e}")
        return json.dumps({"error": str(e), "status": "failed"})
```

### Pattern 3: Verification Tool (Quality Assurance)

```python
@mcp.tool
def verify_fact_consistency(
    answer: str,
    context: str,
    max_claims: int = 10,
    ctx: Context = None
) -> str:
    """Verify answer against knowledge base.
    
    ❗ CRITICAL FOR LLM:
    - Returns verified: bool (LLM knows if answer is reliable)
    - Returns confidence: 0-1 (LLM knows how much to trust)
    
    Args:
        answer: Generated answer to verify
        context: Fact-checking context
        max_claims: Max claims to check (1-20)
    
    Returns JSON:
        {
            "verified": true,
            "confidence": 0.85,
            "claims_verified": 5,
            "claims_total": 6,
            "reason": "5 out of 6 claims supported by context",
            "recommendation": "answer is reliable"
        }
    
    When to use:
        - Quality assurance on generated answers
        - Preventing hallucinations
        - Multi-fact answers that need validation
    
    Examples:
        A: "Einstein published in 1905 and 1915"
        Context: "Einstein published Special Relativity in 1905..."
        → Verifies both claims
    """
    try:
        if not answer or not context:
            return json.dumps({"error": "Answer and context required"})
        
        if not 1 <= max_claims <= 20:
            return json.dumps({"error": "Max claims must be 1-20"})
        
        await ctx.info(f"Verifying answer ({len(answer)} chars)")
        
        # Run verification
        result = verify_tool.verify(answer, context, max_claims)
        
        # Return decision-relevant info
        return json.dumps({
            "verified": result["verified"],
            "confidence": round(result["confidence"], 3),
            "claims_verified": result["claims_verified"],
            "claims_total": result["claims_total"],
            "verification_percentage": round(
                result["claims_verified"] / result["claims_total"] * 100
                if result["claims_total"] > 0 else 0
            ),
            "reason": result["reason"],
            "recommendation": "RELIABLE" if result["verified"] else "REVIEW NEEDED"
        })
    
    except Exception as e:
        await ctx.error(f"Verification failed: {e}")
        return json.dumps({"error": str(e), "status": "failed"})
```

---

## **CRITICAL: LLM-FRIENDLY JSON RETURNS**

LLMs parse tool returns. Make sure they're:

```python
# ✅ GOOD - LLM understands structure
return json.dumps({
    "status": "success",
    "result": answer,
    "confidence": 0.85,
    "metrics": {
        "compression_ratio": 0.3,
        "tokens_saved": 500
    }
})

# ❌ BAD - LLM confused
return f"Answer is {answer}"
return answer  # No structure
return str({"result": answer})  # String, not JSON

# ✅ GOOD - Error handling
return json.dumps({"error": "Invalid input", "code": "INPUT_ERROR"})

# ❌ BAD - Raises exception in tool
raise Exception("Invalid input")  # MCP catches but less structured
```

---

## **FASTMCP 2.0 UNIQUE FEATURES**

### 1. Context Parameter (LLM-Safe Logging)

```python
@mcp.tool
def example_tool(param: str, ctx: Context = None) -> str:
    """Tool using context."""
    
    # Log to client (shows in Claude/ChatGPT interface)
    await ctx.info(f"Starting with {param}")
    
    # Do work
    result = compute(param)
    
    # Report progress
    await ctx.report_progress(50, "Processing complete")
    
    # Log warnings
    if warning_condition:
        await ctx.warning("Potential issue detected")
    
    # Error logging
    if error_condition:
        await ctx.error("Something failed")
    
    return result
```

### 2. Server Composition (Modular Design)

```python
# parent_server.py
from fastmcp import FastMCP

parent = FastMCP("Main Server")

# Mount child servers
parent.mount(compression_server)
parent.mount(reasoning_server)

# Tools now available with prefixes:
# - compression_compress_prompt
# - reasoning_matrix_of_thought
```

### 3. Authentication (Production-Ready)

```python
from fastmcp.server.auth.providers.google import GoogleProvider

auth = GoogleProvider(
    client_id="...",
    client_secret="...",
    base_url="https://myserver.com"
)

# Protected server
mcp = FastMCP("Protected Server", auth=auth)

# Clients use: Client(url, auth="oauth") for auto-setup
```

### 4. OpenAPI Integration (Auto API from Tools)

```python
# Auto-generate FastAPI app from tools
mcp_app = FastMCP.from_fastapi(existing_fastapi_app)

# Now has both:
# - MCP tools via /mcp endpoint
# - Original FastAPI routes
```

---

## **PERFORMANCE TIPS FOR FastMCP**

### 1. Async Everything

```python
# ✅ Async (non-blocking)
@mcp.tool
async def fast_tool(input: str) -> str:
    result = await expensive_async_operation(input)
    return result

# ❌ Sync (blocking)
@mcp.tool
def slow_tool(input: str) -> str:
    result = expensive_sync_operation(input)  # Blocks
    return result
```

### 2. Streaming for Long Operations

```python
@mcp.tool
async def streaming_tool(items: List[str], ctx: Context) -> str:
    results = []
    for i, item in enumerate(items):
        result = process(item)
        results.append(result)
        
        # Report progress to client every 10%
        if (i + 1) % max(1, len(items) // 10) == 0:
            await ctx.report_progress(
                (i + 1) / len(items) * 100,
                f"Processed {i+1}/{len(items)}"
            )
    
    return json.dumps(results)
```

### 3. Caching Repeated Operations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(key: str) -> str:
    # Only computed once per unique key
    return complex_computation(key)

@mcp.tool
def cached_tool(input: str) -> str:
    result = expensive_operation(input)
    return result
```

---

## **DEPLOYMENT: LOCAL VS CLOUD VS SELF-HOSTED**

### Local Development (Stdio)

```bash
# Simplest - good for testing
fastmcp run src/server.py

# Server outputs to stdout, clients connect via stdio
```

### Production HTTP (Self-Hosted)

```bash
# HTTP server on port 8000
python src/server.py

# curl http://localhost:8000/mcp/tools  (list tools)
# curl -X POST http://localhost:8000/mcp/call_tool ...
```

### FastMCP Cloud (Managed)

```bash
# Deploy automatically
fastmcp deploy src/server.py --public

# Get HTTPS endpoint
# https://your-server.fastmcp.cloud/mcp

# Auto-scaling, authentication, monitoring included
```

---

## **TESTING FastMCP TOOLS**

### In-Memory Testing (No Subprocess)

```python
import pytest
from fastmcp import Client

@pytest.mark.asyncio
async def test_tool_directly():
    # In-memory connection to FastMCP server
    async with Client(mcp) as client:  # mcp = FastMCP instance
        result = await client.call_tool(
            "my_tool",
            {"param": "test"}
        )
        
        assert result
```

### CLI Testing

```bash
# Test via CLI
fastmcp test src/server.py compress_prompt \
    --context "Long text" \
    --question "What is X?" \
    --compression-ratio 0.3
```

---

## **SUMMARY: FastMCP 2.0 vs Original MCP SDK**

| Feature | FastMCP 2.0 | SDK |
|---------|-----------|-----|
| **Learning curve** | Minimal (decorators) | Complex (protocol) |
| **Tool definition** | @mcp.tool | Manual schemas |
| **Error handling** | Built-in try-catch | Manual |
| **Auth** | Enterprise (Google, GitHub, Azure) | None |
| **Deployment** | Managed cloud option | DIY |
| **Testing** | In-memory, CLI | Process spawning |
| **Performance** | Optimized | Baseline |
| **Production ready** | Yes (used at scale) | Yes |

---

## **QUICK MIGRATION: SDK → FastMCP 2.0**

```python
# OLD (SDK)
from mcp.server import Server

server = Server("name")

@server.list_tools()
def list_tools():
    return [{"name": "tool1", ...}]

@server.call_tool()
def call_tool(name, args):
    if name == "tool1":
        return tool1(args)

# NEW (FastMCP 2.0)
from fastmcp import FastMCP

mcp = FastMCP("name")

@mcp.tool
def tool1(param: str) -> str:
    return f"Result: {param}"

# That's it! FastMCP handles everything
```

---

## **WHEN TO USE EACH PATTERN**

| Use Case | Tool Type | Example |
|----------|-----------|---------|
| Reduce token cost | Compression | Long documents → compress first |
| Complex reasoning | MoT | Multi-hop QA, constraints |
| Serial logic | Long chain | Step-by-step proofs, arithmetic |
| Quality check | Verification | Hallucination prevention |
| Strategy selection | Recommendation | Route to optimal method |

---

**End of FastMCP 2.0 Quick Reference**

Use FastMCP for production systems. It's the standard framework recommended by Anthropic for MCP development in 2024-2025.
