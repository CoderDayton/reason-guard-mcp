"""Reason Guard MCP Server.

FastMCP 2.0 implementation providing reasoning state management tools.
The calling LLM does all reasoning; these tools track and organize the process.

Tools:
1. think - Unified reasoning tool (auto-selects chain/matrix/hybrid mode)
2. compress - Semantic context compression
3. status - Server/session status

Run with: uvx reason-guard
Or: python -m src.server
"""

# Note: We intentionally do NOT use `from __future__ import annotations` here
# because it causes issues with Pydantic/FastMCP type resolution at decorator time.
# Python 3.11+ supports PEP 604 union syntax (X | Y) natively.

import asyncio
import hashlib
import os
import secrets
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial, wraps
from typing import Any, Literal, TypeVar

import orjson

# Note: uvloop is installed automatically when importing src.utils.session
# (via AsyncSessionManager -> unified_reasoner imports)
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import get_http_request
from loguru import logger

from src.models.model_manager import ModelManager
from src.tools.compress import ContextAwareCompressionTool
from src.tools.unified_reasoner import (
    ReasoningMode,
    ResponseVerbosity,
    SessionStatus,
    ThoughtType,
    UnifiedReasonerManager,
    get_unified_manager,
    init_unified_manager,
)
from src.utils.errors import ModelNotReadyException, ReasonGuardException, ToolExecutionError
from src.utils.observability import get_metrics_store
from src.utils.session import SessionNotFoundError

# Load environment variables from .env file (for local development)
load_dotenv()


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable, treating empty string as unset.

    Also checks Docker secrets path for sensitive values.
    """
    # Check Docker secrets first for sensitive keys
    secrets_path = f"/run/secrets/{key.lower()}"
    if os.path.isfile(secrets_path):
        try:
            with open(secrets_path) as f:
                value = f.read().strip()
                if value:
                    return value
        except Exception:  # nosec B110
            pass  # Fall through to env var

    value = os.getenv(key, default)
    return value if value else default


def _get_env_int(key: str, default: int) -> int:
    """Get environment variable as integer."""
    value = os.getenv(key)
    if value:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer for {key}: {value}, using default {default}")
    return default


def _json(data: dict[str, Any] | None, *, indent: bool = True) -> str:
    """Serialize data to JSON string with proper typing.

    Type-safe wrapper around orjson.dumps that returns str.
    """
    if data is None:
        data = {}
    opts = orjson.OPT_INDENT_2 if indent else 0
    result: bytes = orjson.dumps(data, option=opts, default=str)
    return result.decode("utf-8")


# =============================================================================
# Configuration from Environment Variables
# =============================================================================

# =============================================================================
# Authentication Configuration (V-001: CWE-306 Mitigation)
# =============================================================================

# API Key Authentication
# Keys can be provided via:
#   1. REASONGUARD_API_KEYS env var (comma-separated list of keys)
#   2. REASONGUARD_API_KEYS_FILE env var (path to file with one key per line)
#   3. Docker secrets at /run/secrets/reasonguard_api_keys
# SECURITY: Default to true for network transports, false only for stdio (local dev)
# This prevents unauthenticated access when server is exposed to network (CWE-306)
_auth_default = "false" if _get_env("SERVER_TRANSPORT", "stdio") == "stdio" else "true"
AUTH_ENABLED = _get_env("AUTH_ENABLED", _auth_default).lower() == "true"
AUTH_BYPASS_LOCALHOST = _get_env("AUTH_BYPASS_LOCALHOST", "true").lower() == "true"

# Trusted proxies for X-Forwarded-For header validation (V-002: CWE-348 Mitigation)
# Only trust X-Forwarded-For when the direct client is in this list
# Comma-separated list of IP addresses or CIDR ranges (e.g., "10.0.0.0/8,172.16.0.0/12")
_TRUSTED_PROXIES_RAW = _get_env("TRUSTED_PROXIES", "")


def _parse_trusted_proxies(raw: str) -> frozenset[str]:
    """Parse and validate TRUSTED_PROXIES, warning on invalid entries."""
    import ipaddress

    valid_proxies: list[str] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            if "/" in entry:
                # Validate CIDR notation
                ipaddress.ip_network(entry, strict=False)
            else:
                # Validate IP address
                ipaddress.ip_address(entry)
            valid_proxies.append(entry)
        except ValueError as e:
            # Log at module load time - admin needs to know config is broken
            import logging

            logging.getLogger(__name__).warning(
                f"Invalid TRUSTED_PROXIES entry '{entry}': {e}. Entry will be ignored."
            )
    return frozenset(valid_proxies)


TRUSTED_PROXIES: frozenset[str] = _parse_trusted_proxies(_TRUSTED_PROXIES_RAW)


def _load_api_keys() -> set[str]:
    """Load API keys from environment or secrets file.

    Priority order:
    1. Docker secrets file (/run/secrets/reasonguard_api_keys)
    2. REASONGUARD_API_KEYS_FILE environment variable
    3. REASONGUARD_API_KEYS environment variable (comma-separated)

    Returns:
        Set of valid API key hashes (SHA-256).

    """
    keys: set[str] = set()

    # Try Docker secrets first (most secure in containerized environments)
    secrets_path = "/run/secrets/reasonguard_api_keys"
    if os.path.isfile(secrets_path):
        try:
            with open(secrets_path) as f:
                for line in f:
                    key = line.strip()
                    if key and not key.startswith("#"):
                        # Store hash of key, not the key itself
                        keys.add(hashlib.sha256(key.encode()).hexdigest())
            logger.info(f"Loaded {len(keys)} API key(s) from Docker secrets")
            return keys
        except Exception as e:
            logger.warning(f"Failed to load Docker secrets: {e}")

    # Try file path from environment
    keys_file = _get_env("REASONGUARD_API_KEYS_FILE")
    if keys_file and os.path.isfile(keys_file):
        try:
            with open(keys_file) as f:
                for line in f:
                    key = line.strip()
                    if key and not key.startswith("#"):
                        keys.add(hashlib.sha256(key.encode()).hexdigest())
            logger.info(f"Loaded {len(keys)} API key(s) from {keys_file}")
            return keys
        except Exception as e:
            logger.warning(f"Failed to load keys from {keys_file}: {e}")

    # Fall back to environment variable (least secure, for development)
    keys_env = _get_env("REASONGUARD_API_KEYS")
    if keys_env:
        for key in keys_env.split(","):
            key = key.strip()
            if key:
                keys.add(hashlib.sha256(key.encode()).hexdigest())
        logger.info(f"Loaded {len(keys)} API key(s) from environment variable")
        return keys

    if AUTH_ENABLED:
        logger.warning(
            "AUTH_ENABLED=true but no API keys configured. "
            "Set REASONGUARD_API_KEYS, REASONGUARD_API_KEYS_FILE, or use Docker secrets."
        )

    return keys


# Load API keys at startup
_API_KEY_HASHES: set[str] = set()


def _init_api_keys() -> None:
    """Initialize API keys (called at server startup)."""
    global _API_KEY_HASHES
    _API_KEY_HASHES = _load_api_keys()


def validate_api_key(api_key: str | None) -> tuple[bool, str]:
    """Validate an API key.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        api_key: The API key to validate.

    Returns:
        Tuple of (is_valid, error_message).

    """
    if not AUTH_ENABLED:
        return True, ""

    if not api_key:
        return False, "API key required. Set Authorization header with Bearer token."

    # Strip "Bearer " prefix if present
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]

    # Hash the provided key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Constant-time comparison against all valid key hashes
    # We iterate all hashes to prevent timing attacks from revealing key count
    is_valid = False
    for valid_hash in _API_KEY_HASHES:
        if secrets.compare_digest(key_hash, valid_hash):
            is_valid = True
            # Don't break - continue checking all hashes for constant time

    if not is_valid:
        return False, "Invalid API key"

    return True, ""


def generate_api_key() -> str:
    """Generate a new secure API key.

    Returns:
        A URL-safe 32-byte random key.

    """
    return secrets.token_urlsafe(32)


# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def require_auth(func: F) -> F:
    """Decorator to require API key authentication for a tool.

    Note: In FastMCP 2.0, Context provides limited access to request headers.
    This decorator checks if auth is enabled and validates when possible.
    For HTTP transport, clients should pass api_key in tool arguments or
    configure auth at the transport level.

    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not AUTH_ENABLED:
            return await func(*args, **kwargs)

        # Check for api_key in kwargs (passed by client)
        api_key = kwargs.pop("api_key", None)

        # Also check ctx for any auth info (transport-dependent)
        ctx = kwargs.get("ctx")
        if ctx is not None and hasattr(ctx, "request_context"):
            # Some transports may provide request context
            req_ctx = getattr(ctx, "request_context", {})
            if isinstance(req_ctx, dict):
                api_key = api_key or req_ctx.get("authorization")

        is_valid, error = validate_api_key(api_key)
        if not is_valid:
            return _json({"error": "authentication_required", "message": error}, indent=False)

        return await func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


# Model Configuration
EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-xs")

# Server Configuration
SERVER_NAME = _get_env("SERVER_NAME", "Reason-Guard-MCP")
SERVER_TRANSPORT = _get_env("SERVER_TRANSPORT", "stdio")
# Bind to localhost by default for security (CWE-306: Missing Authentication)
# Use 0.0.0.0 only in production with proper authentication/firewall
SERVER_HOST = _get_env("SERVER_HOST", "127.0.0.1")
SERVER_PORT = _get_env_int("SERVER_PORT", 8000)

# Rate Limiting Configuration
RATE_LIMIT_MAX_SESSIONS = _get_env_int("RATE_LIMIT_MAX_SESSIONS", 100)
RATE_LIMIT_WINDOW_SECONDS = _get_env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
MAX_TOTAL_SESSIONS = _get_env_int("MAX_TOTAL_SESSIONS", 500)

# Session Cleanup Configuration
SESSION_MAX_AGE_MINUTES = _get_env_int("SESSION_MAX_AGE_MINUTES", 30)
CLEANUP_INTERVAL_SECONDS = _get_env_int("CLEANUP_INTERVAL_SECONDS", 60)

# =============================================================================
# Input Size Limits (CWE-400: Uncontrolled Resource Consumption)
# =============================================================================
MAX_PROBLEM_SIZE = _get_env_int("MAX_PROBLEM_SIZE", 50000)  # 50KB
MAX_THOUGHT_SIZE = _get_env_int("MAX_THOUGHT_SIZE", 10000)  # 10KB
MAX_CONTEXT_SIZE = _get_env_int("MAX_CONTEXT_SIZE", 100000)  # 100KB
MAX_ALTERNATIVES = _get_env_int("MAX_ALTERNATIVES", 10)  # Max alternatives for MPPA
MAX_THOUGHTS_PER_SESSION = _get_env_int("MAX_THOUGHTS_PER_SESSION", 1000)  # Memory exhaustion guard

# RAG Configuration - Enabled by default for semantic scoring and thought retrieval
ENABLE_RAG = _get_env("ENABLE_RAG", "true").lower() == "true"
VECTOR_DB_PATH = _get_env("VECTOR_DB_PATH", "reasonguard_thoughts.db")


def _validate_vector_db_path() -> str:
    """Validate VECTOR_DB_PATH to prevent path traversal (CWE-22)."""
    from src.utils.weight_store import validate_db_path

    if VECTOR_DB_PATH == ":memory:":
        return VECTOR_DB_PATH

    try:
        validated = validate_db_path(VECTOR_DB_PATH)
        return str(validated)
    except ValueError as e:
        logger.error(f"Invalid VECTOR_DB_PATH: {e}")
        logger.warning("Falling back to in-memory vector store")
        return ":memory:"


# =============================================================================
# Initialize Embedding Model Manager
# =============================================================================


def _get_embedding_model_name() -> str:
    """Get full embedding model name."""
    model_name = EMBEDDING_MODEL
    if "/" not in model_name:
        model_name = f"sentence-transformers/{model_name}"
    return model_name


def _init_model_manager() -> None:
    """Initialize the model manager with embedding model."""
    model_manager = ModelManager.get_instance()
    model_name = _get_embedding_model_name()
    logger.info(f"Preloading embedding model: {model_name}")
    model_manager.initialize(model_name, blocking=True)
    logger.info("Embedding model ready")


# =============================================================================
# Thread Pool for CPU-bound Operations
# =============================================================================
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reasonguard-worker")


# =============================================================================
# Rate Limiting and Session Management
# =============================================================================


class SessionRateLimiter:
    """Sliding window rate limiter for session creation.

    Prevents resource exhaustion by limiting the rate of new session creation.
    Uses a sliding window algorithm for accurate rate limiting.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        """Initialize rate limiter.

        Args:
            max_requests: Maximum sessions allowed in the time window
            window_seconds: Time window in seconds

        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _cleanup_old_timestamps(self) -> None:
        """Remove timestamps outside the current window."""
        cutoff = time.monotonic() - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    async def check_rate_limit(self) -> tuple[bool, dict[str, Any]]:
        """Check if a new session can be created.

        Returns:
            Tuple of (allowed, info_dict).
            If allowed is False, info_dict contains error details.

        """
        async with self._lock:
            self._cleanup_old_timestamps()

            if len(self._timestamps) >= self._max_requests:
                # Calculate when the oldest request will expire
                oldest = self._timestamps[0]
                retry_after = self._window_seconds - (time.monotonic() - oldest)
                return False, {
                    "error": "rate_limit_exceeded",
                    "message": (
                        f"Too many sessions created. "
                        f"Max {self._max_requests} per {self._window_seconds}s."
                    ),
                    "retry_after_seconds": max(1, int(retry_after)),
                    "current_count": len(self._timestamps),
                    "max_allowed": self._max_requests,
                }

            return True, {"remaining": self._max_requests - len(self._timestamps)}

    async def record_request(self) -> None:
        """Record a successful session creation."""
        async with self._lock:
            self._timestamps.append(time.monotonic())

    def get_stats(self) -> dict[str, Any]:
        """Get current rate limiter statistics."""
        self._cleanup_old_timestamps()
        return {
            "current_count": len(self._timestamps),
            "max_allowed": self._max_requests,
            "window_seconds": self._window_seconds,
            "remaining": max(0, self._max_requests - len(self._timestamps)),
        }


# Global rate limiter instance
_rate_limiter: SessionRateLimiter | None = None


def get_rate_limiter() -> SessionRateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SessionRateLimiter(
            max_requests=RATE_LIMIT_MAX_SESSIONS,
            window_seconds=RATE_LIMIT_WINDOW_SECONDS,
        )
    return _rate_limiter


# =============================================================================
# IP-Based Rate Limiting (V-003)
# =============================================================================

# Configuration for IP-based rate limiting
IP_RATE_LIMIT_MAX_REQUESTS = int(_get_env("IP_RATE_LIMIT_MAX_REQUESTS", "100"))
IP_RATE_LIMIT_WINDOW_SECONDS = int(_get_env("IP_RATE_LIMIT_WINDOW_SECONDS", "60"))
IP_RATE_LIMIT_ENABLED = _get_env("IP_RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")


class IPRateLimiter:
    """Per-IP sliding window rate limiter.

    V-003: Prevents abuse by limiting requests per client IP address.
    Uses asyncio.Lock for non-blocking concurrent access.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        """Initialize IP rate limiter.

        Args:
            max_requests: Maximum requests per IP in the time window
            window_seconds: Time window in seconds

        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._ip_timestamps: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    def _cleanup_old_timestamps(self, ip: str) -> None:
        """Remove timestamps outside the current window for an IP."""
        if ip not in self._ip_timestamps:
            return
        cutoff = time.monotonic() - self._window_seconds
        timestamps = self._ip_timestamps[ip]
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        # Remove empty deques to prevent memory leak
        if not timestamps:
            del self._ip_timestamps[ip]

    async def check_rate_limit(self, ip: str) -> tuple[bool, dict[str, Any]]:
        """Check if a request from this IP is allowed.

        Args:
            ip: Client IP address

        Returns:
            Tuple of (allowed, info_dict).
            If allowed is False, info_dict contains error details.

        """
        async with self._lock:
            self._cleanup_old_timestamps(ip)

            timestamps = self._ip_timestamps.get(ip, deque())

            if len(timestamps) >= self._max_requests:
                # Calculate when the oldest request will expire
                oldest = timestamps[0]
                retry_after = self._window_seconds - (time.monotonic() - oldest)
                return False, {
                    "error": "ip_rate_limit_exceeded",
                    "message": (
                        f"Too many requests from this IP. "
                        f"Max {self._max_requests} per {self._window_seconds}s."
                    ),
                    "retry_after_seconds": max(1, int(retry_after)),
                    "current_count": len(timestamps),
                    "max_allowed": self._max_requests,
                }

            return True, {"remaining": self._max_requests - len(timestamps)}

    async def record_request(self, ip: str) -> None:
        """Record a request from an IP."""
        async with self._lock:
            if ip not in self._ip_timestamps:
                self._ip_timestamps[ip] = deque()
            self._ip_timestamps[ip].append(time.monotonic())

    def get_stats(self) -> dict[str, Any]:
        """Get current rate limiter statistics."""
        # Cleanup all IPs first
        for ip in list(self._ip_timestamps.keys()):
            self._cleanup_old_timestamps(ip)
        return {
            "active_ips": len(self._ip_timestamps),
            "max_requests_per_ip": self._max_requests,
            "window_seconds": self._window_seconds,
        }


# Global IP rate limiter instance
_ip_rate_limiter: IPRateLimiter | None = None


def get_ip_rate_limiter() -> IPRateLimiter:
    """Get or create IP rate limiter instance."""
    global _ip_rate_limiter
    if _ip_rate_limiter is None:
        _ip_rate_limiter = IPRateLimiter(
            max_requests=IP_RATE_LIMIT_MAX_REQUESTS,
            window_seconds=IP_RATE_LIMIT_WINDOW_SECONDS,
        )
    return _ip_rate_limiter


def _is_trusted_proxy(ip: str) -> bool:
    """Check if an IP is in the trusted proxies list.

    Supports exact IP matching and basic CIDR notation.
    """
    if not TRUSTED_PROXIES:
        return False

    import ipaddress

    try:
        client_addr = ipaddress.ip_address(ip)
    except ValueError:
        return False

    for proxy in TRUSTED_PROXIES:
        try:
            if "/" in proxy:
                # CIDR notation
                if client_addr in ipaddress.ip_network(proxy, strict=False):
                    return True
            else:
                # Exact IP match
                if client_addr == ipaddress.ip_address(proxy):
                    return True
        except ValueError:
            continue  # Skip invalid entries

    return False


def get_client_ip() -> str:
    """Get client IP from HTTP request, with proxy header support.

    Only trusts X-Forwarded-For when the direct client is in TRUSTED_PROXIES.
    This prevents IP spoofing attacks via header injection.

    Returns:
        Client IP address, or "unknown" if not available.

    """
    try:
        request = get_http_request()
        direct_ip = request.client.host if request.client else None

        # Only trust forwarded headers if direct client is a trusted proxy
        if direct_ip and _is_trusted_proxy(direct_ip):
            # Check X-Forwarded-For header (for reverse proxies)
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                # Take the first IP in the chain (original client)
                client_ip = forwarded_for.split(",")[0].strip()
                # Reject localhost values in forwarded headers (spoofing attempt)
                if client_ip not in ("127.0.0.1", "::1", "localhost"):
                    return str(client_ip)
                else:
                    logger.warning(
                        f"Rejected spoofed localhost in X-Forwarded-For from {direct_ip}"
                    )
            # Check X-Real-IP header (nginx)
            real_ip = request.headers.get("x-real-ip")
            if real_ip:
                ip = real_ip.strip()
                if ip not in ("127.0.0.1", "::1", "localhost"):
                    return str(ip)
                else:
                    logger.warning(f"Rejected spoofed localhost in X-Real-IP from {direct_ip}")

        # Fall back to direct client IP
        if direct_ip:
            return str(direct_ip)
        return "unknown"
    except Exception:
        # Not in HTTP context (e.g., stdio transport)
        return "unknown"


async def check_ip_rate_limit() -> tuple[bool, dict[str, Any] | None]:
    """Check IP-based rate limit for current request.

    Returns:
        Tuple of (allowed, error_info or None)

    """
    if not IP_RATE_LIMIT_ENABLED:
        return True, None

    client_ip = get_client_ip()
    if client_ip == "unknown":
        # Can't rate limit without IP, allow through
        return True, None

    # Skip localhost if AUTH_BYPASS_LOCALHOST is enabled
    if AUTH_BYPASS_LOCALHOST and client_ip in ("127.0.0.1", "::1", "localhost"):
        return True, None

    limiter = get_ip_rate_limiter()
    allowed, info = await limiter.check_rate_limit(client_ip)
    if not allowed:
        logger.warning(f"IP rate limit exceeded for {client_ip}")
        return False, info

    await limiter.record_request(client_ip)
    return True, None


# =============================================================================
# Automatic Session Cleanup
# =============================================================================

_cleanup_task: asyncio.Task[None] | None = None


async def _cleanup_stale_sessions() -> None:
    """Background task to clean up stale sessions."""
    max_age = timedelta(minutes=SESSION_MAX_AGE_MINUTES)
    logger.info(
        f"Session cleanup task started (max_age={SESSION_MAX_AGE_MINUTES}m, "
        f"interval={CLEANUP_INTERVAL_SECONDS}s)"
    )

    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)

            manager = get_unified_manager()
            removed = await manager.cleanup_stale(max_age)

            if removed:
                logger.info(f"Cleaned up {len(removed)} stale sessions: {removed}")

        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            # Continue running despite errors


def _start_cleanup_task() -> None:
    """Start the background cleanup task if not already running."""
    global _cleanup_task
    try:
        loop = asyncio.get_running_loop()
        if _cleanup_task is None or _cleanup_task.done():
            _cleanup_task = loop.create_task(_cleanup_stale_sessions())
            logger.debug("Cleanup task scheduled")
    except RuntimeError:
        # No running event loop - will be started when server runs
        logger.debug("No event loop available, cleanup task will start with server")


def _stop_cleanup_task() -> None:
    """Stop the background cleanup task."""
    global _cleanup_task
    if _cleanup_task is not None and not _cleanup_task.done():
        _cleanup_task.cancel()
        _cleanup_task = None
        logger.debug("Cleanup task stopped")


# =============================================================================
# Initialize FastMCP Server
# =============================================================================

mcp = FastMCP(
    name=SERVER_NAME,
    instructions="""Reason Guard MCP Server - Dual-paradigm reasoning state manager.

ARCHITECTURE: You (the LLM) do ALL reasoning. These tools TRACK, ORGANIZE, and optionally ENFORCE.

=== CHOOSE YOUR PARADIGM ===

Use paradigm_hint(problem) first! It analyzes your problem and recommends which paradigm to use.

GUIDANCE MODE (think tool): Server provides suggestions, warnings, rewards.
  - Use when: You want flexibility with helpful feedback
  - Bad steps get warnings but are still recorded

ENFORCEMENT MODE (atomic router): Server REJECTS invalid steps.
  - Use when: Problem requires disciplined reasoning (proofs, traps, paradoxes)
  - Bad steps are REJECTED - you must fix and retry

=== PARADIGM SELECTION TOOL ===

0. paradigm_hint(problem) - Analyze problem, get recommendation
   Returns: recommendation (guidance/enforcement), confidence, suggested_tools

=== GUIDANCE MODE TOOLS ===

1. think(action, ...) - Unified reasoning with auto-mode selection
   Actions: start, continue, branch, revise, synthesize, verify, finish
   Modes: auto (default), chain, matrix, hybrid

2. compress(context, query, ratio) - Semantic context compression

3. status(session_id?) - Server/session status

GUIDANCE WORKFLOW:
1. think(action="start", problem="Solve X")
2. think(action="continue", session_id=ID, thought="Step 1...")
   -> Receives guidance, blind spot warnings, reward signals
3. think(action="finish", session_id=ID, thought="Answer...")

=== ENFORCEMENT MODE TOOLS (Atomic Router) ===

4. initialize_reasoning(problem, complexity) - Start enforced session
   Complexity: low (2-5 steps), medium (4-8), high (6-12), auto

5. submit_step(session_id, step_type, content, confidence) - Submit step
   Step types: premise -> hypothesis -> verification -> synthesis
   Returns: ACCEPTED, REJECTED, BRANCH_REQUIRED, or VERIFICATION_REQUIRED

6. create_branch(session_id, alternatives) - Required when confidence too low

7. verify_claims(session_id, claims, evidence) - Verify before synthesis

8. router_status(session_id?) - Router/session state

ENFORCEMENT RULES:
- Rule A: Cannot synthesize until min_steps reached
- Rule B: Low confidence requires branching (create_branch)
- Rule C: Must verify before synthesis
- Rule D: Must follow state machine (premise->hypothesis->verification->synthesis)
- Rule E: Must synthesize at max_steps

ENFORCEMENT WORKFLOW:
1. initialize_reasoning("Prove the Monty Hall solution", "high")
   -> Gets trap warning, min=6, max=12 steps
2. submit_step(id, "premise", "There are 3 doors...", 0.95)
   -> ACCEPTED
3. submit_step(id, "hypothesis", "Switching wins 2/3...", 0.5)
   -> BRANCH_REQUIRED (confidence < 0.75)
4. create_branch(id, ["Switching wins 2/3", "Staying wins 1/2", "Equal odds"])
   -> ACCEPTED, now explore alternatives
5. submit_step(id, "verification", "Enumerating cases...", 0.9)
   -> ACCEPTED
6. submit_step(id, "synthesis", "Switching wins 2/3 proven", 0.95)
   -> ACCEPTED (session complete)

=== WHEN TO USE WHICH ===

| Problem Type              | Paradigm    | Why                           |
|---------------------------|-------------|-------------------------------|
| Open-ended analysis       | Guidance    | Flexibility, exploration      |
| Math proofs               | Enforcement | Prevents jumping to conclusion|
| Paradoxes (Monty Hall)    | Enforcement | Traps require discipline      |
| Code debugging            | Guidance    | Iterative, exploratory        |
| Logical arguments         | Enforcement | Forces verification           |
| Creative brainstorming    | Guidance    | No strict structure needed    |
""",
)

# =============================================================================
# Tool Instances
# =============================================================================

_compression_tool: ContextAwareCompressionTool | None = None


def get_compression_tool() -> ContextAwareCompressionTool:
    """Get or create compression tool instance."""
    global _compression_tool
    if _compression_tool is None:
        model_name = _get_embedding_model_name()
        _compression_tool = ContextAwareCompressionTool(model_name=model_name)
    return _compression_tool


# =============================================================================
# Type Definitions
# =============================================================================

ThinkAction = Literal[
    "start",
    "continue",
    "branch",
    "revise",
    "synthesize",
    "verify",
    "finish",
    "resolve",
    "analyze",
    "suggest",
    "feedback",  # S2: Record suggestion outcome
    "auto",  # S3: Auto-execute suggestion
]
ThinkModeStr = Literal["auto", "chain", "matrix", "hybrid", "verify"]
ResolveStrategy = Literal["revise", "branch", "reconcile", "backtrack"]
SuggestionOutcome = Literal["accepted", "rejected"]


# =============================================================================
# Input Validation Helpers (CWE-400 Prevention)
# =============================================================================


def _validate_input_sizes(
    problem: str | None = None,
    thought: str | None = None,
    context: str | None = None,
    alternatives: list[str] | None = None,
) -> dict[str, Any] | None:
    """Validate input sizes to prevent resource exhaustion.

    Returns:
        None if valid, or dict with error details if invalid.

    """
    if problem and len(problem) > MAX_PROBLEM_SIZE:
        return {
            "error": "input_too_large",
            "field": "problem",
            "max_size": MAX_PROBLEM_SIZE,
            "actual_size": len(problem),
            "message": f"Problem exceeds maximum size ({MAX_PROBLEM_SIZE:,} chars)",
        }

    if thought and len(thought) > MAX_THOUGHT_SIZE:
        return {
            "error": "input_too_large",
            "field": "thought",
            "max_size": MAX_THOUGHT_SIZE,
            "actual_size": len(thought),
            "message": f"Thought exceeds maximum size ({MAX_THOUGHT_SIZE:,} chars)",
        }

    if context and len(context) > MAX_CONTEXT_SIZE:
        return {
            "error": "input_too_large",
            "field": "context",
            "max_size": MAX_CONTEXT_SIZE,
            "actual_size": len(context),
            "message": f"Context exceeds maximum size ({MAX_CONTEXT_SIZE:,} chars)",
        }

    if alternatives:
        if len(alternatives) > MAX_ALTERNATIVES:
            return {
                "error": "too_many_alternatives",
                "max_alternatives": MAX_ALTERNATIVES,
                "actual_count": len(alternatives),
                "message": f"Too many alternatives ({len(alternatives)} > {MAX_ALTERNATIVES})",
            }
        # Also validate each alternative's size
        for i, alt in enumerate(alternatives):
            if len(alt) > MAX_THOUGHT_SIZE:
                return {
                    "error": "input_too_large",
                    "field": f"alternatives[{i}]",
                    "max_size": MAX_THOUGHT_SIZE,
                    "actual_size": len(alt),
                    "message": f"Alternative {i} exceeds maximum size ({MAX_THOUGHT_SIZE:,} chars)",
                }

    return None


# =============================================================================
# TOOL 1: THINK (Unified Reasoning)
# =============================================================================


@mcp.tool
async def think(
    action: ThinkAction,
    mode: ThinkModeStr | None = None,
    session_id: str | None = None,
    problem: str | None = None,
    context: str | None = None,
    thought: str | None = None,
    expected_steps: int = 10,
    rows: int | None = None,
    cols: int | None = None,
    row: int | None = None,
    col: int | None = None,
    branch_from: str | None = None,
    revises: str | None = None,
    confidence: float | None = None,
    alternatives: list[str] | None = None,
    alternative_confidences: list[float] | None = None,
    # Contradiction resolution parameters
    resolve_strategy: ResolveStrategy | None = None,
    contradicting_thought_id: str | None = None,
    # Backwards compatibility for verify mode
    claim_id: int | None = None,
    verdict: str | None = None,
    evidence: str | None = None,
    # S2: Feedback parameters for recording suggestion outcomes
    suggestion_id: str | None = None,
    suggestion_outcome: SuggestionOutcome | None = None,
    actual_action: str | None = None,
    # S3: Auto-execute parameters
    max_auto_steps: int = 5,
    stop_on_high_risk: bool = True,
    # OPT4: Response verbosity (minimal by default for ~50% token reduction)
    verbosity: Literal["minimal", "normal", "full"] = "minimal",
    ctx: Context | None = None,
) -> str:
    """Unified reasoning tool with auto-mode selection.

    Actions:
        start: Begin a new reasoning session (requires problem, mode defaults to auto)
        continue: Add a reasoning step (requires session_id and thought)
        branch: Branch from a thought (chain/hybrid, requires branch_from)
        revise: Revise a thought (chain/hybrid, requires revises)
        synthesize: Synthesize a column (matrix/hybrid, requires col and thought)
        resolve: Resolve a detected contradiction (requires resolve_strategy and thought)
        analyze: Get mid-session analysis with metrics and recommendations (requires session_id)
        suggest: Get AI-recommended next action based on session state (requires session_id)
        feedback: Record outcome of a suggestion for weight learning (requires suggestion_id, suggestion_outcome)
        auto: Auto-execute the top suggested action (requires session_id)
        finish: Complete reasoning (requires session_id)

    Modes:
        auto: Auto-select based on problem complexity (default)
        chain: Sequential chain-of-thought reasoning
        matrix: Multi-perspective matrix reasoning (rows x cols)
        hybrid: Adaptive chain -> matrix escalation

    Args:
        action: The action to perform (required)
        mode: Reasoning mode for "start" action (default: auto)
        session_id: Session ID for continuing/finishing
        problem: Problem statement for "start" action
        context: Background context (optional)
        thought: Your reasoning content
        expected_steps: Expected steps for chain mode (default 10)
        rows: Matrix rows (auto-detected if None)
        cols: Matrix columns (auto-detected if None)
        row: Matrix row index (0-based)
        col: Matrix column index (0-based)
        branch_from: Thought ID to branch from
        revises: Thought ID to revise
        confidence: Your confidence in this thought (0-1)
        alternatives: MPPA - Alternative reasoning paths to evaluate
        alternative_confidences: CISC - Your confidence scores for alternatives
        resolve_strategy: Strategy for resolving contradiction (revise/branch/reconcile/backtrack)
        contradicting_thought_id: ID of the thought to address in resolution
        suggestion_id: ID of suggestion to record feedback for (feedback action)
        suggestion_outcome: Whether suggestion was accepted or rejected (feedback action)
        actual_action: What action was actually taken, if different from suggestion (feedback action)
        max_auto_steps: Maximum steps for auto-execute (default 5, auto action)
        stop_on_high_risk: Stop auto-execution at high-risk checkpoints (default True, auto action)
        verbosity: Response detail level - minimal (default, ~50% fewer tokens), normal, or full

    Returns:
        JSON with action result, session state, guidance, and any warnings

    """
    try:
        # V-003: Check IP-based rate limit
        ip_allowed, ip_error = await check_ip_rate_limit()
        if not ip_allowed:
            return _json(ip_error, indent=False)

        # Validate input sizes (CWE-400 prevention)
        validation_error = _validate_input_sizes(
            problem=problem,
            thought=thought,
            context=context,
            alternatives=alternatives,
        )
        if validation_error:
            return _json(validation_error, indent=False)

        manager = get_unified_manager()

        if action == "start":
            return await _think_start(
                manager=manager,
                mode=mode,
                problem=problem,
                context=context,
                expected_steps=expected_steps,
                rows=rows,
                cols=cols,
                ctx=ctx,
            )
        elif action == "continue":
            return await _think_continue(
                manager=manager,
                session_id=session_id,
                thought=thought,
                row=row,
                col=col,
                confidence=confidence,
                alternatives=alternatives,
                alternative_confidences=alternative_confidences,
                verbosity=ResponseVerbosity(verbosity),
                ctx=ctx,
            )
        elif action == "branch":
            return await _think_branch(
                manager=manager,
                session_id=session_id,
                thought=thought,
                branch_from=branch_from,
                ctx=ctx,
            )
        elif action == "revise":
            return await _think_revise(
                manager=manager,
                session_id=session_id,
                thought=thought,
                revises=revises,
                ctx=ctx,
            )
        elif action == "synthesize":
            return await _think_synthesize(
                manager=manager,
                session_id=session_id,
                col=col,
                thought=thought,
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "verify":
            # Backwards compatibility: verify action adds a verification thought
            return await _think_verify(
                manager=manager,
                session_id=session_id,
                thought=thought or evidence or "",
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "finish":
            return await _think_finish(
                manager=manager,
                session_id=session_id,
                thought=thought,
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "resolve":
            return await _think_resolve(
                manager=manager,
                session_id=session_id,
                thought=thought,
                resolve_strategy=resolve_strategy,
                contradicting_thought_id=contradicting_thought_id,
                confidence=confidence,
                ctx=ctx,
            )
        elif action == "analyze":
            return await _think_analyze(
                manager=manager,
                session_id=session_id,
                ctx=ctx,
            )
        elif action == "suggest":
            return await _think_suggest(
                manager=manager,
                session_id=session_id,
                ctx=ctx,
            )
        elif action == "feedback":
            return await _think_feedback(
                manager=manager,
                session_id=session_id,
                suggestion_id=suggestion_id,
                suggestion_outcome=suggestion_outcome,
                actual_action=actual_action,
                ctx=ctx,
            )
        elif action == "auto":
            return await _think_auto(
                manager=manager,
                session_id=session_id,
                max_auto_steps=max_auto_steps,
                stop_on_high_risk=stop_on_high_risk,
                ctx=ctx,
            )
        else:
            return _json({"error": f"Unknown action: {action}"}, indent=False)

    except ValueError as e:
        return _json({"error": str(e)}, indent=False)
    except Exception as e:
        error = ToolExecutionError("think", str(e), {"action": action})
        logger.error(f"Think action '{action}' failed: {e}")
        return _json(error.to_dict(), indent=False)


async def _think_start(
    manager: UnifiedReasonerManager,
    mode: ThinkModeStr | None,
    problem: str | None,
    context: str | None,
    expected_steps: int,
    rows: int | None,
    cols: int | None,
    ctx: Context | None,
) -> str:
    """Handle think start action."""
    if not problem:
        return _json({"error": "problem is required for start action"}, indent=False)

    # Check rate limit
    rate_limiter = get_rate_limiter()
    allowed, info = await rate_limiter.check_rate_limit()
    if not allowed:
        if ctx:
            await ctx.warning(f"Rate limit exceeded: {info['message']}")
        return _json(info, indent=False)

    # Check total session count (eventual consistency is fine for this check)
    total_sessions = len(manager.get_all_sessions_snapshot())
    if total_sessions >= MAX_TOTAL_SESSIONS:
        error_info = {
            "error": "max_sessions_exceeded",
            "message": (
                f"Maximum total sessions ({MAX_TOTAL_SESSIONS}) reached. "
                "Wait for cleanup or finish existing sessions."
            ),
            "current_sessions": total_sessions,
            "max_sessions": MAX_TOTAL_SESSIONS,
        }
        if ctx:
            await ctx.warning(f"Max sessions exceeded: {total_sessions}/{MAX_TOTAL_SESSIONS}")
        return _json(error_info, indent=False)

    # Record the request for rate limiting
    await rate_limiter.record_request()

    # Convert mode string to enum
    # Note: "verify" mode is mapped to "chain" for backwards compatibility
    reasoning_mode = ReasoningMode.AUTO
    if mode:
        if mode == "verify":
            # Backwards compatibility: verify mode uses chain reasoning
            reasoning_mode = ReasoningMode.CHAIN
        else:
            try:
                reasoning_mode = ReasoningMode(mode)
            except ValueError:
                return _json({"error": f"Unknown mode: {mode}"}, indent=False)

    # Start session with observability
    store = get_metrics_store()
    with store.trace("think_start") as span:
        span.set_attribute("mode", mode or "auto")
        span.set_attribute("problem_length", len(problem))

        result = await manager.start_session(
            problem=problem,
            context=context or "",
            mode=reasoning_mode,
            expected_steps=expected_steps,
            rows=rows,
            cols=cols,
        )

        # Record session in metrics
        session_id = result.get("session_id", "")
        actual_mode = result.get("actual_mode", "unknown")
        span.set_attribute("session_id", session_id)
        span.set_attribute("actual_mode", actual_mode)
        store.record_session_start(
            session_id=session_id,
            problem=problem[:500],
            mode=actual_mode,
            complexity=result.get("domain", "unknown"),
        )

    # Backwards compatibility: if user requested "verify", show that in response
    if mode == "verify":
        result["mode"] = "verify"

    if ctx:
        actual_mode = result.get("actual_mode", "unknown")
        domain = result.get("domain", "unknown")
        await ctx.info(
            f"Started session {result['session_id']} (mode={actual_mode}, domain={domain})"
        )

    return _json(result)


async def _think_continue(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    row: int | None,
    col: int | None,
    confidence: float | None,
    alternatives: list[str] | None,
    alternative_confidences: list[float] | None,
    verbosity: ResponseVerbosity,
    ctx: Context | None,
) -> str:
    """Handle think continue action."""
    if not session_id:
        return _json({"error": "session_id is required for continue action"}, indent=False)
    if not thought:
        return _json({"error": "thought is required for continue action"}, indent=False)

    try:
        result = await manager.add_thought(
            session_id=session_id,
            content=thought,
            thought_type=ThoughtType.CONTINUATION,
            row=row,
            col=col,
            confidence=confidence,
            alternatives=alternatives,
            alternative_confidences=alternative_confidences,
            verbosity=verbosity,
        )
    except SessionNotFoundError:
        return _json({"error": "Invalid or expired session"}, indent=False)

    if ctx:
        step = result.get("step", "?")
        score = result.get("survival_score", 0)
        await ctx.info(f"Added step {step} (score={score:.2f})")

        # Warn about blind spots (key changed from blind_spots_detected)
        if "blind_spots" in result:
            count = len(result["blind_spots"])
            await ctx.warning(f"Detected {count} blind spot(s) in this step")

    return _json(result)


async def _think_branch(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    branch_from: str | None,
    ctx: Context | None,
) -> str:
    """Handle think branch action."""
    if not session_id:
        return _json({"error": "session_id is required for branch action"}, indent=False)
    if not thought:
        return _json({"error": "thought is required for branch action"}, indent=False)
    if not branch_from:
        return _json({"error": "branch_from is required for branch action"}, indent=False)

    try:
        result = await manager.add_thought(
            session_id=session_id,
            content=thought,
            thought_type=ThoughtType.BRANCH,
            branch_from=branch_from,
        )
    except SessionNotFoundError:
        return _json({"error": "Invalid or expired session"}, indent=False)

    if ctx:
        await ctx.info(f"Branched from thought {branch_from}")

    return _json(result)


async def _think_revise(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    revises: str | None,
    ctx: Context | None,
) -> str:
    """Handle think revise action."""
    if not session_id:
        return _json({"error": "session_id is required for revise action"}, indent=False)
    if not thought:
        return _json({"error": "thought is required for revise action"}, indent=False)
    if not revises:
        return _json({"error": "revises is required for revise action"}, indent=False)

    result = await manager.add_thought(
        session_id=session_id,
        content=thought,
        thought_type=ThoughtType.REVISION,
        revises=revises,
    )

    if ctx:
        await ctx.info(f"Revised thought {revises}")

    return _json(result)


async def _think_synthesize(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    col: int | None,
    thought: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think synthesize action (matrix/hybrid only)."""
    if not session_id:
        return _json({"error": "session_id is required for synthesize action"}, indent=False)
    if col is None:
        return _json({"error": "col is required for synthesize action"}, indent=False)
    if not thought:
        return _json({"error": "thought is required for synthesize action"}, indent=False)

    result = await manager.synthesize(
        session_id=session_id,
        col=col,
        content=thought,
        confidence=confidence,
    )

    if ctx:
        await ctx.info(f"Synthesized column {col}")

    return _json(result)


async def _think_verify(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think verify action (backwards compatibility).

    In the unified reasoner, verify is treated as a verification thought type.
    """
    if not session_id:
        return _json({"error": "session_id is required for verify action"}, indent=False)
    if not thought:
        return _json({"error": "thought/evidence is required for verify action"}, indent=False)

    result = await manager.add_thought(
        session_id=session_id,
        content=thought,
        thought_type=ThoughtType.VERIFICATION,
        confidence=confidence,
    )

    if ctx:
        step = result.get("step", "?")
        await ctx.info(f"Added verification step {step}")

    return _json(result)


async def _think_finish(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think finish action."""
    if not session_id:
        return _json({"error": "session_id is required for finish action"}, indent=False)

    store = get_metrics_store()
    with store.trace("think_finish", session_id=session_id) as span:
        result = await manager.finalize(
            session_id=session_id,
            answer=thought or "",
            confidence=confidence,
        )

        total_steps = result.get("total_steps", 0)
        mode = result.get("mode_used", "unknown")
        total_reward = result.get("total_reward", 0)

        span.set_attribute("total_steps", total_steps)
        span.set_attribute("mode", mode)
        span.set_attribute("total_reward", total_reward)

        # Record session end in metrics
        duration_ms = result.get("duration_ms", 0)
        store.record_session_end(session_id, status="completed", total_steps=total_steps)
        if duration_ms:
            store.record_histogram("session.duration_ms", duration_ms, {"mode": mode})

    if ctx:
        await ctx.info(f"Finalized: {total_steps} steps, mode={mode}, reward={total_reward:.2f}")

        # Warn about unaddressed blind spots
        if "unaddressed_blind_spots" in result:
            count = len(result["unaddressed_blind_spots"])
            await ctx.warning(f"Warning: {count} blind spot(s) were not addressed")

    return _json(result)


async def _think_resolve(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    thought: str | None,
    resolve_strategy: ResolveStrategy | None,
    contradicting_thought_id: str | None,
    confidence: float | None,
    ctx: Context | None,
) -> str:
    """Handle think resolve action for contradiction resolution.

    Strategies:
        revise: Modify the current thought to resolve the contradiction
        branch: Create separate reasoning branches for each possibility
        reconcile: Find a higher-level synthesis resolving the contradiction
        backtrack: Abandon the contradicting line of reasoning

    """
    if not session_id:
        return _json({"error": "session_id is required for resolve action"}, indent=False)

    if not resolve_strategy:
        return _json(
            {
                "error": "resolve_strategy is required for resolve action",
                "valid_strategies": ["revise", "branch", "reconcile", "backtrack"],
            },
            indent=False,
        )

    if not thought:
        return _json({"error": "thought is required for resolve action"}, indent=False)

    result = await manager.resolve_contradiction(
        session_id=session_id,
        strategy=resolve_strategy,
        resolution_content=thought,
        contradicting_thought_id=contradicting_thought_id,
        confidence=confidence,
    )

    if ctx:
        strategy = result.get("strategy_applied", resolve_strategy)
        await ctx.info(f"Resolved contradiction using '{strategy}' strategy")

        if result.get("remaining_contradictions", 0) > 0:
            count = result["remaining_contradictions"]
            await ctx.warning(f"Note: {count} contradiction(s) still remain in the session")

    return _json(result)


async def _think_analyze(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    ctx: Context | None,
) -> str:
    """Handle think analyze action for mid-session analysis.

    Returns consolidated metrics, quality scores, and actionable recommendations
    without duplicating raw session data from status/get_status.
    """
    if not session_id:
        return _json({"error": "session_id is required for analyze action"}, indent=False)

    try:
        analytics = await manager.analyze_session(session_id)
    except SessionNotFoundError:
        return _json({"error": "Invalid or expired session"}, indent=False)

    result = analytics.to_dict()

    if ctx:
        # Summarize key findings
        quality = result["quality"]["overall"]
        risk = result["risk"]["level"]
        issues = result["issues"]

        await ctx.info(f"Analysis complete: quality={quality:.2f}, risk={risk}")

        if issues["unresolved_contradictions"] > 0:
            await ctx.warning(
                f"Found {issues['unresolved_contradictions']} unresolved contradiction(s)"
            )

        if issues["blind_spots_unaddressed"] > 0:
            await ctx.warning(
                f"Found {issues['blind_spots_unaddressed']} unaddressed blind spot(s)"
            )

        # Show top recommendation
        if result["recommendations"]:
            await ctx.info(f"Top recommendation: {result['recommendations'][0]}")

    return _json(result)


async def _think_suggest(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    ctx: Context | None,
) -> str:
    """Handle think suggest action for AI-recommended next action.

    Analyzes session state and returns the recommended next action
    with parameters and reasoning, reducing LLM cognitive load.
    """
    if not session_id:
        return _json({"error": "session_id is required for suggest action"}, indent=False)

    try:
        suggestion = await manager.suggest_next_action(session_id)
    except SessionNotFoundError:
        return _json({"error": "Invalid or expired session"}, indent=False)

    if ctx:
        action = suggestion["suggested_action"]
        urgency = suggestion["urgency"]
        reason = suggestion["reason"]

        if urgency == "high":
            await ctx.warning(f"Suggested: {action} (urgent) - {reason}")
        else:
            await ctx.info(f"Suggested: {action} - {reason}")

        # Show guidance
        await ctx.info(f"Guidance: {suggestion['guidance']}")

        # Show alternatives if any
        if suggestion["alternatives"]:
            alt_actions = ", ".join(a["action"] for a in suggestion["alternatives"])
            await ctx.info(f"Alternatives: {alt_actions}")

    return _json(suggestion)


async def _think_feedback(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    suggestion_id: str | None,
    suggestion_outcome: SuggestionOutcome | None,
    actual_action: str | None,
    ctx: Context | None,
) -> str:
    """Handle think feedback action for recording suggestion outcomes.

    Records whether a suggestion was accepted or rejected, and what action
    was actually taken. This data is used to adjust suggestion weights
    for future recommendations.
    """
    if not session_id:
        return _json({"error": "session_id is required for feedback action"}, indent=False)

    if not suggestion_id:
        return _json({"error": "suggestion_id is required for feedback action"}, indent=False)

    if not suggestion_outcome:
        return _json({"error": "suggestion_outcome is required for feedback action"}, indent=False)

    # Map string outcome to the expected type
    outcome: Literal["accepted", "rejected"] = suggestion_outcome

    result = await manager.record_suggestion_outcome(
        session_id=session_id,
        suggestion_id=suggestion_id,
        outcome=outcome,
        actual_action=actual_action,
    )

    if ctx:
        if result["success"]:
            await ctx.info(
                f"Recorded feedback for suggestion {suggestion_id}: {suggestion_outcome}"
            )
            if actual_action and actual_action != result.get("recorded_action"):
                await ctx.info(f"Actual action taken: {actual_action}")
            await ctx.info(f"Updated weights: {result.get('updated_weights', {})}")
        else:
            await ctx.warning(f"Failed to record feedback: {result.get('error')}")

    return _json(result)


async def _think_auto(
    manager: UnifiedReasonerManager,
    session_id: str | None,
    max_auto_steps: int,
    stop_on_high_risk: bool,
    ctx: Context | None,
) -> str:
    """Handle think auto action for auto-executing suggested actions.

    Automatically executes the top suggested action up to max_auto_steps times,
    stopping at high-risk checkpoints if stop_on_high_risk is True.

    Note: For actions that require thought content (continue, resolve, synthesize),
    auto-execution will generate placeholder content. For full LLM integration,
    use the UnifiedReasonerManager.auto_execute_suggestion() method directly
    with a thought_generator callback.
    """
    if not session_id:
        return _json({"error": "session_id is required for auto action"}, indent=False)

    if max_auto_steps < 1:
        return _json({"error": "max_auto_steps must be at least 1"}, indent=False)

    if max_auto_steps > 20:
        return _json({"error": "max_auto_steps cannot exceed 20"}, indent=False)

    # Execute auto steps
    try:
        result = await manager.auto_execute_suggestion(
            session_id=session_id,
            max_auto_steps=max_auto_steps,
            stop_on_high_risk=stop_on_high_risk,
            thought_generator=None,  # No LLM integration at server level
        )
    except SessionNotFoundError:
        return _json({"error": "Invalid or expired session"}, indent=False)

    if ctx:
        actions_executed = result.get("actions_executed", [])
        total_executed = len(actions_executed)

        if total_executed > 0:
            await ctx.info(f"Auto-executed {total_executed} action(s)")
            for i, action_info in enumerate(actions_executed, 1):
                action_name = action_info.get("action", "unknown")
                success = action_info.get("success", False)
                status = "✓" if success else "✗"
                await ctx.info(f"  {i}. {action_name} {status}")
        else:
            await ctx.info("No actions auto-executed")

        # Report stop reason
        stop_reason = result.get("stop_reason")
        if stop_reason:
            if stop_reason == "high_risk_checkpoint":
                await ctx.warning("Stopped at high-risk checkpoint (requires human review)")
            elif stop_reason == "max_steps_reached":
                await ctx.info(f"Reached max auto steps ({max_auto_steps})")
            elif stop_reason == "session_finished":
                await ctx.info("Session finished")
            elif stop_reason == "no_suggestion":
                await ctx.info("No more suggestions available")
            else:
                await ctx.info(f"Stopped: {stop_reason}")

        # Show session summary
        session_state = result.get("session_state", {})
        if session_state:
            progress = session_state.get("progress", 0)
            total_thoughts = session_state.get("total_thoughts", 0)
            await ctx.info(f"Progress: {progress:.0%} ({total_thoughts} thoughts)")

    return _json(result)


# =============================================================================
# TOOL 2: COMPRESS
# =============================================================================


@mcp.tool
async def compress(
    context: str,
    query: str,
    ratio: float = 0.3,
    ctx: Context | None = None,
) -> str:
    """Compress long context using semantic-level sentence filtering.

    Reduces token count while preserving semantic relevance to the query.
    Uses sentence embeddings to score and select most relevant content.

    Args:
        context: Long text to compress (required)
        query: Query to determine relevance (required)
        ratio: Target compression ratio 0.1-1.0 (default 0.3)

    Returns:
        JSON with compressed_context, compression_ratio, tokens_saved

    """
    try:
        # V-003: Check IP-based rate limit
        ip_allowed, ip_error = await check_ip_rate_limit()
        if not ip_allowed:
            return _json(ip_error, indent=False)

        if not query:
            return _json({"error": "query is required for compression"}, indent=False)

        if ctx:
            await ctx.info(f"Compressing {len(context)} characters...")

        tool = get_compression_tool()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            partial(
                tool.compress,
                context=context,
                question=query,
                compression_ratio=ratio,
                preserve_order=True,
            ),
        )

        if ctx:
            tokens_saved = result.original_tokens - result.compressed_tokens
            await ctx.info(
                f"Compressed to {result.compression_ratio:.1%} ({tokens_saved} tokens saved)"
            )

        # Convert result to dict for JSON serialization
        return _json(
            {
                "compressed": result.compressed_context,
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "tokens_saved": result.original_tokens - result.compressed_tokens,
                "max_relevance_score": max(
                    (score for _, score in result.relevance_scores), default=0.0
                ),
                "mean_relevance_score": sum(score for _, score in result.relevance_scores)
                / len(result.relevance_scores)
                if result.relevance_scores
                else 0.0,
            }
        )

    except ModelNotReadyException as e:
        error = ToolExecutionError("compress", str(e), {"retry_after_seconds": 30})
        logger.warning(f"Model not ready: {e}")
        return _json(error.to_dict(), indent=False)
    except ReasonGuardException as e:
        error = ToolExecutionError("compress", str(e))
        logger.error(f"Compression failed: {e}")
        return _json(error.to_dict(), indent=False)
    except Exception as e:
        error = ToolExecutionError("compress", str(e), {"type": type(e).__name__})
        logger.error(f"Unexpected error: {e}")
        return _json(error.to_dict(), indent=False)


# =============================================================================
# TOOL 3: STATUS
# =============================================================================


@mcp.tool
async def status(
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Get server status or specific session status.

    Args:
        session_id: Optional session ID to get specific session status

    Returns:
        JSON with server info and session counts, or specific session state

    """
    try:
        manager = get_unified_manager()

        # If session_id provided, return that session's status
        if session_id:
            try:
                result = await manager.get_status(session_id)
                return _json(result)
            except SessionNotFoundError:
                return _json({"error": f"Session not found: {session_id}"}, indent=False)

        # Otherwise return server status
        model_manager = ModelManager.get_instance()
        model_status = model_manager.get_status()
        rate_limiter = get_rate_limiter()

        # Count sessions by status (eventual consistency is fine for status)
        active_count = 0
        completed_count = 0
        for session in manager.get_all_sessions_snapshot().values():
            if session.status == SessionStatus.ACTIVE:
                active_count += 1
            elif session.status == SessionStatus.COMPLETED:
                completed_count += 1

        status_result: dict[str, Any] = {
            "server": {
                "name": SERVER_NAME,
                "transport": SERVER_TRANSPORT,
                "tools": [
                    # Guidance mode
                    "think",
                    "compress",
                    "status",
                    "paradigm_hint",
                    # Enforcement mode (atomic router)
                    "initialize_reasoning",
                    "submit_step",
                    "create_branch",
                    "verify_claims",
                    "router_status",
                ],
                "version": "2.0.0",  # Unified reasoner version
            },
            "model": model_status,
            "embedding_model": _get_embedding_model_name(),
            "features": {
                "auto_mode_selection": True,
                "blind_spot_detection": True,
                "rlvr_rewards": True,
                "mppa_integration": True,
                "cisc_integration": True,
                "rag_enabled": ENABLE_RAG,
            },
            "sessions": {
                "total": active_count + completed_count,
                "active": active_count,
                "completed": completed_count,
                "max_total": MAX_TOTAL_SESSIONS,
            },
            # Backwards compatibility alias
            "active_sessions": {
                "total": active_count + completed_count,
                "active": active_count,
                "completed": completed_count,
                "max_total": MAX_TOTAL_SESSIONS,
            },
            "rate_limit": rate_limiter.get_stats(),
            "ip_rate_limit": {
                "enabled": IP_RATE_LIMIT_ENABLED,
                **get_ip_rate_limiter().get_stats(),
            },
            "cleanup": {
                "max_age_minutes": SESSION_MAX_AGE_MINUTES,
                "interval_seconds": CLEANUP_INTERVAL_SECONDS,
                "task_running": _cleanup_task is not None and not _cleanup_task.done(),
            },
        }

        if ctx:
            state = model_status.get("state", "unknown")
            await ctx.info(f"Server ready, model state: {state}")

        return _json(status_result)

    except Exception as e:
        error = ToolExecutionError("status", str(e))
        logger.error(f"Status check failed: {e}")
        return _json(error.to_dict(), indent=False)


# =============================================================================
# TOOL 4: PARADIGM_HINT (Recommend guidance vs enforcement)
# =============================================================================

# Keywords that suggest enforcement mode is better
_ENFORCEMENT_KEYWORDS = frozenset(
    [
        # Logical/mathematical rigor
        "prove",
        "proof",
        "theorem",
        "lemma",
        "qed",
        "therefore",
        "hence",
        "implies",
        "contradiction",
        "contrapositive",
        "induction",
        # Paradoxes and traps
        "paradox",
        "monty hall",
        "birthday problem",
        "simpson",
        "base rate",
        "gambler's fallacy",
        "conjunction fallacy",
        "survivorship",
        # Formal reasoning
        "valid",
        "invalid",
        "sound",
        "unsound",
        "fallacy",
        "syllogism",
        "modus ponens",
        "modus tollens",
        "deduction",
        "inference",
        # High-stakes verification
        "verify",
        "certify",
        "audit",
        "compliance",
        "safety-critical",
        # Specific problem types
        "probability",
        "conditional probability",
        "bayes",
    ]
)

# Keywords that suggest guidance mode is better
_GUIDANCE_KEYWORDS = frozenset(
    [
        # Creative/exploratory
        "brainstorm",
        "explore",
        "ideas",
        "creative",
        "design",
        "imagine",
        "possibilities",
        "alternatives",
        "options",
        "suggest",
        # Analysis/understanding
        "explain",
        "summarize",
        "analyze",
        "compare",
        "contrast",
        "describe",
        "understand",
        "clarify",
        "interpret",
        "review",
        # Iterative work
        "debug",
        "troubleshoot",
        "fix",
        "refactor",
        "improve",
        "optimize",
        "iterate",
        "prototype",
        "draft",
        "sketch",
        # Open-ended
        "discuss",
        "opinion",
        "perspective",
        "viewpoint",
        "thoughts",
        "consider",
        "reflect",
        "ponder",
    ]
)


def _analyze_problem_for_paradigm(problem: str) -> dict[str, Any]:
    """Analyze a problem statement and recommend guidance vs enforcement.

    Returns dict with recommendation, confidence, and reasoning.
    """
    problem_lower = problem.lower()

    # Count keyword matches
    enforcement_matches = [kw for kw in _ENFORCEMENT_KEYWORDS if kw in problem_lower]
    guidance_matches = [kw for kw in _GUIDANCE_KEYWORDS if kw in problem_lower]

    # Check for question patterns
    has_prove_pattern = any(
        p in problem_lower
        for p in [
            "prove that",
            "prove the",
            "show that",
            "demonstrate that",
            "verify that",
            "confirm that",
            "is it true that",
        ]
    )
    has_open_pattern = any(
        p in problem_lower
        for p in [
            "what are",
            "how might",
            "could you",
            "help me",
            "i want to",
            "let's",
            "can we",
            "what if",
            "why does",
            "how does",
        ]
    )

    # Calculate scores
    enforcement_score = len(enforcement_matches) * 2 + (3 if has_prove_pattern else 0)
    guidance_score = len(guidance_matches) * 2 + (3 if has_open_pattern else 0)

    # Determine recommendation
    total = enforcement_score + guidance_score
    if total == 0:
        # No strong signals - default to guidance (more flexible)
        return {
            "recommendation": "guidance",
            "confidence": 0.5,
            "reasoning": "No strong indicators found. Guidance mode is more flexible for general problems.",
            "enforcement_signals": [],
            "guidance_signals": [],
        }

    if enforcement_score > guidance_score:
        confidence = min(0.95, 0.6 + (enforcement_score - guidance_score) * 0.1)
        return {
            "recommendation": "enforcement",
            "confidence": round(confidence, 2),
            "reasoning": (
                "Problem contains formal reasoning indicators. "
                "Enforcement mode prevents premature conclusions."
            ),
            "enforcement_signals": enforcement_matches[:5],  # Top 5
            "guidance_signals": guidance_matches[:3],
        }
    else:
        confidence = min(0.95, 0.6 + (guidance_score - enforcement_score) * 0.1)
        return {
            "recommendation": "guidance",
            "confidence": round(confidence, 2),
            "reasoning": (
                "Problem appears exploratory or creative. Guidance mode allows flexible iteration."
            ),
            "enforcement_signals": enforcement_matches[:3],
            "guidance_signals": guidance_matches[:5],
        }


@mcp.tool
async def paradigm_hint(
    problem: str,
    ctx: Context | None = None,
) -> str:
    """Analyze a problem and recommend guidance vs enforcement paradigm.

    Call this before starting reasoning to get a recommendation on which
    paradigm (think tool vs atomic router) is better suited for your problem.

    Args:
        problem: The problem statement to analyze

    Returns:
        JSON with:
        - recommendation: "guidance" or "enforcement"
        - confidence: 0.0-1.0 confidence in recommendation
        - reasoning: Why this paradigm was recommended
        - enforcement_signals: Keywords suggesting enforcement
        - guidance_signals: Keywords suggesting guidance
        - suggested_tools: Which tools to use

    Example:
        >>> paradigm_hint("Prove the Monty Hall solution mathematically")
        {
            "recommendation": "enforcement",
            "confidence": 0.85,
            "reasoning": "Problem contains formal reasoning indicators...",
            "suggested_tools": ["initialize_reasoning", "submit_step", ...]
        }

    """
    try:
        if not problem or not problem.strip():
            return _json({"error": "problem is required"}, indent=False)

        result = _analyze_problem_for_paradigm(problem)

        # Add suggested tools based on recommendation
        if result["recommendation"] == "enforcement":
            result["suggested_tools"] = [
                "initialize_reasoning",
                "submit_step",
                "create_branch",
                "verify_claims",
            ]
            result["workflow_hint"] = (
                "1. initialize_reasoning(problem, complexity='auto')\n"
                "2. submit_step(id, 'premise', ..., confidence)\n"
                "3. submit_step(id, 'hypothesis', ..., confidence)\n"
                "4. If BRANCH_REQUIRED: create_branch(id, alternatives)\n"
                "5. submit_step(id, 'verification', ..., confidence)\n"
                "6. submit_step(id, 'synthesis', ..., confidence)"
            )
        else:
            result["suggested_tools"] = ["think", "compress"]
            result["workflow_hint"] = (
                "1. think(action='start', problem=...)\n"
                "2. think(action='continue', session_id, thought=...)\n"
                "3. Repeat step 2 as needed\n"
                "4. think(action='finish', session_id, thought=...)"
            )

        if ctx:
            rec = result["recommendation"]
            conf = result["confidence"]
            await ctx.info(f"Paradigm hint: {rec} (confidence={conf:.0%})")

        return _json(result)

    except Exception as e:
        error = ToolExecutionError("paradigm_hint", str(e))
        logger.error(f"Paradigm hint failed: {e}")
        return _json(error.to_dict(), indent=False)


# =============================================================================
# ATOMIC ROUTER TOOLS (New enforced reasoning system)
# =============================================================================

from src.tools.atomic_router import (  # noqa: E402
    create_branch as router_create_branch,
)
from src.tools.atomic_router import (  # noqa: E402
    get_router_stats,
)
from src.tools.atomic_router import (  # noqa: E402
    get_session_state as router_get_session_state,
)
from src.tools.atomic_router import (  # noqa: E402
    initialize_reasoning as router_initialize,
)
from src.tools.atomic_router import (  # noqa: E402
    submit_atomic_step as router_submit_step,
)
from src.tools.atomic_router import (  # noqa: E402
    verify_claims as router_verify_claims,
)


@mcp.tool
async def initialize_reasoning(
    problem: str,
    complexity: Literal["low", "medium", "high", "auto"] = "auto",
    ctx: Context | None = None,
) -> str:
    """Initialize a new enforced reasoning session.

    Unlike the 'think' tool which guides, this ENFORCES reasoning discipline
    through rejection. Steps that violate rules are REJECTED, not warned.

    Args:
        problem: The problem to reason about (required)
        complexity: Complexity level - determines min/max steps and confidence threshold
            - low: 2-5 steps, 60% confidence threshold
            - medium: 4-8 steps, 70% confidence threshold
            - high: 6-12 steps, 75% confidence threshold
            - auto: Detect from problem keywords (default)

    Returns:
        JSON with session_id, complexity, constraints, and trap warnings

    Example:
        >>> initialize_reasoning("Explain the Monty Hall paradox", "auto")
        {
            "session_id": "abc123",
            "complexity": "high",
            "min_steps": 6,
            "max_steps": 12,
            "confidence_threshold": 0.75,
            "guidance": "TRAP WARNING: Monty Hall problem..."
        }

    """
    try:
        # V-003: Check IP-based rate limit
        ip_allowed, ip_error = await check_ip_rate_limit()
        if not ip_allowed:
            return _json(ip_error, indent=False)

        # Validate input size
        validation_error = _validate_input_sizes(problem=problem)
        if validation_error:
            return _json(validation_error, indent=False)

        store = get_metrics_store()
        with store.trace("initialize_reasoning") as span:
            span.set_attribute("complexity", complexity)
            span.set_attribute("problem_length", len(problem))

            result = router_initialize(problem=problem, complexity=complexity)

            span.set_attribute("session_id", result.session_id)
            span.set_attribute("detected_complexity", result.complexity)
            store.record_session_start(
                session_id=result.session_id,
                problem=problem[:500],
                mode="enforcement",
                complexity=result.complexity,
            )

        if ctx:
            await ctx.info(
                f"Initialized router session {result.session_id} "
                f"(complexity={result.complexity}, min={result.min_steps}, max={result.max_steps})"
            )
            if "TRAP WARNING" in result.guidance:
                await ctx.warning("Reasoning trap detected - see guidance")

        return _json(result.model_dump())

    except Exception as e:
        error = ToolExecutionError("initialize_reasoning", str(e))
        logger.error(f"Initialize reasoning failed: {e}")
        return _json(error.to_dict(), indent=False)


@mcp.tool
async def submit_step(
    session_id: str,
    step_type: Literal["premise", "hypothesis", "verification", "synthesis"],
    content: str,
    confidence: float,
    evidence: list[str] | None = None,
    ctx: Context | None = None,
) -> str:
    """Submit a single reasoning step. May be REJECTED.

    The router enforces reasoning discipline through 5 rules:
    - Rule A: Cannot synthesize until min_steps reached
    - Rule B: Low confidence requires branching (BRANCH_REQUIRED)
    - Rule C: Cannot synthesize without verification (VERIFICATION_REQUIRED)
    - Rule D: Must follow state machine transitions
    - Rule E: Must synthesize at max_steps

    State machine transitions:
        (start) -> premise -> hypothesis -> verification -> synthesis
                    ↓           ↓              ↓
                 premise    hypothesis      hypothesis

    Args:
        session_id: Session ID from initialize_reasoning
        step_type: Type of reasoning step
            - premise: Establish facts/assumptions
            - hypothesis: Form testable hypothesis
            - verification: Test hypothesis with evidence
            - synthesis: Final conclusion (terminal)
        content: Your reasoning content
        confidence: Your confidence in this step (0.0-1.0, required!)
        evidence: Optional evidence supporting this step

    Returns:
        JSON with status (ACCEPTED/REJECTED/BRANCH_REQUIRED/VERIFICATION_REQUIRED),
        rejection_reason if rejected, and next valid steps

    Example:
        >>> submit_step("abc123", "premise", "There are 3 doors", 0.95)
        {"status": "ACCEPTED", "step_id": "step-1", "valid_next_steps": ["premise", "hypothesis"], ...}

        >>> submit_step("abc123", "synthesis", "Answer is X", 0.9)
        {"status": "REJECTED", "rejection_reason": "Need 4 more reasoning steps", ...}

    """
    try:
        # V-003: Check IP-based rate limit
        ip_allowed, ip_error = await check_ip_rate_limit()
        if not ip_allowed:
            return _json(ip_error, indent=False)

        # Validate input size
        validation_error = _validate_input_sizes(thought=content)
        if validation_error:
            return _json(validation_error, indent=False)

        result = router_submit_step(
            session_id=session_id,
            step_type=step_type,
            content=content,
            confidence=confidence,
            evidence=evidence,
        )

        if ctx:
            if result.status.value == "ACCEPTED":
                await ctx.info(
                    f"Step accepted: {step_type} (steps={result.steps_taken}, "
                    f"remaining={result.steps_remaining})"
                )
            elif result.status.value == "REJECTED":
                await ctx.warning(f"Step REJECTED: {result.rejection_reason}")
            elif result.status.value == "BRANCH_REQUIRED":
                await ctx.warning(
                    f"BRANCH_REQUIRED: confidence {confidence:.2f} too low. "
                    "Use create_branch() with 2-4 alternatives."
                )
            elif result.status.value == "VERIFICATION_REQUIRED":
                await ctx.warning(
                    "VERIFICATION_REQUIRED: Add a verification step before synthesis."
                )

            if result.can_synthesize:
                await ctx.info("Session ready for synthesis")
            elif result.synthesis_blockers:
                await ctx.info(f"Synthesis blockers: {', '.join(result.synthesis_blockers)}")

        return _json(result.model_dump())

    except Exception as e:
        error = ToolExecutionError("submit_step", str(e))
        logger.error(f"Submit step failed: {e}")
        return _json(error.to_dict(), indent=False)


@mcp.tool
async def create_branch(
    session_id: str,
    alternatives: list[str],
    ctx: Context | None = None,
) -> str:
    """Create alternative reasoning branches.

    Required when submit_step returns BRANCH_REQUIRED (confidence below threshold).
    Creates 2-4 alternative hypotheses to explore in parallel.

    After branching:
    1. Submit verification steps for each hypothesis
    2. Compare evidence and confidence across branches
    3. Select the hypothesis with strongest support
    4. Continue reasoning from that branch

    Args:
        session_id: Session ID from initialize_reasoning
        alternatives: 2-4 alternative hypotheses to explore

    Returns:
        JSON with branch_ids and guidance on how to proceed

    Example:
        >>> create_branch("abc123", [
        ...     "Probability stays 1/3 for original door",
        ...     "Probability becomes 1/2 for both doors"
        ... ])
        {"branch_ids": ["br-1", "br-2"], "guidance": "..."}

    """
    try:
        # V-003: Check IP-based rate limit
        ip_allowed, ip_error = await check_ip_rate_limit()
        if not ip_allowed:
            return _json(ip_error, indent=False)

        result = router_create_branch(session_id=session_id, alternatives=alternatives)

        if ctx:
            if result.branch_ids:
                await ctx.info(f"Created {len(result.branch_ids)} branches")
            else:
                await ctx.warning(f"Branch creation failed: {result.guidance}")

        return _json(result.model_dump())

    except Exception as e:
        error = ToolExecutionError("create_branch", str(e))
        logger.error(f"Create branch failed: {e}")
        return _json(error.to_dict(), indent=False)


@mcp.tool
async def verify_claims(
    session_id: str,
    claims: list[str],
    evidence: list[str],
    ctx: Context | None = None,
) -> str:
    """Verify claims against evidence and check for contradictions.

    Use this before synthesis to ensure your conclusions are supported.
    Detects logical contradictions between claims and missing evidence.

    Args:
        session_id: Session ID from initialize_reasoning
        claims: List of claims to verify
        evidence: List of evidence to check against

    Returns:
        JSON with verified claims, contradictions, missing evidence,
        and whether synthesis is now allowed

    Example:
        >>> verify_claims("abc123",
        ...     claims=["Switching wins 2/3", "Staying wins 1/3"],
        ...     evidence=["By Bayes theorem...", "Initial probability 1/3..."]
        ... )
        {
            "verified": ["Switching wins 2/3", "Staying wins 1/3"],
            "contradictions": [],
            "missing_evidence": [],
            "can_synthesize": true
        }

    """
    try:
        # V-003: Check IP-based rate limit
        ip_allowed, ip_error = await check_ip_rate_limit()
        if not ip_allowed:
            return _json(ip_error, indent=False)

        result = router_verify_claims(session_id=session_id, claims=claims, evidence=evidence)

        if ctx:
            if result.verified:
                await ctx.info(f"Verified {len(result.verified)} claim(s)")
            if result.contradictions:
                await ctx.warning(f"Detected {len(result.contradictions)} contradiction(s)!")
            if result.missing_evidence:
                await ctx.warning(f"Missing evidence for {len(result.missing_evidence)} claim(s)")
            if result.can_synthesize:
                await ctx.info("All checks passed - ready for synthesis")

        return _json(result.model_dump())

    except Exception as e:
        error = ToolExecutionError("verify_claims", str(e))
        logger.error(f"Verify claims failed: {e}")
        return _json(error.to_dict(), indent=False)


@mcp.tool
async def router_status(
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Get atomic router status or session state.

    Args:
        session_id: Optional session ID for specific session state

    Returns:
        JSON with router stats or session state

    """
    try:
        if session_id:
            state = router_get_session_state(session_id)
            if state is None:
                return _json(
                    {"error": f"Session '{session_id}' not found or expired"}, indent=False
                )
            return _json(state)

        # Return global router stats
        stats = get_router_stats()
        return _json(
            {
                "router": {
                    "name": "Atomic Reasoning Router",
                    "tools": [
                        "initialize_reasoning",
                        "submit_step",
                        "create_branch",
                        "verify_claims",
                        "router_status",
                    ],
                    "rules": [
                        "A: Minimum depth",
                        "B: Confidence branching",
                        "C: Verification required",
                        "D: State machine",
                        "E: Maximum steps",
                    ],
                },
                "sessions": stats,
            }
        )

    except Exception as e:
        error = ToolExecutionError("router_status", str(e))
        logger.error(f"Router status failed: {e}")
        return _json(error.to_dict(), indent=False)


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================


async def _init_unified_manager_async() -> None:
    """Initialize the unified manager with vector store and encoder for semantic scoring."""
    vector_store = None
    encoder = None

    # Initialize encoder for semantic scoring
    try:
        from src.models.context_encoder import ContextEncoder, EncoderConfig

        encoder_config = EncoderConfig(model_name=_get_embedding_model_name())
        encoder = ContextEncoder(encoder_config)
        logger.info(f"Initialized context encoder with {encoder_config.model_name}")
    except Exception as e:
        logger.warning(f"Failed to initialize context encoder: {e}")
        encoder = None

    if ENABLE_RAG:
        try:
            from src.models.vector_store import AsyncVectorStore, VectorStoreConfig

            # Validate path before use (CWE-22 prevention)
            validated_path = _validate_vector_db_path()
            config = VectorStoreConfig(db_path=validated_path)
            vector_store = AsyncVectorStore(config)
            await vector_store.__aenter__()
            logger.info(f"RAG enabled with vector store at {validated_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize vector store, RAG disabled: {e}")
            vector_store = None

    await init_unified_manager(
        vector_store=vector_store,
        encoder=encoder,
        enable_semantic_scoring=encoder is not None,
    )
    logger.info(
        f"Unified reasoner manager initialized "
        f"(semantic_scoring={encoder is not None}, rag={vector_store is not None})"
    )


def main() -> None:
    """Run the Reason Guard MCP server."""
    logger.info(f"Starting {SERVER_NAME} (transport: {SERVER_TRANSPORT})")

    # Initialize API key authentication (V-001: CWE-306)
    _init_api_keys()
    if AUTH_ENABLED:
        logger.info(f"API key authentication enabled ({len(_API_KEY_HASHES)} key(s) loaded)")
    else:
        logger.warning("API key authentication disabled - enable with AUTH_ENABLED=true")

    # Initialize embedding model
    _init_model_manager()

    # Initialize unified manager (sync wrapper for async init)
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_init_unified_manager_async())
        loop.close()
    except Exception as e:
        logger.warning(f"Async init failed, using default manager: {e}")
        # Fallback to sync initialization without vector store
        get_unified_manager()

    # Start background cleanup task
    _start_cleanup_task()

    try:
        if SERVER_TRANSPORT == "stdio":
            mcp.run(transport="stdio")
        elif SERVER_TRANSPORT == "http":
            mcp.run(transport="streamable-http", host=SERVER_HOST, port=SERVER_PORT)
        elif SERVER_TRANSPORT == "sse":
            mcp.run(transport="sse", host=SERVER_HOST, port=SERVER_PORT)
        else:
            logger.warning(f"Unknown transport '{SERVER_TRANSPORT}', falling back to stdio")
            mcp.run(transport="stdio")
    finally:
        _stop_cleanup_task()


if __name__ == "__main__":
    main()
