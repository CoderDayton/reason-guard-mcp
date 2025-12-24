"""Utility modules for Reason Guard MCP."""

import warnings

from .complexity import (
    ComplexityLevel,
    ComplexityResult,
    clear_complexity_cache,
    detect_complexity,
    get_cache_stats,
)
from .errors import (
    CompressionException,
    ConfigException,
    LLMException,
    ReasonGuardException,
    ReasoningException,
    ToolExecutionError,
    VerificationException,
)
from .retry import retry_with_backoff
from .schema import (
    CompressionResult,
    ReasoningResult,
    ReasoningStrategy,
    StrategyRecommendation,
    safe_json_serialize,
)
from .scoring import (
    calculate_cell_survival_score,
    calculate_survival_score,
    semantic_survival_score,
)
from .session import SessionManager, SessionNotFoundError
from .session_signing import (
    SessionSigner,
    TokenError,
    TokenExpiredError,
    TokenInvalidError,
    TokenPayload,
    get_session_signer,
    hash_client_id,
    reset_session_signer,
)
from .weight_store import (
    DEFAULT_WEIGHTS,
    DecayConfig,
    DomainWeights,
    WeightStore,
    get_weight_store,
    reset_weight_store,
)


def __getattr__(name: str) -> type[ReasonGuardException]:
    """Lazy attribute access for deprecated names."""
    if name == "MatrixMindException":
        warnings.warn(
            "MatrixMindException is deprecated, use ReasonGuardException instead. "
            "MatrixMindException will be removed in version 1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ReasonGuardException
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ReasoningStrategy",
    "CompressionResult",
    "ReasoningResult",
    "StrategyRecommendation",
    "safe_json_serialize",
    "ReasonGuardException",
    "MatrixMindException",  # Deprecated alias
    "CompressionException",
    "ReasoningException",
    "VerificationException",
    "LLMException",
    "ConfigException",
    "ToolExecutionError",
    "retry_with_backoff",
    "ComplexityLevel",
    "ComplexityResult",
    "detect_complexity",
    "clear_complexity_cache",
    "get_cache_stats",
    "SessionManager",
    "SessionNotFoundError",
    "SessionSigner",
    "TokenError",
    "TokenExpiredError",
    "TokenInvalidError",
    "TokenPayload",
    "get_session_signer",
    "hash_client_id",
    "reset_session_signer",
    "calculate_cell_survival_score",
    "calculate_survival_score",
    "semantic_survival_score",
    "DEFAULT_WEIGHTS",
    "DecayConfig",
    "DomainWeights",
    "WeightStore",
    "get_weight_store",
    "reset_weight_store",
]
