"""Type definitions and schemas for Reason Guard MCP."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReasoningStrategy(str, Enum):
    """Recommended reasoning strategy based on problem type."""

    LONG_CHAIN = "long_chain"
    MATRIX = "matrix"
    PARALLEL = "parallel_voting"


@dataclass
class CompressionResult:
    """Result from prompt compression."""

    compressed_context: str
    compression_ratio: float
    original_tokens: int
    compressed_tokens: int
    sentences_kept: int
    sentences_removed: int
    relevance_scores: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "compressed_context": self.compressed_context,
            "compression_ratio": round(self.compression_ratio, 3),
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.original_tokens - self.compressed_tokens,
            "sentences_kept": self.sentences_kept,
            "sentences_removed": self.sentences_removed,
            "top_relevance_scores": [
                {
                    "sentence": s[:80] + "..." if len(s) > 80 else s,
                    "score": round(sc, 3),
                }
                for s, sc in self.relevance_scores[:5]
            ],
        }


@dataclass
class ReasoningResult:
    """Result from reasoning operation."""

    answer: str
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)
    verification_results: dict[str, Any] | None = None
    reasoning_trace: dict[str, Any] | None = None
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "num_reasoning_steps": len(self.reasoning_steps),
            "reasoning_steps": self.reasoning_steps[:5],
            "tokens_used": self.tokens_used,
            "has_verification": self.verification_results is not None,
            "verification_results": self.verification_results,
            "reasoning_trace": self.reasoning_trace,
        }


@dataclass
class StrategyRecommendation:
    """Strategy recommendation for a problem."""

    strategy: ReasoningStrategy
    estimated_depth: int
    estimated_tokens: int
    expressiveness_guarantee: bool
    confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "recommended_strategy": self.strategy.value,
            "estimated_depth_steps": self.estimated_depth,
            "estimated_tokens_needed": self.estimated_tokens,
            "expressiveness_guarantee": self.expressiveness_guarantee,
            "strategy_confidence": round(self.confidence, 3),
            "explanation": self.reasoning,
        }


@dataclass
class VerificationResult:
    """Result from fact verification."""

    verified: bool
    confidence: float
    claims_verified: int
    claims_total: int
    reason: str
    claim_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "verified": self.verified,
            "confidence": round(self.confidence, 3),
            "claims_verified": self.claims_verified,
            "claims_total": self.claims_total,
            "verification_percentage": (
                round(self.claims_verified / self.claims_total * 100, 1)
                if self.claims_total > 0
                else 0.0
            ),
            "reason": self.reason,
            "recommendation": "RELIABLE" if self.verified else "REVIEW NEEDED",
            "claim_details": self.claim_details[:5],
        }


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize objects to JSON.

    Args:
        obj: Object to serialize. Can be a dataclass with to_dict() method,
             a dict, or any JSON-serializable object.

    Returns:
        JSON string representation of the object.

    """
    try:
        if hasattr(obj, "to_dict"):
            return json.dumps(obj.to_dict(), indent=2, ensure_ascii=False)
        return json.dumps(obj, default=str, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {e!s}", "type": type(obj).__name__})
