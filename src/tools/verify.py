"""Fact verification state manager.

Implements a state management tool for tracking claim verification.
The calling LLM does all reasoning; this tool tracks claims,
verification status, and provides consistency analysis.

Architecture:
    - Tool receives claims and verifications FROM the calling LLM
    - Tracks claim status (pending, verified, contradicted, unclear)
    - Provides heuristic consistency checks
    - Returns verification summary
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from src.utils.session import SessionManager


class ClaimStatus(Enum):
    """Status of a claim verification."""

    PENDING = "pending"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNCLEAR = "unclear"


class SessionStatus(Enum):
    """Status of a verification session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class Claim:
    """A factual claim to verify."""

    claim_id: int
    content: str
    status: ClaimStatus = ClaimStatus.PENDING
    evidence: str | None = None
    confidence: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim_id": self.claim_id,
            "content": self.content,
            "status": self.status.value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VerificationState:
    """State of a verification session."""

    session_id: str
    answer: str
    context: str
    claims: list[Claim] = field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        verified = sum(1 for c in self.claims if c.status == ClaimStatus.SUPPORTED)
        contradicted = sum(1 for c in self.claims if c.status == ClaimStatus.CONTRADICTED)
        pending = sum(1 for c in self.claims if c.status == ClaimStatus.PENDING)

        return {
            "session_id": self.session_id,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "context_length": len(self.context),
            "claims": [c.to_dict() for c in self.claims],
            "status": self.status.value,
            "summary": {
                "total_claims": len(self.claims),
                "supported": verified,
                "contradicted": contradicted,
                "pending": pending,
                "unclear": len(self.claims) - verified - contradicted - pending,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class VerificationManager(SessionManager[VerificationState]):
    """Manages fact verification sessions.

    This is a STATE MANAGER, not a verifier. The calling LLM extracts
    claims and verifies them; this tool tracks the verification state.

    Example usage flow:
        1. Agent calls: start_verification(answer="...", context="...")
        2. Agent extracts claims: add_claim(content="Einstein was born in 1879")
        3. Agent verifies each: verify_claim(claim_id=0, status="supported", evidence="...")
        4. Agent finalizes: finalize()

    """

    def __init__(self) -> None:
        """Initialize the verification manager."""
        super().__init__()

    def start_verification(
        self,
        answer: str,
        context: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new verification session.

        Args:
            answer: The answer to verify.
            context: The context to verify against.
            metadata: Optional metadata.

        Returns:
            Session info and guidance.

        """
        session_id = str(uuid.uuid4())[:8]
        state = VerificationState(
            session_id=session_id,
            answer=answer,
            context=context,
            metadata=metadata or {},
        )
        self._register_session(session_id, state)

        # Provide heuristic claim suggestions
        suggested_claims = self._suggest_claims(answer)

        logger.debug(f"Started verification session {session_id}")

        return {
            "session_id": session_id,
            "status": "started",
            "answer_length": len(answer),
            "context_length": len(context),
            "suggested_claims": suggested_claims,
            "instruction": (
                "Extract factual claims from the answer. "
                "Call add_claim() for each distinct factual assertion, "
                "then verify_claim() for each against the context."
            ),
        }

    def add_claim(
        self,
        session_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Add a claim to verify.

        Args:
            session_id: Session to add claim to.
            content: The factual claim to verify.

        Returns:
            Updated state with claim ID.

        """
        with self.session(session_id) as state:
            if state.status != SessionStatus.ACTIVE:
                return {"error": f"Session is {state.status.value}"}

            claim_id = len(state.claims)
            claim = Claim(claim_id=claim_id, content=content)
            state.claims.append(claim)
            state.updated_at = datetime.now()

            # Check for potential issues
            issues = self._check_claim(state, claim)

            response = {
                "session_id": session_id,
                "claim_id": claim_id,
                "claim": content,
                "total_claims": len(state.claims),
                "pending_verification": sum(
                    1 for c in state.claims if c.status == ClaimStatus.PENDING
                ),
            }

            if issues:
                response["warnings"] = issues

            response["instruction"] = (
                f"Verify claim {claim_id} against the context. "
                f"Call verify_claim(claim_id={claim_id}, "
                "status='supported'|'contradicted'|'unclear', "
                "evidence='relevant text from context')."
            )

            logger.debug(f"Added claim {claim_id} to session {session_id}")

            return response

    def verify_claim(
        self,
        session_id: str,
        claim_id: int,
        status: str,
        evidence: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Verify a claim.

        Args:
            session_id: Session containing the claim.
            claim_id: ID of the claim to verify.
            status: Verification status (supported, contradicted, unclear).
            evidence: Supporting/contradicting evidence from context.
            confidence: Confidence in the verification (0-1).

        Returns:
            Updated verification state.

        """
        with self.session(session_id) as state:
            if state.status != SessionStatus.ACTIVE:
                return {"error": f"Session is {state.status.value}"}

            if claim_id < 0 or claim_id >= len(state.claims):
                return {"error": f"Invalid claim_id {claim_id}"}

            # Update claim
            try:
                claim_status = ClaimStatus(status.lower())
            except ValueError:
                return {
                    "error": f"Invalid status '{status}'. Use: supported, contradicted, unclear"
                }

            claim = state.claims[claim_id]
            claim.status = claim_status
            claim.evidence = evidence
            claim.confidence = confidence
            claim.timestamp = datetime.now()
            state.updated_at = datetime.now()

            # Check evidence quality
            issues = []
            if (
                evidence
                and claim_status == ClaimStatus.SUPPORTED
                and evidence.lower() not in state.context.lower()
            ):
                issues.append(
                    "Evidence text not found verbatim in context. "
                    "Ensure you're quoting directly from the provided context."
                )

            # Summary
            pending = sum(1 for c in state.claims if c.status == ClaimStatus.PENDING)
            supported = sum(1 for c in state.claims if c.status == ClaimStatus.SUPPORTED)
            contradicted = sum(1 for c in state.claims if c.status == ClaimStatus.CONTRADICTED)

            response = {
                "session_id": session_id,
                "claim_id": claim_id,
                "status": claim_status.value,
                "summary": {
                    "total": len(state.claims),
                    "supported": supported,
                    "contradicted": contradicted,
                    "pending": pending,
                },
            }

            if issues:
                response["warnings"] = issues

            if pending > 0:
                next_pending = next(c for c in state.claims if c.status == ClaimStatus.PENDING)
                response["instruction"] = (
                    f"{pending} claims still pending. "
                    f"Verify claim {next_pending.claim_id}: '{next_pending.content[:50]}...'"
                )
                response["next_claim"] = {
                    "id": next_pending.claim_id,
                    "content": next_pending.content,
                }
            else:
                response["instruction"] = (
                    "All claims verified. Call finalize() to complete verification."
                )

            logger.debug(
                f"Verified claim {claim_id} as {claim_status.value} in session {session_id}"
            )

            return response

    def get_status(self, session_id: str) -> dict[str, Any]:
        """Get verification status.

        Args:
            session_id: Session to retrieve.

        Returns:
            Full verification state.

        """
        with self.session(session_id) as state:
            return state.to_dict()

    def finalize(self, session_id: str) -> dict[str, Any]:
        """Finalize the verification.

        Args:
            session_id: Session to finalize.

        Returns:
            Final verification result.

        """
        with self.session(session_id) as state:
            # Check for pending claims
            pending = [c for c in state.claims if c.status == ClaimStatus.PENDING]
            if pending:
                return {
                    "error": f"{len(pending)} claims still pending verification",
                    "pending_claims": [{"id": c.claim_id, "content": c.content} for c in pending],
                }

            state.status = SessionStatus.COMPLETED
            state.updated_at = datetime.now()

            # Calculate overall result
            supported = sum(1 for c in state.claims if c.status == ClaimStatus.SUPPORTED)
            contradicted = sum(1 for c in state.claims if c.status == ClaimStatus.CONTRADICTED)
            total = len(state.claims)

            if total == 0:
                overall_confidence = 1.0
                verified = True
            else:
                overall_confidence = supported / total if total > 0 else 0.0
                verified = contradicted == 0 and overall_confidence >= 0.5

            claims_dict = [c.to_dict() for c in state.claims]

        logger.info(f"Finalized verification session {session_id}: {supported}/{total} supported")

        return {
            "session_id": session_id,
            "status": "completed",
            "verified": verified,
            "confidence": round(overall_confidence, 2),
            "summary": {
                "total_claims": total,
                "supported": supported,
                "contradicted": contradicted,
                "unclear": total - supported - contradicted,
            },
            "claims": claims_dict,
            "recommendation": (
                "Answer appears factually consistent with context."
                if verified
                else f"Answer has {contradicted} contradicted claim(s). Review and revise."
            ),
        }

    def abandon(self, session_id: str, reason: str = "") -> dict[str, Any]:
        """Abandon a verification session."""
        with self.session(session_id) as state:
            state.status = SessionStatus.ABANDONED
            state.metadata["abandon_reason"] = reason
            state.updated_at = datetime.now()
            claims_count = len(state.claims)

        return {
            "session_id": session_id,
            "status": "abandoned",
            "reason": reason,
            "claims_added": claims_count,
        }

    def list_sessions(self) -> dict[str, Any]:
        """List all verification sessions."""
        with self.locked() as sessions:
            result = []
            for sid, state in sessions.items():
                result.append(
                    {
                        "session_id": sid,
                        "status": state.status.value,
                        "claims": len(state.claims),
                        "created": state.created_at.isoformat(),
                    }
                )
            return {"sessions": result, "total": len(result)}

    def _suggest_claims(self, answer: str) -> list[str]:
        """Suggest potential claims to extract (heuristic)."""
        suggestions = []

        # Look for sentences with factual indicators
        sentences = re.split(r"[.!?]+", answer)
        factual_patterns = [
            r"\b\d{4}\b",  # Years
            r"\b\d+(?:\.\d+)?%?\b",  # Numbers/percentages
            r"\b(?:is|was|are|were|has|have|had)\b",  # Being/having verbs
            r"\b(?:founded|created|invented|discovered|published)\b",  # Action verbs
            r"\b(?:located|born|died|started|ended)\b",  # State verbs
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum meaningful length
                for pattern in factual_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        suggestions.append(sentence)
                        break

            if len(suggestions) >= 5:  # Limit suggestions
                break

        return suggestions

    def _check_claim(self, state: VerificationState, claim: Claim) -> list[str]:
        """Check claim for potential issues."""
        issues = []
        content = claim.content.lower()

        # Check if claim is in the answer
        if claim.content.lower() not in state.answer.lower():
            # Fuzzy check - at least some overlap
            claim_words = set(content.split())
            answer_words = set(state.answer.lower().split())
            overlap = len(claim_words & answer_words) / len(claim_words) if claim_words else 0
            if overlap < 0.5:
                issues.append(
                    "Claim may not be derived from the answer. "
                    "Ensure you're extracting claims that appear in the answer text."
                )

        # Check for vague claims
        vague_phrases = ["something", "someone", "somewhere", "somehow", "some kind"]
        if any(phrase in content for phrase in vague_phrases):
            issues.append("Claim contains vague language. Extract specific factual assertions.")

        # Check for opinion vs fact
        opinion_phrases = ["i think", "i believe", "probably", "might be", "could be"]
        if any(phrase in content for phrase in opinion_phrases):
            issues.append("Claim appears to be an opinion. Extract verifiable facts only.")

        # Check for duplicate claims
        for existing in state.claims[:-1]:  # Exclude current
            existing_words = set(existing.content.lower().split())
            claim_words = set(content.split())
            if len(existing_words) > 3 and len(claim_words) > 3:
                overlap = len(existing_words & claim_words) / len(existing_words | claim_words)
                if overlap > 0.8:
                    issues.append(
                        f"Claim similar to existing claim {existing.claim_id}. Avoid duplicates."
                    )
                    break

        return issues


# Global instance for session persistence
_verification_manager = VerificationManager()


def get_verification_manager() -> VerificationManager:
    """Get the global verification manager instance."""
    return _verification_manager
