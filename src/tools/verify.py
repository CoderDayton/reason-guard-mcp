"""Fact verification tool for quality assurance.

Verifies generated answers against context by extracting and
checking individual claims. Helps prevent hallucinations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.utils.errors import VerificationException
from src.utils.retry import retry_with_backoff
from src.utils.schema import VerificationResult

if TYPE_CHECKING:
    from src.models.llm_client import LLMClient


class FactVerificationTool:
    """Fact consistency verification for generated answers.

    Extracts factual claims from an answer and verifies each
    against the provided context. Returns verification confidence
    and details.

    Attributes:
        llm: LLM client for claim extraction and verification.

    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize verification tool.

        Args:
            llm_client: LLM client for text generation.

        """
        self.llm = llm_client

    @retry_with_backoff(max_attempts=2, base_delay=1.0)
    def verify(
        self,
        answer: str,
        context: str,
        max_claims: int = 10,
        confidence_threshold: float = 0.7,
    ) -> VerificationResult:
        """Verify answer claims against context.

        Algorithm:
        1. Extract factual claims from answer
        2. For each claim, check if supported by context
        3. Calculate overall confidence and verification status

        Args:
            answer: Answer text to verify.
            context: Factual context for verification.
            max_claims: Maximum claims to extract and verify.
            confidence_threshold: Threshold for "verified" status (0-1).

        Returns:
            VerificationResult with verification status and details.

        Raises:
            VerificationException: If verification fails due to invalid input.

        Example:
            >>> tool = FactVerificationTool(llm_client)
            >>> result = tool.verify(
            ...     answer="Einstein published relativity in 1905.",
            ...     context="Albert Einstein published special relativity in 1905..."
            ... )
            >>> print(result.verified)  # True
            >>> print(result.confidence)  # 0.9

        """
        # Validate inputs
        if not answer or not answer.strip():
            raise VerificationException("Answer cannot be empty")

        if not context or not context.strip():
            raise VerificationException("Context cannot be empty")

        if not 1 <= max_claims <= 20:
            raise VerificationException(f"max_claims must be 1-20, got {max_claims}")

        try:
            # Extract claims from answer
            claims = self._extract_claims(answer, max_claims)

            if not claims:
                return VerificationResult(
                    verified=True,
                    confidence=0.5,
                    claims_verified=0,
                    claims_total=0,
                    reason="No verifiable claims found in answer",
                    claim_details=[],
                )

            # Verify each claim
            claim_details: list[dict[str, Any]] = []
            verified_count = 0

            for claim in claims:
                result = self._verify_claim(claim, context)
                claim_details.append(result)

                if result.get("supported", False):
                    verified_count += 1

            # Calculate confidence
            confidence = verified_count / len(claims) if claims else 0.5
            is_verified = confidence >= confidence_threshold

            return VerificationResult(
                verified=is_verified,
                confidence=confidence,
                claims_verified=verified_count,
                claims_total=len(claims),
                reason=f"Verified {verified_count}/{len(claims)} claims against context",
                claim_details=claim_details,
            )

        except VerificationException:
            raise
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            raise VerificationException(f"Verification failed: {e!s}") from e

    def _extract_claims(self, text: str, max_claims: int) -> list[str]:
        """Extract factual claims from text.

        Uses LLM to identify distinct factual statements that can be verified.

        Args:
            text: Text to extract claims from.
            max_claims: Maximum number of claims.

        Returns:
            List of claim strings.

        """
        try:
            prompt = f"""Extract the key FACTUAL CLAIMS from this text that can be verified.
Each claim should be a single, specific statement of fact.

Text: {text}

List up to {max_claims} factual claims, one per line (just the claim, no numbering):"""

            response = self.llm.generate(prompt, max_tokens=500, temperature=0.3)

            # Parse response into claims
            lines = response.strip().split("\n")
            claims = []

            for line in lines:
                # Clean up line
                claim = line.strip().lstrip("0123456789.-) ").strip()
                if claim and len(claim.split()) >= 3:  # At least 3 words
                    claims.append(claim)

                if len(claims) >= max_claims:
                    break

            return claims

        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            # Fallback: simple sentence splitting
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            return [s for s in sentences if len(s.split()) >= 5][:max_claims]

    def _verify_claim(self, claim: str, context: str) -> dict[str, Any]:
        """Verify a single claim against context.

        Args:
            claim: Claim to verify.
            context: Context to check against.

        Returns:
            Dict with claim, supported (bool), confidence, and reason.

        """
        try:
            prompt = f"""Is this claim supported by the given context?

CLAIM: {claim}

CONTEXT: {context[:1500]}{"..." if len(context) > 1500 else ""}

Answer with:
- SUPPORTED: if the context clearly supports this claim
- CONTRADICTED: if the context contradicts this claim
- UNCLEAR: if the context doesn't contain relevant information

Then briefly explain why (1 sentence):"""

            response = self.llm.generate(prompt, max_tokens=100, temperature=0.2)
            response_lower = response.lower()

            # Parse response
            if "supported" in response_lower[:50]:
                supported = True
                confidence = 0.9
            elif "contradicted" in response_lower[:50]:
                supported = False
                confidence = 0.9
            else:
                supported = False
                confidence = 0.5

            return {
                "claim": claim[:100] + "..." if len(claim) > 100 else claim,
                "supported": supported,
                "confidence": confidence,
                "reason": response[:150] if response else "No explanation",
            }

        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return {
                "claim": claim[:100],
                "supported": False,
                "confidence": 0.5,
                "reason": f"Verification error: {e!s}",
            }
