"""Long Chain-of-Thought reasoning tool.

Implements deep sequential reasoning with optional intermediate verification.
Optimized for problems with strong serial dependencies.

Based on paper 2505.21825 - achieves exponential advantage over parallel
methods for serial problems (66% vs 36% on constraint tasks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.utils.errors import ReasoningException
from src.utils.retry import retry_with_backoff
from src.utils.schema import ReasoningResult

if TYPE_CHECKING:
    from src.models.llm_client import LLMClient


class LongChainOfThoughtTool:
    """Long-chain sequential reasoning with verification checkpoints.

    Implements deep reasoning where each step builds fundamentally
    on previous steps. Includes optional intermediate verification
    to catch errors early.

    Best for:
    - Graph connectivity problems
    - Constraint satisfaction
    - Arithmetic with dependencies
    - Proof verification

    Attributes:
        llm: LLM client for generating reasoning steps.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize long chain tool.

        Args:
            llm_client: LLM client for text generation.
        """
        self.llm = llm_client

    @retry_with_backoff(max_attempts=2, base_delay=1.0)
    def reason(
        self,
        problem: str,
        num_steps: int = 15,
        verify_intermediate: bool = True,
        verification_frequency: int = 3,
    ) -> ReasoningResult:
        """Execute long-chain sequential reasoning.

        Algorithm:
        1. Initialize reasoning chain with problem
        2. For each step:
           a. Generate next reasoning step based on previous
           b. Optionally verify step consistency
           c. If verification fails, flag but continue
        3. Extract final answer from chain

        Args:
            problem: Problem statement to solve.
            num_steps: Number of reasoning steps (1-50).
            verify_intermediate: Whether to verify steps periodically.
            verification_frequency: Verify every N steps.

        Returns:
            ReasoningResult with answer, steps, and verification info.

        Raises:
            ReasoningException: If reasoning fails due to invalid parameters
                               or LLM errors.

        Example:
            >>> tool = LongChainOfThoughtTool(llm_client)
            >>> result = tool.reason(
            ...     problem="Make 24 from 3, 4, 5, 6",
            ...     num_steps=10,
            ...     verify_intermediate=True
            ... )
            >>> print(result.answer)
            >>> print(result.verification_results)
        """
        # Validate inputs
        if not problem or not problem.strip():
            raise ReasoningException("Problem cannot be empty")

        if not 1 <= num_steps <= 50:
            raise ReasoningException(f"num_steps must be 1-50, got {num_steps}")

        try:
            # Initialize reasoning chain
            reasoning_chain: list[str] = []
            verifications: list[dict[str, Any]] = []
            verification_passed = 0
            verification_total = 0

            # Generate reasoning steps
            for step_num in range(1, num_steps + 1):
                # Build context from recent steps
                recent_context = self._build_recent_context(reasoning_chain, max_steps=5)

                # Generate next step
                next_step = self._generate_step(
                    problem=problem,
                    step_num=step_num,
                    total_steps=num_steps,
                    recent_context=recent_context,
                )

                if not next_step:
                    logger.warning(f"Step {step_num} generation failed, stopping chain")
                    break

                reasoning_chain.append(next_step)

                # Periodic verification
                if verify_intermediate and step_num % verification_frequency == 0:
                    verification = self._verify_step(
                        problem=problem,
                        chain=reasoning_chain,
                        step_num=step_num,
                    )
                    verifications.append(verification)
                    verification_total += 1

                    if verification.get("is_valid", False):
                        verification_passed += 1
                    else:
                        reason = verification.get("reason", "unknown")
                        logger.warning(f"Step {step_num} verification failed: {reason}")

            # Extract final answer
            final_answer = self._extract_answer(problem, reasoning_chain)

            # Calculate confidence
            if verification_total > 0:
                verification_ratio = verification_passed / verification_total
                confidence = 0.5 + 0.4 * verification_ratio
            else:
                confidence = 0.7  # Default when no verification

            return ReasoningResult(
                answer=final_answer,
                confidence=confidence,
                reasoning_steps=reasoning_chain,
                verification_results={
                    "total_verifications": verification_total,
                    "passed": verification_passed,
                    "failed": verification_total - verification_passed,
                    "details": verifications[-3:] if verifications else [],  # Last 3
                },
                tokens_used=self.llm.estimate_tokens("\n".join(reasoning_chain)),
                reasoning_trace={
                    "total_steps": len(reasoning_chain),
                    "requested_steps": num_steps,
                    "verify_enabled": verify_intermediate,
                },
            )

        except ReasoningException:
            raise
        except Exception as e:
            logger.error(f"Long chain reasoning failed: {e}")
            raise ReasoningException(f"Long chain reasoning failed: {e!s}") from e

    def _generate_step(
        self,
        problem: str,
        step_num: int,
        total_steps: int,
        recent_context: str,
    ) -> str | None:
        """Generate a single reasoning step.

        Args:
            problem: Original problem.
            step_num: Current step number.
            total_steps: Total expected steps.
            recent_context: Context from recent steps.

        Returns:
            Generated step text, or None if failed.
        """
        try:
            prompt = f"""Problem: {problem}

Previous reasoning:
{recent_context if recent_context else "(Starting fresh)"}

Generate Step {step_num}/{total_steps}:
- Build directly on previous reasoning
- Make one clear logical advancement
- Be specific and precise (2-4 sentences)

Step {step_num}:"""

            response = self.llm.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=0.5,  # Lower temperature for consistency
            )

            return response.strip() if response else None

        except Exception as e:
            logger.warning(f"Step {step_num} generation error: {e}")
            return None

    def _verify_step(
        self,
        problem: str,
        chain: list[str],
        step_num: int,
    ) -> dict[str, Any]:
        """Verify a reasoning step for logical consistency.

        Args:
            problem: Original problem.
            chain: Full reasoning chain so far.
            step_num: Step to verify.

        Returns:
            Verification result dict with is_valid, reason, step.
        """
        try:
            current_step = chain[step_num - 1] if step_num <= len(chain) else ""
            prev_step = chain[step_num - 2] if step_num > 1 and step_num - 1 <= len(chain) else ""

            prompt = f"""Verify this reasoning step for logical consistency:

Problem: {problem}

Previous step: {prev_step[:200] if prev_step else "(none)"}

Current step to verify: {current_step[:300]}

Is this step:
1. Logically sound (no logical errors)?
2. Consistent with previous reasoning?
3. Making valid progress toward the solution?

Answer with YES if valid, NO if problematic. Then briefly explain (1 sentence):"""

            response = self.llm.generate(prompt, max_tokens=150, temperature=0.3)

            is_valid = response.lower().startswith("yes") or "yes" in response.lower()[:20]

            return {
                "step": step_num,
                "is_valid": is_valid,
                "reason": response[:100] if response else "No response",
            }

        except Exception as e:
            logger.warning(f"Verification failed for step {step_num}: {e}")
            return {
                "step": step_num,
                "is_valid": True,  # Fail open
                "reason": f"Verification error: {e!s}",
            }

    def _extract_answer(self, problem: str, chain: list[str]) -> str:
        """Extract final answer from reasoning chain.

        Args:
            problem: Original problem.
            chain: Complete reasoning chain.

        Returns:
            Extracted answer string.
        """
        if not chain:
            return "Unable to generate answer"

        try:
            # Use last few steps as context
            last_steps = "\n".join(chain[-3:]) if len(chain) >= 3 else "\n".join(chain)

            prompt = f"""Problem: {problem}

Final reasoning steps:
{last_steps}

Based on the reasoning chain above, what is the FINAL ANSWER to the problem?
Provide a direct, concise answer (1-2 sentences):"""

            response = self.llm.generate(prompt, max_tokens=200, temperature=0.3)
            return response.strip() if response else chain[-1].split("\n")[0]

        except Exception as e:
            logger.warning(f"Answer extraction failed: {e}")
            return chain[-1].split("\n")[0] if chain else "Unable to extract answer"

    def _build_recent_context(self, chain: list[str], max_steps: int = 5) -> str:
        """Build context string from recent reasoning steps.

        Args:
            chain: Full reasoning chain.
            max_steps: Maximum steps to include.

        Returns:
            Formatted context string.
        """
        if not chain:
            return ""

        recent = chain[-max_steps:]
        start_idx = len(chain) - len(recent) + 1

        return "\n".join(
            f"Step {start_idx + i}: {step[:150]}..."
            if len(step) > 150
            else f"Step {start_idx + i}: {step}"
            for i, step in enumerate(recent)
        )
