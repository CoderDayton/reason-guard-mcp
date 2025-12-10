"""Long Chain-of-Thought reasoning tool.

Implements deep sequential reasoning with optional intermediate verification
and STaR (Self-Taught Reasoner) iterations for improved accuracy.

Based on paper 2505.21825 - achieves exponential advantage over parallel
methods for serial problems requiring deep sequential dependencies.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.utils.errors import ReasoningException
from src.utils.retry import retry_with_backoff
from src.utils.schema import ReasoningResult

if TYPE_CHECKING:
    from src.models.llm_client import LLMClientProtocol


class LongChainOfThoughtTool:
    """Long-chain sequential reasoning with verification checkpoints.

    Implements deep reasoning where each step builds fundamentally
    on previous steps. Includes optional intermediate verification
    to catch errors early.

    Supports STaR (Self-Taught Reasoner) iterations which generate
    multiple reasoning chains with varying temperatures and select
    the best one based on verification scores.

    Best for:
    - Graph connectivity problems
    - Constraint satisfaction
    - Arithmetic with dependencies
    - Proof verification

    Attributes:
        llm: LLM client for generating reasoning steps.

    """

    def __init__(self, llm_client: LLMClientProtocol) -> None:
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
        star_iterations: int = 0,
    ) -> ReasoningResult:
        """Execute long-chain sequential reasoning.

        Algorithm:
        1. Initialize reasoning chain with problem
        2. For each step:
           a. Generate next reasoning step based on previous
           b. Optionally verify step consistency
           c. If verification fails, flag but continue
        3. Extract final answer from chain

        If star_iterations > 0 (STaR mode):
        1. Generate multiple reasoning chains with varying temperatures
        2. Score each chain based on verification ratio + final answer validity
        3. Return the best-scoring chain
        4. Early exit if a fully verified chain is found

        Args:
            problem: Problem statement to solve.
            num_steps: Number of reasoning steps (1-50).
            verify_intermediate: Whether to verify steps periodically.
            verification_frequency: Verify every N steps.
            star_iterations: Number of STaR iterations (0=disabled, 1-5 recommended).
                            Higher values increase accuracy but also latency.

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
            ...     verify_intermediate=True,
            ...     star_iterations=3  # Run up to 3 iterations
            ... )
            >>> print(result.answer)
            >>> print(result.reasoning_trace["star_iterations_used"])

        """
        # Validate inputs
        if not problem or not problem.strip():
            raise ReasoningException("Problem cannot be empty")

        if not 1 <= num_steps <= 50:
            raise ReasoningException(f"num_steps must be 1-50, got {num_steps}")

        if not 0 <= star_iterations <= 10:
            raise ReasoningException(f"star_iterations must be 0-10, got {star_iterations}")

        try:
            # STaR mode: run multiple iterations and select best
            if star_iterations > 0:
                return self._run_star_iterations(
                    problem=problem,
                    num_steps=num_steps,
                    verify_intermediate=verify_intermediate,
                    verification_frequency=verification_frequency,
                    star_iterations=star_iterations,
                )

            # Standard single-chain mode
            return self._run_single_chain(
                problem=problem,
                num_steps=num_steps,
                verify_intermediate=verify_intermediate,
                verification_frequency=verification_frequency,
                temperature=0.5,
            )

        except ReasoningException:
            raise
        except Exception as e:
            logger.error(f"Long chain reasoning failed: {e}")
            raise ReasoningException(f"Long chain reasoning failed: {e!s}") from e

    def _run_single_chain(
        self,
        problem: str,
        num_steps: int,
        verify_intermediate: bool,
        verification_frequency: int,
        temperature: float = 0.5,
    ) -> ReasoningResult:
        """Run a single reasoning chain.

        Args:
            problem: Problem statement.
            num_steps: Number of reasoning steps.
            verify_intermediate: Whether to verify steps.
            verification_frequency: Verify every N steps.
            temperature: LLM temperature for generation.

        Returns:
            ReasoningResult from this chain.

        """
        reasoning_chain: list[str] = []
        verifications: list[dict[str, Any]] = []
        verification_passed = 0
        verification_total = 0

        # Generate reasoning steps
        for step_num in range(1, num_steps + 1):
            recent_context = self._build_recent_context(reasoning_chain, max_steps=5)

            next_step = self._generate_step(
                problem=problem,
                step_num=step_num,
                total_steps=num_steps,
                recent_context=recent_context,
                temperature=temperature,
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
                    logger.debug(f"Step {step_num} verification failed: {reason}")

        # Extract final answer
        final_answer = self._extract_answer(problem, reasoning_chain)

        # Calculate confidence
        if verification_total > 0:
            verification_ratio = verification_passed / verification_total
            confidence = 0.5 + 0.4 * verification_ratio
        else:
            confidence = 0.7

        return ReasoningResult(
            answer=final_answer,
            confidence=confidence,
            reasoning_steps=reasoning_chain,
            verification_results={
                "total_verifications": verification_total,
                "passed": verification_passed,
                "failed": verification_total - verification_passed,
                "details": verifications[-3:] if verifications else [],
            },
            tokens_used=self.llm.estimate_tokens("\n".join(reasoning_chain)),
            reasoning_trace={
                "total_steps": len(reasoning_chain),
                "requested_steps": num_steps,
                "verify_enabled": verify_intermediate,
                "temperature": temperature,
            },
        )

    def _run_star_iterations(
        self,
        problem: str,
        num_steps: int,
        verify_intermediate: bool,
        verification_frequency: int,
        star_iterations: int,
    ) -> ReasoningResult:
        """Run STaR iterations to find best reasoning chain.

        Generates multiple reasoning chains with varying temperatures,
        scores each based on verification results and final answer validity,
        and returns the best one.

        Args:
            problem: Problem statement.
            num_steps: Number of reasoning steps.
            verify_intermediate: Whether to verify steps.
            verification_frequency: Verify every N steps.
            star_iterations: Number of iterations to run.

        Returns:
            Best ReasoningResult from all iterations.

        """
        best_result: ReasoningResult | None = None
        best_score = -1.0
        iterations_used = 0

        for iteration in range(star_iterations):
            iterations_used = iteration + 1

            # Vary temperature to explore different reasoning paths
            # Range: 0.4 to 0.8 across iterations
            temp = 0.4 + (iteration * 0.4 / max(star_iterations - 1, 1))

            logger.debug(f"STaR iteration {iterations_used}/{star_iterations}, temp={temp:.2f}")

            result = self._run_single_chain(
                problem=problem,
                num_steps=num_steps,
                verify_intermediate=verify_intermediate,
                verification_frequency=verification_frequency,
                temperature=temp,
            )

            # Score the chain
            score = self._score_chain(problem, result, verify_intermediate)

            logger.debug(f"STaR iteration {iterations_used} score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_result = result

            # Early exit if we got a high-confidence result
            if score >= 1.4:  # Verification passed + answer valid
                logger.debug(f"STaR early exit at iteration {iterations_used} (score={score:.3f})")
                break

        if best_result is None:
            raise ReasoningException("STaR iterations produced no valid result")

        # Add STaR metadata to reasoning trace
        trace = dict(best_result.reasoning_trace) if best_result.reasoning_trace else {}
        trace["star_enabled"] = True
        trace["star_iterations_requested"] = star_iterations
        trace["star_iterations_used"] = iterations_used
        trace["star_best_score"] = round(best_score, 3)

        return ReasoningResult(
            answer=best_result.answer,
            confidence=best_result.confidence,
            reasoning_steps=best_result.reasoning_steps,
            verification_results=best_result.verification_results,
            tokens_used=best_result.tokens_used,
            reasoning_trace=trace,
        )

    def _score_chain(
        self,
        problem: str,
        result: ReasoningResult,
        verify_intermediate: bool,
    ) -> float:
        """Score a reasoning chain for STaR selection.

        Score components:
        - Verification ratio (0-1): How many intermediate steps passed
        - Answer validity (0-0.5): Self-verification of final answer

        Args:
            problem: Original problem.
            result: ReasoningResult to score.
            verify_intermediate: Whether verification was enabled.

        Returns:
            Score between 0 and ~1.5.

        """
        score = 0.0

        # Component 1: Verification ratio
        if verify_intermediate and result.verification_results:
            vr = result.verification_results
            total = vr.get("total_verifications", 0)
            if total > 0:
                passed = vr.get("passed", 0)
                score += passed / total

        # Component 2: Final answer self-verification
        answer_valid = self._verify_final_answer(problem, result)
        if answer_valid:
            score += 0.5

        return score

    def _verify_final_answer(self, problem: str, result: ReasoningResult) -> bool:
        """Self-verify that the final answer is correct and consistent.

        Args:
            problem: Original problem.
            result: ReasoningResult containing answer and reasoning.

        Returns:
            True if answer appears valid, False otherwise.

        """
        if not result.answer or result.answer == "Unable to generate answer":
            return False

        try:
            # Build reasoning summary from last steps
            last_steps = result.reasoning_steps[-3:] if result.reasoning_steps else []
            reasoning_summary = " → ".join(s[:80] + "..." if len(s) > 80 else s for s in last_steps)

            prompt = f"""Problem: {problem}

Proposed answer: {result.answer}

Reasoning path: {reasoning_summary}

Is this answer CORRECT and CONSISTENT with the reasoning shown?
Answer YES if the answer logically follows from the reasoning.
Answer NO if there are errors or inconsistencies.

Your verdict (YES/NO):"""

            response = self.llm.generate(prompt, max_tokens=50, temperature=0.2)
            is_valid = response.strip().upper().startswith("YES")

            logger.debug(f"Final answer verification: {is_valid}")
            return is_valid

        except Exception as e:
            logger.warning(f"Final answer verification failed: {e}")
            return False  # Fail closed for scoring

    async def reason_async(
        self,
        problem: str,
        num_steps: int = 15,
        verify_intermediate: bool = True,
        verification_frequency: int = 3,
        star_iterations: int = 0,
    ) -> ReasoningResult:
        """Execute long-chain sequential reasoning with async verification.

        P2 Optimization: Verification is performed as fire-and-forget tasks,
        allowing step generation to continue without waiting for verification
        results. Verifications are collected at the end.

        When star_iterations > 0 (STaR mode):
        - Runs multiple reasoning chains concurrently with varying temperatures
        - Scores each chain based on verification ratio + final answer validity
        - Returns the best-scoring chain
        - Early exit if high-confidence chain found

        Args:
            problem: Problem statement to solve.
            num_steps: Number of reasoning steps (1-50).
            verify_intermediate: Whether to verify steps periodically.
            verification_frequency: Verify every N steps.
            star_iterations: Number of STaR iterations (0=disabled, 1-10).
                            Higher values increase accuracy but also latency.

        Returns:
            ReasoningResult with answer, steps, and verification info.

        Raises:
            ReasoningException: If reasoning fails.

        """
        # Validate inputs
        if not problem or not problem.strip():
            raise ReasoningException("Problem cannot be empty")

        if not 1 <= num_steps <= 50:
            raise ReasoningException(f"num_steps must be 1-50, got {num_steps}")

        if not 0 <= star_iterations <= 10:
            raise ReasoningException(f"star_iterations must be 0-10, got {star_iterations}")

        try:
            # STaR mode: run multiple iterations concurrently
            if star_iterations > 0:
                return await self._run_star_iterations_async(
                    problem=problem,
                    num_steps=num_steps,
                    verify_intermediate=verify_intermediate,
                    verification_frequency=verification_frequency,
                    star_iterations=star_iterations,
                )

            # Standard single-chain mode
            return await self._run_single_chain_async(
                problem=problem,
                num_steps=num_steps,
                verify_intermediate=verify_intermediate,
                verification_frequency=verification_frequency,
                temperature=0.5,
            )

        except ReasoningException:
            raise
        except Exception as e:
            logger.error(f"Long chain async reasoning failed: {e}")
            raise ReasoningException(f"Long chain reasoning failed: {e!s}") from e

    async def _run_star_iterations_async(
        self,
        problem: str,
        num_steps: int,
        verify_intermediate: bool,
        verification_frequency: int,
        star_iterations: int,
    ) -> ReasoningResult:
        """Run STaR iterations asynchronously with concurrent chain generation.

        Strategy: Run first iteration to get a baseline, then run remaining
        iterations concurrently. Uses early-exit signaling via asyncio.Event
        to cancel pending iterations when a high-confidence result is found,
        saving compute on expensive LLM calls.

        Early-exit flow:
        1. First iteration runs alone (baseline)
        2. If score >= 1.4, return immediately
        3. Otherwise, spawn remaining iterations concurrently
        4. As each completes, score it immediately
        5. If any scores >= 1.4, signal cancellation to others
        6. Return best result found

        Args:
            problem: Problem statement.
            num_steps: Number of reasoning steps.
            verify_intermediate: Whether to verify steps.
            verification_frequency: Verify every N steps.
            star_iterations: Number of iterations to run.

        Returns:
            Best ReasoningResult from all iterations.

        """
        # Calculate temperatures for each iteration (0.4 to 0.8)
        temperatures = [
            0.4 + (i * 0.4 / max(star_iterations - 1, 1)) for i in range(star_iterations)
        ]

        # Run first iteration to check for early exit
        first_result = await self._run_single_chain_async(
            problem=problem,
            num_steps=num_steps,
            verify_intermediate=verify_intermediate,
            verification_frequency=verification_frequency,
            temperature=temperatures[0],
        )
        first_score = await self._score_chain_async(problem, first_result, verify_intermediate)

        logger.debug(f"STaR async iteration 1/{star_iterations}, score={first_score:.3f}")

        # Early exit if first iteration is high-confidence
        if first_score >= 1.4:
            logger.debug(f"STaR async early exit at iteration 1 (score={first_score:.3f})")
            return self._add_star_metadata(first_result, star_iterations, 1, first_score)

        # If only one iteration requested, return it
        if star_iterations == 1:
            return self._add_star_metadata(first_result, star_iterations, 1, first_score)

        # Early-exit signaling: when one iteration finds high-confidence result,
        # signal others to stop
        early_exit_event = asyncio.Event()
        best_result = first_result
        best_score = first_score
        iterations_completed = 1  # First iteration already done

        # Lock to protect best_result/best_score updates
        result_lock = asyncio.Lock()

        async def run_and_score_iteration(idx: int, temp: float) -> None:
            """Run iteration and score immediately, signaling early exit if high-confidence."""
            nonlocal best_result, best_score, iterations_completed

            # Check if we should abort before starting
            if early_exit_event.is_set():
                logger.debug(f"STaR iteration {idx} skipped (early exit signaled)")
                return

            try:
                # Run the chain with cancellation check
                result = await self._run_single_chain_with_cancellation(
                    problem=problem,
                    num_steps=num_steps,
                    verify_intermediate=verify_intermediate,
                    verification_frequency=verification_frequency,
                    temperature=temp,
                    cancel_event=early_exit_event,
                )

                # If cancelled mid-chain, result is None
                if result is None:
                    logger.debug(f"STaR iteration {idx} cancelled mid-chain")
                    return

                # Score immediately
                score = await self._score_chain_async(problem, result, verify_intermediate)
                logger.debug(f"STaR async iteration {idx}/{star_iterations}, score={score:.3f}")

                # Update best result under lock
                async with result_lock:
                    iterations_completed += 1
                    if score > best_score:
                        best_score = score
                        best_result = result

                    # Signal early exit if we found a high-confidence result
                    if score >= 1.4 and not early_exit_event.is_set():
                        logger.debug(
                            f"STaR early exit signaled at iteration {idx} (score={score:.3f})"
                        )
                        early_exit_event.set()

            except asyncio.CancelledError:
                logger.debug(f"STaR iteration {idx} cancelled")
                raise
            except Exception as e:
                logger.warning(f"STaR async iteration {idx} failed: {e}")

        # Run remaining iterations concurrently with early-exit support
        tasks = [
            asyncio.create_task(run_and_score_iteration(i + 2, temp))
            for i, temp in enumerate(temperatures[1:])
        ]

        # Wait for all tasks (they handle their own cancellation via the event)
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate actual iterations used (completed + 1 for first)
        iterations_used = iterations_completed

        return self._add_star_metadata(best_result, star_iterations, iterations_used, best_score)

    async def _run_single_chain_with_cancellation(
        self,
        problem: str,
        num_steps: int,
        verify_intermediate: bool,
        verification_frequency: int,
        temperature: float,
        cancel_event: asyncio.Event,
    ) -> ReasoningResult | None:
        """Run a single async reasoning chain with cancellation support.

        Checks the cancel_event between steps. If set, returns None immediately
        to save compute on remaining steps.

        Args:
            problem: Problem statement.
            num_steps: Number of reasoning steps.
            verify_intermediate: Whether to verify steps.
            verification_frequency: Verify every N steps.
            temperature: LLM temperature for generation.
            cancel_event: Event that signals cancellation when set.

        Returns:
            ReasoningResult if completed, None if cancelled.

        """
        reasoning_chain: list[str] = []
        pending_verifications: list[asyncio.Task[dict[str, Any]]] = []

        # Generate reasoning steps with cancellation checks
        for step_num in range(1, num_steps + 1):
            # Check for cancellation between steps
            if cancel_event.is_set():
                # Cancel pending verifications
                for task in pending_verifications:
                    task.cancel()
                return None

            recent_context = self._build_recent_context(reasoning_chain, max_steps=5)

            next_step = await self._generate_step_async(
                problem=problem,
                step_num=step_num,
                total_steps=num_steps,
                recent_context=recent_context,
                temperature=temperature,
            )

            if not next_step:
                logger.warning(f"Step {step_num} generation failed, stopping chain")
                break

            reasoning_chain.append(next_step)

            # Fire-and-forget verification
            if verify_intermediate and step_num % verification_frequency == 0:
                task = asyncio.create_task(
                    self._verify_step_async(
                        problem=problem,
                        chain=reasoning_chain.copy(),
                        step_num=step_num,
                    )
                )
                pending_verifications.append(task)

        # Final cancellation check before expensive answer extraction
        if cancel_event.is_set():
            for task in pending_verifications:
                task.cancel()
            return None

        # Collect verification results
        verifications: list[dict[str, Any]] = []
        verification_passed = 0
        verification_total = 0

        if pending_verifications:
            results = await asyncio.gather(*pending_verifications, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Verification task failed: {result}")
                    verifications.append(
                        {"step": -1, "is_valid": True, "reason": f"Error: {result}"}
                    )
                    verification_total += 1
                elif isinstance(result, dict):
                    verifications.append(result)
                    verification_total += 1
                    if result.get("is_valid", False):
                        verification_passed += 1

        # Extract final answer
        final_answer = await self._extract_answer_async(problem, reasoning_chain)

        # Calculate confidence
        if verification_total > 0:
            verification_ratio = verification_passed / verification_total
            confidence = 0.5 + 0.4 * verification_ratio
        else:
            confidence = 0.7

        return ReasoningResult(
            answer=final_answer,
            confidence=confidence,
            reasoning_steps=reasoning_chain,
            verification_results={
                "total_verifications": verification_total,
                "passed": verification_passed,
                "failed": verification_total - verification_passed,
                "details": verifications[-3:] if verifications else [],
            },
            tokens_used=self.llm.estimate_tokens("\n".join(reasoning_chain)),
            reasoning_trace={
                "total_steps": len(reasoning_chain),
                "requested_steps": num_steps,
                "verify_enabled": verify_intermediate,
                "temperature": temperature,
                "async_verification": True,
            },
        )

    def _add_star_metadata(
        self,
        result: ReasoningResult,
        star_iterations_requested: int,
        star_iterations_used: int,
        best_score: float,
    ) -> ReasoningResult:
        """Add STaR metadata to reasoning trace.

        Args:
            result: Base reasoning result.
            star_iterations_requested: Total iterations requested.
            star_iterations_used: Iterations actually completed.
            best_score: Best score achieved.

        Returns:
            New ReasoningResult with STaR metadata.

        """
        trace = dict(result.reasoning_trace) if result.reasoning_trace else {}
        trace["star_enabled"] = True
        trace["star_iterations_requested"] = star_iterations_requested
        trace["star_iterations_used"] = star_iterations_used
        trace["star_best_score"] = round(best_score, 3)
        trace["star_async"] = True
        # Track if early exit saved compute
        trace["star_early_exit"] = star_iterations_used < star_iterations_requested

        return ReasoningResult(
            answer=result.answer,
            confidence=result.confidence,
            reasoning_steps=result.reasoning_steps,
            verification_results=result.verification_results,
            tokens_used=result.tokens_used,
            reasoning_trace=trace,
        )

    async def _run_single_chain_async(
        self,
        problem: str,
        num_steps: int,
        verify_intermediate: bool,
        verification_frequency: int,
        temperature: float = 0.5,
    ) -> ReasoningResult:
        """Run a single async reasoning chain with configurable temperature.

        Args:
            problem: Problem statement.
            num_steps: Number of reasoning steps.
            verify_intermediate: Whether to verify steps.
            verification_frequency: Verify every N steps.
            temperature: LLM temperature for generation.

        Returns:
            ReasoningResult from this chain.

        """
        reasoning_chain: list[str] = []
        pending_verifications: list[asyncio.Task[dict[str, Any]]] = []

        # Generate reasoning steps
        for step_num in range(1, num_steps + 1):
            recent_context = self._build_recent_context(reasoning_chain, max_steps=5)

            next_step = await self._generate_step_async(
                problem=problem,
                step_num=step_num,
                total_steps=num_steps,
                recent_context=recent_context,
                temperature=temperature,
            )

            if not next_step:
                logger.warning(f"Step {step_num} generation failed, stopping chain")
                break

            reasoning_chain.append(next_step)

            # Fire-and-forget verification
            if verify_intermediate and step_num % verification_frequency == 0:
                task = asyncio.create_task(
                    self._verify_step_async(
                        problem=problem,
                        chain=reasoning_chain.copy(),
                        step_num=step_num,
                    )
                )
                pending_verifications.append(task)

        # Collect verification results
        verifications: list[dict[str, Any]] = []
        verification_passed = 0
        verification_total = 0

        if pending_verifications:
            results = await asyncio.gather(*pending_verifications, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Verification task failed: {result}")
                    verifications.append(
                        {"step": -1, "is_valid": True, "reason": f"Error: {result}"}
                    )
                    verification_total += 1
                elif isinstance(result, dict):
                    verifications.append(result)
                    verification_total += 1
                    if result.get("is_valid", False):
                        verification_passed += 1

        # Extract final answer
        final_answer = await self._extract_answer_async(problem, reasoning_chain)

        # Calculate confidence
        if verification_total > 0:
            verification_ratio = verification_passed / verification_total
            confidence = 0.5 + 0.4 * verification_ratio
        else:
            confidence = 0.7

        return ReasoningResult(
            answer=final_answer,
            confidence=confidence,
            reasoning_steps=reasoning_chain,
            verification_results={
                "total_verifications": verification_total,
                "passed": verification_passed,
                "failed": verification_total - verification_passed,
                "details": verifications[-3:] if verifications else [],
            },
            tokens_used=self.llm.estimate_tokens("\n".join(reasoning_chain)),
            reasoning_trace={
                "total_steps": len(reasoning_chain),
                "requested_steps": num_steps,
                "verify_enabled": verify_intermediate,
                "temperature": temperature,
                "async_verification": True,
            },
        )

    async def _score_chain_async(
        self,
        problem: str,
        result: ReasoningResult,
        verify_intermediate: bool,
    ) -> float:
        """Score a reasoning chain asynchronously for STaR selection.

        Score components:
        - Verification ratio (0-1): How many intermediate steps passed
        - Answer validity (0-0.5): Self-verification of final answer

        Args:
            problem: Original problem.
            result: ReasoningResult to score.
            verify_intermediate: Whether verification was enabled.

        Returns:
            Score between 0 and ~1.5.

        """
        score = 0.0

        # Component 1: Verification ratio
        if verify_intermediate and result.verification_results:
            vr = result.verification_results
            total = vr.get("total_verifications", 0)
            if total > 0:
                passed = vr.get("passed", 0)
                score += passed / total

        # Component 2: Final answer self-verification (async)
        answer_valid = await self._verify_final_answer_async(problem, result)
        if answer_valid:
            score += 0.5

        return score

    async def _verify_final_answer_async(self, problem: str, result: ReasoningResult) -> bool:
        """Async self-verification that the final answer is correct and consistent.

        Args:
            problem: Original problem.
            result: ReasoningResult containing answer and reasoning.

        Returns:
            True if answer appears valid, False otherwise.

        """
        if not result.answer or result.answer == "Unable to generate answer":
            return False

        try:
            last_steps = result.reasoning_steps[-3:] if result.reasoning_steps else []
            reasoning_summary = " → ".join(s[:80] + "..." if len(s) > 80 else s for s in last_steps)

            prompt = f"""Problem: {problem}

Proposed answer: {result.answer}

Reasoning path: {reasoning_summary}

Is this answer CORRECT and CONSISTENT with the reasoning shown?
Answer YES if the answer logically follows from the reasoning.
Answer NO if there are errors or inconsistencies.

Your verdict (YES/NO):"""

            response = await self.llm.generate_async(prompt, max_tokens=50, temperature=0.2)
            is_valid = response.strip().upper().startswith("YES")

            logger.debug(f"Async final answer verification: {is_valid}")
            return is_valid

        except Exception as e:
            logger.warning(f"Async final answer verification failed: {e}")
            return False

    async def _generate_step_async(
        self,
        problem: str,
        step_num: int,
        total_steps: int,
        recent_context: str,
        temperature: float = 0.5,
    ) -> str | None:
        """Async version of step generation.

        Args:
            problem: Original problem.
            step_num: Current step number.
            total_steps: Total expected steps.
            recent_context: Context from recent steps.
            temperature: LLM temperature for generation.

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

            response = await self.llm.generate_async(
                prompt=prompt,
                max_tokens=400,
                temperature=temperature,
            )

            return response.strip() if response else None

        except Exception as e:
            logger.warning(f"Step {step_num} generation error: {e}")
            return None

    async def _verify_step_async(
        self,
        problem: str,
        chain: list[str],
        step_num: int,
    ) -> dict[str, Any]:
        """Async version of step verification.

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

            response = await self.llm.generate_async(prompt, max_tokens=150, temperature=0.3)

            is_valid = response.lower().startswith("yes") or "yes" in response.lower()[:20]

            return {
                "step": step_num,
                "is_valid": is_valid,
                "reason": response[:100] if response else "No response",
            }

        except Exception as e:
            logger.warning(f"Async verification failed for step {step_num}: {e}")
            return {
                "step": step_num,
                "is_valid": True,  # Fail open
                "reason": f"Verification error: {e!s}",
            }

    async def _extract_answer_async(self, problem: str, chain: list[str]) -> str:
        """Async version of answer extraction.

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

            response = await self.llm.generate_async(prompt, max_tokens=200, temperature=0.3)
            return response.strip() if response else chain[-1].split("\n")[0]

        except Exception as e:
            logger.warning(f"Async answer extraction failed: {e}")
            return chain[-1].split("\n")[0] if chain else "Unable to extract answer"

    def _generate_step(
        self,
        problem: str,
        step_num: int,
        total_steps: int,
        recent_context: str,
        temperature: float = 0.5,
    ) -> str | None:
        """Generate a single reasoning step.

        Args:
            problem: Original problem.
            step_num: Current step number.
            total_steps: Total expected steps.
            recent_context: Context from recent steps.
            temperature: LLM temperature for generation.

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
                temperature=temperature,
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
