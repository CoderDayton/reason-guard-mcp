"""Shared LLM Client for MatrixMind Examples.

Provides async LLM client supporting both OpenAI-compatible and native Ollama APIs.
For qwen3 models in thinking mode, uses native Ollama API with think=false.

Usage:
    from llm_client import init_llm_client, close_llm_client, get_llm_answer

    # Initialize once
    await init_llm_client(base_url="http://localhost:11434/v1", model="llama3.2")

    # Use helpers
    answer = await get_llm_answer("What is 2+2?")

    # Cleanup
    await close_llm_client()

Environment Variables:
    LLM_URL: Base URL for LLM API (default: http://localhost:11434/v1)
    LLM_MODEL: Model name (default: llama3.2)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

# Add parent directory to path for imports from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.confidence import cisc_vote  # noqa: E402

if TYPE_CHECKING:
    from reasoning_comparison import ReasoningProblem

# =============================================================================
# LLM Client
# =============================================================================


@dataclass
class LLMClient:
    """Async client supporting both OpenAI-compatible and native Ollama APIs.

    For qwen3 models in thinking mode, uses native Ollama API with think=false.
    """

    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.2"
    timeout: float = 60.0
    _client: httpx.AsyncClient = field(init=False, repr=False)
    _use_native_ollama: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Initialize HTTP client, detecting Ollama/qwen3 for native API."""
        # Detect if this is an Ollama endpoint and model needs native API
        # qwen3 models use "thinking" mode which puts output in reasoning field
        self._use_native_ollama = "11434" in self.base_url and (
            "qwen3" in self.model.lower() or "qwq" in self.model.lower()
        )

        # For native Ollama API, use base URL without /v1
        if self._use_native_ollama:
            native_url = self.base_url.replace("/v1", "")
            self._client = httpx.AsyncClient(base_url=native_url, timeout=self.timeout)
        else:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> LLMClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def complete(
        self,
        prompt: str,
        system: str = "You are a helpful assistant. Answer concisely.",
        max_tokens: int = 256,
    ) -> str:
        """Generate a completion from the LLM."""
        try:
            if self._use_native_ollama:
                # Use native Ollama API with think=false for qwen3 models
                response = await self._client.post(
                    "/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "think": False,  # Disable thinking mode for direct answers
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": 0.1,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"].strip()
            else:
                # OpenAI-compatible API
                response = await self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                    },
                )
                response.raise_for_status()
                data = response.json()
                msg = data["choices"][0]["message"]
                # Some models put content in "reasoning" field (thinking mode)
                content = msg.get("content", "") or msg.get("reasoning", "")
                return content.strip()
        except Exception as e:
            return f"[LLM Error: {e}]"


# =============================================================================
# Global Client Management
# =============================================================================

_llm_client: LLMClient | None = None


def get_default_llm_url() -> str:
    """Get default LLM URL from environment or use localhost."""
    return os.environ.get("LLM_URL", "http://localhost:11434/v1")


def get_default_llm_model() -> str:
    """Get default LLM model from environment or use llama3.2."""
    return os.environ.get("LLM_MODEL", "llama3.2")


async def init_llm_client(
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
) -> LLMClient:
    """Initialize the global LLM client.

    Args:
        base_url: LLM API base URL (default: from LLM_URL env or localhost:11434/v1)
        model: Model name (default: from LLM_MODEL env or llama3.2)
        timeout: Request timeout in seconds

    Returns:
        The initialized LLMClient instance

    """
    global _llm_client

    if _llm_client is not None:
        await _llm_client.close()

    _llm_client = LLMClient(
        base_url=base_url or get_default_llm_url(),
        model=model or get_default_llm_model(),
        timeout=timeout,
    )
    return _llm_client


async def close_llm_client() -> None:
    """Close the global LLM client."""
    global _llm_client
    if _llm_client is not None:
        await _llm_client.close()
        _llm_client = None


def get_llm_client() -> LLMClient | None:
    """Get the current LLM client (or None if not initialized)."""
    return _llm_client


def is_llm_enabled() -> bool:
    """Check if LLM client is initialized."""
    return _llm_client is not None


# =============================================================================
# LLM Helper Functions
# =============================================================================


async def get_llm_answer(problem_text: str, context: str = "") -> str:
    """Get an answer from the LLM for a problem.

    Args:
        problem_text: The problem/question to answer
        context: Optional context to include

    Returns:
        The LLM's answer, or empty string if LLM not initialized

    """
    if _llm_client is None:
        return ""

    full_prompt = problem_text
    if context:
        full_prompt = f"Context: {context}\n\nQuestion: {problem_text}"

    system = (
        "You are solving reasoning problems. "
        "Give only the final answer, no explanation. "
        "Be concise - just the answer value."
    )
    return await _llm_client.complete(full_prompt, system=system, max_tokens=64)


async def get_llm_confidence(answer: str, problem: str, context: str = "") -> float:
    """Get LLM self-assessed confidence for an answer (CISC verbal_0_100 method).

    Asks the LLM to rate its confidence in the answer on a 0-100 scale.
    This implements the verbal_0_100 confidence method from the CISC paper.

    Args:
        answer: The answer to assess confidence for
        problem: The original problem/question
        context: Optional problem context

    Returns:
        Confidence score normalized to 0.0-1.0 range, or 0.5 if extraction fails

    """
    if _llm_client is None:
        return 0.5

    prompt = f"""Problem: {problem}
{f"Context: {context}" if context else ""}

Proposed answer: {answer}

Rate your confidence that this answer is correct on a scale of 0-100.
Reply with ONLY a number (e.g., "85"), nothing else."""

    system = "You are assessing answer quality. Reply only with a confidence number 0-100."

    try:
        response = await _llm_client.complete(prompt, system=system, max_tokens=8)
        # Extract number from response
        import re

        match = re.search(r"\d+", response)
        if match:
            confidence = int(match.group()) / 100.0
            return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    except Exception:
        pass

    return 0.5  # Default neutral confidence


async def get_llm_alternatives(
    problem: str,
    primary_thought: str,
    num_alternatives: int = 2,
) -> list[tuple[str, float]]:
    """Generate alternative reasoning paths with confidence scores.

    Asks the LLM to generate alternative approaches and rate each one.
    Used for MPPA multi-path exploration with CISC confidence scoring.

    Args:
        problem: The problem being solved
        primary_thought: The primary reasoning step
        num_alternatives: Number of alternatives to generate

    Returns:
        List of (thought, confidence) tuples for alternatives

    """
    if _llm_client is None:
        return []

    prompt = f"""Problem: {problem}

Current approach: {primary_thought}

Generate {num_alternatives} different alternative approaches to this problem.
For each alternative:
1. Give the alternative reasoning (1-2 sentences)
2. Rate your confidence it's better than the current approach (0-100)

Format each as:
ALTERNATIVE: <reasoning>
CONFIDENCE: <number>"""

    system = "You are exploring alternative reasoning paths. Be creative but realistic."

    try:
        response = await _llm_client.complete(prompt, system=system, max_tokens=300)

        # Parse alternatives
        import re

        alternatives = []
        alt_pattern = r"ALTERNATIVE:\s*(.+?)\s*CONFIDENCE:\s*(\d+)"
        matches = re.findall(alt_pattern, response, re.IGNORECASE | re.DOTALL)

        for thought, conf_str in matches[:num_alternatives]:
            conf = int(conf_str) / 100.0
            alternatives.append((thought.strip(), max(0.0, min(1.0, conf))))

        return alternatives
    except Exception:
        pass

    return []


async def get_llm_chain_step(
    problem: ReasoningProblem,
    step_num: int,
    previous_steps: list[str],
) -> str:
    """Generate a single chain reasoning step using the LLM.

    This simulates a real MCP tool call flow where each step builds on previous ones.

    Args:
        problem: The reasoning problem being solved
        step_num: Current step number (1-indexed)
        previous_steps: List of previous step contents

    Returns:
        The generated step content, or empty string if LLM not initialized

    """
    if _llm_client is None:
        return ""

    # Build context from previous steps
    if previous_steps:
        history = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(previous_steps))
        context_block = f"\nYour reasoning so far:\n{history}\n"
    else:
        context_block = ""

    # Step-specific prompts based on position in chain
    total_steps = problem.steps_hint
    if step_num == 1:
        instruction = "Start by identifying the key information and what you need to find."
    elif step_num == total_steps:
        instruction = "Now give your final answer based on your reasoning."
    elif step_num == 2:
        instruction = "Set up the approach or formula you'll use to solve this."
    else:
        instruction = "Continue reasoning towards the solution."

    prompt = f"""Problem: {problem.question}
{f"Context: {problem.context}" if problem.context else ""}
{context_block}
{instruction}

Provide Step {step_num} of your reasoning (1-2 sentences, be specific):"""

    system = (
        "You are a step-by-step reasoning assistant. "
        "Each step should build logically on previous steps. "
        "Be specific and show your work. Keep each step concise."
    )
    return await _llm_client.complete(prompt, system=system, max_tokens=150)


async def get_llm_mot_cell(
    problem: ReasoningProblem,
    perspective: str,
    criterion: str,
    other_cells: list[tuple[str, str, str]],  # (perspective, criterion, content)
) -> str:
    """Generate a single MoT matrix cell using the LLM.

    Each cell analyzes the problem from a specific perspective and criterion.
    This simulates multi-perspective analysis in MoT.

    Args:
        problem: The reasoning problem being analyzed
        perspective: The perspective to analyze from (e.g., "mathematical", "logical")
        criterion: The criterion to evaluate (e.g., "accuracy", "completeness")
        other_cells: List of (perspective, criterion, content) tuples for context

    Returns:
        The generated cell content, or empty string if LLM not initialized

    """
    if _llm_client is None:
        return ""

    # Build context from other cells
    if other_cells:
        others = "\n".join(f"- [{p}×{c}]: {content[:100]}..." for p, c, content in other_cells)
        context_block = f"\nOther perspectives already analyzed:\n{others}\n"
    else:
        context_block = ""

    prompt = f"""Problem: {problem.question}
{f"Context: {problem.context}" if problem.context else ""}
{context_block}
Analyze this problem from the perspective of "{perspective}" using the criterion "{criterion}".

Provide a focused analysis (2-3 sentences):"""

    system = (
        "You are analyzing a problem from multiple perspectives. "
        "Focus specifically on the given perspective and criterion. "
        "Be insightful and specific to this angle of analysis."
    )
    return await _llm_client.complete(prompt, system=system, max_tokens=150)


async def get_llm_mot_synthesis(
    problem: ReasoningProblem,
    column_cells: list[tuple[str, str]],  # (perspective, content)
    criterion: str,
) -> str:
    """Synthesize a column of MoT cells into a unified insight.

    This combines multiple perspectives for a single criterion.

    Args:
        problem: The reasoning problem being analyzed
        column_cells: List of (perspective, content) tuples to synthesize
        criterion: The criterion being synthesized

    Returns:
        The synthesized insight, or empty string if LLM not initialized

    """
    if _llm_client is None:
        return ""

    cells_text = "\n".join(f"- {perspective}: {content}" for perspective, content in column_cells)

    prompt = f"""Problem: {problem.question}

For the criterion "{criterion}", synthesize these perspectives:
{cells_text}

Provide a unified synthesis (1-2 sentences):"""

    system = (
        "You are synthesizing multiple perspectives into a coherent insight. "
        "Find the common thread and resolve any contradictions."
    )
    return await _llm_client.complete(prompt, system=system, max_tokens=100)


# =============================================================================
# CLI Argument Helpers
# =============================================================================


def add_llm_args(parser: Any) -> None:
    """Add standard LLM CLI arguments to an argument parser.

    Adds:
        --llm: Enable LLM-based reasoning
        --llm-url: LLM API URL
        --llm-model: Model name

    Args:
        parser: argparse.ArgumentParser instance

    """
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use real LLM calls for answer generation (requires Ollama or compatible API)",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=None,
        help=f"LLM API base URL (default: $LLM_URL or {get_default_llm_url()})",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=f"LLM model name (default: $LLM_MODEL or {get_default_llm_model()})",
    )


# =============================================================================
# True CISC: Self-Consistency with Confidence-Weighted Voting
# =============================================================================


@dataclass
class CISCSample:
    """A single reasoning sample for CISC voting."""

    reasoning: str  # The reasoning chain/thought process
    answer: str  # The extracted final answer
    confidence: float  # LLM self-assessed confidence (0-1)


@dataclass
class CISCSolveResult:
    """Result of CISC self-consistency solving."""

    winner: str  # The winning answer after CISC voting
    winner_weight: float  # Total weight for winning answer
    winner_reasoning: str  # Best reasoning path for winning answer
    samples: list[CISCSample]  # All samples generated
    answer_weights: dict[str, float]  # Weight per unique answer
    num_unique_answers: int  # Answer diversity
    majority_winner: str  # What simple majority would have picked
    cisc_changed_answer: bool  # Whether CISC changed the result
    temperature: float  # Softmax temperature used


async def get_llm_reasoning_sample(
    problem: str,
    context: str = "",
    temperature: float = 0.7,
) -> tuple[str, str]:
    """Generate a single reasoning chain and extract final answer.

    Uses higher temperature to get answer diversity across samples.

    Args:
        problem: The problem to solve
        context: Optional context
        temperature: Sampling temperature (higher = more diverse)

    Returns:
        Tuple of (reasoning_chain, final_answer)

    """
    if _llm_client is None:
        return "", ""

    full_prompt = problem
    if context:
        full_prompt = f"Context: {context}\n\nQuestion: {problem}"

    # First get the reasoning chain
    reasoning_system = (
        "You are a careful step-by-step reasoner. "
        "Think through the problem systematically, showing your work. "
        "End with 'ANSWER: <your final answer>'"
    )

    try:
        # Use native Ollama API with custom temperature if available
        if _llm_client._use_native_ollama:
            response = await _llm_client._client.post(
                "/api/chat",
                json={
                    "model": _llm_client.model,
                    "messages": [
                        {"role": "system", "content": reasoning_system},
                        {"role": "user", "content": full_prompt},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {
                        "num_predict": 512,
                        "temperature": temperature,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            reasoning = data["message"]["content"].strip()
        else:
            # OpenAI-compatible API - would need custom temperature support
            reasoning = await _llm_client.complete(
                full_prompt, system=reasoning_system, max_tokens=512
            )

        # Extract final answer with multiple strategies
        import re

        answer = ""

        # Strategy 0: Look for LaTeX \boxed{} format (qwen3 and many LLMs use this)
        # JSON strings escape backslashes, so \\boxed in JSON = \boxed in LaTeX
        # In Python: r"\\\\boxed" matches the literal string "\\boxed"
        boxed_match = re.search(r"\\\\boxed\{([^}]+)\}", reasoning)
        if boxed_match:
            answer = boxed_match.group(1).strip()

        # Strategy 1: Look for explicit ANSWER: marker (handle markdown bold **)
        if not answer:
            answer_match = re.search(
                r"(?:^|\n)\s*\*?\*?ANSWER:\s*(\d+(?:\.\d+)?)", reasoning, re.IGNORECASE
            )
            if answer_match:
                answer = answer_match.group(1).strip()

        # Strategy 2: Look for "the answer is X" pattern
        if not answer or len(answer) < 1:
            answer_is_match = re.search(
                r"(?:the\s+)?answer\s+is[:\s]+(\S+)", reasoning, re.IGNORECASE
            )
            if answer_is_match:
                answer = answer_is_match.group(1).strip()

        # Strategy 3: Look for "= X" at end of line (math results)
        if not answer or len(answer) < 1:
            equals_match = re.search(
                r"=\s*([\d./]+)\s*(?:$|\n|hours?|dollars?|girls?|boys?)", reasoning, re.IGNORECASE
            )
            if equals_match:
                answer = equals_match.group(1).strip()

        # Strategy 4: Extract last numeric value from the text
        if not answer or len(answer) < 1:
            # Find all numbers (including fractions and decimals)
            numbers = re.findall(r"\b(\d+(?:\.\d+)?(?:/\d+)?)\b", reasoning)
            if numbers:
                # Use the last number that appears in the reasoning
                answer = numbers[-1]

        # Strategy 5: Last line fallback
        if not answer or len(answer) < 1:
            lines = [l.strip() for l in reasoning.split("\n") if l.strip()]
            if lines:
                answer = lines[-1][:50]  # Limit length

        # Clean up answer - remove markdown formatting and common noise
        answer = re.sub(r"[\*\$\#\[\]]+", "", answer)  # Remove markdown chars
        answer = re.sub(r"^[:\s]+|[:\s\.]+$", "", answer)  # Remove leading/trailing chars
        answer = answer.strip()

        return reasoning, answer

    except Exception as e:
        return f"[Error: {e}]", ""


async def cisc_solve(
    problem: str,
    context: str = "",
    num_samples: int = 5,
    temperature: float = 1.0,
    sampling_temperature: float = 0.7,
) -> CISCSolveResult:
    """Solve a problem using true CISC: multiple samples with weighted voting.

    This implements the actual CISC algorithm from the paper:
    1. Generate N complete reasoning chains with diverse answers
    2. Extract confidence for each complete chain
    3. Apply softmax-normalized weighted majority voting

    Args:
        problem: The problem to solve
        context: Optional context
        num_samples: Number of reasoning chains to generate (paper uses 5-30)
        temperature: Softmax temperature for vote weighting
        sampling_temperature: LLM temperature for answer diversity

    Returns:
        CISCSolveResult with winning answer and all samples

    """
    if _llm_client is None:
        return CISCSolveResult(
            winner="",
            winner_weight=0.0,
            winner_reasoning="",
            samples=[],
            answer_weights={},
            num_unique_answers=0,
            majority_winner="",
            cisc_changed_answer=False,
            temperature=temperature,
        )

    samples: list[CISCSample] = []

    # Step 1: Generate N reasoning chains
    for i in range(num_samples):
        # Vary temperature slightly for more diversity
        sample_temp = sampling_temperature + (i * 0.05)
        reasoning, answer = await get_llm_reasoning_sample(problem, context, sample_temp)

        if answer:
            # Step 2: Get confidence for this complete reasoning chain
            confidence = await get_llm_confidence(answer, problem, context)
            samples.append(CISCSample(reasoning=reasoning, answer=answer, confidence=confidence))

    if not samples:
        return CISCSolveResult(
            winner="",
            winner_weight=0.0,
            winner_reasoning="",
            samples=[],
            answer_weights={},
            num_unique_answers=0,
            majority_winner="",
            cisc_changed_answer=False,
            temperature=temperature,
        )

    # Step 3: Apply CISC weighted voting
    answers = [s.answer for s in samples]
    confidences = [s.confidence for s in samples]

    vote_result = cisc_vote(answers, confidences, temperature=temperature)

    # Find best reasoning for winning answer
    def norm(s: str) -> str:
        return s.strip().lower()

    winner_norm = norm(vote_result.winner)
    best_reasoning = ""
    best_conf = -1.0
    for sample in samples:
        if norm(sample.answer) == winner_norm and sample.confidence > best_conf:
            best_conf = sample.confidence
            best_reasoning = sample.reasoning

    # Simple majority winner for comparison
    from collections import Counter

    answer_counts = Counter(norm(s.answer) for s in samples)
    majority_norm = answer_counts.most_common(1)[0][0]
    majority_winner = next(s.answer for s in samples if norm(s.answer) == majority_norm)

    return CISCSolveResult(
        winner=vote_result.winner,
        winner_weight=vote_result.winner_weight,
        winner_reasoning=best_reasoning,
        samples=samples,
        answer_weights=vote_result.answer_weights,
        num_unique_answers=vote_result.num_unique_answers,
        majority_winner=majority_winner,
        cisc_changed_answer=norm(vote_result.winner) != norm(majority_winner),
        temperature=temperature,
    )
