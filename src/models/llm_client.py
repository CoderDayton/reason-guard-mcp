"""LLM Client wrapper with retry logic and timeout protection."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from src.utils.errors import LLMException
from src.utils.retry import retry_with_backoff

if TYPE_CHECKING:
    pass


class LLMClient:
    """Wrapper around OpenAI LLM with retry logic and structured error handling.

    This client provides:
    - Automatic retry with exponential backoff
    - Timeout protection on all calls
    - Both sync and async interfaces
    - Token estimation utilities
    - Structured error handling

    Example:
        client = LLMClient(model="gpt-4-turbo")
        response = client.generate("What is 2+2?")
        print(response)  # "4"

    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4-turbo",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Initialize LLM client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            base_url: Optional custom API base URL for OpenAI-compatible endpoints.
            model: Model name to use for completions.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.

        Raises:
            LLMException: If API key is not provided and not in environment.

        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMException("OPENAI_API_KEY environment variable not set")

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize sync client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
            timeout=timeout,
        )

        # Initialize async client
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
            timeout=timeout,
        )

        logger.info(f"LLM client initialized with model: {model}")

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using LLM (synchronous).

        Args:
            prompt: User prompt/question.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            top_p: Top-p (nucleus) sampling parameter.
            system_prompt: Optional system message to set context.

        Returns:
            Generated text response.

        Raises:
            LLMException: If generation fails after retries.

        """
        try:
            messages: list[dict[str, str]] = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            content = response.choices[0].message.content
            return content or ""

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMException(f"Generation failed: {e!s}") from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using LLM (asynchronous).

        Args:
            prompt: User prompt/question.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            top_p: Top-p (nucleus) sampling parameter.
            system_prompt: Optional system message to set context.

        Returns:
            Generated text response.

        Raises:
            LLMException: If generation fails after retries.

        """
        try:
            messages: list[dict[str, str]] = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            logger.debug(f"Sending async request to {self.model} with {len(prompt)} char prompt")

            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Extract content with fallback for non-standard API responses
            content: str | None = None
            finish_reason: str | None = None

            if response.choices:
                choice = response.choices[0]
                finish_reason = choice.finish_reason
                content = choice.message.content

                # Some APIs return content in alternative fields
                if not content:
                    msg = choice.message
                    msg_dict: dict[str, str | None] = {}

                    # Get all fields from message object
                    if hasattr(msg, "model_dump"):
                        msg_dict = msg.model_dump()

                    # Priority order for content extraction:
                    # 1. reasoning_content - used by reasoning models (MiniMax-M2, o1, etc.)
                    # 2. text - used by some providers
                    # 3. output - used by some providers
                    reasoning_content = msg_dict.get("reasoning_content")
                    if reasoning_content:
                        # For reasoning models, extract the conclusion from reasoning
                        # The reasoning_content contains chain-of-thought, we want the final answer
                        content = self._extract_answer_from_reasoning(reasoning_content)
                        logger.debug(
                            f"Extracted from reasoning_content "
                            f"({len(reasoning_content)} chars → {len(content)} chars)"
                        )
                    elif hasattr(msg, "text") and msg.text:  # type: ignore[attr-defined]
                        content = msg.text  # type: ignore[attr-defined]
                    elif msg_dict.get("text"):
                        content = msg_dict["text"]
                    elif msg_dict.get("output"):
                        content = msg_dict["output"]

                logger.debug(
                    f"Response: finish_reason={finish_reason}, "
                    f"content_len={len(content) if content else 0}, "
                    f"content_preview={content[:100] if content else 'EMPTY'}..."
                )

                # Warn on suspicious empty response with length finish reason
                if not content and finish_reason == "length":
                    raw_msg = (
                        choice.message.model_dump()
                        if hasattr(choice.message, "model_dump")
                        else str(choice.message)
                    )
                    logger.warning(
                        f"API returned finish_reason='length' but empty content. "
                        f"Raw message: {raw_msg}"
                    )
            else:
                logger.warning(f"No choices in response: {response}")

            return content or ""

        except Exception as e:
            logger.error(f"Async LLM generation failed: {e}")
            raise LLMException(f"Async generation failed: {e!s}") from e

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation.

        Uses the approximation that 1 token ≈ 4 characters for English text.
        For more accurate counts, use tiktoken library.

        Args:
            text: Text to estimate token count for.

        Returns:
            Estimated token count.

        """
        return len(text) // 4

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate total tokens in a message list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Estimated total token count.

        """
        total = 0
        for msg in messages:
            # Add overhead for message structure (~4 tokens per message)
            total += 4
            total += self.estimate_tokens(msg.get("content", ""))
        return total

    def _extract_answer_from_reasoning(self, reasoning_content: str) -> str:
        """Extract the final answer from reasoning model output.

        Reasoning models (like MiniMax-M2, o1) return their chain-of-thought
        in reasoning_content, with the actual answer being the conclusion.
        This method extracts a usable answer from the reasoning trace.

        Args:
            reasoning_content: The raw reasoning chain from the model.

        Returns:
            Extracted answer or the last substantive part of reasoning.

        """
        if not reasoning_content:
            return ""

        # Common patterns that indicate the start of a final answer
        answer_markers = [
            "Thus we can produce:",
            "Thus:",
            "So we can write:",
            "The answer is:",
            "Final answer:",
            "In conclusion:",
            "Therefore:",
            "So the answer is:",
            "Potential answer:",
        ]

        # Try to find explicit answer markers
        reasoning_lower = reasoning_content.lower()
        for marker in answer_markers:
            marker_lower = marker.lower()
            if marker_lower in reasoning_lower:
                idx = reasoning_lower.rfind(marker_lower)
                # Extract everything after the marker
                answer_part = reasoning_content[idx + len(marker) :].strip()
                # Clean up: take until next paragraph or end
                if "\n\n" in answer_part:
                    answer_part = answer_part.split("\n\n")[0]
                # Remove quotes if present
                answer_part = answer_part.strip("\"'")
                if len(answer_part) > 20:  # Ensure we have substantial content
                    return answer_part

        # Fallback: Look for quoted content which often contains the answer
        import re

        quoted: list[str] = re.findall(r'"([^"]{30,})"', reasoning_content)
        if quoted:
            # Return the last substantial quote (likely the final answer attempt)
            return quoted[-1]

        # Last resort: return the last paragraph of reasoning
        paragraphs = [p.strip() for p in reasoning_content.split("\n\n") if p.strip()]
        if paragraphs:
            last_para = paragraphs[-1]
            # Clean up meta-commentary
            if last_para.startswith(("We need to", "The user", "So we")) and len(paragraphs) > 1:
                # Try the second-to-last paragraph
                return paragraphs[-2][:500]
            return last_para[:500]

        return reasoning_content[:500]
