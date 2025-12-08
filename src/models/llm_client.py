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

            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            content = response.choices[0].message.content
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
