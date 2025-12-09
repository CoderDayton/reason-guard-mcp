"""LLM Client wrapper with retry logic and timeout protection."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from src.models.model_config import ModelConfig, TruncationStrategy, get_model_config
from src.utils.errors import LLMException
from src.utils.retry import retry_with_backoff

if TYPE_CHECKING:
    pass

# Patterns that identify reasoning models needing higher token limits
# Note: This is kept for backward compatibility with token scaling logic.
# For comprehensive model detection, use model_config.get_model_config()
REASONING_MODEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"minimax", re.IGNORECASE),
    re.compile(r"deepseek", re.IGNORECASE),
    re.compile(r"qwen", re.IGNORECASE),
    re.compile(r"\bo1\b", re.IGNORECASE),  # o1, o1-preview, o1-mini
    re.compile(r"reasoning", re.IGNORECASE),
    re.compile(r"think", re.IGNORECASE),
]


class LLMClient:
    """Wrapper around OpenAI LLM with retry logic and structured error handling.

    This client provides:
    - Automatic retry with exponential backoff
    - Timeout protection on all calls
    - Both sync and async interfaces
    - Token estimation utilities
    - Structured error handling
    - Auto-detection of reasoning models with token multiplier
    - Model-specific optimal sampling parameters

    Example:
        client = LLMClient(model="gpt-5.1")
        response = client.generate("What is 2+2?")
        print(response)  # "4"

        # For reasoning models, tokens are auto-scaled:
        client = LLMClient(model="deepseek-reasoner")
        # max_tokens=300 becomes 900 (3x multiplier)

        # Model-specific configs are auto-applied:
        client = LLMClient(model="claude-3-opus")
        # Uses Claude's optimal params, handles mutual exclusion of temp/top_p

    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-5.1",
        timeout: int = 60,
        max_retries: int = 3,
        reasoning_token_multiplier: float | None = None,
        default_temperature: float | None = None,
    ) -> None:
        """Initialize LLM client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            base_url: Optional custom API base URL for OpenAI-compatible endpoints.
            model: Model name to use for completions.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.
            reasoning_token_multiplier: Multiplier for max_tokens when using
                reasoning models. If None, auto-detects based on model name
                (3.0x for reasoning models, 1.0x for standard models).
            default_temperature: Default sampling temperature (0.0-2.0).
                If None, uses model-specific optimal temperature from config.
                Lower = more deterministic, higher = more creative.
                Can be overridden per-call.

        Raises:
            LLMException: If API key is not provided and not in environment.

        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMException("OPENAI_API_KEY environment variable not set")

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Get model-specific configuration
        self._model_config: ModelConfig = get_model_config(model)

        # Temperature priority: explicit arg > model config > fallback 0.7
        if default_temperature is not None:
            self.default_temperature = default_temperature
        elif self._model_config.temperature is not None:
            self.default_temperature = self._model_config.temperature
        else:
            self.default_temperature = 0.7

        # Auto-detect if this is a reasoning model (use both methods for token scaling)
        self._is_reasoning_model = (
            self._detect_reasoning_model(model) or self._model_config.is_reasoning_model
        )

        # Set token multiplier: explicit > auto-detect > default
        if reasoning_token_multiplier is not None:
            self._token_multiplier = reasoning_token_multiplier
        elif self._is_reasoning_model:
            self._token_multiplier = 3.0  # Reasoning models need ~3x tokens for CoT
        else:
            self._token_multiplier = 1.0

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

        if self._is_reasoning_model:
            logger.info(
                f"LLM client initialized with reasoning model: {model} "
                f"(token multiplier: {self._token_multiplier}x, "
                f"config: {self._model_config.name})"
            )
        else:
            logger.info(
                f"LLM client initialized with model: {model} "
                f"(config: {self._model_config.name}, temp: {self.default_temperature})"
            )

    @staticmethod
    def _detect_reasoning_model(model_name: str) -> bool:
        """Detect if model is a reasoning model based on name patterns.

        Args:
            model_name: The model identifier string.

        Returns:
            True if model appears to be a reasoning model.

        """
        return any(pattern.search(model_name) for pattern in REASONING_MODEL_PATTERNS)

    def _scale_tokens(self, max_tokens: int) -> int:
        """Scale max_tokens by the reasoning multiplier.

        Args:
            max_tokens: Original token limit requested.

        Returns:
            Scaled token limit for reasoning models.

        """
        return int(max_tokens * self._token_multiplier)

    def _build_sampling_params(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Build sampling parameters respecting model capabilities.

        Uses model-specific config to determine which parameters to include
        and handles edge cases like:
        - Models that don't support temperature (o1, o3)
        - Models with mutually exclusive temp/top_p (Claude)
        - Models with optimal top_k values (Gemma, Qwen)

        Args:
            temperature: User-specified temperature (None = use config default).
            top_p: User-specified top_p (None = use config default).

        Returns:
            Dict with API-compatible sampling parameters.

        """
        # Get effective temperature: user override > default_temperature
        eff_temp = temperature if temperature is not None else self.default_temperature

        # Get model-aware params using config
        return self._model_config.to_api_params(
            temperature_override=eff_temp,
            top_p_override=top_p,
            prefer_temperature=True,  # Prefer temp when mutually exclusive
        )

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        auto_truncate: bool = True,
        truncation_strategy: TruncationStrategy = TruncationStrategy.KEEP_END,
    ) -> str:
        """Generate text using LLM (synchronous).

        Args:
            prompt: User prompt/question.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0). If None, uses
                the model's optimal temperature from config.
            top_p: Top-p (nucleus) sampling parameter. If None, uses
                the model's optimal top_p from config.
            system_prompt: Optional system message to set context.
            auto_truncate: If True, automatically truncate long prompts to fit
                context window. If False, raise LLMException on overflow.
            truncation_strategy: How to truncate if prompt exceeds context.
                KEEP_END keeps recent context (default, usually most relevant).
                KEEP_START keeps beginning. KEEP_BOTH_ENDS keeps start and end.

        Returns:
            Generated text response.

        Raises:
            LLMException: If generation fails after retries, or if auto_truncate
                is False and prompt exceeds context window.

        """
        try:
            # Prepare messages with automatic truncation if needed
            messages = self._prepare_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                truncation_strategy=truncation_strategy,
                auto_truncate=auto_truncate,
            )

            # Scale tokens for reasoning models
            scaled_tokens = self._scale_tokens(max_tokens)

            # Build sampling params using model config
            sampling_params = self._build_sampling_params(temperature, top_p)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=scaled_tokens,
                **sampling_params,
            )

            content = response.choices[0].message.content
            return content or ""

        except LLMException:
            # Re-raise LLMException (e.g., from _prepare_messages)
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMException(f"Generation failed: {e!s}") from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        auto_truncate: bool = True,
        truncation_strategy: TruncationStrategy = TruncationStrategy.KEEP_END,
    ) -> str:
        """Generate text using LLM (asynchronous).

        Args:
            prompt: User prompt/question.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0). If None, uses
                the model's optimal temperature from config.
            top_p: Top-p (nucleus) sampling parameter. If None, uses
                the model's optimal top_p from config.
            system_prompt: Optional system message to set context.
            auto_truncate: If True, automatically truncate long prompts to fit
                context window. If False, raise LLMException on overflow.
            truncation_strategy: How to truncate if prompt exceeds context.
                KEEP_END keeps recent context (default, usually most relevant).
                KEEP_START keeps beginning. KEEP_BOTH_ENDS keeps start and end.

        Returns:
            Generated text response.

        Raises:
            LLMException: If generation fails after retries, or if auto_truncate
                is False and prompt exceeds context window.

        """
        try:
            # Prepare messages with automatic truncation if needed
            messages = self._prepare_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                truncation_strategy=truncation_strategy,
                auto_truncate=auto_truncate,
            )

            # Scale tokens for reasoning models
            scaled_tokens = self._scale_tokens(max_tokens)

            # Build sampling params using model config
            sampling_params = self._build_sampling_params(temperature, top_p)

            logger.debug(
                f"Sending async request to {self.model} with {len(prompt)} char prompt"
                + (
                    f" (tokens: {max_tokens}→{scaled_tokens})"
                    if scaled_tokens != max_tokens
                    else ""
                )
                + f" (params: {sampling_params})"
            )

            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=scaled_tokens,
                **sampling_params,
            )

            # Extract content with fallback for non-standard API responses
            content: str | None = None
            finish_reason: str | None = None

            if response.choices:
                choice = response.choices[0]
                finish_reason = choice.finish_reason
                content = choice.message.content

                # Handle <think> tags in content (DeepSeek, Qwen, etc.)
                # These models return: <think>reasoning...</think>actual answer
                if content:
                    content = self._strip_think_tags(content)

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

                # Validate content quality - detect garbage responses
                if content:
                    content = self._validate_and_clean_content(content)

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

        except LLMException:
            # Re-raise LLMException (e.g., from _prepare_messages)
            raise
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

    def _prepare_messages(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2000,
        truncation_strategy: TruncationStrategy = TruncationStrategy.KEEP_END,
        auto_truncate: bool = True,
    ) -> list[dict[str, str]]:
        """Prepare messages with context-aware truncation.

        Automatically truncates the user prompt if it would exceed the model's
        context window, leaving room for the system prompt and expected output.

        Args:
            prompt: User prompt/question.
            system_prompt: Optional system message to set context.
            max_tokens: Maximum tokens reserved for response.
            truncation_strategy: How to truncate if prompt exceeds context.
                KEEP_END is default since recent context is usually more relevant.
            auto_truncate: If True, automatically truncate long prompts.
                If False, raise LLMException when prompt exceeds context.

        Returns:
            List of message dicts ready for API call.

        Raises:
            LLMException: If auto_truncate=False and prompt exceeds context.

        """
        messages: list[dict[str, str]] = []

        # Calculate tokens used by system prompt
        system_tokens = 0
        if system_prompt:
            system_tokens = self.estimate_tokens(system_prompt) + 4  # +4 for message overhead
            messages.append({"role": "system", "content": system_prompt})

        # Calculate available tokens for user prompt
        # Reserve: output tokens + system tokens + message overhead
        scaled_output_tokens = self._scale_tokens(max_tokens)
        additional_tokens = system_tokens + 4  # +4 for user message overhead

        # Check if prompt fits
        if not self._model_config.fits_in_context(
            prompt,
            reserve_for_output=scaled_output_tokens,
            additional_tokens=additional_tokens,
        ):
            estimated_prompt_tokens = self._model_config.estimate_tokens(prompt)
            available = self._model_config.get_available_tokens(
                reserve_for_output=scaled_output_tokens,
                used_tokens=additional_tokens,
            )

            if auto_truncate:
                # Use ERROR strategy internally if auto_truncate is False
                effective_strategy = truncation_strategy
                truncated_prompt = self._model_config.truncate_to_fit(
                    prompt,
                    reserve_for_output=scaled_output_tokens,
                    additional_tokens=additional_tokens,
                    strategy=effective_strategy,
                )

                logger.warning(
                    f"Prompt truncated to fit context window: "
                    f"{estimated_prompt_tokens} → "
                    f"{self._model_config.estimate_tokens(truncated_prompt)} tokens "
                    f"(available: {available}, strategy: {truncation_strategy.value})"
                )
                prompt = truncated_prompt
            else:
                raise LLMException(
                    f"Prompt ({estimated_prompt_tokens} tokens) exceeds available context "
                    f"({available} tokens) for model {self.model}. "
                    f"Context window: {self._model_config.max_context_length}, "
                    f"reserved for output: {scaled_output_tokens}, "
                    f"system prompt: {system_tokens} tokens."
                )

        messages.append({"role": "user", "content": prompt})
        return messages

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

    def _strip_think_tags(self, content: str) -> str:
        """Strip <think>...</think> tags and return the answer portion.

        Some reasoning models (DeepSeek, Qwen, etc.) return responses in format:
        <think>chain of thought reasoning...</think>actual answer

        This method extracts just the answer after the closing </think> tag.
        If the content is entirely within think tags with no answer after,
        it extracts the answer from the thinking content.

        Args:
            content: Raw response content that may contain think tags.

        Returns:
            The answer portion, or extracted answer from thinking if no
            content exists after the tags.

        """
        import re

        # Check for <think> tags (case-insensitive, handles whitespace)
        think_pattern = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL | re.IGNORECASE)
        match = think_pattern.search(content)

        if match:
            thinking_content = match.group(1).strip()
            answer_content = match.group(2).strip()

            if answer_content:
                # There's content after </think>, that's the answer
                logger.debug(
                    f"Stripped <think> tags: {len(thinking_content)} chars thinking, "
                    f"{len(answer_content)} chars answer"
                )
                return answer_content
            elif thinking_content:
                # Only thinking content, no answer - extract from thinking
                logger.debug(
                    f"Only <think> content found ({len(thinking_content)} chars), "
                    "extracting answer from thinking"
                )
                return self._extract_answer_from_reasoning(thinking_content)

        # Check for unclosed <think> tag (model hit token limit during thinking)
        unclosed_pattern = re.compile(r"<think>\s*(.*)", re.DOTALL | re.IGNORECASE)
        unclosed_match = unclosed_pattern.search(content)

        if unclosed_match and "</think>" not in content.lower():
            thinking_content = unclosed_match.group(1).strip()
            if thinking_content:
                logger.debug(
                    f"Unclosed <think> tag found ({len(thinking_content)} chars), "
                    "extracting answer from partial thinking"
                )
                return self._extract_answer_from_reasoning(thinking_content)

        # No think tags, return content as-is
        return content

    def _validate_and_clean_content(self, content: str) -> str:
        """Validate response content quality and clean garbage responses.

        Detects common failure modes where models return garbage:
        - Strings of repeated punctuation/whitespace
        - Very low alphanumeric character ratio
        - Repeated character sequences

        Args:
            content: The response content to validate.

        Returns:
            The original content if valid, empty string if garbage detected.

        """
        if not content:
            return ""

        # Calculate alphanumeric ratio
        alphanumeric_count = sum(1 for c in content if c.isalnum())
        total_chars = len(content)

        if total_chars == 0:
            return ""

        alphanumeric_ratio = alphanumeric_count / total_chars

        # Content should have at least 20% alphanumeric characters
        # Valid prose typically has 70-85%, even code has 40-60%
        min_ratio = 0.20

        if alphanumeric_ratio < min_ratio:
            logger.warning(
                f"Garbage response detected: alphanumeric ratio {alphanumeric_ratio:.2%} "
                f"(threshold: {min_ratio:.0%}). Content preview: {content[:100]!r}"
            )
            return ""

        # Check for excessive character repetition (e.g., ",,,,,,," or "      ")
        # Count unique characters vs total
        unique_chars = len(set(content))
        repetition_ratio = unique_chars / total_chars

        # If very few unique characters relative to length, likely garbage
        # (e.g., ",   ,   ,   ," has ~4 unique chars but could be 100+ length)
        if total_chars > 50 and repetition_ratio < 0.05:
            logger.warning(
                f"Garbage response detected: repetition ratio {repetition_ratio:.2%} "
                f"(only {unique_chars} unique chars in {total_chars}). "
                f"Content preview: {content[:100]!r}"
            )
            return ""

        return content
