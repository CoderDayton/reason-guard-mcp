"""Model-specific configuration for optimal sampling parameters.

This module provides optimal default parameters for different LLM models/families.
Research shows that different models perform best with different sampling settings.

Key insights from model documentation:
- Reasoning models (o1, o3, R1) often don't support temperature/top_p
- Some providers (Claude) reject requests with both temp AND top_p
- DeepSeek V3 API auto-scales temperature (user's 1.0 → internal 0.3)
- Qwen has different optimal settings for thinking vs non-thinking modes

Usage:
    config = get_model_config("deepseek-chat")
    # Returns ModelConfig with optimal temperature, top_p, etc.

    params = config.to_api_params()
    # Returns dict suitable for API calls, excluding unsupported params
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelCapability(Enum):
    """Flags for model capabilities that affect parameter handling."""

    SUPPORTS_TEMPERATURE = "supports_temperature"
    SUPPORTS_TOP_P = "supports_top_p"
    SUPPORTS_TOP_K = "supports_top_k"
    SUPPORTS_FREQUENCY_PENALTY = "supports_frequency_penalty"
    SUPPORTS_PRESENCE_PENALTY = "supports_presence_penalty"
    IS_REASONING_MODEL = "is_reasoning_model"
    # Some models don't allow both temp and top_p simultaneously
    MUTUALLY_EXCLUSIVE_SAMPLING = "mutually_exclusive_sampling"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model or model family.

    Attributes:
        name: Human-readable name for the model/family.
        temperature: Optimal temperature setting (None if not supported).
        top_p: Optimal top_p/nucleus sampling (None if not supported).
        top_k: Optimal top_k sampling (None if not supported).
        frequency_penalty: Penalty for token frequency (None if not supported).
        presence_penalty: Penalty for token presence (None if not supported).
        capabilities: Set of ModelCapability flags.
        notes: Human-readable notes about optimal usage.

    """

    name: str
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    capabilities: frozenset[ModelCapability] = field(
        default_factory=lambda: frozenset(
            {ModelCapability.SUPPORTS_TEMPERATURE, ModelCapability.SUPPORTS_TOP_P}
        )
    )
    notes: str = ""

    def supports(self, capability: ModelCapability) -> bool:
        """Check if model supports a specific capability."""
        return capability in self.capabilities

    @property
    def is_reasoning_model(self) -> bool:
        """Check if this is a reasoning model (extended thinking)."""
        return ModelCapability.IS_REASONING_MODEL in self.capabilities

    @property
    def supports_temperature(self) -> bool:
        """Check if model supports temperature parameter."""
        return ModelCapability.SUPPORTS_TEMPERATURE in self.capabilities

    @property
    def supports_top_p(self) -> bool:
        """Check if model supports top_p parameter."""
        return ModelCapability.SUPPORTS_TOP_P in self.capabilities

    @property
    def has_mutually_exclusive_sampling(self) -> bool:
        """Check if temp and top_p cannot be used together."""
        return ModelCapability.MUTUALLY_EXCLUSIVE_SAMPLING in self.capabilities

    def to_api_params(
        self,
        *,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
        prefer_temperature: bool = True,
    ) -> dict[str, Any]:
        """Convert config to API-compatible parameter dict.

        Args:
            temperature_override: User-specified temperature (takes precedence).
            top_p_override: User-specified top_p (takes precedence).
            prefer_temperature: When model has mutually exclusive sampling,
                prefer temperature over top_p. Default True.

        Returns:
            Dict with only supported parameters for API calls.

        """
        params: dict[str, Any] = {}

        # Determine effective values (override > config default)
        eff_temp = temperature_override if temperature_override is not None else self.temperature
        eff_top_p = top_p_override if top_p_override is not None else self.top_p

        # Handle mutually exclusive sampling (e.g., Claude)
        if self.has_mutually_exclusive_sampling:
            if prefer_temperature and self.supports_temperature and eff_temp is not None:
                params["temperature"] = eff_temp
            elif self.supports_top_p and eff_top_p is not None:
                params["top_p"] = eff_top_p
        else:
            # Normal case: add both if supported
            if self.supports_temperature and eff_temp is not None:
                params["temperature"] = eff_temp
            if self.supports_top_p and eff_top_p is not None:
                params["top_p"] = eff_top_p

        # Add other parameters if supported
        if ModelCapability.SUPPORTS_TOP_K in self.capabilities and self.top_k is not None:
            params["top_k"] = self.top_k

        if (
            ModelCapability.SUPPORTS_FREQUENCY_PENALTY in self.capabilities
            and self.frequency_penalty is not None
        ):
            params["frequency_penalty"] = self.frequency_penalty

        if (
            ModelCapability.SUPPORTS_PRESENCE_PENALTY in self.capabilities
            and self.presence_penalty is not None
        ):
            params["presence_penalty"] = self.presence_penalty

        return params


# =============================================================================
# Model Configuration Registry
# =============================================================================

# Default configuration for unknown models
DEFAULT_CONFIG = ModelConfig(
    name="default",
    temperature=0.7,
    top_p=0.9,
    notes="Default settings for unknown models",
)

# Standard capabilities for most models
_STANDARD_CAPS = frozenset(
    {
        ModelCapability.SUPPORTS_TEMPERATURE,
        ModelCapability.SUPPORTS_TOP_P,
        ModelCapability.SUPPORTS_FREQUENCY_PENALTY,
        ModelCapability.SUPPORTS_PRESENCE_PENALTY,
    }
)

# Reasoning model capabilities (no sampling params)
_REASONING_CAPS = frozenset({ModelCapability.IS_REASONING_MODEL})

# Capabilities with top_k support
_WITH_TOP_K = frozenset(
    {
        ModelCapability.SUPPORTS_TEMPERATURE,
        ModelCapability.SUPPORTS_TOP_P,
        ModelCapability.SUPPORTS_TOP_K,
    }
)


# =============================================================================
# Model Family Configurations
# =============================================================================

# OpenAI GPT-5 family (latest)
GPT5_CONFIG = ModelConfig(
    name="OpenAI GPT-5",
    temperature=0.7,
    top_p=0.95,
    capabilities=_STANDARD_CAPS,
    notes="General: 0.7, Coding: 0.0-0.3. Alter temp OR top_p, not both recommended.",
)

GPT5_CODING_CONFIG = ModelConfig(
    name="OpenAI GPT-5 (Coding)",
    temperature=0.2,
    top_p=0.95,
    capabilities=_STANDARD_CAPS,
    notes="Lower temperature for deterministic code generation.",
)

# OpenAI GPT-4 family (legacy, same settings as GPT-5)
GPT4_CONFIG = GPT5_CONFIG  # Alias for backward compatibility

GPT4_CODING_CONFIG = GPT5_CODING_CONFIG  # Alias for backward compatibility

# OpenAI o1/o3 reasoning models - fixed sampling, no params supported
O1_CONFIG = ModelConfig(
    name="OpenAI o1/o3 Reasoning",
    temperature=None,
    top_p=None,
    capabilities=_REASONING_CAPS,
    notes="Reasoning models use fixed temperature=1, top_p=1. Sampling params not supported.",
)

# DeepSeek V3 (chat model)
DEEPSEEK_V3_CONFIG = ModelConfig(
    name="DeepSeek V3",
    temperature=0.3,
    top_p=None,
    capabilities=frozenset({ModelCapability.SUPPORTS_TEMPERATURE}),
    notes="API auto-scales temp (1.0→0.3). Coding: 0.0, Conversation: 1.3.",
)

# DeepSeek R1 (reasoning model)
DEEPSEEK_R1_CONFIG = ModelConfig(
    name="DeepSeek R1",
    temperature=0.6,
    top_p=0.95,
    capabilities=frozenset(
        {
            ModelCapability.IS_REASONING_MODEL,
            ModelCapability.SUPPORTS_TEMPERATURE,
            ModelCapability.SUPPORTS_TOP_P,
        }
    ),
    notes="Reasoning model with thinking tokens. temp=0.6, top_p=0.95 recommended.",
)

# Mistral family
MISTRAL_SMALL_CONFIG = ModelConfig(
    name="Mistral Small 3.2",
    temperature=0.15,
    top_p=None,
    capabilities=frozenset({ModelCapability.SUPPORTS_TEMPERATURE}),
    notes="Very low temperature recommended for Mistral Small.",
)

MISTRAL_LARGE_CONFIG = ModelConfig(
    name="Mistral Large/Medium",
    temperature=0.7,
    top_p=0.95,
    capabilities=_STANDARD_CAPS,
    notes="Standard settings for Mistral Large/Medium models.",
)

MAGISTRAL_CONFIG = ModelConfig(
    name="Mistral Magistral (Reasoning)",
    temperature=0.7,
    top_p=0.95,
    capabilities=frozenset(
        {
            ModelCapability.IS_REASONING_MODEL,
            ModelCapability.SUPPORTS_TEMPERATURE,
            ModelCapability.SUPPORTS_TOP_P,
        }
    ),
    notes="Reasoning variant of Mistral.",
)

# Qwen family
QWEN_THINKING_CONFIG = ModelConfig(
    name="Qwen3 (Thinking Mode)",
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    capabilities=frozenset(
        {
            ModelCapability.IS_REASONING_MODEL,
            ModelCapability.SUPPORTS_TEMPERATURE,
            ModelCapability.SUPPORTS_TOP_P,
            ModelCapability.SUPPORTS_TOP_K,
        }
    ),
    notes="For thinking mode with /think or enable_thinking=True.",
)

QWEN_NON_THINKING_CONFIG = ModelConfig(
    name="Qwen3 (Non-Thinking Mode)",
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    presence_penalty=1.5,
    capabilities=frozenset(
        {
            ModelCapability.SUPPORTS_TEMPERATURE,
            ModelCapability.SUPPORTS_TOP_P,
            ModelCapability.SUPPORTS_TOP_K,
            ModelCapability.SUPPORTS_PRESENCE_PENALTY,
        }
    ),
    notes="For non-thinking mode. presence_penalty=1.5 reduces repetition.",
)

# Llama family
LLAMA4_CONFIG = ModelConfig(
    name="Llama 4",
    temperature=0.6,
    top_p=0.9,
    capabilities=_STANDARD_CAPS,
    notes="From Meta's generation_config.json.",
)

LLAMA3_CONFIG = ModelConfig(
    name="Llama 3",
    temperature=0.6,
    top_p=0.9,
    capabilities=_STANDARD_CAPS,
    notes="Standard settings for Llama 3 family.",
)

# Google Gemma family
GEMMA_CONFIG = ModelConfig(
    name="Google Gemma 3",
    temperature=1.0,
    top_p=0.96,
    top_k=64,
    capabilities=_WITH_TOP_K,
    notes="Higher temperature than typical. From HuggingFace config.",
)

# Google Gemini family
GEMINI_CONFIG = ModelConfig(
    name="Google Gemini",
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    capabilities=_WITH_TOP_K,
    notes="Standard Gemini settings with top_k support.",
)

# Anthropic Claude family
CLAUDE_CONFIG = ModelConfig(
    name="Anthropic Claude",
    temperature=0.7,
    top_p=0.9,
    capabilities=frozenset(
        {
            ModelCapability.SUPPORTS_TEMPERATURE,
            ModelCapability.SUPPORTS_TOP_P,
            ModelCapability.MUTUALLY_EXCLUSIVE_SAMPLING,
        }
    ),
    notes="Cannot specify both temperature AND top_p simultaneously.",
)

# Microsoft Phi family
PHI_CONFIG = ModelConfig(
    name="Microsoft Phi-4",
    temperature=0.7,
    top_p=0.95,
    capabilities=_STANDARD_CAPS,
    notes="Standard Phi-4 settings.",
)

PHI_REASONING_CONFIG = ModelConfig(
    name="Microsoft Phi-4 Reasoning",
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    capabilities=frozenset(
        {
            ModelCapability.IS_REASONING_MODEL,
            ModelCapability.SUPPORTS_TEMPERATURE,
            ModelCapability.SUPPORTS_TOP_P,
            ModelCapability.SUPPORTS_TOP_K,
        }
    ),
    notes="Reasoning variant with extended thinking.",
)

# Cohere Command family
COHERE_CONFIG = ModelConfig(
    name="Cohere Command R+",
    temperature=0.3,
    top_p=0.75,
    capabilities=_STANDARD_CAPS,
    notes="Conservative defaults from Cohere docs.",
)

# MiniMax family
MINIMAX_CONFIG = ModelConfig(
    name="MiniMax",
    temperature=0.7,
    top_p=0.9,
    capabilities=frozenset(
        {
            ModelCapability.IS_REASONING_MODEL,
            ModelCapability.SUPPORTS_TEMPERATURE,
            ModelCapability.SUPPORTS_TOP_P,
        }
    ),
    notes="MiniMax reasoning models (M1, M2).",
)

# Groq-hosted models (use underlying model configs, but note Groq limits)
GROQ_CONFIG = ModelConfig(
    name="Groq",
    temperature=0.7,
    top_p=0.9,
    capabilities=_STANDARD_CAPS,
    notes="Groq-hosted models. Apply underlying model's optimal config.",
)


# =============================================================================
# Pattern Matching for Model Identification
# =============================================================================

# Order matters: more specific patterns should come first
MODEL_PATTERNS: list[tuple[re.Pattern[str], ModelConfig]] = [
    # OpenAI reasoning models (o1, o3, o1-mini, o1-preview, o3-mini)
    (re.compile(r"\b(o1|o3)(-mini|-preview)?\b", re.IGNORECASE), O1_CONFIG),
    # OpenAI GPT-5 variants (latest)
    (re.compile(r"gpt-5.*codex|gpt-5.*code", re.IGNORECASE), GPT5_CODING_CONFIG),
    (re.compile(r"gpt-5|gpt5", re.IGNORECASE), GPT5_CONFIG),
    # OpenAI GPT-4 variants (legacy)
    (re.compile(r"gpt-4.*codex|gpt-4.*code", re.IGNORECASE), GPT4_CODING_CONFIG),
    (re.compile(r"gpt-4|gpt4", re.IGNORECASE), GPT4_CONFIG),
    (re.compile(r"gpt-3\.5|gpt3\.5", re.IGNORECASE), GPT4_CONFIG),  # Similar settings
    # DeepSeek
    (re.compile(r"deepseek.*r1|deepseek.*reason", re.IGNORECASE), DEEPSEEK_R1_CONFIG),
    (re.compile(r"deepseek", re.IGNORECASE), DEEPSEEK_V3_CONFIG),
    # Mistral
    (re.compile(r"magistral|mistral.*reason", re.IGNORECASE), MAGISTRAL_CONFIG),
    (re.compile(r"mistral.*small", re.IGNORECASE), MISTRAL_SMALL_CONFIG),
    (re.compile(r"mistral", re.IGNORECASE), MISTRAL_LARGE_CONFIG),
    # Qwen - check for thinking mode indicators
    (re.compile(r"qwen.*think|qwq", re.IGNORECASE), QWEN_THINKING_CONFIG),
    (re.compile(r"qwen", re.IGNORECASE), QWEN_NON_THINKING_CONFIG),
    # Llama
    (re.compile(r"llama.*4|llama-4", re.IGNORECASE), LLAMA4_CONFIG),
    (re.compile(r"llama", re.IGNORECASE), LLAMA3_CONFIG),
    # Google
    (re.compile(r"gemma", re.IGNORECASE), GEMMA_CONFIG),
    (re.compile(r"gemini", re.IGNORECASE), GEMINI_CONFIG),
    # Anthropic
    (re.compile(r"claude", re.IGNORECASE), CLAUDE_CONFIG),
    # Microsoft
    (re.compile(r"phi.*reason", re.IGNORECASE), PHI_REASONING_CONFIG),
    (re.compile(r"phi", re.IGNORECASE), PHI_CONFIG),
    # Cohere
    (re.compile(r"command|cohere", re.IGNORECASE), COHERE_CONFIG),
    # MiniMax
    (re.compile(r"minimax|abab", re.IGNORECASE), MINIMAX_CONFIG),
    # Groq (fallback - will use default config but flag as Groq)
    (re.compile(r"groq", re.IGNORECASE), GROQ_CONFIG),
]


def get_model_config(model_name: str) -> ModelConfig:
    """Get optimal configuration for a model based on its name.

    Args:
        model_name: The model identifier string (e.g., "gpt-5.1",
            "deepseek-chat", "claude-3-opus").

    Returns:
        ModelConfig with optimal parameters for the model.
        Returns DEFAULT_CONFIG if no specific config is found.

    Example:
        >>> config = get_model_config("deepseek-r1")
        >>> config.temperature
        0.6
        >>> config.is_reasoning_model
        True

    """
    for pattern, config in MODEL_PATTERNS:
        if pattern.search(model_name):
            return config

    return DEFAULT_CONFIG


def get_api_params(
    model_name: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    prefer_temperature: bool = True,
) -> dict[str, Any]:
    """Get API-ready parameters for a model with optional overrides.

    Convenience function that combines get_model_config and to_api_params.

    Args:
        model_name: The model identifier string.
        temperature: User-specified temperature override.
        top_p: User-specified top_p override.
        prefer_temperature: When model has mutually exclusive sampling,
            prefer temperature over top_p.

    Returns:
        Dict with API-compatible sampling parameters.

    Example:
        >>> params = get_api_params("claude-3-opus", temperature=0.5)
        >>> params
        {'temperature': 0.5}  # top_p excluded due to mutual exclusion

    """
    config = get_model_config(model_name)
    return config.to_api_params(
        temperature_override=temperature,
        top_p_override=top_p,
        prefer_temperature=prefer_temperature,
    )


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model based on its name.

    Args:
        model_name: The model identifier string.

    Returns:
        True if model is identified as a reasoning model.

    """
    config = get_model_config(model_name)
    return config.is_reasoning_model


def get_effective_temperature(
    model_name: str,
    user_temperature: float | None = None,
    default_temperature: float = 0.7,
) -> float | None:
    """Get the effective temperature for a model.

    Args:
        model_name: The model identifier string.
        user_temperature: User-specified temperature (highest priority).
        default_temperature: Fallback if model config has no temperature.

    Returns:
        The temperature to use, or None if model doesn't support it.

    """
    config = get_model_config(model_name)

    if not config.supports_temperature:
        return None

    if user_temperature is not None:
        return user_temperature

    if config.temperature is not None:
        return config.temperature

    return default_temperature
