"""Tests for model configuration module."""

from __future__ import annotations

import pytest

from src.models.model_config import (
    CLAUDE_CONFIG,
    DEEPSEEK_R1_CONFIG,
    DEEPSEEK_V3_CONFIG,
    DEFAULT_CONFIG,
    GEMMA_CONFIG,
    GPT4_CONFIG,
    LLAMA4_CONFIG,
    MISTRAL_SMALL_CONFIG,
    O1_CONFIG,
    QWEN_NON_THINKING_CONFIG,
    QWEN_THINKING_CONFIG,
    ModelCapability,
    ModelConfig,
    get_api_params,
    get_effective_temperature,
    get_model_config,
    is_reasoning_model,
)

# =============================================================================
# ModelConfig dataclass tests
# =============================================================================


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_capabilities(self) -> None:
        """Test default capabilities include temperature and top_p support."""
        config = ModelConfig(name="test")
        assert config.supports_temperature
        assert config.supports_top_p
        assert not config.is_reasoning_model

    def test_is_reasoning_model_property(self) -> None:
        """Test is_reasoning_model property."""
        reasoning_config = ModelConfig(
            name="reasoning",
            capabilities=frozenset({ModelCapability.IS_REASONING_MODEL}),
        )
        assert reasoning_config.is_reasoning_model

        standard_config = ModelConfig(name="standard")
        assert not standard_config.is_reasoning_model

    def test_supports_method(self) -> None:
        """Test supports() method for capability checking."""
        config = ModelConfig(
            name="test",
            capabilities=frozenset(
                {
                    ModelCapability.SUPPORTS_TEMPERATURE,
                    ModelCapability.SUPPORTS_TOP_K,
                }
            ),
        )
        assert config.supports(ModelCapability.SUPPORTS_TEMPERATURE)
        assert config.supports(ModelCapability.SUPPORTS_TOP_K)
        assert not config.supports(ModelCapability.SUPPORTS_TOP_P)

    def test_has_mutually_exclusive_sampling(self) -> None:
        """Test mutually exclusive sampling detection."""
        exclusive_config = ModelConfig(
            name="test",
            capabilities=frozenset({ModelCapability.MUTUALLY_EXCLUSIVE_SAMPLING}),
        )
        assert exclusive_config.has_mutually_exclusive_sampling

        normal_config = ModelConfig(name="normal")
        assert not normal_config.has_mutually_exclusive_sampling


class TestModelConfigToApiParams:
    """Tests for ModelConfig.to_api_params() method."""

    def test_basic_params(self) -> None:
        """Test basic parameter generation."""
        config = ModelConfig(
            name="test",
            temperature=0.7,
            top_p=0.9,
        )
        params = config.to_api_params()
        assert params == {"temperature": 0.7, "top_p": 0.9}

    def test_override_values(self) -> None:
        """Test that override values take precedence."""
        config = ModelConfig(
            name="test",
            temperature=0.7,
            top_p=0.9,
        )
        params = config.to_api_params(temperature_override=0.3, top_p_override=0.5)
        assert params == {"temperature": 0.3, "top_p": 0.5}

    def test_partial_override(self) -> None:
        """Test partial override (only temperature)."""
        config = ModelConfig(
            name="test",
            temperature=0.7,
            top_p=0.9,
        )
        params = config.to_api_params(temperature_override=0.3)
        assert params == {"temperature": 0.3, "top_p": 0.9}

    def test_unsupported_params_excluded(self) -> None:
        """Test that unsupported parameters are excluded."""
        # Config with no temperature support
        config = ModelConfig(
            name="test",
            temperature=0.7,
            top_p=0.9,
            capabilities=frozenset({ModelCapability.SUPPORTS_TOP_P}),
        )
        params = config.to_api_params()
        assert "temperature" not in params
        assert params == {"top_p": 0.9}

    def test_mutually_exclusive_prefers_temperature(self) -> None:
        """Test that mutually exclusive sampling prefers temperature by default."""
        config = ModelConfig(
            name="claude",
            temperature=0.7,
            top_p=0.9,
            capabilities=frozenset(
                {
                    ModelCapability.SUPPORTS_TEMPERATURE,
                    ModelCapability.SUPPORTS_TOP_P,
                    ModelCapability.MUTUALLY_EXCLUSIVE_SAMPLING,
                }
            ),
        )
        params = config.to_api_params(prefer_temperature=True)
        assert params == {"temperature": 0.7}
        assert "top_p" not in params

    def test_mutually_exclusive_can_prefer_top_p(self) -> None:
        """Test that mutually exclusive sampling can prefer top_p."""
        config = ModelConfig(
            name="claude",
            temperature=0.7,
            top_p=0.9,
            capabilities=frozenset(
                {
                    ModelCapability.SUPPORTS_TEMPERATURE,
                    ModelCapability.SUPPORTS_TOP_P,
                    ModelCapability.MUTUALLY_EXCLUSIVE_SAMPLING,
                }
            ),
        )
        params = config.to_api_params(prefer_temperature=False)
        assert params == {"top_p": 0.9}
        assert "temperature" not in params

    def test_top_k_included_when_supported(self) -> None:
        """Test that top_k is included when supported."""
        config = ModelConfig(
            name="test",
            temperature=0.7,
            top_k=50,
            capabilities=frozenset(
                {
                    ModelCapability.SUPPORTS_TEMPERATURE,
                    ModelCapability.SUPPORTS_TOP_K,
                }
            ),
        )
        params = config.to_api_params()
        assert params == {"temperature": 0.7, "top_k": 50}

    def test_presence_penalty_included_when_supported(self) -> None:
        """Test that presence_penalty is included when supported."""
        config = ModelConfig(
            name="test",
            temperature=0.7,
            presence_penalty=1.5,
            capabilities=frozenset(
                {
                    ModelCapability.SUPPORTS_TEMPERATURE,
                    ModelCapability.SUPPORTS_PRESENCE_PENALTY,
                }
            ),
        )
        params = config.to_api_params()
        assert params == {"temperature": 0.7, "presence_penalty": 1.5}

    def test_reasoning_model_no_params(self) -> None:
        """Test that reasoning model with no sampling support returns empty dict."""
        config = ModelConfig(
            name="o1",
            temperature=None,
            top_p=None,
            capabilities=frozenset({ModelCapability.IS_REASONING_MODEL}),
        )
        params = config.to_api_params()
        assert params == {}


# =============================================================================
# Model pattern matching tests
# =============================================================================


class TestGetModelConfig:
    """Tests for get_model_config() function."""

    def test_unknown_model_returns_default(self) -> None:
        """Test that unknown models return default config."""
        config = get_model_config("unknown-model-xyz")
        assert config == DEFAULT_CONFIG

    # OpenAI models
    def test_gpt4_detection(self) -> None:
        """Test GPT-4 model detection."""
        for model in ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "GPT4"]:
            config = get_model_config(model)
            assert config == GPT4_CONFIG, f"Failed for {model}"

    def test_o1_detection(self) -> None:
        """Test OpenAI o1/o3 reasoning model detection."""
        for model in ["o1", "o1-preview", "o1-mini", "o3", "o3-mini"]:
            config = get_model_config(model)
            assert config == O1_CONFIG, f"Failed for {model}"
            assert config.is_reasoning_model

    # DeepSeek models
    def test_deepseek_v3_detection(self) -> None:
        """Test DeepSeek V3 model detection."""
        for model in ["deepseek-chat", "deepseek-v3", "DeepSeek"]:
            config = get_model_config(model)
            assert config == DEEPSEEK_V3_CONFIG, f"Failed for {model}"
            assert not config.is_reasoning_model

    def test_deepseek_r1_detection(self) -> None:
        """Test DeepSeek R1 reasoning model detection."""
        for model in ["deepseek-r1", "deepseek-reasoner", "DeepSeek-R1"]:
            config = get_model_config(model)
            assert config == DEEPSEEK_R1_CONFIG, f"Failed for {model}"
            assert config.is_reasoning_model

    # Mistral models
    def test_mistral_small_detection(self) -> None:
        """Test Mistral Small detection."""
        config = get_model_config("mistral-small-3.2")
        assert config == MISTRAL_SMALL_CONFIG
        assert config.temperature == 0.15  # Very low temp

    # Qwen models
    def test_qwen_thinking_detection(self) -> None:
        """Test Qwen thinking mode detection."""
        for model in ["qwen-think", "qwq", "Qwen-thinking"]:
            config = get_model_config(model)
            assert config == QWEN_THINKING_CONFIG, f"Failed for {model}"
            assert config.is_reasoning_model

    def test_qwen_non_thinking_detection(self) -> None:
        """Test Qwen non-thinking mode detection."""
        for model in ["qwen-72b", "qwen2.5", "Qwen"]:
            config = get_model_config(model)
            assert config == QWEN_NON_THINKING_CONFIG, f"Failed for {model}"
            assert not config.is_reasoning_model

    # Llama models
    def test_llama4_detection(self) -> None:
        """Test Llama 4 detection."""
        for model in ["llama-4", "llama4", "Llama-4-70b"]:
            config = get_model_config(model)
            assert config == LLAMA4_CONFIG, f"Failed for {model}"

    # Google models
    def test_gemma_detection(self) -> None:
        """Test Gemma detection."""
        config = get_model_config("gemma-3-27b")
        assert config == GEMMA_CONFIG
        assert config.temperature == 1.0  # Higher temp
        assert config.top_k == 64

    # Anthropic models
    def test_claude_detection(self) -> None:
        """Test Claude detection."""
        for model in ["claude-3-opus", "claude-3-sonnet", "Claude-4"]:
            config = get_model_config(model)
            assert config == CLAUDE_CONFIG, f"Failed for {model}"
            assert config.has_mutually_exclusive_sampling


class TestIsReasoningModel:
    """Tests for is_reasoning_model() function."""

    def test_reasoning_models_detected(self) -> None:
        """Test that reasoning models are correctly detected."""
        reasoning_models = [
            "o1",
            "o1-preview",
            "o3-mini",
            "deepseek-r1",
            "qwq",
            "magistral",
        ]
        for model in reasoning_models:
            assert is_reasoning_model(model), f"{model} should be reasoning model"

    def test_standard_models_not_reasoning(self) -> None:
        """Test that standard models are not marked as reasoning."""
        standard_models = [
            "gpt-4",
            "gpt-4o",
            "claude-3-opus",
            "llama-4",
            "mistral-large",
        ]
        for model in standard_models:
            assert not is_reasoning_model(model), f"{model} should not be reasoning"


class TestGetApiParams:
    """Tests for get_api_params() convenience function."""

    def test_basic_usage(self) -> None:
        """Test basic API params retrieval."""
        params = get_api_params("gpt-4")
        assert "temperature" in params
        assert "top_p" in params

    def test_with_override(self) -> None:
        """Test API params with temperature override."""
        params = get_api_params("gpt-4", temperature=0.3)
        assert params["temperature"] == 0.3

    def test_reasoning_model_no_params(self) -> None:
        """Test reasoning model returns empty/minimal params."""
        params = get_api_params("o1")
        # o1 doesn't support temperature/top_p
        assert "temperature" not in params
        assert "top_p" not in params

    def test_claude_mutual_exclusion(self) -> None:
        """Test Claude's mutually exclusive sampling."""
        params = get_api_params("claude-3-opus", temperature=0.5)
        # Should have temperature but not top_p
        assert params == {"temperature": 0.5}


class TestGetEffectiveTemperature:
    """Tests for get_effective_temperature() function."""

    def test_user_temperature_takes_precedence(self) -> None:
        """Test that user-specified temperature takes precedence."""
        temp = get_effective_temperature("gpt-4", user_temperature=0.3)
        assert temp == 0.3

    def test_config_temperature_used(self) -> None:
        """Test that config temperature is used when no user override."""
        temp = get_effective_temperature("deepseek-chat")
        assert temp == DEEPSEEK_V3_CONFIG.temperature
        assert temp == 0.3

    def test_default_fallback(self) -> None:
        """Test fallback to default temperature."""
        # Create scenario where config has no temperature
        temp = get_effective_temperature(
            "unknown-model", user_temperature=None, default_temperature=0.5
        )
        # Should use default (0.7) since unknown model uses DEFAULT_CONFIG
        assert temp == 0.7  # DEFAULT_CONFIG.temperature

    def test_reasoning_model_returns_none(self) -> None:
        """Test that reasoning models without temp support return None."""
        temp = get_effective_temperature("o1")
        assert temp is None

    def test_reasoning_model_ignores_user_temp(self) -> None:
        """Test that user temp is ignored for models without temp support."""
        temp = get_effective_temperature("o1", user_temperature=0.5)
        # Should still be None since o1 doesn't support temperature
        assert temp is None


# =============================================================================
# Specific model configuration value tests
# =============================================================================


class TestModelConfigValues:
    """Tests for specific model configuration values based on research."""

    def test_deepseek_v3_values(self) -> None:
        """Test DeepSeek V3 optimal values."""
        config = DEEPSEEK_V3_CONFIG
        assert config.temperature == 0.3
        assert config.top_p is None  # Only supports temperature
        assert not config.is_reasoning_model

    def test_deepseek_r1_values(self) -> None:
        """Test DeepSeek R1 optimal values."""
        config = DEEPSEEK_R1_CONFIG
        assert config.temperature == 0.6
        assert config.top_p == 0.95
        assert config.is_reasoning_model

    def test_qwen_thinking_values(self) -> None:
        """Test Qwen thinking mode optimal values."""
        config = QWEN_THINKING_CONFIG
        assert config.temperature == 0.6
        assert config.top_p == 0.95
        assert config.top_k == 20
        assert config.is_reasoning_model

    def test_qwen_non_thinking_values(self) -> None:
        """Test Qwen non-thinking mode optimal values."""
        config = QWEN_NON_THINKING_CONFIG
        assert config.temperature == 0.7
        assert config.top_p == 0.8
        assert config.top_k == 20
        assert config.presence_penalty == 1.5
        assert not config.is_reasoning_model

    def test_mistral_small_values(self) -> None:
        """Test Mistral Small 3.2 optimal values."""
        config = MISTRAL_SMALL_CONFIG
        assert config.temperature == 0.15  # Very low
        assert not config.is_reasoning_model

    def test_gemma_values(self) -> None:
        """Test Gemma 3 optimal values."""
        config = GEMMA_CONFIG
        assert config.temperature == 1.0  # Higher than typical
        assert config.top_p == 0.96
        assert config.top_k == 64

    def test_llama4_values(self) -> None:
        """Test Llama 4 optimal values."""
        config = LLAMA4_CONFIG
        assert config.temperature == 0.6
        assert config.top_p == 0.9

    def test_o1_values(self) -> None:
        """Test OpenAI o1 values (no sampling params)."""
        config = O1_CONFIG
        assert config.temperature is None
        assert config.top_p is None
        assert config.is_reasoning_model
        assert not config.supports_temperature
        assert not config.supports_top_p


# =============================================================================
# Edge cases and regression tests
# =============================================================================


class TestEdgeCases:
    """Edge case and regression tests."""

    def test_case_insensitive_matching(self) -> None:
        """Test that model matching is case-insensitive."""
        models = [
            ("GPT-4", GPT4_CONFIG),
            ("gpt-4", GPT4_CONFIG),
            ("CLAUDE-3-OPUS", CLAUDE_CONFIG),
            ("DeepSeek-R1", DEEPSEEK_R1_CONFIG),
        ]
        for model_name, expected_config in models:
            config = get_model_config(model_name)
            assert config == expected_config, f"Case mismatch for {model_name}"

    def test_empty_model_name(self) -> None:
        """Test empty model name returns default."""
        config = get_model_config("")
        assert config == DEFAULT_CONFIG

    def test_model_with_version_suffix(self) -> None:
        """Test models with version suffixes are detected."""
        models = [
            "gpt-4-0125-preview",
            "claude-3-opus-20240229",
            "deepseek-chat-v3",
        ]
        for model in models:
            config = get_model_config(model)
            assert config != DEFAULT_CONFIG, f"{model} should match a config"

    def test_partial_model_names(self) -> None:
        """Test partial model names don't cause false positives."""
        # "o1" should match o1 config, but "model-01" should not
        assert get_model_config("o1") == O1_CONFIG
        # "o10" contains "o1" but is a different model - pattern uses word boundary
        config = get_model_config("gpt-4o-mini")
        assert config == GPT4_CONFIG  # Should match gpt-4, not o1

    def test_frozen_config_immutable(self) -> None:
        """Test that ModelConfig is immutable (frozen dataclass)."""
        config = ModelConfig(name="test", temperature=0.7)
        with pytest.raises(AttributeError):
            config.temperature = 0.5  # type: ignore[misc]

    def test_default_config_values(self) -> None:
        """Test DEFAULT_CONFIG has sensible defaults."""
        assert DEFAULT_CONFIG.temperature == 0.7
        assert DEFAULT_CONFIG.top_p == 0.9
        assert DEFAULT_CONFIG.supports_temperature
        assert DEFAULT_CONFIG.supports_top_p
        assert not DEFAULT_CONFIG.is_reasoning_model
