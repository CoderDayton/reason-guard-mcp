"""Unit tests for src/models/llm_client.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.llm_client import LLMClient
from src.utils.errors import LLMException


class TestLLMClientInit:
    """Test LLMClient initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        with (
            patch("src.models.llm_client.OpenAI") as mock_openai,
            patch("src.models.llm_client.AsyncOpenAI") as mock_async_openai,
        ):
            client = LLMClient(api_key="sk-test-key")

            assert client.model == "gpt-5.1"
            assert client.timeout == 60
            assert client.max_retries == 3

            mock_openai.assert_called_once()
            mock_async_openai.assert_called_once()

    def test_init_with_env_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization reads API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        with (
            patch("src.models.llm_client.OpenAI") as mock_openai,
            patch("src.models.llm_client.AsyncOpenAI"),
        ):
            LLMClient()

            # Verify OpenAI was called with env key
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["api_key"] == "sk-env-key"

    def test_init_without_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization raises if no API key available."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(LLMException, match="OPENAI_API_KEY"):
            LLMClient()

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="gpt-3.5-turbo")
            assert client.model == "gpt-3.5-turbo"

    def test_init_with_custom_timeout(self) -> None:
        """Test initialization with custom timeout."""
        with (
            patch("src.models.llm_client.OpenAI") as mock_openai,
            patch("src.models.llm_client.AsyncOpenAI"),
        ):
            client = LLMClient(api_key="sk-test", timeout=120)
            assert client.timeout == 120

            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["timeout"] == 120

    def test_init_with_custom_base_url(self) -> None:
        """Test initialization with custom base URL."""
        with (
            patch("src.models.llm_client.OpenAI") as mock_openai,
            patch("src.models.llm_client.AsyncOpenAI"),
        ):
            LLMClient(api_key="sk-test", base_url="https://custom.api.com")

            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["base_url"] == "https://custom.api.com"

    def test_init_with_max_retries(self) -> None:
        """Test initialization with custom max retries."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", max_retries=5)
            assert client.max_retries == 5


class TestLLMClientReasoningModelDetection:
    """Test reasoning model auto-detection and token scaling."""

    def test_standard_model_no_scaling(self) -> None:
        """Test standard models have 1.0x token multiplier."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="gpt-5.1")
            assert client._is_reasoning_model is False
            assert client._token_multiplier == 1.0
            assert client._scale_tokens(300) == 300

    def test_minimax_detected_as_reasoning(self) -> None:
        """Test MiniMax models are detected as reasoning models."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="MiniMaxAI/MiniMax-M2")
            assert client._is_reasoning_model is True
            assert client._token_multiplier == 3.0
            assert client._scale_tokens(300) == 900

    def test_deepseek_detected_as_reasoning(self) -> None:
        """Test DeepSeek models are detected as reasoning models."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="deepseek-reasoner")
            assert client._is_reasoning_model is True
            assert client._token_multiplier == 3.0

    def test_qwen_detected_as_reasoning(self) -> None:
        """Test Qwen models are detected as reasoning models."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="qwen-2.5-coder")
            assert client._is_reasoning_model is True

    def test_o1_detected_as_reasoning(self) -> None:
        """Test o1 models are detected as reasoning models."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            for model in ["o1", "o1-preview", "o1-mini"]:
                client = LLMClient(api_key="sk-test", model=model)
                assert client._is_reasoning_model is True, f"{model} should be detected"

    def test_default_temperature(self) -> None:
        """Test default temperature is stored and used."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            # Default temperature
            client = LLMClient(api_key="sk-test")
            assert client.default_temperature == 0.7

            # Custom temperature
            client = LLMClient(api_key="sk-test", default_temperature=0.3)
            assert client.default_temperature == 0.3

    def test_explicit_multiplier_overrides_auto(self) -> None:
        """Test explicit multiplier overrides auto-detection."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            # Standard model with explicit high multiplier
            client = LLMClient(
                api_key="sk-test",
                model="gpt-5.1",
                reasoning_token_multiplier=2.5,
            )
            assert client._token_multiplier == 2.5
            assert client._scale_tokens(400) == 1000

    def test_explicit_multiplier_on_reasoning_model(self) -> None:
        """Test explicit multiplier can override reasoning model default."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(
                api_key="sk-test",
                model="deepseek-r1",
                reasoning_token_multiplier=4.0,  # Override default 3.0x
            )
            assert client._is_reasoning_model is True
            assert client._token_multiplier == 4.0


class TestLLMClientGenerate:
    """Test LLMClient.generate method."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        """Create a mock LLM client."""
        with (
            patch("src.models.llm_client.OpenAI") as mock_openai,
            patch("src.models.llm_client.AsyncOpenAI"),
        ):
            client = LLMClient(api_key="sk-test")

            # Setup mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"

            mock_openai.return_value.chat.completions.create.return_value = mock_response
            client.client = mock_openai.return_value

            return client

    def test_generate_basic(self, mock_client: LLMClient) -> None:
        """Test basic text generation."""
        result = mock_client.generate("What is 2+2?")

        assert result == "Test response"
        mock_client.client.chat.completions.create.assert_called_once()

    def test_generate_with_system_prompt(self, mock_client: LLMClient) -> None:
        """Test generation with system prompt."""
        mock_client.generate("Hello", system_prompt="You are a helpful assistant")

        call_kwargs = mock_client.client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"

    def test_generate_with_custom_params(self, mock_client: LLMClient) -> None:
        """Test generation with custom parameters."""
        mock_client.generate(
            "Test",
            max_tokens=500,
            temperature=0.5,
            top_p=0.8,
        )

        call_kwargs = mock_client.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.8

    def test_generate_empty_content_returns_empty_string(self) -> None:
        """Test generation returns empty string when content is None."""
        with (
            patch("src.models.llm_client.OpenAI") as mock_openai,
            patch("src.models.llm_client.AsyncOpenAI"),
        ):
            client = LLMClient(api_key="sk-test")

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None

            mock_openai.return_value.chat.completions.create.return_value = mock_response
            client.client = mock_openai.return_value

            result = client.generate("Test")
            assert result == ""

    def test_generate_failure_raises_llm_exception(self) -> None:
        """Test generation failure raises LLMException."""
        with (
            patch("src.models.llm_client.OpenAI") as mock_openai,
            patch("src.models.llm_client.AsyncOpenAI"),
        ):
            client = LLMClient(api_key="sk-test")

            mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
            client.client = mock_openai.return_value

            with pytest.raises(LLMException, match="Generation failed"):
                client.generate("Test")


class TestLLMClientGenerateAsync:
    """Test LLMClient.generate_async method."""

    @pytest.fixture
    def mock_async_client(self) -> LLMClient:
        """Create a mock LLM client with async support."""
        with (
            patch("src.models.llm_client.OpenAI"),
            patch("src.models.llm_client.AsyncOpenAI") as mock_async_openai,
        ):
            client = LLMClient(api_key="sk-test")

            # Setup mock async response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Async response"

            mock_async_openai.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            client.async_client = mock_async_openai.return_value

            return client

    @pytest.mark.asyncio
    async def test_generate_async_basic(self, mock_async_client: LLMClient) -> None:
        """Test basic async text generation."""
        result = await mock_async_client.generate_async("What is 2+2?")

        assert result == "Async response"
        mock_async_client.async_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_async_with_system_prompt(self, mock_async_client: LLMClient) -> None:
        """Test async generation with system prompt."""
        await mock_async_client.generate_async("Hello", system_prompt="You are helpful")

        call_kwargs = mock_async_client.async_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_generate_async_empty_content_returns_empty_string(self) -> None:
        """Test async generation returns empty string when content is None."""
        with (
            patch("src.models.llm_client.OpenAI"),
            patch("src.models.llm_client.AsyncOpenAI") as mock_async_openai,
        ):
            client = LLMClient(api_key="sk-test")

            # Create a proper mock that doesn't have 'text' attribute
            mock_message = MagicMock(spec=["content", "model_dump"])
            mock_message.content = None
            mock_message.model_dump.return_value = {"content": None, "role": "assistant"}

            mock_choice = MagicMock(spec=["message", "finish_reason"])
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            mock_async_openai.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            client.async_client = mock_async_openai.return_value

            result = await client.generate_async("Test")
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_async_failure_raises_llm_exception(self) -> None:
        """Test async generation failure raises LLMException."""
        with (
            patch("src.models.llm_client.OpenAI"),
            patch("src.models.llm_client.AsyncOpenAI") as mock_async_openai,
        ):
            client = LLMClient(api_key="sk-test")

            mock_async_openai.return_value.chat.completions.create = AsyncMock(
                side_effect=Exception("Async API Error")
            )
            client.async_client = mock_async_openai.return_value

            with pytest.raises(LLMException, match="Async generation failed"):
                await client.generate_async("Test")


class TestLLMClientTokenEstimation:
    """Test LLMClient token estimation methods."""

    @pytest.fixture
    def client(self) -> LLMClient:
        """Create a mock LLM client."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            return LLMClient(api_key="sk-test")

    def test_estimate_tokens_empty_string(self, client: LLMClient) -> None:
        """Test token estimation for empty string."""
        assert client.estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self, client: LLMClient) -> None:
        """Test token estimation for short text."""
        # 8 characters / 4 = 2 tokens
        assert client.estimate_tokens("Hello!!") == 1

    def test_estimate_tokens_longer_text(self, client: LLMClient) -> None:
        """Test token estimation for longer text."""
        # ~100 characters = ~25 tokens
        text = "a" * 100
        assert client.estimate_tokens(text) == 25

    def test_count_messages_tokens_empty(self, client: LLMClient) -> None:
        """Test message token count for empty list."""
        assert client.count_messages_tokens([]) == 0

    def test_count_messages_tokens_single_message(self, client: LLMClient) -> None:
        """Test message token count for single message."""
        messages = [{"role": "user", "content": "Hello world"}]  # 11 chars = 2 tokens + 4 overhead
        result = client.count_messages_tokens(messages)
        assert result == 6  # 4 overhead + 11/4

    def test_count_messages_tokens_multiple_messages(self, client: LLMClient) -> None:
        """Test message token count for multiple messages."""
        messages = [
            {"role": "system", "content": "Hi"},  # 4 overhead + 2/4=0 = 4
            {"role": "user", "content": "Test"},  # 4 overhead + 4/4=1 = 5
        ]
        result = client.count_messages_tokens(messages)
        assert result == 9  # 4+0 + 4+1 = 9

    def test_count_messages_tokens_missing_content(self, client: LLMClient) -> None:
        """Test message token count handles missing content key."""
        messages: list[dict[str, str]] = [{"role": "user"}]  # No content
        result = client.count_messages_tokens(messages)
        assert result == 4  # Just overhead, no content tokens


class TestLLMClientReasoningExtraction:
    """Test LLMClient reasoning content extraction."""

    @pytest.fixture
    def client(self) -> LLMClient:
        """Create LLM client for testing."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            return LLMClient(api_key="sk-test")

    def test_extract_empty_reasoning(self, client: LLMClient) -> None:
        """Test extraction from empty reasoning returns empty string."""
        result = client._extract_answer_from_reasoning("")
        assert result == ""

    def test_extract_with_thus_marker(self, client: LLMClient) -> None:
        """Test extraction finds content after 'Thus:' marker."""
        reasoning = """The user wants X. We analyze Y.
        Thus: Einstein's key contributions include special and general relativity.
        Some more meta text."""
        result = client._extract_answer_from_reasoning(reasoning)
        assert "Einstein" in result
        assert "relativity" in result

    def test_extract_with_quoted_answer(self, client: LLMClient) -> None:
        """Test extraction finds quoted content as answer."""
        reasoning = """We need to produce an answer. The answer should be:
        "Einstein revolutionized physics with special relativity in 1905."
        That satisfies the requirement."""
        result = client._extract_answer_from_reasoning(reasoning)
        assert "Einstein" in result
        assert "1905" in result

    def test_extract_fallback_to_last_paragraph(self, client: LLMClient) -> None:
        """Test extraction falls back to last paragraph when no markers found."""
        reasoning = """First paragraph of analysis.

        Second paragraph with more details.

        Final paragraph: Einstein's work on relativity changed physics forever."""
        result = client._extract_answer_from_reasoning(reasoning)
        assert "Einstein" in result
        assert "relativity" in result

    def test_extract_skips_meta_commentary(self, client: LLMClient) -> None:
        """Test extraction skips paragraphs starting with meta phrases."""
        reasoning = """Einstein developed special relativity in 1905.

        We need to produce a single step answer."""
        result = client._extract_answer_from_reasoning(reasoning)
        assert "Einstein" in result
        assert "We need to" not in result


class TestLLMClientThinkTagStripping:
    """Test LLMClient <think> tag handling."""

    @pytest.fixture
    def client(self) -> LLMClient:
        """Create LLM client for testing."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            return LLMClient(api_key="sk-test")

    def test_strip_no_think_tags(self, client: LLMClient) -> None:
        """Test content without think tags is returned as-is."""
        content = "Einstein developed special relativity in 1905."
        result = client._strip_think_tags(content)
        assert result == content

    def test_strip_think_tags_with_answer(self, client: LLMClient) -> None:
        """Test extraction of answer after </think> tag."""
        content = """<think>
        Let me analyze this question about Einstein.
        He made many contributions to physics.
        </think>Einstein's key contributions include special and general relativity."""
        result = client._strip_think_tags(content)
        assert "Einstein's key contributions" in result
        assert "<think>" not in result
        assert "Let me analyze" not in result

    def test_strip_think_tags_only_thinking(self, client: LLMClient) -> None:
        """Test extraction when only thinking content exists."""
        content = """<think>
        Einstein developed special relativity in 1905.
        Thus: His key contribution was unifying space and time.
        </think>"""
        result = client._strip_think_tags(content)
        # Should extract answer from thinking content
        assert "unifying space and time" in result or "1905" in result

    def test_strip_unclosed_think_tag(self, client: LLMClient) -> None:
        """Test handling of unclosed <think> tag (token limit hit)."""
        content = """<think>
        Einstein's theory of relativity changed physics.
        Thus: He unified space and time into spacetime."""
        result = client._strip_think_tags(content)
        # Should extract from partial thinking
        assert "spacetime" in result or "relativity" in result

    def test_strip_think_tags_case_insensitive(self, client: LLMClient) -> None:
        """Test that think tag matching is case-insensitive."""
        content = """<THINK>Some reasoning here</THINK>The actual answer."""
        result = client._strip_think_tags(content)
        assert result == "The actual answer."

    def test_strip_think_tags_with_whitespace(self, client: LLMClient) -> None:
        """Test handling of whitespace around think tags."""
        content = """<think>

        Some reasoning

        </think>

        Clean answer here."""
        result = client._strip_think_tags(content)
        assert result == "Clean answer here."


class TestLLMClientGarbageDetection:
    """Test LLMClient garbage response detection."""

    @pytest.fixture
    def client(self) -> LLMClient:
        """Create LLM client for testing."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            return LLMClient(api_key="sk-test")

    def test_valid_prose_passes(self, client: LLMClient) -> None:
        """Test that valid prose content passes validation."""
        content = "Einstein developed special relativity in 1905."
        result = client._validate_and_clean_content(content)
        assert result == content

    def test_valid_code_passes(self, client: LLMClient) -> None:
        """Test that valid code content passes validation."""
        content = """def hello_world():
    print("Hello, world!")
    return 42"""
        result = client._validate_and_clean_content(content)
        assert result == content

    def test_empty_content_returns_empty(self, client: LLMClient) -> None:
        """Test that empty content returns empty string."""
        assert client._validate_and_clean_content("") == ""

    def test_garbage_punctuation_detected(self, client: LLMClient) -> None:
        """Test that strings of punctuation/whitespace are detected as garbage."""
        garbage = ",   ,           ,   , ,       ,   ,           ,   , ,"
        result = client._validate_and_clean_content(garbage)
        assert result == ""

    def test_garbage_repeated_chars_detected(self, client: LLMClient) -> None:
        """Test that highly repetitive content is detected as garbage."""
        # 100 commas with a few spaces - very low unique char ratio
        garbage = "," * 80 + " " * 20
        result = client._validate_and_clean_content(garbage)
        assert result == ""

    def test_low_alphanumeric_ratio_detected(self, client: LLMClient) -> None:
        """Test that content with very low alphanumeric ratio is garbage."""
        # Only 10% alphanumeric
        garbage = "abc" + "!@#$%^&*()_+-=[]{}|;':\",./<>?" * 3
        result = client._validate_and_clean_content(garbage)
        assert result == ""

    def test_borderline_content_passes(self, client: LLMClient) -> None:
        """Test that borderline but valid content passes."""
        # JSON-like content has lower alphanumeric ratio but should still pass
        content = '{"key": "value", "number": 42, "array": [1, 2, 3]}'
        result = client._validate_and_clean_content(content)
        assert result == content

    def test_short_punctuation_passes(self, client: LLMClient) -> None:
        """Test that short punctuation strings are not flagged as repetitive garbage."""
        # Short content shouldn't trigger repetition check
        content = "..."
        result = client._validate_and_clean_content(content)
        # This will fail alphanumeric ratio check, not repetition
        assert result == ""

    def test_markdown_with_formatting_passes(self, client: LLMClient) -> None:
        """Test that markdown with special chars passes."""
        content = """# Heading

**Bold text** and *italic text*

- List item 1
- List item 2

```python
print("code")
```"""
        result = client._validate_and_clean_content(content)
        assert result == content


class TestLLMClientModelConfig:
    """Test LLMClient model configuration integration."""

    def test_model_config_stored(self) -> None:
        """Test that model config is stored on client initialization."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="gpt-5.1")
            assert hasattr(client, "_model_config")
            assert client._model_config.name == "OpenAI GPT-5"

    def test_model_config_temperature_used(self) -> None:
        """Test that model config temperature is used as default."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            # DeepSeek V3 has optimal temp of 0.3
            client = LLMClient(api_key="sk-test", model="deepseek-chat")
            assert client.default_temperature == 0.3

    def test_explicit_temperature_overrides_config(self) -> None:
        """Test that explicit temperature overrides model config."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            # DeepSeek V3 has optimal temp of 0.3, but we override to 0.8
            client = LLMClient(api_key="sk-test", model="deepseek-chat", default_temperature=0.8)
            assert client.default_temperature == 0.8

    def test_build_sampling_params_basic(self) -> None:
        """Test _build_sampling_params returns correct parameters."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="gpt-5.1")
            params = client._build_sampling_params()
            assert "temperature" in params
            assert params["temperature"] == 0.7  # GPT-4 default
            assert "top_p" in params

    def test_build_sampling_params_with_override(self) -> None:
        """Test _build_sampling_params with user overrides."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="gpt-5.1")
            params = client._build_sampling_params(temperature=0.3, top_p=0.5)
            assert params["temperature"] == 0.3
            assert params["top_p"] == 0.5

    def test_build_sampling_params_o1_model(self) -> None:
        """Test _build_sampling_params for o1 model (no temp/top_p support)."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="o1")
            params = client._build_sampling_params()
            # o1 doesn't support temperature or top_p
            assert "temperature" not in params
            assert "top_p" not in params

    def test_build_sampling_params_claude_mutual_exclusion(self) -> None:
        """Test _build_sampling_params for Claude (mutually exclusive temp/top_p)."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="claude-3-opus")
            params = client._build_sampling_params()
            # Claude supports both but they're mutually exclusive - should only have one
            assert "temperature" in params
            assert "top_p" not in params  # prefer_temperature=True

    def test_model_config_qwen_non_thinking(self) -> None:
        """Test Qwen non-thinking mode config."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="qwen-72b")
            assert client.default_temperature == 0.7
            # Qwen non-thinking is NOT marked as reasoning model in llm_client patterns
            # but model_config knows it's not a reasoning model
            assert client._model_config.is_reasoning_model is False

    def test_model_config_deepseek_r1_reasoning(self) -> None:
        """Test DeepSeek R1 is detected as reasoning via model config."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="deepseek-r1")
            assert client._is_reasoning_model is True
            assert client._model_config.is_reasoning_model is True
            assert client.default_temperature == 0.6  # R1 optimal

    def test_unknown_model_uses_default_config(self) -> None:
        """Test unknown models use default configuration."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="unknown-model-xyz")
            assert client._model_config.name == "default"
            assert client.default_temperature == 0.7

    def test_gemma_uses_higher_temperature(self) -> None:
        """Test Gemma uses its optimal higher temperature."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="gemma-3-27b")
            assert client.default_temperature == 1.0  # Gemma's optimal is higher


class TestLLMClientContextTruncation:
    """Test LLMClient automatic context truncation."""

    @pytest.fixture
    def client(self) -> LLMClient:
        """Create LLM client for testing."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            return LLMClient(api_key="sk-test", model="gpt-5.1")

    def test_prepare_messages_no_truncation_needed(self, client: LLMClient) -> None:
        """Test _prepare_messages with short prompt (no truncation)."""
        messages = client._prepare_messages(
            prompt="What is 2+2?",
            system_prompt="You are a math assistant.",
            max_tokens=1000,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a math assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"

    def test_prepare_messages_auto_truncates_long_prompt(self, client: LLMClient) -> None:
        """Test _prepare_messages automatically truncates long prompts."""
        # GPT-5 has 128K context, but let's create a prompt that would exceed it
        # if we request a large output reservation
        # With 128K context and 16K max output, we have ~112K for input
        # Create a prompt that would be too long
        long_text = "x" * 500000  # ~125K tokens at 4 chars/token

        messages = client._prepare_messages(
            prompt=long_text,
            max_tokens=16000,
            auto_truncate=True,
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        # Prompt should be truncated
        assert len(messages[0]["content"]) < len(long_text)

    def test_prepare_messages_raises_when_auto_truncate_false(self, client: LLMClient) -> None:
        """Test _prepare_messages raises when auto_truncate=False and prompt too long."""
        from src.utils.errors import LLMException

        long_text = "x" * 600000  # Way too long

        with pytest.raises(LLMException, match="exceeds available context"):
            client._prepare_messages(
                prompt=long_text,
                max_tokens=16000,
                auto_truncate=False,
            )

    def test_prepare_messages_with_system_prompt_reserves_space(self, client: LLMClient) -> None:
        """Test that system prompt space is reserved when truncating."""
        # Create a prompt that's close to the limit
        long_text = "x" * 400000

        # With system prompt, less space available for user prompt
        with_system = client._prepare_messages(
            prompt=long_text,
            system_prompt="You are a helpful assistant." * 100,  # ~500 chars
            max_tokens=16000,
            auto_truncate=True,
        )

        without_system = client._prepare_messages(
            prompt=long_text,
            max_tokens=16000,
            auto_truncate=True,
        )

        # With system prompt, user content should be shorter
        user_content_with_system = with_system[-1]["content"]
        user_content_without_system = without_system[-1]["content"]

        # The difference should account for system prompt space
        assert len(user_content_with_system) <= len(user_content_without_system)

    def test_prepare_messages_truncation_strategy_keep_start(self, client: LLMClient) -> None:
        """Test truncation with KEEP_START strategy."""
        from src.models.model_config import TruncationStrategy

        # Create prompt with distinct start and end
        long_text = "START_MARKER_" + "x" * 500000 + "_END_MARKER"

        messages = client._prepare_messages(
            prompt=long_text,
            max_tokens=16000,
            truncation_strategy=TruncationStrategy.KEEP_START,
            auto_truncate=True,
        )

        content = messages[0]["content"]
        assert content.startswith("START_MARKER_")
        assert "_END_MARKER" not in content

    def test_prepare_messages_truncation_strategy_keep_end(self, client: LLMClient) -> None:
        """Test truncation with KEEP_END strategy (default)."""
        from src.models.model_config import TruncationStrategy

        # Create prompt with distinct start and end
        long_text = "START_MARKER_" + "x" * 500000 + "_END_MARKER"

        messages = client._prepare_messages(
            prompt=long_text,
            max_tokens=16000,
            truncation_strategy=TruncationStrategy.KEEP_END,
            auto_truncate=True,
        )

        content = messages[0]["content"]
        assert "START_MARKER_" not in content
        assert content.endswith("_END_MARKER")

    def test_prepare_messages_accounts_for_token_scaling(self) -> None:
        """Test that reasoning model token scaling is considered in truncation."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            # DeepSeek R1 has 3x token multiplier
            client = LLMClient(api_key="sk-test", model="deepseek-r1")

            # With 64K context and 8K max output (scaled to 24K), available is ~40K
            long_text = "x" * 200000  # ~50K tokens

            messages = client._prepare_messages(
                prompt=long_text,
                max_tokens=8000,  # Will be scaled to 24000
                auto_truncate=True,
            )

            # Should be truncated more aggressively due to token scaling
            assert len(messages[0]["content"]) < len(long_text)

    def test_generate_with_auto_truncate(self) -> None:
        """Test generate() method respects auto_truncate parameter."""
        with (
            patch("src.models.llm_client.OpenAI") as mock_openai,
            patch("src.models.llm_client.AsyncOpenAI"),
        ):
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="42"))]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_key="sk-test", model="gpt-5.1")

            # Very long prompt that needs truncation
            long_prompt = "x" * 600000

            result = client.generate(
                prompt=long_prompt,
                max_tokens=16000,
                auto_truncate=True,
            )

            assert result == "42"
            # Verify the API was called (truncation happened internally)
            mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_async_with_auto_truncate(self) -> None:
        """Test generate_async() method respects auto_truncate parameter."""
        with (
            patch("src.models.llm_client.OpenAI"),
            patch("src.models.llm_client.AsyncOpenAI") as mock_async_openai,
        ):
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="42"))]
            mock_client.chat.completions.create.return_value = mock_response
            mock_async_openai.return_value = mock_client

            client = LLMClient(api_key="sk-test", model="gpt-5.1")

            # Very long prompt that needs truncation
            long_prompt = "x" * 600000

            result = await client.generate_async(
                prompt=long_prompt,
                max_tokens=16000,
                auto_truncate=True,
            )

            assert result == "42"
            # Verify the API was called (truncation happened internally)
            mock_client.chat.completions.create.assert_called_once()

    def test_generate_raises_without_auto_truncate(self) -> None:
        """Test generate() raises when auto_truncate=False and prompt too long."""
        from src.utils.errors import LLMException

        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            client = LLMClient(api_key="sk-test", model="gpt-5.1")

            long_prompt = "x" * 600000

            with pytest.raises(LLMException, match="exceeds available context"):
                client.generate(
                    prompt=long_prompt,
                    max_tokens=16000,
                    auto_truncate=False,
                )
