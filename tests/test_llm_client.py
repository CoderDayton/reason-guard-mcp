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

            assert client.model == "gpt-4-turbo"
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
            client = LLMClient(api_key="sk-test", model="gpt-4-turbo")
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

    def test_explicit_multiplier_overrides_auto(self) -> None:
        """Test explicit multiplier overrides auto-detection."""
        with patch("src.models.llm_client.OpenAI"), patch("src.models.llm_client.AsyncOpenAI"):
            # Standard model with explicit high multiplier
            client = LLMClient(
                api_key="sk-test",
                model="gpt-4-turbo",
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
