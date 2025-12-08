"""pytest configuration and fixtures."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up mock environment variables for testing."""
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-for-testing")


@pytest.fixture
def sample_context() -> str:
    """Provide sample context for testing."""
    return """
    Albert Einstein was a theoretical physicist who developed the theory of relativity.
    He was born in Germany in 1879 and later became a Swiss and American citizen.
    Einstein published his special theory of relativity in 1905, which introduced E=mc².
    His general theory of relativity was published in 1915 and describes gravity.
    Einstein received the Nobel Prize in Physics in 1921 for his work on photoelectric effect.
    He worked at the Institute for Advanced Study in Princeton from 1933 until his death in 1955.
    """


@pytest.fixture
def sample_question() -> str:
    """Provide sample question for testing."""
    return "When did Einstein publish his theory of relativity?"


@pytest.fixture
def sample_answer() -> str:
    """Provide sample answer for testing."""
    return "Einstein published special relativity in 1905 and general relativity in 1915."


@pytest.fixture
def mock_llm_response() -> MagicMock:
    """Create a mock LLM response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "This is a mock response from the LLM."
    return response
