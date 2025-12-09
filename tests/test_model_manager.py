"""Tests for model_manager module.

Tests model lifecycle management, caching, and error handling.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.model_manager import (
    DISK_SPACE_BUFFER_MB,
    MODEL_SIZES_MB,
    ModelManager,
    ModelState,
    _check_disk_space,
    _estimate_model_size,
    _get_default_cache_dir,
    _get_model_size_mb,
)
from src.utils.errors import ModelNotReadyException


class TestHelperFunctions:
    """Test helper functions for model manager."""

    def test_get_default_cache_dir_without_env(self) -> None:
        """Test default cache directory when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove EMBEDDING_CACHE_DIR if it exists
            os.environ.pop("EMBEDDING_CACHE_DIR", None)
            result = _get_default_cache_dir()
            assert result == Path.home() / ".cache" / "matrixmind-mcp" / "models"

    def test_get_default_cache_dir_with_env(self) -> None:
        """Test default cache directory when env var is set."""
        with patch.dict(os.environ, {"EMBEDDING_CACHE_DIR": "/custom/cache"}):
            result = _get_default_cache_dir()
            assert result == Path("/custom/cache")

    def test_get_model_size_mb_known_model(self) -> None:
        """Test getting size for known model."""
        result = _get_model_size_mb("sentence-transformers/all-MiniLM-L6-v2")
        assert result == 80

    def test_get_model_size_mb_partial_match(self) -> None:
        """Test getting size for partial model name match."""
        # Test partial match - the model name ends with a known suffix
        result = _get_model_size_mb("my-org/custom-all-MiniLM-L6-v2")
        assert result == 80  # Should match all-MiniLM-L6-v2

    def test_get_model_size_mb_unknown_model(self) -> None:
        """Test getting size for unknown model returns default."""
        result = _get_model_size_mb("totally-unknown-model/xyz")
        assert result == 500  # Default

    def test_estimate_model_size_known_small(self) -> None:
        """Test size estimate for known small model."""
        result = _estimate_model_size("sentence-transformers/all-MiniLM-L6-v2")
        assert result == "~80MB"

    def test_estimate_model_size_known_large(self) -> None:
        """Test size estimate for known large model (>=1GB)."""
        result = _estimate_model_size("BAAI/bge-m3")
        assert result == "~2.2GB"

    def test_estimate_model_size_unknown(self) -> None:
        """Test size estimate for unknown model returns empty string."""
        result = _estimate_model_size("totally-unknown/model")
        assert result == ""

    def test_check_disk_space_sufficient(self) -> None:
        """Test disk space check with sufficient space."""
        with tempfile.TemporaryDirectory() as tmpdir:
            has_space, available_mb, required = _check_disk_space(Path(tmpdir), 10)
            assert has_space is True
            assert available_mb > 0
            assert required == 10

    def test_check_disk_space_insufficient(self) -> None:
        """Test disk space check with insufficient space."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Request more than any disk has
            has_space, available_mb, required = _check_disk_space(
                Path(tmpdir), 100_000_000
            )  # 100 TB
            assert has_space is False
            assert available_mb > 0
            assert required == 100_000_000

    def test_check_disk_space_creates_directory(self) -> None:
        """Test disk space check creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "subdir" / "nested"
            has_space, _, _ = _check_disk_space(new_path, 10)
            assert new_path.exists()
            assert has_space is True

    def test_check_disk_space_oserror(self) -> None:
        """Test disk space check handles OSError gracefully."""
        with (
            patch("shutil.disk_usage", side_effect=OSError("Permission denied")),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            has_space, available_mb, required = _check_disk_space(Path(tmpdir), 100)
            # Should assume space is available on error
            assert has_space is True
            assert available_mb == 0
            assert required == 100


class TestModelState:
    """Test ModelState enum."""

    def test_state_values(self) -> None:
        """Test all state values exist."""
        assert ModelState.NOT_STARTED.value == "not_started"
        assert ModelState.DOWNLOADING.value == "downloading"
        assert ModelState.LOADING.value == "loading"
        assert ModelState.READY.value == "ready"
        assert ModelState.FAILED.value == "failed"


class TestModelManager:
    """Test ModelManager class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> Generator[None, None, None]:
        """Reset singleton before each test."""
        ModelManager.reset_instance()
        yield
        ModelManager.reset_instance()

    def test_get_instance_singleton(self) -> None:
        """Test singleton pattern."""
        instance1 = ModelManager.get_instance()
        instance2 = ModelManager.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self) -> None:
        """Test reset instance clears singleton."""
        instance1 = ModelManager.get_instance()
        ModelManager.reset_instance()
        instance2 = ModelManager.get_instance()
        assert instance1 is not instance2

    def test_initial_state(self) -> None:
        """Test initial state of manager."""
        manager = ModelManager.get_instance()
        assert manager.state == ModelState.NOT_STARTED
        assert manager.model_name is None
        assert manager.model is None
        assert manager.tokenizer is None
        assert manager.is_ready() is False

    def test_get_model_not_started_raises(self) -> None:
        """Test get_model raises when not started."""
        manager = ModelManager.get_instance()
        with pytest.raises(ModelNotReadyException, match="not initialized"):
            manager.get_model()

    def test_get_model_downloading_raises(self) -> None:
        """Test get_model raises when downloading."""
        manager = ModelManager.get_instance()
        manager.state = ModelState.DOWNLOADING
        manager.model_name = "test-model"
        with pytest.raises(ModelNotReadyException, match="still downloading"):
            manager.get_model()

    def test_get_model_loading_raises(self) -> None:
        """Test get_model raises when loading."""
        manager = ModelManager.get_instance()
        manager.state = ModelState.LOADING
        manager.model_name = "test-model"
        with pytest.raises(ModelNotReadyException, match="loading into memory"):
            manager.get_model()

    def test_get_model_failed_raises(self) -> None:
        """Test get_model raises when failed."""
        manager = ModelManager.get_instance()
        manager.state = ModelState.FAILED
        manager._error_message = "Download error"
        with pytest.raises(ModelNotReadyException, match="failed to load"):
            manager.get_model()

    def test_get_model_ready_but_none_raises(self) -> None:
        """Test get_model raises when state is ready but model is None."""
        manager = ModelManager.get_instance()
        manager.state = ModelState.READY
        manager.model = None
        manager.tokenizer = None
        with pytest.raises(ModelNotReadyException, match="not available"):
            manager.get_model()

    def test_get_model_success(self) -> None:
        """Test get_model returns model when ready."""
        manager = ModelManager.get_instance()
        manager.state = ModelState.READY
        manager.model = MagicMock()
        manager.tokenizer = MagicMock()

        model, tokenizer = manager.get_model()
        assert model is manager.model
        assert tokenizer is manager.tokenizer

    def test_is_ready(self) -> None:
        """Test is_ready reflects state correctly."""
        manager = ModelManager.get_instance()

        manager.state = ModelState.NOT_STARTED
        assert manager.is_ready() is False

        manager.state = ModelState.DOWNLOADING
        assert manager.is_ready() is False

        manager.state = ModelState.LOADING
        assert manager.is_ready() is False

        manager.state = ModelState.FAILED
        assert manager.is_ready() is False

        manager.state = ModelState.READY
        assert manager.is_ready() is True

    def test_get_status_basic(self) -> None:
        """Test get_status returns expected fields."""
        manager = ModelManager.get_instance()
        manager.model_name = "test-model"
        manager.state = ModelState.DOWNLOADING

        status = manager.get_status()

        assert status["state"] == "downloading"
        assert status["model_name"] == "test-model"
        assert status["ready"] is False
        assert "device" in status
        assert "cache_dir" in status

    def test_get_status_with_disk_info(self) -> None:
        """Test get_status includes disk info."""
        manager = ModelManager.get_instance()
        status = manager.get_status()

        # Should have disk info (might be None if error, but key exists)
        assert "disk_free_mb" in status
        assert "disk_total_mb" in status

    def test_get_status_disk_error(self) -> None:
        """Test get_status handles disk error gracefully."""
        manager = ModelManager.get_instance()

        with patch("shutil.disk_usage", side_effect=OSError("Permission denied")):
            status = manager.get_status()
            assert status["disk_free_mb"] is None
            assert status["disk_total_mb"] is None

    def test_get_status_with_model_size(self) -> None:
        """Test get_status includes model size estimate."""
        manager = ModelManager.get_instance()
        manager.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        status = manager.get_status()
        assert status["model_size_mb"] == 80

    def test_get_status_with_error(self) -> None:
        """Test get_status includes error message."""
        manager = ModelManager.get_instance()
        manager.state = ModelState.FAILED
        manager._error_message = "Connection timeout"

        status = manager.get_status()
        assert status["error"] == "Connection timeout"

    def test_initialize_already_loaded_same_model(self) -> None:
        """Test initialize skips reload for same model."""
        manager = ModelManager.get_instance()
        manager.model_name = "test-model"
        manager.state = ModelState.READY

        # Should return early without re-loading
        with patch.object(manager, "_load_model") as mock_load:
            manager.initialize("test-model")
            mock_load.assert_not_called()

    def test_initialize_with_custom_cache_dir(self) -> None:
        """Test initialize with custom cache directory."""
        manager = ModelManager.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_cache = Path(tmpdir) / "custom_cache"

            # Mock the load to avoid actual download
            with patch.object(manager, "_load_model"):
                manager.initialize("test-model", cache_dir=custom_cache)

            assert manager.cache_dir == custom_cache
            assert custom_cache.exists()

    def test_initialize_insufficient_disk_space(self) -> None:
        """Test initialize raises on insufficient disk space."""
        manager = ModelManager.get_instance()

        with (
            patch(
                "src.models.model_manager._check_disk_space",
                return_value=(False, 100, 1000),
            ),
            pytest.raises(ModelNotReadyException, match="Insufficient disk space"),
        ):
            manager.initialize("test-model")

        assert manager.state == ModelState.FAILED

    def test_initialize_non_blocking(self) -> None:
        """Test initialize with non-blocking mode."""
        manager = ModelManager.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(manager, "_load_model"):
            manager.initialize("test-model", cache_dir=tmpdir, blocking=False)
            # Give thread a moment to start
            import time

            time.sleep(0.1)
            # In non-blocking mode, _load_model should be called in a thread
            # (but might not be called yet due to thread timing)

    def test_load_model_no_model_name(self) -> None:
        """Test _load_model handles missing model name."""
        manager = ModelManager.get_instance()
        manager.model_name = None

        manager._load_model()

        assert manager.state == ModelState.FAILED
        assert manager._error_message == "Model name not set"

    def test_load_model_exception(self) -> None:
        """Test _load_model handles exceptions."""
        manager = ModelManager.get_instance()
        manager.model_name = "test-model"

        with patch(
            "src.models.model_manager.AutoTokenizer.from_pretrained",
            side_effect=Exception("Network error"),
        ):
            manager._load_model()

        assert manager.state == ModelState.FAILED
        assert "Network error" in str(manager._error_message)

    def test_get_hidden_size_not_ready(self) -> None:
        """Test get_hidden_size raises when not ready."""
        manager = ModelManager.get_instance()
        manager.state = ModelState.NOT_STARTED

        with pytest.raises(ModelNotReadyException):
            manager.get_hidden_size()

    def test_get_hidden_size_success(self) -> None:
        """Test get_hidden_size returns correct value."""
        manager = ModelManager.get_instance()
        manager.state = ModelState.READY

        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        manager.model = mock_model
        manager.tokenizer = MagicMock()

        result = manager.get_hidden_size()
        assert result == 768


class TestModelSizesConstant:
    """Test MODEL_SIZES_MB constant."""

    def test_known_models_have_sizes(self) -> None:
        """Test that known models have defined sizes."""
        assert "sentence-transformers/all-MiniLM-L6-v2" in MODEL_SIZES_MB
        assert "BAAI/bge-m3" in MODEL_SIZES_MB
        assert "Snowflake/snowflake-arctic-embed-xs" in MODEL_SIZES_MB

    def test_sizes_are_reasonable(self) -> None:
        """Test that sizes are reasonable (between 10MB and 10GB)."""
        for model, size in MODEL_SIZES_MB.items():
            assert 10 <= size <= 10000, f"Model {model} has unreasonable size {size}MB"

    def test_buffer_constant(self) -> None:
        """Test disk space buffer constant."""
        assert DISK_SPACE_BUFFER_MB == 100
