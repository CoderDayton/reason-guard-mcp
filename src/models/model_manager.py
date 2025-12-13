"""Model Manager for embedding model lifecycle.

Handles downloading, caching, and loading of embedding models with proper
error handling for cases where models are not yet ready.

Cache location: ~/.cache/matrixmind-mcp/models/ (configurable via EMBEDDING_CACHE_DIR)
"""

from __future__ import annotations

import os
import shutil
import threading
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from src.utils.errors import ModelNotReadyException

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class ModelState(str, Enum):
    """State of model loading."""

    NOT_STARTED = "not_started"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


# Known model sizes (approximate, in MB) for common embedding models
# Used to provide helpful download estimates and disk space checks
MODEL_SIZES_MB: dict[str, int] = {
    # Snowflake Arctic models (lightweight, efficient)
    "Snowflake/snowflake-arctic-embed-xs": 90,
    "Snowflake/snowflake-arctic-embed-s": 130,
    "Snowflake/snowflake-arctic-embed-m": 440,
    "Snowflake/snowflake-arctic-embed-l": 1100,
    # Sentence Transformers
    "sentence-transformers/all-mpnet-base-v2": 420,
    "sentence-transformers/all-MiniLM-L6-v2": 80,
    "sentence-transformers/all-MiniLM-L12-v2": 120,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 80,
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": 420,
    # BAAI BGE models
    "BAAI/bge-m3": 2200,
    "BAAI/bge-large-en-v1.5": 1300,
    "BAAI/bge-base-en-v1.5": 420,
    "BAAI/bge-small-en-v1.5": 130,
    # Other popular models
    "nomic-ai/nomic-embed-text-v1.5": 550,
    "intfloat/e5-large-v2": 1300,
    "intfloat/e5-base-v2": 420,
    "intfloat/e5-small-v2": 130,
}

# Minimum disk space buffer (MB) beyond model size
DISK_SPACE_BUFFER_MB = 100


def _get_default_cache_dir() -> Path:
    """Get default cache directory, respecting EMBEDDING_CACHE_DIR env var."""
    env_cache_dir = os.getenv("EMBEDDING_CACHE_DIR")
    if env_cache_dir:
        return Path(env_cache_dir)
    return Path.home() / ".cache" / "matrixmind-mcp" / "models"


def _get_model_size_mb(model_name: str) -> int:
    """Get estimated model size in MB.

    Args:
        model_name: HuggingFace model name.

    Returns:
        Estimated size in MB, or 500 as default for unknown models.

    """
    # Check exact match first
    if model_name in MODEL_SIZES_MB:
        return MODEL_SIZES_MB[model_name]

    # Check partial matches
    model_lower = model_name.lower()
    for known_model, size_mb in MODEL_SIZES_MB.items():
        known_short = known_model.split("/")[-1].lower()
        if known_short in model_lower or model_lower.endswith(known_short):
            return size_mb

    # Default estimate for unknown models
    return 500


def _estimate_model_size(model_name: str) -> str:
    """Get estimated model size string for download messages.

    Args:
        model_name: HuggingFace model name.

    Returns:
        Human-readable size estimate or empty string if unknown.

    """
    size_mb = _get_model_size_mb(model_name)
    # Only return string for known models
    if model_name in MODEL_SIZES_MB or any(
        model_name.lower().endswith(k.split("/")[-1].lower()) for k in MODEL_SIZES_MB
    ):
        if size_mb >= 1000:
            return f"~{size_mb / 1000:.1f}GB"
        return f"~{size_mb}MB"
    return ""


def _check_disk_space(cache_dir: Path, required_mb: int) -> tuple[bool, int, int]:
    """Check if sufficient disk space is available.

    Args:
        cache_dir: Directory where model will be cached.
        required_mb: Required space in MB.

    Returns:
        Tuple of (has_space, available_mb, required_mb).

    """
    try:
        # Ensure parent directory exists for statvfs
        cache_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(cache_dir)
        available_mb = usage.free // (1024 * 1024)
        return available_mb >= required_mb, int(available_mb), required_mb
    except OSError as e:
        logger.warning(f"Could not check disk space: {e}")
        # If we can't check, assume there's space and let download fail naturally
        return True, 0, required_mb


class ModelManager:
    """Manages embedding model lifecycle with download caching.

    Thread-safe singleton that handles model downloading, caching, and loading.
    Returns proper errors if tools are called before the model is ready.

    Attributes:
        model_name: Name of the embedding model.
        cache_dir: Directory for model cache.
        state: Current loading state.
        model: Loaded model (if ready).
        tokenizer: Loaded tokenizer (if ready).
        device: Device model is loaded on.

    Example:
        >>> manager = ModelManager.get_instance()
        >>> manager.initialize("sentence-transformers/all-mpnet-base-v2")
        >>> # Later, when tool is called:
        >>> model, tokenizer = manager.get_model()  # Raises if not ready

    """

    _instance: ModelManager | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize model manager (use get_instance() instead)."""
        self.model_name: str | None = None
        self.cache_dir: Path = _get_default_cache_dir()
        self.state: ModelState = ModelState.NOT_STARTED
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._error_message: str | None = None
        self._init_lock = threading.Lock()
        self._inference_lock = threading.Lock()  # Protects tokenizer/model during inference

    @classmethod
    def get_instance(cls) -> ModelManager:
        """Get singleton instance of ModelManager.

        Returns:
            The global ModelManager instance.

        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def initialize(
        self,
        model_name: str,
        cache_dir: Path | str | None = None,
        blocking: bool = True,
    ) -> None:
        """Initialize and load the embedding model.

        Downloads the model if not cached, then loads it into memory.
        Validates disk space before attempting download.

        Args:
            model_name: HuggingFace model name (e.g., "sentence-transformers/all-mpnet-base-v2").
            cache_dir: Custom cache directory. Defaults to EMBEDDING_CACHE_DIR env var,
                      or ~/.cache/matrixmind-mcp/models/ if not set.
            blocking: If True, wait for model to load. If False, load in background thread.

        Raises:
            ModelNotReadyException: If initialization fails or insufficient disk space.

        """
        with self._init_lock:
            # Already initialized with same model
            if self.model_name == model_name and self.state == ModelState.READY:
                logger.debug(f"Model {model_name} already loaded")
                return

            self.model_name = model_name
            if cache_dir:
                self.cache_dir = Path(cache_dir)

            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Check disk space before attempting download
            required_mb = _get_model_size_mb(model_name) + DISK_SPACE_BUFFER_MB
            has_space, available_mb, _ = _check_disk_space(self.cache_dir, required_mb)

            if not has_space:
                self.state = ModelState.FAILED
                self._error_message = (
                    f"Insufficient disk space. Need ~{required_mb}MB, "
                    f"but only {available_mb}MB available in {self.cache_dir}"
                )
                logger.error(self._error_message)
                raise ModelNotReadyException(self._error_message)

            if blocking:
                self._load_model()
            else:
                thread = threading.Thread(target=self._load_model, daemon=True)
                thread.start()

    def _load_model(self) -> None:
        """Internal method to download and load the model."""
        if self.model_name is None:
            self.state = ModelState.FAILED
            self._error_message = "Model name not set"
            return

        try:
            self.state = ModelState.DOWNLOADING
            size_hint = _estimate_model_size(self.model_name)
            size_msg = f" ({size_hint})" if size_hint else ""
            logger.info(f"Downloading embedding model: {self.model_name}{size_msg}")
            logger.info(f"Cache directory: {self.cache_dir}")
            logger.info("This may take a few minutes on first run...")

            # Set HuggingFace cache environment variable
            os.environ["HF_HOME"] = str(self.cache_dir)
            os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)

            # Load tokenizer and model (will download if not cached)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            logger.info("Download complete. Loading model into memory...")

            self.state = ModelState.LOADING
            # Suppress "Some weights not initialized" warning for embedding models
            # The pooler layer is unused (we use mean pooling, not CLS token)
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Some weights of.*were not initialized",
                    category=FutureWarning,
                )
                model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )

            # Move to device and set eval mode
            model.to(self.device)
            model.eval()
            self.model = model

            self.state = ModelState.READY
            logger.info(f"Embedding model ready on {self.device}")

        except Exception as e:
            self.state = ModelState.FAILED
            self._error_message = str(e)
            logger.error(f"Failed to load embedding model: {e}")

    def get_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Get the loaded model and tokenizer.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            ModelNotReadyException: If model is not ready (downloading, loading, or failed).

        """
        if self.state == ModelState.NOT_STARTED:
            raise ModelNotReadyException(
                "Embedding model not initialized. Please wait for server startup to complete."
            )

        if self.state == ModelState.DOWNLOADING:
            size_hint = _estimate_model_size(self.model_name or "")
            size_msg = f" ({size_hint})" if size_hint else ""
            raise ModelNotReadyException(
                f"Embedding model '{self.model_name}'{size_msg} is still downloading. "
                "This may take a few minutes on first run. Please try again shortly."
            )

        if self.state == ModelState.LOADING:
            raise ModelNotReadyException(
                f"Embedding model '{self.model_name}' is loading into memory. "
                "Please try again in a few seconds."
            )

        if self.state == ModelState.FAILED:
            raise ModelNotReadyException(
                f"Embedding model failed to load: {self._error_message}. "
                "Check your network connection and model name."
            )

        if self.model is None or self.tokenizer is None:
            raise ModelNotReadyException("Embedding model not available. Please check server logs.")

        return self.model, self.tokenizer

    def is_ready(self) -> bool:
        """Check if model is ready for use.

        Returns:
            True if model is loaded and ready.

        """
        return self.state == ModelState.READY

    def inference_lock(self) -> threading.Lock:
        """Get the inference lock for thread-safe tokenizer/model access.

        The HuggingFace tokenizer's Rust backend is not thread-safe and will
        raise "Already borrowed" errors under concurrent access. Use this lock
        to protect tokenization and model inference calls.

        Returns:
            The inference lock (use with `with` statement).

        Example:
            >>> manager = ModelManager.get_instance()
            >>> with manager.inference_lock():
            ...     inputs = tokenizer(texts, return_tensors="pt")
            ...     outputs = model(**inputs)

        """
        return self._inference_lock

    def get_status(self) -> dict[str, str | int | bool | None]:
        """Get current model and system status.

        Returns comprehensive status including model state, device info,
        memory usage, and disk space. Useful for debugging deployment issues.

        Returns:
            Dictionary with state, model_name, device, memory, disk, and ready status.

        """
        status: dict[str, str | int | bool | None] = {
            "state": self.state.value,
            "model_name": self.model_name or "not set",
            "device": self.device,
            "cache_dir": str(self.cache_dir),
            "ready": self.is_ready(),
            "error": self._error_message,
        }

        # Add disk space info
        try:
            usage = shutil.disk_usage(self.cache_dir)
            status["disk_free_mb"] = int(usage.free // (1024 * 1024))
            status["disk_total_mb"] = int(usage.total // (1024 * 1024))
        except OSError:
            status["disk_free_mb"] = None
            status["disk_total_mb"] = None

        # Add GPU memory info if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                status["gpu_name"] = torch.cuda.get_device_name(0)
                status["gpu_memory_allocated_mb"] = int(
                    torch.cuda.memory_allocated(0) // (1024 * 1024)
                )
                status["gpu_memory_reserved_mb"] = int(
                    torch.cuda.memory_reserved(0) // (1024 * 1024)
                )
                total_mem = torch.cuda.get_device_properties(0).total_memory
                status["gpu_memory_total_mb"] = int(total_mem // (1024 * 1024))
            except Exception:  # noqa: BLE001  # nosec B110
                # GPU info is best-effort; failure here should not break status
                pass

        # Add model size estimate
        if self.model_name:
            status["model_size_mb"] = _get_model_size_mb(self.model_name)

        return status

    def get_hidden_size(self) -> int:
        """Get the embedding dimension size.

        Returns:
            Hidden size of the model.

        Raises:
            ModelNotReadyException: If model is not ready.

        """
        model, _ = self.get_model()
        return int(model.config.hidden_size)
