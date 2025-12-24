"""Reason Guard MCP Configuration.

Centralized configuration management with environment variable support,
Docker secrets integration, and security defaults.

Usage:
    from src.config import config
    print(config.server_name)
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from loguru import logger


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable, treating empty string as unset.

    Also checks Docker secrets path for sensitive values.
    """
    secrets_path = f"/run/secrets/{key.lower()}"
    if os.path.isfile(secrets_path):
        try:
            with Path(secrets_path).open() as f:
                value = f.read().strip()
                if value:
                    return value
        except Exception:  # nosec B110
            pass  # Fall through to env var

    value = os.getenv(key, default)
    return value if value else default


def _get_env_int(key: str, default: int) -> int:
    """Get environment variable as integer."""
    value = os.getenv(key)
    if value:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer for {key}: {value}, using default {default}")
    return default


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean."""
    value = _get_env(key, "").lower()
    if not value:
        return default
    return value in ("true", "1", "yes")


@dataclass(frozen=True)
class AuthConfig:
    """Authentication configuration."""

    enabled: bool = field(default_factory=lambda: _get_env_bool("AUTH_ENABLED", False))
    bypass_localhost: bool = field(
        default_factory=lambda: _get_env_bool("AUTH_BYPASS_LOCALHOST", True)
    )

    @cached_property
    def api_key_hashes(self) -> frozenset[str]:
        """Load API keys from environment or secrets file."""
        keys: set[str] = set()

        # Try Docker secrets first
        secrets_path = "/run/secrets/reasonguard_api_keys"
        if os.path.isfile(secrets_path):
            try:
                with Path(secrets_path).open() as f:
                    for line in f:
                        key = line.strip()
                        if key and not key.startswith("#"):
                            keys.add(hashlib.sha256(key.encode()).hexdigest())
                if keys:
                    logger.info(f"Loaded {len(keys)} API keys from Docker secrets")
                    return frozenset(keys)
            except Exception as e:
                logger.warning(f"Failed to read Docker secrets: {e}")

        # Try file path
        keys_file = _get_env("REASONGUARD_API_KEYS_FILE")
        if keys_file and os.path.isfile(keys_file):
            try:
                with Path(keys_file).open() as f:
                    for line in f:
                        key = line.strip()
                        if key and not key.startswith("#"):
                            keys.add(hashlib.sha256(key.encode()).hexdigest())
                if keys:
                    logger.info(f"Loaded {len(keys)} API keys from {keys_file}")
                    return frozenset(keys)
            except Exception as e:
                logger.warning(f"Failed to read API keys file: {e}")

        # Try environment variable
        keys_env = _get_env("REASONGUARD_API_KEYS")
        if keys_env:
            for key in keys_env.split(","):
                key = key.strip()
                if key:
                    keys.add(hashlib.sha256(key.encode()).hexdigest())
            if keys:
                logger.info(f"Loaded {len(keys)} API keys from environment")
                return frozenset(keys)

        if self.enabled:
            logger.warning(
                "AUTH_ENABLED=true but no API keys configured. "
                "Set REASONGUARD_API_KEYS, REASONGUARD_API_KEYS_FILE, or use Docker secrets."
            )

        return frozenset(keys)

    def validate_api_key(self, api_key: str | None) -> tuple[bool, str]:
        """Validate an API key using constant-time comparison."""
        if not self.enabled:
            return True, ""

        if not api_key:
            return False, "API key required. Set Authorization header with Bearer token."

        if api_key.startswith("Bearer "):
            api_key = api_key[7:]

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Constant-time comparison
        is_valid = False
        for valid_hash in self.api_key_hashes:
            if secrets.compare_digest(key_hash, valid_hash):
                is_valid = True

        if not is_valid:
            return False, "Invalid API key"

        return True, ""


@dataclass(frozen=True)
class ProxyConfig:
    """Proxy and IP forwarding configuration (CWE-348 mitigation)."""

    _trusted_proxies_raw: str = field(default_factory=lambda: _get_env("TRUSTED_PROXIES", ""))

    @cached_property
    def trusted_proxies(self) -> frozenset[str]:
        """Parse trusted proxies list."""
        return frozenset(ip.strip() for ip in self._trusted_proxies_raw.split(",") if ip.strip())

    def is_trusted_proxy(self, ip: str) -> bool:
        """Check if an IP is in the trusted proxies list."""
        if not self.trusted_proxies:
            return False

        import ipaddress

        try:
            client_addr = ipaddress.ip_address(ip)
        except ValueError:
            return False

        for proxy in self.trusted_proxies:
            try:
                if "/" in proxy:
                    if client_addr in ipaddress.ip_network(proxy, strict=False):
                        return True
                else:
                    if client_addr == ipaddress.ip_address(proxy):
                        return True
            except ValueError:
                continue

        return False


@dataclass(frozen=True)
class ServerConfig:
    """Server runtime configuration."""

    name: str = field(default_factory=lambda: _get_env("SERVER_NAME", "Reason-Guard-MCP"))
    transport: str = field(default_factory=lambda: _get_env("SERVER_TRANSPORT", "stdio"))
    host: str = field(default_factory=lambda: _get_env("SERVER_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _get_env_int("SERVER_PORT", 8000))


@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limiting configuration."""

    max_sessions: int = field(default_factory=lambda: _get_env_int("RATE_LIMIT_MAX_SESSIONS", 100))
    window_seconds: int = field(
        default_factory=lambda: _get_env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
    )
    max_total_sessions: int = field(default_factory=lambda: _get_env_int("MAX_TOTAL_SESSIONS", 500))

    # IP-based rate limiting
    ip_enabled: bool = field(default_factory=lambda: _get_env_bool("IP_RATE_LIMIT_ENABLED", True))
    ip_max_requests: int = field(
        default_factory=lambda: _get_env_int("IP_RATE_LIMIT_MAX_REQUESTS", 100)
    )
    ip_window_seconds: int = field(
        default_factory=lambda: _get_env_int("IP_RATE_LIMIT_WINDOW_SECONDS", 60)
    )


@dataclass(frozen=True)
class SessionConfig:
    """Session management configuration."""

    max_age_minutes: int = field(
        default_factory=lambda: _get_env_int("SESSION_MAX_AGE_MINUTES", 30)
    )
    cleanup_interval_seconds: int = field(
        default_factory=lambda: _get_env_int("CLEANUP_INTERVAL_SECONDS", 60)
    )


@dataclass(frozen=True)
class InputLimitsConfig:
    """Input size limits (CWE-400 mitigation)."""

    max_problem_size: int = field(default_factory=lambda: _get_env_int("MAX_PROBLEM_SIZE", 50000))
    max_thought_size: int = field(default_factory=lambda: _get_env_int("MAX_THOUGHT_SIZE", 10000))
    max_context_size: int = field(default_factory=lambda: _get_env_int("MAX_CONTEXT_SIZE", 100000))
    max_alternatives: int = field(default_factory=lambda: _get_env_int("MAX_ALTERNATIVES", 10))
    max_thoughts_per_session: int = field(
        default_factory=lambda: _get_env_int("MAX_THOUGHTS_PER_SESSION", 1000)
    )


@dataclass(frozen=True)
class RAGConfig:
    """RAG and vector store configuration."""

    enabled: bool = field(default_factory=lambda: _get_env_bool("ENABLE_RAG", True))
    vector_db_path: str = field(
        default_factory=lambda: _get_env("VECTOR_DB_PATH", "reasonguard_thoughts.db")
    )

    def get_validated_db_path(self) -> str:
        """Validate and return the vector DB path (CWE-22 mitigation)."""
        from src.utils.weight_store import validate_db_path

        if self.vector_db_path == ":memory:":
            return self.vector_db_path

        try:
            validated = validate_db_path(self.vector_db_path)
            return str(validated)
        except ValueError as e:
            logger.error(f"Invalid VECTOR_DB_PATH: {e}")
            logger.warning("Falling back to in-memory vector store")
            return ":memory:"


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    embedding_model: str = field(
        default_factory=lambda: _get_env("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-xs")
    )

    @property
    def full_model_name(self) -> str:
        """Get full model name with provider prefix."""
        name = self.embedding_model
        if "/" not in name:
            name = f"sentence-transformers/{name}"
        return name


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    auth: AuthConfig = field(default_factory=AuthConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    input_limits: InputLimitsConfig = field(default_factory=InputLimitsConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary (for logging/debugging)."""
        return {
            "server": {
                "name": self.server.name,
                "transport": self.server.transport,
                "host": self.server.host,
                "port": self.server.port,
            },
            "auth": {
                "enabled": self.auth.enabled,
                "bypass_localhost": self.auth.bypass_localhost,
                "keys_loaded": len(self.auth.api_key_hashes),
            },
            "proxy": {
                "trusted_proxies": list(self.proxy.trusted_proxies),
            },
            "rate_limit": {
                "max_sessions": self.rate_limit.max_sessions,
                "window_seconds": self.rate_limit.window_seconds,
                "ip_enabled": self.rate_limit.ip_enabled,
            },
            "session": {
                "max_age_minutes": self.session.max_age_minutes,
            },
            "rag": {
                "enabled": self.rag.enabled,
                "vector_db_path": self.rag.vector_db_path,
            },
            "model": {
                "embedding_model": self.model.full_model_name,
            },
        }


# Global config instance (lazy-loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Reload configuration (for testing)."""
    global _config
    _config = Config()
    return _config


# Convenience alias
config = get_config()
