"""HMAC-based session token signing and verification.

Provides cryptographic binding between session IDs and client identities
to prevent session hijacking and fixation attacks.

Usage:
    from src.utils.session_signing import SessionSigner

    signer = SessionSigner()  # Uses env var or generates ephemeral key

    # Create signed token
    token = signer.create_token(session_id="abc123", client_id="user@example.com")

    # Verify token
    payload = signer.verify_token(token)  # Returns {"session_id": ..., "client_id": ...}
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


class TokenError(Exception):
    """Base exception for token operations."""

    pass


class TokenExpiredError(TokenError):
    """Token has expired."""

    pass


class TokenInvalidError(TokenError):
    """Token is malformed or signature is invalid."""

    pass


@dataclass(frozen=True)
class TokenPayload:
    """Verified token payload."""

    session_id: str
    client_id: str
    issued_at: float
    expires_at: float

    def is_expired(self, *, now: float | None = None) -> bool:
        """Check if token is expired."""
        if now is None:
            now = time.time()
        return now > self.expires_at


class SessionSigner:
    """HMAC-SHA256 session token signer.

    Generates tokens that cryptographically bind:
    - Session ID (the session being accessed)
    - Client ID (the client accessing the session)
    - Expiration time (prevents replay attacks with old tokens)

    The secret key is loaded from:
    1. SESSION_SIGNING_KEY environment variable
    2. Docker secret at /run/secrets/session_signing_key
    3. Ephemeral key generated at startup (lost on restart)

    Token format: base64url(header.payload.signature)
    """

    # Default token lifetime: 1 hour
    DEFAULT_TOKEN_TTL_SECONDS = 3600

    def __init__(
        self,
        secret_key: bytes | None = None,
        *,
        token_ttl_seconds: int | None = None,
    ) -> None:
        """Initialize signer with secret key.

        Args:
            secret_key: HMAC secret key. If None, loads from env or generates.
            token_ttl_seconds: Token lifetime in seconds. Defaults to 1 hour.

        """
        self._secret_key = secret_key or self._load_or_generate_key()
        self._token_ttl = token_ttl_seconds or self.DEFAULT_TOKEN_TTL_SECONDS

    @staticmethod
    def _load_or_generate_key() -> bytes:
        """Load secret key from environment or generate ephemeral key."""
        # Try environment variable first
        key_str = os.getenv("SESSION_SIGNING_KEY")
        if key_str:
            return key_str.encode("utf-8")

        # Try Docker secret (nosec B105 - this is a path, not a hardcoded password)
        secret_path = "/run/secrets/session_signing_key"  # nosec B105
        if os.path.isfile(secret_path):
            try:
                with open(secret_path, "rb") as f:
                    key = f.read().strip()
                    if len(key) >= 32:
                        return key
                    logger.warning("SESSION_SIGNING_KEY from Docker secret is too short")
            except Exception as e:
                logger.warning(f"Failed to read Docker secret: {e}")

        # Generate ephemeral key (32 bytes = 256 bits)
        logger.warning(
            "No SESSION_SIGNING_KEY found. Using ephemeral key. "
            "Sessions will be invalidated on restart."
        )
        return secrets.token_bytes(32)

    def _sign(self, data: bytes) -> bytes:
        """Create HMAC-SHA256 signature."""
        return hmac.new(self._secret_key, data, hashlib.sha256).digest()

    def _verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify HMAC-SHA256 signature using constant-time comparison."""
        expected = self._sign(data)
        return hmac.compare_digest(expected, signature)

    def create_token(
        self,
        session_id: str,
        client_id: str,
        *,
        ttl_seconds: int | None = None,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        """Create a signed session token.

        Args:
            session_id: The session being accessed.
            client_id: Identifier for the client (IP hash, user ID, etc.).
            ttl_seconds: Override default token lifetime.
            extra_claims: Additional claims to include in payload.

        Returns:
            Base64url-encoded signed token.

        """
        now = time.time()
        ttl = ttl_seconds or self._token_ttl

        payload = {
            "sid": session_id,
            "cid": client_id,
            "iat": int(now),
            "exp": int(now + ttl),
        }
        if extra_claims:
            payload["ext"] = extra_claims

        # Encode payload as JSON
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        # Create signature
        signature = self._sign(payload_bytes)

        # Combine payload and signature
        token_bytes = payload_bytes + b"." + signature

        # Encode as base64url
        return base64.urlsafe_b64encode(token_bytes).decode("ascii")

    def verify_token(
        self,
        token: str,
        *,
        expected_session_id: str | None = None,
        expected_client_id: str | None = None,
        allow_expired: bool = False,
    ) -> TokenPayload:
        """Verify a signed session token.

        Args:
            token: The token to verify.
            expected_session_id: If provided, verify session ID matches.
            expected_client_id: If provided, verify client ID matches.
            allow_expired: If True, don't raise on expired tokens.

        Returns:
            TokenPayload with verified claims.

        Raises:
            TokenInvalidError: If token is malformed or signature invalid.
            TokenExpiredError: If token has expired (unless allow_expired=True).

        """
        try:
            # Decode base64url
            token_bytes = base64.urlsafe_b64decode(token.encode("ascii"))
        except Exception as e:
            raise TokenInvalidError(f"Invalid token encoding: {e}") from e

        # Split payload and signature
        try:
            payload_bytes, signature = token_bytes.rsplit(b".", 1)
        except ValueError as e:
            raise TokenInvalidError("Invalid token format") from e

        # Verify signature first (constant-time)
        if not self._verify_signature(payload_bytes, signature):
            raise TokenInvalidError("Invalid token signature")

        # Parse payload
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise TokenInvalidError(f"Invalid token payload: {e}") from e

        # Extract claims
        try:
            session_id = str(payload["sid"])
            client_id = str(payload["cid"])
            issued_at = float(payload["iat"])
            expires_at = float(payload["exp"])
        except KeyError as e:
            raise TokenInvalidError(f"Missing required claim: {e}") from e
        except (TypeError, ValueError) as e:
            raise TokenInvalidError(f"Invalid claim value: {e}") from e

        # Check expiration
        now = time.time()
        if not allow_expired and now > expires_at:
            raise TokenExpiredError(f"Token expired at {expires_at}, current time is {now}")

        # Validate expected values if provided
        if expected_session_id is not None and session_id != expected_session_id:
            raise TokenInvalidError(
                f"Session ID mismatch: expected {expected_session_id}, got {session_id}"
            )
        if expected_client_id is not None and client_id != expected_client_id:
            raise TokenInvalidError(
                f"Client ID mismatch: expected {expected_client_id}, got {client_id}"
            )

        return TokenPayload(
            session_id=session_id,
            client_id=client_id,
            issued_at=issued_at,
            expires_at=expires_at,
        )

    def refresh_token(
        self,
        token: str,
        *,
        ttl_seconds: int | None = None,
    ) -> str:
        """Refresh a token by creating a new one with the same claims.

        Args:
            token: The token to refresh.
            ttl_seconds: Override default token lifetime for new token.

        Returns:
            New signed token with extended expiration.

        Raises:
            TokenInvalidError: If token is invalid.
            TokenExpiredError: If token has expired.

        """
        payload = self.verify_token(token)
        return self.create_token(
            session_id=payload.session_id,
            client_id=payload.client_id,
            ttl_seconds=ttl_seconds,
        )


def hash_client_id(ip: str, user_agent: str | None = None) -> str:
    """Create a stable hash of client identifiers.

    Creates a SHA-256 hash of client IP and optional user-agent to use as
    client_id in session tokens. This provides some privacy while still
    allowing session binding.

    Args:
        ip: Client IP address.
        user_agent: Optional User-Agent header.

    Returns:
        Hex-encoded SHA-256 hash (first 16 chars for brevity).

    """
    data = ip
    if user_agent:
        data += f":{user_agent}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


# Global signer instance (lazy-loaded)
_signer: SessionSigner | None = None


def get_session_signer() -> SessionSigner:
    """Get the global session signer instance."""
    global _signer
    if _signer is None:
        _signer = SessionSigner()
    return _signer


def reset_session_signer() -> None:
    """Reset the global session signer (for testing)."""
    global _signer
    _signer = None
