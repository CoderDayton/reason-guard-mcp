"""Persistent weight storage for suggestion learning.

Stores learned suggestion weights per domain using SQLite.
Weights are loaded when a session starts and saved when feedback is recorded.

Schema Design:
- Weights are stored per domain (math, code, logic, factual, general)
- Each domain has its own learned preference profile
- Weights persist across server restarts
- Thread-safe for concurrent access

Weight Decay:
- Old feedback influence decays over time toward default values
- Configurable decay rate and threshold (days before decay starts)
- Allows system to adapt when user preferences change
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".reasonguard" / "weights.db"

# Allowed base directories for database files (security: prevent path traversal)
# Users can override via REASONGUARD_ALLOWED_DB_DIRS env var (colon-separated)
_DEFAULT_ALLOWED_DIRS = [
    Path.home() / ".reasonguard",
    Path.home() / ".local" / "share" / "reasonguard",
    Path("/tmp"),  # nosec B108 - intentionally allowed for dev/testing
    Path.cwd(),
]


def _get_allowed_db_dirs() -> list[Path]:
    """Get list of allowed directories for database files."""
    env_dirs = os.getenv("REASONGUARD_ALLOWED_DB_DIRS")
    if env_dirs:
        return [Path(d).resolve() for d in env_dirs.split(":") if d]
    return [d.resolve() for d in _DEFAULT_ALLOWED_DIRS]


def validate_db_path(db_path: Path | str) -> Path:
    """Validate and sanitize database path to prevent path traversal.

    Args:
        db_path: Proposed database path.

    Returns:
        Validated, resolved Path object.

    Raises:
        ValueError: If path is outside allowed directories or contains traversal.

    """
    # Allow :memory: for in-memory databases
    if str(db_path) == ":memory:":
        return Path(":memory:")

    path = Path(db_path).resolve()  # Resolve to absolute, canonical path

    # Check for path traversal attempts in the original string
    path_str = str(db_path)
    if ".." in path_str or path_str.startswith("/etc") or path_str.startswith("/root"):
        raise ValueError(f"Invalid database path: suspicious pattern detected in '{db_path}'")

    # Ensure path is within allowed directories
    allowed_dirs = _get_allowed_db_dirs()
    is_allowed = any(
        path == allowed_dir or allowed_dir in path.parents for allowed_dir in allowed_dirs
    )

    if not is_allowed:
        allowed_list = ", ".join(str(d) for d in allowed_dirs)
        raise ValueError(
            f"Database path '{path}' is outside allowed directories. "
            f"Allowed: {allowed_list}. "
            f"Set REASONGUARD_ALLOWED_DB_DIRS to add custom directories."
        )

    return path


# Default weight values (used for decay target)
DEFAULT_WEIGHTS = {
    "resolve": 1.0,
    "continue_blind_spot": 0.9,
    "verify": 0.8,
    "continue_depth": 0.7,
    "synthesize": 0.6,
    "finish": 0.5,
    "continue_default": 0.4,
}

# Decay configuration
DEFAULT_DECAY_RATE = 0.99  # Per-day decay factor (1.0 = no decay)
DEFAULT_DECAY_THRESHOLD_DAYS = 7  # Days before decay starts


@dataclass
class DecayConfig:
    """Configuration for weight decay over time.

    Weight decay allows the system to adapt when user preferences change
    by gradually moving learned weights back toward defaults.

    Attributes:
        decay_rate: Factor applied per day (0.99 = 1% decay toward default per day)
        threshold_days: Days of inactivity before decay starts
        enabled: Whether decay is active

    Example:
        With decay_rate=0.99 and threshold_days=7:
        - Weight of 1.2 (default 1.0) after 30 days idle:
          offset = 0.2 * (0.99 ^ 23) = 0.2 * 0.79 = 0.158
          new_weight = 1.0 + 0.158 = 1.158

    """

    decay_rate: float = DEFAULT_DECAY_RATE
    threshold_days: int = DEFAULT_DECAY_THRESHOLD_DAYS
    enabled: bool = True

    def calculate_decay_factor(self, days_since_update: float) -> float:
        """Calculate decay factor based on days since last update.

        Args:
            days_since_update: Days since weights were last updated

        Returns:
            Decay factor (1.0 = no decay, 0.0 = full decay to default)

        """
        if not self.enabled or days_since_update <= self.threshold_days:
            return 1.0

        # Apply exponential decay for days beyond threshold
        decay_days = days_since_update - self.threshold_days
        return float(self.decay_rate**decay_days)


@dataclass
class DomainWeights:
    """Weights for a specific domain.

    Mirrors the SuggestionWeights structure for serialization.
    Supports time-based decay toward default values.
    """

    domain: str
    resolve: float = 1.0
    continue_blind_spot: float = 0.9
    verify: float = 0.8
    continue_depth: float = 0.7
    synthesize: float = 0.6
    finish: float = 0.5
    continue_default: float = 0.4
    # Metadata
    total_feedback: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, float]:
        """Return weights as dictionary (excluding metadata)."""
        return {
            "resolve": self.resolve,
            "continue_blind_spot": self.continue_blind_spot,
            "verify": self.verify,
            "continue_depth": self.continue_depth,
            "synthesize": self.synthesize,
            "finish": self.finish,
            "continue_default": self.continue_default,
        }

    def acceptance_rate(self) -> float:
        """Calculate suggestion acceptance rate."""
        if self.total_feedback == 0:
            return 0.0
        return self.accepted_count / self.total_feedback

    def days_since_update(self) -> float:
        """Calculate days since last weight update."""
        delta = datetime.now() - self.last_updated
        return delta.total_seconds() / 86400  # seconds per day

    def apply_decay(self, config: DecayConfig | None = None) -> DomainWeights:
        """Apply time-based decay toward default values.

        Creates a new DomainWeights with decayed values. Does not modify self.

        Args:
            config: Decay configuration. Uses defaults if None.

        Returns:
            New DomainWeights with decay applied

        """
        if config is None:
            config = DecayConfig()

        days = self.days_since_update()
        factor = config.calculate_decay_factor(days)

        if factor >= 1.0:
            # No decay needed
            return self

        # Apply decay: move each weight toward its default
        decayed = DomainWeights(
            domain=self.domain,
            total_feedback=self.total_feedback,
            accepted_count=self.accepted_count,
            rejected_count=self.rejected_count,
            last_updated=self.last_updated,
        )

        for attr, default in DEFAULT_WEIGHTS.items():
            current = getattr(self, attr)
            # Decay the offset from default, not the absolute value
            offset = current - default
            decayed_offset = offset * factor
            setattr(decayed, attr, default + decayed_offset)

        return decayed

    def decay_info(self, config: DecayConfig | None = None) -> dict[str, Any]:
        """Get information about decay status.

        Args:
            config: Decay configuration. Uses defaults if None.

        Returns:
            Dictionary with decay status information

        """
        if config is None:
            config = DecayConfig()

        days = self.days_since_update()
        factor = config.calculate_decay_factor(days)
        days_until_decay = max(0, config.threshold_days - days)

        return {
            "days_since_update": round(days, 2),
            "decay_factor": round(factor, 4),
            "decay_active": factor < 1.0,
            "days_until_decay_starts": round(days_until_decay, 2),
            "threshold_days": config.threshold_days,
            "decay_rate": config.decay_rate,
        }


class WeightStore:
    """Thread-safe SQLite store for learned suggestion weights.

    Provides persistent storage for domain-specific weight profiles
    that survive server restarts. Supports time-based weight decay.

    Usage:
        store = WeightStore()  # Uses default path
        weights = store.load_weights("math")  # Load or create
        weights.verify = 0.9
        store.save_weights(weights)  # Persist changes

    Decay Usage:
        store = WeightStore(decay_config=DecayConfig(decay_rate=0.98))
        weights = store.load_weights("math", apply_decay=True)
        # Weights are automatically decayed based on time since last update

    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        decay_config: DecayConfig | None = None,
    ) -> None:
        """Initialize weight store.

        Args:
            db_path: Path to SQLite database. If None, uses default path.
                    Use ":memory:" for in-memory database (testing).
            decay_config: Configuration for weight decay. Uses defaults if None.

        Raises:
            ValueError: If db_path is outside allowed directories.

        """
        if db_path is None:
            self.db_path = DEFAULT_DB_PATH
        else:
            # Validate path to prevent traversal attacks (CWE-22)
            self.db_path = validate_db_path(db_path)

        self.decay_config = decay_config or DecayConfig()
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            # Ensure directory exists for file-based databases
            if str(self.db_path) != ":memory:":
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,  # We handle threading ourselves
                timeout=30.0,
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = self._get_connection()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS domain_weights (
                    domain TEXT PRIMARY KEY,
                    resolve REAL DEFAULT 1.0,
                    continue_blind_spot REAL DEFAULT 0.9,
                    verify REAL DEFAULT 0.8,
                    continue_depth REAL DEFAULT 0.7,
                    synthesize REAL DEFAULT 0.6,
                    finish REAL DEFAULT 0.5,
                    continue_default REAL DEFAULT 0.4,
                    total_feedback INTEGER DEFAULT 0,
                    accepted_count INTEGER DEFAULT 0,
                    rejected_count INTEGER DEFAULT 0,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS feedback_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    suggestion_id TEXT NOT NULL,
                    suggested_action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    actual_action TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (domain) REFERENCES domain_weights(domain)
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_domain
                    ON feedback_history(domain);
                CREATE INDEX IF NOT EXISTS idx_feedback_session
                    ON feedback_history(session_id);
                """
            )
            conn.commit()

    def load_weights(self, domain: str, apply_decay: bool = True) -> DomainWeights:
        """Load weights for a domain, creating defaults if not exists.

        Args:
            domain: Domain name (math, code, logic, factual, general)
            apply_decay: Whether to apply time-based decay to loaded weights

        Returns:
            DomainWeights with current learned values (optionally decayed)

        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM domain_weights WHERE domain = ?", (domain,))
            row = cursor.fetchone()

            if row is None:
                # Create default weights for this domain
                weights = DomainWeights(domain=domain)
                self.save_weights(weights)
                return weights

            weights = DomainWeights(
                domain=row["domain"],
                resolve=row["resolve"],
                continue_blind_spot=row["continue_blind_spot"],
                verify=row["verify"],
                continue_depth=row["continue_depth"],
                synthesize=row["synthesize"],
                finish=row["finish"],
                continue_default=row["continue_default"],
                total_feedback=row["total_feedback"],
                accepted_count=row["accepted_count"],
                rejected_count=row["rejected_count"],
                last_updated=datetime.fromisoformat(row["last_updated"]),
            )

            if apply_decay and self.decay_config.enabled:
                return weights.apply_decay(self.decay_config)

            return weights

    def save_weights(self, weights: DomainWeights, update_timestamp: bool = True) -> None:
        """Save weights for a domain.

        Args:
            weights: DomainWeights to persist
            update_timestamp: If True, sets last_updated to now. If False, preserves
                the timestamp from the weights object.

        """
        with self._lock:
            conn = self._get_connection()
            timestamp = (
                datetime.now().isoformat() if update_timestamp else weights.last_updated.isoformat()
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO domain_weights (
                    domain, resolve, continue_blind_spot, verify,
                    continue_depth, synthesize, finish, continue_default,
                    total_feedback, accepted_count, rejected_count, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    weights.domain,
                    weights.resolve,
                    weights.continue_blind_spot,
                    weights.verify,
                    weights.continue_depth,
                    weights.synthesize,
                    weights.finish,
                    weights.continue_default,
                    weights.total_feedback,
                    weights.accepted_count,
                    weights.rejected_count,
                    timestamp,
                ),
            )
            conn.commit()

    def record_feedback(
        self,
        domain: str,
        session_id: str,
        suggestion_id: str,
        suggested_action: str,
        outcome: str,
        actual_action: str | None = None,
    ) -> DomainWeights:
        """Record feedback and update weights.

        Args:
            domain: Domain name
            session_id: Session ID
            suggestion_id: Suggestion ID
            suggested_action: What was suggested
            outcome: "accepted" or "rejected"
            actual_action: What user actually did (if different)

        Returns:
            Updated DomainWeights

        """
        with self._lock:
            conn = self._get_connection()

            # Record feedback history
            conn.execute(
                """
                INSERT INTO feedback_history (
                    domain, session_id, suggestion_id,
                    suggested_action, outcome, actual_action
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (domain, session_id, suggestion_id, suggested_action, outcome, actual_action),
            )

            # Load current weights
            weights = self.load_weights(domain)

            # Update counters
            weights.total_feedback += 1
            if outcome == "accepted":
                weights.accepted_count += 1
            else:
                weights.rejected_count += 1

            # Adjust weights based on feedback
            accepted = outcome == "accepted"
            learning_rate = 0.1

            attr_map = {
                "resolve": "resolve",
                "verify": "verify",
                "synthesize": "synthesize",
                "finish": "finish",
                "continue": "continue_default",
            }

            attr = attr_map.get(suggested_action)
            if attr and hasattr(weights, attr):
                current = getattr(weights, attr)
                delta = learning_rate if accepted else -learning_rate
                new_value = max(0.1, min(2.0, current + delta))
                setattr(weights, attr, new_value)

            # If rejected and user took different action, boost that action
            if not accepted and actual_action and actual_action != suggested_action:
                boost_attr = attr_map.get(actual_action)
                if boost_attr and hasattr(weights, boost_attr):
                    current = getattr(weights, boost_attr)
                    new_value = max(0.1, min(2.0, current + 0.05))
                    setattr(weights, boost_attr, new_value)

            # Save updated weights
            self.save_weights(weights)
            conn.commit()

            return weights

    def get_feedback_history(
        self,
        domain: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get feedback history with optional filters.

        Args:
            domain: Filter by domain
            session_id: Filter by session
            limit: Maximum records to return

        Returns:
            List of feedback records

        """
        with self._lock:
            conn = self._get_connection()

            query = "SELECT * FROM feedback_history WHERE 1=1"
            params: list[Any] = []

            if domain:
                query += " AND domain = ?"
                params.append(domain)
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_all_domains(self, apply_decay: bool = True) -> list[DomainWeights]:
        """Get weights for all domains.

        Args:
            apply_decay: Whether to apply time-based decay to loaded weights

        Returns:
            List of DomainWeights for all stored domains

        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT domain FROM domain_weights")
            domains = [row["domain"] for row in cursor.fetchall()]
            return [self.load_weights(domain, apply_decay=apply_decay) for domain in domains]

    def get_statistics(self) -> dict[str, Any]:
        """Get overall statistics including decay configuration.

        Returns:
            Dictionary with statistics across all domains

        """
        with self._lock:
            conn = self._get_connection()

            # Domain stats
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as domain_count,
                    SUM(total_feedback) as total_feedback,
                    SUM(accepted_count) as total_accepted,
                    SUM(rejected_count) as total_rejected
                FROM domain_weights
                """
            )
            row = cursor.fetchone()

            total_feedback = row["total_feedback"] or 0
            total_accepted = row["total_accepted"] or 0

            return {
                "domain_count": row["domain_count"],
                "total_feedback": total_feedback,
                "total_accepted": total_accepted,
                "total_rejected": row["total_rejected"] or 0,
                "acceptance_rate": total_accepted / total_feedback if total_feedback > 0 else 0.0,
                "db_path": str(self.db_path),
                "decay_enabled": self.decay_config.enabled,
                "decay_rate": self.decay_config.decay_rate,
                "decay_threshold_days": self.decay_config.threshold_days,
            }

    def get_decay_status(self) -> dict[str, Any]:
        """Get decay status for all domains.

        Returns:
            Dictionary mapping domain names to their decay info

        """
        status: dict[str, Any] = {
            "config": {
                "enabled": self.decay_config.enabled,
                "decay_rate": self.decay_config.decay_rate,
                "threshold_days": self.decay_config.threshold_days,
            },
            "domains": {},
        }

        # Load raw weights (without decay) to get accurate timestamps
        for weights in self.get_all_domains(apply_decay=False):
            status["domains"][weights.domain] = weights.decay_info(self.decay_config)

        return status

    def set_decay_config(self, config: DecayConfig) -> None:
        """Update decay configuration.

        Args:
            config: New decay configuration

        """
        self.decay_config = config

    def reset_domain(self, domain: str) -> DomainWeights:
        """Reset weights for a domain to defaults.

        Args:
            domain: Domain to reset

        Returns:
            Fresh DomainWeights with defaults

        """
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM domain_weights WHERE domain = ?", (domain,))
            conn.execute("DELETE FROM feedback_history WHERE domain = ?", (domain,))
            conn.commit()

            return self.load_weights(domain)

    def reset_all(self) -> None:
        """Reset all weights and history."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM domain_weights")
            conn.execute("DELETE FROM feedback_history")
            conn.commit()

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None


# Global singleton instance
_weight_store: WeightStore | None = None
_store_lock = threading.Lock()


def get_weight_store(db_path: Path | str | None = None) -> WeightStore:
    """Get or create the global weight store singleton.

    Args:
        db_path: Optional path override. Only used on first call.

    Returns:
        Global WeightStore instance

    """
    global _weight_store
    with _store_lock:
        if _weight_store is None:
            _weight_store = WeightStore(db_path)
        return _weight_store


def reset_weight_store() -> None:
    """Reset the global weight store singleton (for testing)."""
    global _weight_store
    with _store_lock:
        if _weight_store:
            _weight_store.close()
            _weight_store = None
