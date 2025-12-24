"""Tests for weight persistence using WeightStore."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.utils.weight_store import (
    DEFAULT_WEIGHTS,
    DecayConfig,
    DomainWeights,
    WeightStore,
    get_weight_store,
    reset_weight_store,
)


class TestDomainWeights:
    """Tests for DomainWeights dataclass."""

    def test_default_values(self) -> None:
        """Test default weight values."""
        weights = DomainWeights(domain="test")

        assert weights.domain == "test"
        assert weights.resolve == 1.0
        assert weights.verify == 0.8
        assert weights.continue_default == 0.4
        assert weights.finish == 0.5
        assert weights.total_feedback == 0
        assert weights.accepted_count == 0
        assert weights.rejected_count == 0

    def test_to_dict(self) -> None:
        """Test to_dict excludes metadata."""
        weights = DomainWeights(domain="test", total_feedback=10)
        d = weights.to_dict()

        assert "domain" not in d
        assert "total_feedback" not in d
        assert "resolve" in d
        assert "verify" in d
        assert "continue_default" in d

    def test_acceptance_rate(self) -> None:
        """Test acceptance rate calculation."""
        weights = DomainWeights(
            domain="test",
            total_feedback=10,
            accepted_count=7,
            rejected_count=3,
        )

        assert weights.acceptance_rate() == 0.7

    def test_acceptance_rate_zero_feedback(self) -> None:
        """Test acceptance rate with no feedback."""
        weights = DomainWeights(domain="test")
        assert weights.acceptance_rate() == 0.0


class TestWeightStore:
    """Tests for WeightStore SQLite backend."""

    def test_create_in_memory(self) -> None:
        """Test creating in-memory store."""
        store = WeightStore(":memory:")
        assert store is not None
        store.close()

    def test_create_file_based(self) -> None:
        """Test creating file-based store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_weights.db"
            store = WeightStore(db_path)
            assert store is not None
            assert db_path.exists()
            store.close()

    def test_load_creates_default(self) -> None:
        """Test loading non-existent domain creates defaults."""
        store = WeightStore(":memory:")

        weights = store.load_weights("math")

        assert weights.domain == "math"
        assert weights.resolve == 1.0
        assert weights.verify == 0.8
        store.close()

    def test_save_and_load(self) -> None:
        """Test saving and loading weights."""
        store = WeightStore(":memory:")

        # Modify and save
        weights = DomainWeights(domain="code", verify=0.95, finish=0.3)
        store.save_weights(weights)

        # Load and verify
        loaded = store.load_weights("code")
        assert loaded.verify == 0.95
        assert loaded.finish == 0.3
        store.close()

    def test_record_feedback_accepted(self) -> None:
        """Test recording accepted feedback."""
        store = WeightStore(":memory:")

        # Record accepted feedback
        updated = store.record_feedback(
            domain="logic",
            session_id="sess1",
            suggestion_id="sug1",
            suggested_action="verify",
            outcome="accepted",
        )

        assert updated.total_feedback == 1
        assert updated.accepted_count == 1
        assert updated.rejected_count == 0
        # Verify weight increased
        assert updated.verify > 0.8
        store.close()

    def test_record_feedback_rejected(self) -> None:
        """Test recording rejected feedback."""
        store = WeightStore(":memory:")

        # Record rejected feedback
        updated = store.record_feedback(
            domain="math",
            session_id="sess1",
            suggestion_id="sug1",
            suggested_action="continue",
            outcome="rejected",
            actual_action="verify",
        )

        assert updated.total_feedback == 1
        assert updated.rejected_count == 1
        # continue_default should decrease
        assert updated.continue_default < 0.4
        # verify should increase (user's actual action)
        assert updated.verify > 0.8
        store.close()

    def test_feedback_history(self) -> None:
        """Test feedback history retrieval."""
        store = WeightStore(":memory:")

        # Record multiple feedback entries
        store.record_feedback("math", "s1", "sug1", "verify", "accepted")
        store.record_feedback("math", "s1", "sug2", "continue", "rejected")
        store.record_feedback("code", "s2", "sug3", "verify", "accepted")

        # Get all history
        history = store.get_feedback_history()
        assert len(history) == 3

        # Filter by domain
        math_history = store.get_feedback_history(domain="math")
        assert len(math_history) == 2

        # Filter by session
        s1_history = store.get_feedback_history(session_id="s1")
        assert len(s1_history) == 2
        store.close()

    def test_get_all_domains(self) -> None:
        """Test getting all domain weights."""
        store = WeightStore(":memory:")

        # Create some domains
        store.load_weights("math")
        store.load_weights("code")
        store.load_weights("logic")

        domains = store.get_all_domains()
        assert len(domains) == 3
        domain_names = {d.domain for d in domains}
        assert domain_names == {"math", "code", "logic"}
        store.close()

    def test_get_statistics(self) -> None:
        """Test getting overall statistics."""
        store = WeightStore(":memory:")

        # Record some feedback
        store.record_feedback("math", "s1", "sug1", "verify", "accepted")
        store.record_feedback("math", "s1", "sug2", "continue", "rejected")
        store.record_feedback("code", "s2", "sug3", "verify", "accepted")

        stats = store.get_statistics()
        assert stats["domain_count"] == 2  # math and code
        assert stats["total_feedback"] == 3
        assert stats["total_accepted"] == 2
        assert stats["total_rejected"] == 1
        assert abs(stats["acceptance_rate"] - 2 / 3) < 0.01
        store.close()

    def test_reset_domain(self) -> None:
        """Test resetting a domain."""
        store = WeightStore(":memory:")

        # Modify weights
        store.record_feedback("math", "s1", "sug1", "verify", "accepted")
        modified = store.load_weights("math")
        assert modified.verify > 0.8

        # Reset
        reset = store.reset_domain("math")
        assert reset.verify == 0.8
        assert reset.total_feedback == 0
        store.close()

    def test_reset_all(self) -> None:
        """Test resetting all data."""
        store = WeightStore(":memory:")

        # Create some data
        store.record_feedback("math", "s1", "sug1", "verify", "accepted")
        store.record_feedback("code", "s2", "sug2", "verify", "accepted")

        # Reset all
        store.reset_all()

        stats = store.get_statistics()
        assert stats["domain_count"] == 0
        assert stats["total_feedback"] == 0
        store.close()

    def test_weight_clamping(self) -> None:
        """Test weights are clamped between 0.1 and 2.0."""
        store = WeightStore(":memory:")

        # Accept verify many times to push weight high
        for i in range(30):
            store.record_feedback("test", f"s{i}", f"sug{i}", "verify", "accepted")

        weights = store.load_weights("test")
        assert weights.verify <= 2.0

        # Reject finish many times to push weight low
        for i in range(30):
            store.record_feedback("test2", f"s{i}", f"sug{i}", "finish", "rejected")

        weights2 = store.load_weights("test2")
        assert weights2.finish >= 0.1
        store.close()


class TestWeightStoreIntegration:
    """Integration tests with UnifiedReasonerManager."""

    @pytest.fixture
    def weight_store(self) -> WeightStore:
        """Create in-memory weight store for testing."""
        return WeightStore(":memory:")

    @pytest.mark.asyncio
    async def test_manager_loads_weights_on_start(self, weight_store: WeightStore) -> None:
        """Test manager loads persistent weights when session starts."""
        from src.tools.unified_reasoner import UnifiedReasonerManager

        # Pre-set weights for math domain
        weights = DomainWeights(domain="math", verify=1.5, finish=0.2)
        weight_store.save_weights(weights)

        # Create manager with weight store
        manager = UnifiedReasonerManager(
            weight_store=weight_store,
            enable_weight_persistence=True,
            enable_graph=False,
            enable_domain_validation=False,
        )

        # Start session with math problem
        result = await manager.start_session(
            problem="What is 2 + 2?",
            expected_steps=3,
        )
        session_id = result["session_id"]

        # Verify weights were loaded
        async with manager.session(session_id) as session:
            assert session.suggestion_weights.verify == 1.5
            assert session.suggestion_weights.finish == 0.2

        weight_store.close()

    @pytest.mark.asyncio
    async def test_manager_persists_feedback(self, weight_store: WeightStore) -> None:
        """Test manager persists feedback to weight store."""
        from src.tools.unified_reasoner import UnifiedReasonerManager

        manager = UnifiedReasonerManager(
            weight_store=weight_store,
            enable_weight_persistence=True,
            enable_graph=False,
            enable_domain_validation=False,
        )

        # Start session
        result = await manager.start_session(
            problem="What is 2 + 2?",
            expected_steps=3,
        )
        session_id = result["session_id"]

        # Get suggestion
        suggestion = await manager.suggest_next_action(session_id)
        suggestion_id = suggestion.get("suggestion_id")

        if suggestion_id:
            # Record feedback
            await manager.record_suggestion_outcome(
                session_id=session_id,
                suggestion_id=suggestion_id,
                outcome="accepted",
            )

            # Verify feedback was persisted
            history = weight_store.get_feedback_history()
            assert len(history) >= 1
            assert history[0]["outcome"] == "accepted"

        weight_store.close()

    @pytest.mark.asyncio
    async def test_weights_persist_across_sessions(self, weight_store: WeightStore) -> None:
        """Test weights persist and affect future sessions."""
        from src.tools.unified_reasoner import UnifiedReasonerManager

        # Session 1: Record feedback
        manager1 = UnifiedReasonerManager(
            weight_store=weight_store,
            enable_weight_persistence=True,
            enable_graph=False,
            enable_domain_validation=False,
        )

        result1 = await manager1.start_session(problem="What is 2 + 2?", expected_steps=3)
        session_id1 = result1["session_id"]

        suggestion1 = await manager1.suggest_next_action(session_id1)
        if suggestion1.get("suggestion_id"):
            # Accept verify suggestion multiple times
            for _ in range(3):
                suggestion = await manager1.suggest_next_action(session_id1)
                if suggestion.get("suggestion_id"):
                    await manager1.record_suggestion_outcome(
                        session_id=session_id1,
                        suggestion_id=suggestion["suggestion_id"],
                        outcome="accepted",
                    )

        # Get final weights from session 1
        async with manager1.session(session_id1):
            pass

        # Session 2: New manager, same store
        manager2 = UnifiedReasonerManager(
            weight_store=weight_store,
            enable_weight_persistence=True,
            enable_graph=False,
            enable_domain_validation=False,
        )

        result2 = await manager2.start_session(problem="What is 3 + 3?", expected_steps=3)
        session_id2 = result2["session_id"]

        # Verify weights were loaded from persistence
        async with manager2.session(session_id2):
            # Weights should reflect learned preferences
            loaded_weights = weight_store.load_weights("math")
            assert loaded_weights.total_feedback > 0

        weight_store.close()


class TestWeightStoreSingleton:
    """Tests for singleton weight store pattern."""

    def test_get_weight_store_singleton(self) -> None:
        """Test singleton returns same instance."""
        reset_weight_store()

        store1 = get_weight_store()
        store2 = get_weight_store()

        assert store1 is store2
        reset_weight_store()

    def test_reset_weight_store(self) -> None:
        """Test reset clears singleton."""
        reset_weight_store()

        store1 = get_weight_store()
        reset_weight_store()
        store2 = get_weight_store()

        assert store1 is not store2
        reset_weight_store()


class TestDecayConfig:
    """Tests for DecayConfig weight decay configuration."""

    def test_default_values(self) -> None:
        """Test default decay config values."""
        config = DecayConfig()

        assert config.decay_rate == 0.99
        assert config.threshold_days == 7
        assert config.enabled is True

    def test_calculate_decay_factor_within_threshold(self) -> None:
        """Test no decay within threshold period."""
        config = DecayConfig(threshold_days=7)

        # Day 0 - no decay
        assert config.calculate_decay_factor(0) == 1.0
        # Day 5 - still no decay
        assert config.calculate_decay_factor(5) == 1.0
        # Day 7 - exactly at threshold, no decay
        assert config.calculate_decay_factor(7) == 1.0

    def test_calculate_decay_factor_after_threshold(self) -> None:
        """Test decay applies after threshold."""
        config = DecayConfig(decay_rate=0.99, threshold_days=7)

        # Day 8 - 1 day past threshold
        factor_day8 = config.calculate_decay_factor(8)
        assert factor_day8 == pytest.approx(0.99, rel=0.001)

        # Day 17 - 10 days past threshold
        factor_day17 = config.calculate_decay_factor(17)
        assert factor_day17 == pytest.approx(0.99**10, rel=0.001)

        # Day 37 - 30 days past threshold
        factor_day37 = config.calculate_decay_factor(37)
        assert factor_day37 == pytest.approx(0.99**30, rel=0.001)

    def test_decay_disabled(self) -> None:
        """Test decay can be disabled."""
        config = DecayConfig(enabled=False)

        # Even 100 days later, no decay
        assert config.calculate_decay_factor(100) == 1.0


class TestDomainWeightsDecay:
    """Tests for DomainWeights decay functionality."""

    def test_days_since_update(self) -> None:
        """Test calculating days since last update."""
        weights = DomainWeights(
            domain="test",
            last_updated=datetime.now() - timedelta(days=5),
        )

        days = weights.days_since_update()
        assert days == pytest.approx(5.0, abs=0.1)

    def test_apply_decay_no_change_within_threshold(self) -> None:
        """Test decay doesn't change weights within threshold."""
        weights = DomainWeights(
            domain="test",
            verify=1.2,  # Above default of 0.8
            last_updated=datetime.now() - timedelta(days=3),
        )
        config = DecayConfig(threshold_days=7)

        decayed = weights.apply_decay(config)

        # Should be unchanged
        assert decayed.verify == 1.2

    def test_apply_decay_moves_toward_default(self) -> None:
        """Test decay moves weights toward default values."""
        # Weight above default
        weights = DomainWeights(
            domain="test",
            verify=1.2,  # Default is 0.8, offset is +0.4
            continue_default=0.2,  # Default is 0.4, offset is -0.2
            last_updated=datetime.now() - timedelta(days=37),  # 30 days past threshold
        )
        config = DecayConfig(decay_rate=0.99, threshold_days=7)

        decayed = weights.apply_decay(config)

        # Verify should move toward 0.8 (default)
        # Decay factor ≈ 0.99^30 ≈ 0.74
        assert decayed.verify < weights.verify
        assert decayed.verify > DEFAULT_WEIGHTS["verify"]

        # continue_default should move toward 0.4 (default)
        assert decayed.continue_default > weights.continue_default
        assert decayed.continue_default < DEFAULT_WEIGHTS["continue_default"]

    def test_apply_decay_preserves_metadata(self) -> None:
        """Test decay preserves non-weight metadata."""
        weights = DomainWeights(
            domain="test",
            verify=1.2,
            total_feedback=50,
            accepted_count=35,
            rejected_count=15,
            last_updated=datetime.now() - timedelta(days=30),
        )

        decayed = weights.apply_decay()

        assert decayed.domain == "test"
        assert decayed.total_feedback == 50
        assert decayed.accepted_count == 35
        assert decayed.rejected_count == 15

    def test_decay_info(self) -> None:
        """Test decay_info returns useful status."""
        weights = DomainWeights(
            domain="test",
            last_updated=datetime.now() - timedelta(days=10),
        )
        config = DecayConfig(decay_rate=0.99, threshold_days=7)

        info = weights.decay_info(config)

        assert info["days_since_update"] == pytest.approx(10.0, abs=0.1)
        assert info["decay_active"] is True
        assert info["days_until_decay_starts"] == 0
        assert info["decay_factor"] == pytest.approx(0.99**3, abs=0.01)


class TestWeightStoreDecay:
    """Tests for WeightStore decay integration."""

    def test_store_accepts_decay_config(self) -> None:
        """Test WeightStore accepts decay configuration."""
        config = DecayConfig(decay_rate=0.95, threshold_days=14)
        store = WeightStore(":memory:", decay_config=config)

        assert store.decay_config.decay_rate == 0.95
        assert store.decay_config.threshold_days == 14
        store.close()

    def test_load_weights_applies_decay_by_default(self) -> None:
        """Test load_weights applies decay by default."""
        store = WeightStore(":memory:")

        # Manually create weights with old timestamp
        weights = DomainWeights(
            domain="test",
            verify=1.2,
            last_updated=datetime.now() - timedelta(days=30),
        )
        store.save_weights(weights, update_timestamp=False)

        # Load with decay
        loaded = store.load_weights("test", apply_decay=True)

        # Should be decayed
        assert loaded.verify < 1.2
        store.close()

    def test_load_weights_can_skip_decay(self) -> None:
        """Test load_weights can skip decay."""
        store = WeightStore(":memory:")

        # Manually create weights with old timestamp
        weights = DomainWeights(
            domain="test",
            verify=1.2,
            last_updated=datetime.now() - timedelta(days=30),
        )
        store.save_weights(weights, update_timestamp=False)

        # Load without decay
        loaded = store.load_weights("test", apply_decay=False)

        # Should be unchanged
        assert loaded.verify == 1.2
        store.close()

    def test_get_statistics_includes_decay_config(self) -> None:
        """Test statistics include decay configuration."""
        config = DecayConfig(decay_rate=0.98, threshold_days=10)
        store = WeightStore(":memory:", decay_config=config)

        stats = store.get_statistics()

        assert stats["decay_enabled"] is True
        assert stats["decay_rate"] == 0.98
        assert stats["decay_threshold_days"] == 10
        store.close()

    def test_get_decay_status(self) -> None:
        """Test get_decay_status returns per-domain info."""
        store = WeightStore(":memory:")

        # Create domains with different ages
        weights1 = DomainWeights(
            domain="math",
            last_updated=datetime.now() - timedelta(days=3),
        )
        weights2 = DomainWeights(
            domain="code",
            last_updated=datetime.now() - timedelta(days=20),
        )
        store.save_weights(weights1, update_timestamp=False)
        store.save_weights(weights2, update_timestamp=False)

        status = store.get_decay_status()

        assert "config" in status
        assert "domains" in status
        assert "math" in status["domains"]
        assert "code" in status["domains"]

        # Math should not have active decay
        assert status["domains"]["math"]["decay_active"] is False

        # Code should have active decay
        assert status["domains"]["code"]["decay_active"] is True
        store.close()

    def test_set_decay_config(self) -> None:
        """Test decay config can be updated."""
        store = WeightStore(":memory:")

        # Change config
        new_config = DecayConfig(decay_rate=0.90, threshold_days=3, enabled=True)
        store.set_decay_config(new_config)

        assert store.decay_config.decay_rate == 0.90
        assert store.decay_config.threshold_days == 3
        store.close()

    def test_decay_disabled_in_store(self) -> None:
        """Test decay can be disabled at store level."""
        config = DecayConfig(enabled=False)
        store = WeightStore(":memory:", decay_config=config)

        # Create old weights
        weights = DomainWeights(
            domain="test",
            verify=1.5,
            last_updated=datetime.now() - timedelta(days=100),
        )
        store.save_weights(weights, update_timestamp=False)

        # Load - should not decay
        loaded = store.load_weights("test")
        assert loaded.verify == 1.5
        store.close()
