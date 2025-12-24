#!/usr/bin/env python3
"""Basic usage example for Reason Guard MCP.

Demonstrates the core reasoning workflow using UnifiedReasonerManager:
1. Start a reasoning session
2. Add reasoning steps
3. Analyze session quality
4. Finish with final answer

Run: uv run python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
import os

from src.models.context_encoder import ContextEncoder
from src.models.model_manager import ModelManager
from src.tools.unified_reasoner import (
    ReasoningMode,
    ThoughtType,
    UnifiedReasonerManager,
)


def init_encoder() -> ContextEncoder | None:
    """Initialize encoder for semantic scoring."""
    model_name = os.getenv("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-xs")
    try:
        manager = ModelManager.get_instance()
        manager.initialize(model_name, blocking=True)
        return ContextEncoder()
    except Exception as e:
        print(f"Note: Encoder not available ({e}), using word overlap scoring")
        return None


async def main() -> None:
    """Run basic reasoning workflow."""
    print("=" * 60)
    print("Reason Guard Basic Usage Example")
    print("=" * 60)

    # Initialize encoder for better scoring
    print("\nLoading embedding model...")
    encoder = init_encoder()
    if encoder:
        print(f"  Loaded on {encoder.device}")

    # Create manager with encoder
    manager = UnifiedReasonerManager(
        encoder=encoder,
        enable_semantic_scoring=encoder is not None,
    )

    # 1. Start a reasoning session
    print("\n[1] Starting session...")
    session = await manager.start_session(
        problem="A store has 120 apples. They sell 25% on Monday and 1/3 of the remainder on Tuesday. How many apples are left?",
        mode=ReasoningMode.AUTO,
    )
    print(f"    Session: {session['session_id']}")
    print(f"    Mode: {session['actual_mode']} (auto-selected)")
    print(f"    Domain: {session['domain']}")
    print(f"    Complexity: {session['complexity']['complexity_level']}")

    session_id = session["session_id"]

    # 2. Add reasoning steps
    print("\n[2] Adding reasoning steps...")

    steps = [
        "First, calculate 25% of 120 apples sold on Monday: 120 × 0.25 = 30 apples sold",
        "Apples remaining after Monday: 120 - 30 = 90 apples",
        "On Tuesday, 1/3 of the remaining 90 apples are sold: 90 ÷ 3 = 30 apples sold",
        "Therefore, final count: 90 - 30 = 60 apples remaining",
    ]

    for i, thought in enumerate(steps, 1):
        result = await manager.add_thought(
            session_id=session_id,
            content=thought,
            thought_type=ThoughtType.CONTINUATION,
            confidence=0.9,
        )
        score = result.get("survival_score", 0)
        print(f"    Step {i}: score={score:.3f}")

    # 3. Analyze session quality
    print("\n[3] Analyzing session...")
    analysis = await manager.analyze_session(session_id)
    print(f"    Total thoughts: {analysis.total_thoughts}")
    print(f"    Coherence: {analysis.coherence_score:.2f}")
    print(f"    Coverage: {analysis.coverage_score:.2f}")
    print(f"    Risk: {analysis.risk_level}")
    if analysis.recommendations:
        print(f"    Recommendation: {analysis.recommendations[0]}")

    # 4. Get next action suggestion
    print("\n[4] Getting suggestion...")
    suggestion = await manager.suggest_next_action(session_id)
    print(f"    Suggested: {suggestion['suggested_action']}")
    print(f"    Urgency: {suggestion['urgency']}")

    # 5. Finish with answer
    print("\n[5] Finishing session...")
    result = await manager.finalize(
        session_id=session_id,
        answer="60 apples",
        confidence=0.95,
    )
    print(f"    Status: {result['status']}")
    print(f"    Final confidence: {result['confidence']}")
    print(f"    Total steps: {result['total_steps']}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
