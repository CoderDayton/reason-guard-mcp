"""End-to-end workflow demonstrating the suggestion feedback loop.

This example shows the complete autonomous reasoning workflow:
1. Start a reasoning session
2. Get AI suggestions for next actions
3. Execute actions (accept/reject suggestions)
4. Record feedback to train the suggestion weights
5. Observe how weights adapt to preferences

The feedback loop enables the system to learn user preferences over time,
improving suggestion quality for future sessions.

Run directly (uses server module):
    uv run python examples/feedback_loop.py

Or with MCP client after starting server:
    uv run fastmcp run src/server.py
    uv run python examples/feedback_loop.py --client
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# =============================================================================
# Direct Server Usage (No MCP Client)
# =============================================================================


async def demo_direct_workflow() -> None:
    """Demonstrate the feedback loop using direct server function calls."""
    from src.server import think

    print("\n" + "=" * 70)
    print("FEEDBACK LOOP DEMO - Direct Server Access")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Phase 1: Start a reasoning session
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 1: Starting Reasoning Session")
    print("-" * 70)

    problem = """
    A company has 3 warehouses (A, B, C) and needs to supply 4 stores.
    Each warehouse has limited capacity and each store has specific demand.
    Transportation costs vary by route. Find the optimal distribution plan
    that minimizes total cost while meeting all demands.
    """

    result = json.loads(
        await think.fn(
            action="start",
            mode="chain",
            problem=problem.strip(),
            expected_steps=8,
        )
    )

    session_id = result["session_id"]
    print(f"Session ID: {session_id}")
    print(f"Mode: {result['mode']}")
    print(f"Domain: {result.get('domain', 'general')}")

    # -------------------------------------------------------------------------
    # Phase 2: Get initial suggestion
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 2: Getting AI Suggestion")
    print("-" * 70)

    suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

    print(f"Suggested Action: {suggestion['suggested_action']}")
    print(f"Urgency: {suggestion['urgency']}")
    print(f"Reason: {suggestion['reason']}")
    print(f"Guidance: {suggestion['guidance']}")
    if suggestion.get("suggestion_id"):
        print(f"Suggestion ID: {suggestion['suggestion_id']}")

    # Show current weights
    learning = suggestion.get("learning", {})
    if learning.get("weights_applied"):
        print("\nCurrent Suggestion Weights:")
        weights = learning["weights_applied"]
        for action, weight in sorted(weights.items(), key=lambda x: -x[1])[:5]:
            print(f"  {action}: {weight:.2f}")

    # -------------------------------------------------------------------------
    # Phase 3: Accept suggestion and add first thought
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 3: Accepting Suggestion - Adding First Thought")
    print("-" * 70)

    # Record acceptance feedback
    if suggestion.get("suggestion_id"):
        feedback = json.loads(
            await think.fn(
                action="feedback",
                session_id=session_id,
                suggestion_id=suggestion["suggestion_id"],
                suggestion_outcome="accepted",
            )
        )
        print(f"Feedback recorded: {feedback['outcome']}")

    # Execute the suggested action
    thought1 = """
    This is a transportation problem - a classic linear programming scenario.
    Let me identify the key components:
    - Decision variables: x_ij = units shipped from warehouse i to store j
    - Objective: Minimize total transportation cost
    - Constraints: Supply limits at warehouses, demand requirements at stores
    """

    result = json.loads(
        await think.fn(
            action="continue",
            session_id=session_id,
            thought=thought1.strip(),
            confidence=0.85,
        )
    )
    print(f"Added thought: {result['thought_id']}")
    print(f"Step: {result['step']}")

    # -------------------------------------------------------------------------
    # Phase 4: Get next suggestion, but reject it
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 4: Getting Next Suggestion (Will Reject)")
    print("-" * 70)

    suggestion2 = json.loads(await think.fn(action="suggest", session_id=session_id))

    print(f"Suggested Action: {suggestion2['suggested_action']}")
    print(f"Reason: {suggestion2['reason']}")

    # User decides to take a different action (verify instead of continue)
    print("\nUser prefers to VERIFY the problem formulation first...")

    if suggestion2.get("suggestion_id"):
        feedback = json.loads(
            await think.fn(
                action="feedback",
                session_id=session_id,
                suggestion_id=suggestion2["suggestion_id"],
                suggestion_outcome="rejected",
                actual_action="verify",  # What user actually did
            )
        )
        print(f"Feedback recorded: {feedback['outcome']}")
        print(f"Actual action taken: {feedback['actual_action']}")

        # Show updated weights
        updated_weights = feedback.get("weights_updated", {})
        print("\nUpdated Weights (after rejection):")
        for action, weight in sorted(updated_weights.items(), key=lambda x: -x[1])[:5]:
            print(f"  {action}: {weight:.2f}")

    # Execute verify action instead
    verify_thought = """
    Let me verify my problem formulation is correct:
    - Are all warehouses accounted for? Yes (A, B, C)
    - Are all stores considered? Yes (4 stores)
    - Is the objective clear? Yes (minimize cost)
    - Are constraints complete? Need to add non-negativity constraints
    """

    result = json.loads(
        await think.fn(
            action="continue",
            session_id=session_id,
            thought=verify_thought.strip(),
            confidence=0.9,
        )
    )
    print(f"\nAdded verification thought: {result['thought_id']}")

    # -------------------------------------------------------------------------
    # Phase 5: Continue with more reasoning steps
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 5: Building Reasoning Chain")
    print("-" * 70)

    thoughts = [
        (
            "Setting up the mathematical model:\n"
            "- Variables: x_Ai, x_Bi, x_Ci for each store i (1-4)\n"
            "- Objective: min(sum of cost_ij * x_ij)\n"
            "- Supply constraints: sum_j(x_ij) <= capacity_i for each warehouse\n"
            "- Demand constraints: sum_i(x_ij) >= demand_j for each store"
        ),
        (
            "This can be solved using the transportation simplex method or\n"
            "general LP solvers. The northwest corner method can find an\n"
            "initial basic feasible solution."
        ),
        (
            "For implementation, we could use:\n"
            "- scipy.optimize.linprog for Python\n"
            "- OR-Tools for more complex scenarios\n"
            "- Manual stepping stone method for educational purposes"
        ),
    ]

    for i, thought in enumerate(thoughts, 1):
        # Get suggestion
        suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

        # Always accept 'continue' suggestions in this phase
        if suggestion.get("suggestion_id"):
            await think.fn(
                action="feedback",
                session_id=session_id,
                suggestion_id=suggestion["suggestion_id"],
                suggestion_outcome="accepted",
            )

        result = json.loads(
            await think.fn(
                action="continue",
                session_id=session_id,
                thought=thought,
                confidence=0.8 + (i * 0.05),
            )
        )
        print(f"Step {i + 2}: {thought[:50]}... (confidence: {0.8 + i * 0.05:.2f})")

    # -------------------------------------------------------------------------
    # Phase 6: Check for suggestions near end of session
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 6: Final Suggestion Check")
    print("-" * 70)

    final_suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))

    print(f"Suggested Action: {final_suggestion['suggested_action']}")
    print(f"Urgency: {final_suggestion['urgency']}")
    print(f"Reason: {final_suggestion['reason']}")

    session_summary = final_suggestion.get("session_summary", {})
    print("\nSession Summary:")
    print(f"  Thoughts: {session_summary.get('thoughts', 0)}")
    print(f"  Quality: {session_summary.get('quality', 0):.2f}")
    print(f"  Risk: {session_summary.get('risk', 'unknown')}")

    # -------------------------------------------------------------------------
    # Phase 7: Try auto action (returns suggestion without LLM)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 7: Auto Action (Suggestion Mode)")
    print("-" * 70)

    auto_result = json.loads(
        await think.fn(
            action="auto",
            session_id=session_id,
            max_auto_steps=3,
            stop_on_high_risk=True,
        )
    )

    print(f"Auto Status: {auto_result.get('status')}")
    print(f"Reason: {auto_result.get('reason')}")

    auto_suggestion = auto_result.get("suggestion", {})
    if auto_suggestion:
        print(f"Next suggested action: {auto_suggestion.get('suggested_action')}")

    # -------------------------------------------------------------------------
    # Phase 8: Finish the session
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 8: Finishing Session")
    print("-" * 70)

    conclusion = """
    CONCLUSION: The optimal distribution problem can be solved using linear
    programming. The transportation simplex method is efficient for this class
    of problems. Key steps: (1) formulate as LP, (2) find initial BFS using
    northwest corner, (3) iterate using stepping stone or MODI method,
    (4) verify optimality conditions.
    """

    result = json.loads(
        await think.fn(
            action="finish",
            session_id=session_id,
            thought=conclusion.strip(),
            confidence=0.92,
        )
    )

    print(f"Session Status: {result['status']}")
    print(f"Total Thoughts: {result.get('total_thoughts', 0)}")

    # -------------------------------------------------------------------------
    # Phase 9: Final analysis showing learned weights
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 9: Session Analysis (Learned Weights)")
    print("-" * 70)

    analysis = json.loads(await think.fn(action="analyze", session_id=session_id))

    print(f"Quality Score: {analysis.get('quality_score', 0):.2f}")
    print(f"Risk Level: {analysis.get('risk_level', 'unknown')}")

    metrics = analysis.get("metrics", {})
    print("\nMetrics:")
    print(f"  Depth: {metrics.get('depth', 0):.2f}")
    print(f"  Breadth: {metrics.get('breadth', 0):.2f}")
    print(f"  Coherence: {metrics.get('coherence', 0):.2f}")

    recommendations = analysis.get("recommendations", [])
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations[:3]:
            print(f"  - {rec}")

    print("\n" + "=" * 70)
    print("FEEDBACK LOOP DEMO COMPLETE")
    print("=" * 70)
    print("""
Key Takeaways:
1. Suggestions are recorded with unique IDs for tracking
2. Feedback (accepted/rejected) adjusts suggestion weights
3. The system learns user preferences over time
4. Auto action returns suggestions for manual execution at server level
5. Weight adjustments persist within a session

In production, weights would persist across sessions to build
a preference model unique to each user or use case.
""")


# =============================================================================
# MCP Client Usage (When Server is Running Separately)
# =============================================================================


async def demo_client_workflow() -> None:
    """Demonstrate the feedback loop using MCP client."""
    try:
        from fastmcp import Client
    except ImportError:
        print("Please install fastmcp: pip install fastmcp")
        return

    from mcp.types import TextContent

    def parse(result) -> dict[str, Any]:
        content = result.content[0]
        return json.loads(content.text if isinstance(content, TextContent) else "{}")

    print("\n" + "=" * 70)
    print("FEEDBACK LOOP DEMO - MCP Client")
    print("=" * 70)

    async with Client("src/server.py") as client:
        # Start session
        print("\n[1] Starting session...")
        result = await client.call_tool(
            "think",
            {
                "action": "start",
                "mode": "chain",
                "problem": "Analyze the trade-offs between microservices and monolithic architecture.",
                "expected_steps": 5,
            },
        )
        data = parse(result)
        session_id = data["session_id"]
        print(f"    Session: {session_id}")

        # Get suggestion
        print("\n[2] Getting suggestion...")
        result = await client.call_tool("think", {"action": "suggest", "session_id": session_id})
        suggestion = parse(result)
        print(f"    Suggested: {suggestion['suggested_action']}")
        print(f"    Reason: {suggestion['reason'][:60]}...")

        suggestion_id = suggestion.get("suggestion_id")

        # Record acceptance
        if suggestion_id:
            print("\n[3] Recording feedback (accepted)...")
            result = await client.call_tool(
                "think",
                {
                    "action": "feedback",
                    "session_id": session_id,
                    "suggestion_id": suggestion_id,
                    "suggestion_outcome": "accepted",
                },
            )
            feedback = parse(result)
            print(f"    Outcome: {feedback['outcome']}")

        # Add thought
        print("\n[4] Adding thought...")
        result = await client.call_tool(
            "think",
            {
                "action": "continue",
                "session_id": session_id,
                "thought": "Microservices offer independent scaling but add network complexity.",
                "confidence": 0.85,
            },
        )
        data = parse(result)
        print(f"    Step: {data['step']}")

        # Get another suggestion and reject it
        print("\n[5] Getting suggestion (will reject)...")
        result = await client.call_tool("think", {"action": "suggest", "session_id": session_id})
        suggestion2 = parse(result)
        print(f"    Suggested: {suggestion2['suggested_action']}")

        if suggestion2.get("suggestion_id"):
            print("\n[6] Recording feedback (rejected, took 'branch' instead)...")
            result = await client.call_tool(
                "think",
                {
                    "action": "feedback",
                    "session_id": session_id,
                    "suggestion_id": suggestion2["suggestion_id"],
                    "suggestion_outcome": "rejected",
                    "actual_action": "branch",
                },
            )
            feedback = parse(result)
            print(f"    Outcome: {feedback['outcome']}")
            print(f"    Actual action: {feedback['actual_action']}")

        # Try auto action
        print("\n[7] Trying auto action...")
        result = await client.call_tool(
            "think",
            {
                "action": "auto",
                "session_id": session_id,
                "max_auto_steps": 2,
            },
        )
        auto = parse(result)
        print(f"    Status: {auto.get('status')}")
        if auto.get("suggestion"):
            print(f"    Next suggestion: {auto['suggestion'].get('suggested_action')}")

        # Finish
        print("\n[8] Finishing session...")
        result = await client.call_tool(
            "think",
            {
                "action": "finish",
                "session_id": session_id,
                "thought": "Both architectures have merits; choice depends on team size and scaling needs.",
                "confidence": 0.88,
            },
        )
        data = parse(result)
        print(f"    Status: {data['status']}")

    print("\n" + "=" * 70)
    print("MCP CLIENT DEMO COMPLETE")
    print("=" * 70)


# =============================================================================
# Simulated Multi-Session Learning Demo
# =============================================================================


@dataclass
class WeightSnapshot:
    """Snapshot of weights at a point in time."""

    session: int
    continue_weight: float
    verify_weight: float
    resolve_weight: float
    finish_weight: float


async def demo_weight_evolution() -> None:
    """Demonstrate how weights evolve across multiple sessions.

    This simulates what would happen with persistent weight storage.
    """
    from src.tools.unified_reasoner import SuggestionWeights

    print("\n" + "=" * 70)
    print("WEIGHT EVOLUTION DEMO")
    print("Shows how suggestion weights adapt to user preferences")
    print("=" * 70)

    # Simulate starting weights
    weights = SuggestionWeights()
    snapshots: list[WeightSnapshot] = []

    # Record initial state
    snapshots.append(
        WeightSnapshot(
            session=0,
            continue_weight=weights.continue_default,
            verify_weight=weights.verify,
            resolve_weight=weights.resolve,
            finish_weight=weights.finish,
        )
    )

    print("\nSimulating 5 sessions with different user preferences...")
    print("User preference: Prefers verification, dislikes early finishing\n")

    # Simulate sessions with a user who:
    # - Often accepts 'verify' suggestions
    # - Often rejects 'continue' in favor of 'verify'
    # - Rejects 'finish' suggestions when confidence < 0.9

    session_behaviors = [
        # Session 1: User verifies a lot
        [("continue", False, "verify"), ("verify", True, None), ("verify", True, None)],
        # Session 2: User still prefers verification
        [("continue", False, "verify"), ("verify", True, None), ("finish", False, "continue")],
        # Session 3: Mix of actions
        [("continue", True, None), ("verify", True, None), ("synthesize", True, None)],
        # Session 4: Rejects finish suggestions
        [("verify", True, None), ("finish", False, "verify"), ("continue", True, None)],
        # Session 5: Balanced
        [("continue", True, None), ("verify", True, None), ("finish", True, None)],
    ]

    for session_num, behaviors in enumerate(session_behaviors, 1):
        print(f"Session {session_num}:")
        for suggested, accepted, actual in behaviors:
            weights.adjust(suggested, accepted)
            if not accepted and actual:
                weights.adjust(actual, True, learning_rate=0.05)
            status = "accepted" if accepted else f"rejected -> {actual}"
            print(f"  {suggested}: {status}")

        snapshots.append(
            WeightSnapshot(
                session=session_num,
                continue_weight=weights.continue_default,
                verify_weight=weights.verify,
                resolve_weight=weights.resolve,
                finish_weight=weights.finish,
            )
        )

    # Display weight evolution
    print("\n" + "-" * 70)
    print("WEIGHT EVOLUTION TABLE")
    print("-" * 70)
    print(f"{'Session':<10} {'continue':<12} {'verify':<12} {'resolve':<12} {'finish':<12}")
    print("-" * 70)

    for snap in snapshots:
        print(
            f"{snap.session:<10} {snap.continue_weight:<12.3f} {snap.verify_weight:<12.3f} "
            f"{snap.resolve_weight:<12.3f} {snap.finish_weight:<12.3f}"
        )

    # Show final learned preferences
    print("\n" + "-" * 70)
    print("LEARNED PREFERENCES")
    print("-" * 70)

    initial = snapshots[0]
    final = snapshots[-1]

    changes = [
        ("continue", final.continue_weight - initial.continue_weight),
        ("verify", final.verify_weight - initial.verify_weight),
        ("resolve", final.resolve_weight - initial.resolve_weight),
        ("finish", final.finish_weight - initial.finish_weight),
    ]

    for action, change in sorted(changes, key=lambda x: -x[1]):
        direction = "+" if change > 0 else ""
        print(f"  {action}: {direction}{change:.3f}")

    print("\n" + "=" * 70)
    print("WEIGHT EVOLUTION DEMO COMPLETE")
    print("=" * 70)
    print("""
Observations:
- 'verify' weight increased (user often accepted verify suggestions)
- 'continue' weight decreased (user often rejected in favor of verify)
- 'finish' weight decreased (user rejected premature finish suggestions)

This demonstrates how the system learns individual preferences
to provide better suggestions over time.
""")


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> None:
    """Run the feedback loop demonstrations."""
    parser = argparse.ArgumentParser(
        description="Demonstrate the suggestion feedback loop workflow"
    )
    parser.add_argument(
        "--client",
        action="store_true",
        help="Use MCP client instead of direct server access",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Run weight evolution demo only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demos",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MatrixMind MCP - Feedback Loop Workflow Examples")
    print("Demonstrating: Suggestions, Feedback, and Weight Learning")
    print("=" * 70)

    if args.weights:
        await demo_weight_evolution()
    elif args.client:
        await demo_client_workflow()
    elif args.all:
        await demo_direct_workflow()
        await demo_client_workflow()
        await demo_weight_evolution()
    else:
        await demo_direct_workflow()
        await demo_weight_evolution()

    print("\nAll demos completed successfully!")


# =============================================================================
# Persistent Weight Storage Demo
# =============================================================================


async def demo_persistent_weights() -> None:
    """Demonstrate persistent weight storage using SQLite.

    This shows how learned preferences persist across sessions and restarts.
    """
    from pathlib import Path

    from src.utils.weight_store import WeightStore, reset_weight_store

    print("\n" + "=" * 70)
    print("PERSISTENT WEIGHT STORAGE DEMO")
    print("Shows how weights persist to SQLite and survive restarts")
    print("=" * 70)

    # Use a temporary database for the demo
    demo_db = Path("/tmp/matrixmind_demo_weights.db")
    if demo_db.exists():
        demo_db.unlink()

    # Create a fresh store
    reset_weight_store()
    store = WeightStore(db_path=demo_db)

    print(f"\nDatabase location: {demo_db}")

    # -------------------------------------------------------------------------
    # Session 1: Initial learning
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SESSION 1: Learning user preferences")
    print("-" * 70)

    domain = "mathematics"

    # Load initial weights (creates defaults)
    weights1 = store.load_weights(domain)
    print(f"Initial weights for '{domain}':")
    print(f"  verify: {weights1.verify:.3f}")
    print(f"  continue: {weights1.continue_default:.3f}")
    print(f"  finish: {weights1.finish:.3f}")

    # Simulate feedback: user prefers verification
    print("\nSimulating feedback (user prefers 'verify')...")
    store.record_feedback(domain, "session-1", "sug-001", "continue", "rejected", "verify")
    store.record_feedback(domain, "session-1", "sug-002", "verify", "accepted")
    store.record_feedback(domain, "session-1", "sug-003", "verify", "accepted")
    store.record_feedback(domain, "session-1", "sug-004", "finish", "rejected", "continue")

    # Check updated weights
    weights1_updated = store.load_weights(domain)
    print("\nUpdated weights after feedback:")
    print(f"  verify: {weights1_updated.verify:.3f} (was {weights1.verify:.3f})")
    print(
        f"  continue: {weights1_updated.continue_default:.3f} (was {weights1.continue_default:.3f})"
    )
    print(f"  finish: {weights1_updated.finish:.3f} (was {weights1.finish:.3f})")
    print(f"  total_feedback: {weights1_updated.total_feedback}")
    print(f"  acceptance_rate: {weights1_updated.acceptance_rate():.1%}")

    # -------------------------------------------------------------------------
    # Session 2: Simulate restart - weights persist!
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SESSION 2: Simulating application restart")
    print("-" * 70)

    # Create a new store instance (simulates restart)
    print("Creating new WeightStore instance (simulates restart)...")
    store2 = WeightStore(db_path=demo_db)

    # Load weights - should have learned preferences
    weights2 = store2.load_weights(domain)
    print("\nLoaded weights after 'restart':")
    print(f"  verify: {weights2.verify:.3f}")
    print(f"  continue: {weights2.continue_default:.3f}")
    print(f"  finish: {weights2.finish:.3f}")
    print(f"  total_feedback: {weights2.total_feedback}")

    # Verify weights persisted correctly
    assert abs(weights2.verify - weights1_updated.verify) < 0.001
    print("\n✓ Weights persisted correctly across restart!")

    # More feedback in session 2
    print("\nAdding more feedback in session 2...")
    store2.record_feedback(domain, "session-2", "sug-005", "verify", "accepted")
    store2.record_feedback(domain, "session-2", "sug-006", "synthesize", "accepted")

    # -------------------------------------------------------------------------
    # Session 3: Multi-domain learning
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SESSION 3: Multi-domain learning")
    print("-" * 70)

    # Add feedback for different domains
    domains_feedback = [
        ("software", [("continue", "accepted"), ("verify", "rejected")]),
        ("physics", [("verify", "accepted"), ("verify", "accepted"), ("resolve", "accepted")]),
        ("general", [("continue", "accepted"), ("finish", "accepted")]),
    ]

    for d, feedbacks in domains_feedback:
        for i, (action, outcome) in enumerate(feedbacks):
            store2.record_feedback(d, f"session-3-{d}", f"sug-{i}", action, outcome)

    print("Added feedback for multiple domains: software, physics, general")

    # -------------------------------------------------------------------------
    # Statistics and History
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("DATABASE STATISTICS")
    print("-" * 70)

    stats = store2.get_statistics()
    print(f"Total domains: {stats['domain_count']}")
    print(f"Total feedback records: {stats['total_feedback']}")
    print(f"Overall acceptance rate: {stats['acceptance_rate']:.1%}")

    print("\nPer-domain breakdown:")
    all_domains = store2.get_all_domains()
    for dw in sorted(all_domains, key=lambda x: -x.total_feedback):
        print(f"  {dw.domain}:")
        print(
            f"    feedback: {dw.total_feedback} ({dw.accepted_count} accepted, {dw.rejected_count} rejected)"
        )
        print(
            f"    verify: {dw.verify:.3f}, continue: {dw.continue_default:.3f}, finish: {dw.finish:.3f}"
        )

    # -------------------------------------------------------------------------
    # Feedback History
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("FEEDBACK HISTORY (last 5)")
    print("-" * 70)

    history = store2.get_feedback_history(limit=5)
    for record in history:
        actual = f" -> {record['actual_action']}" if record.get("actual_action") else ""
        print(f"  [{record['domain']}] {record['suggested_action']}: {record['outcome']}{actual}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CLEANUP")
    print("-" * 70)

    # Show reset functionality
    print("Demonstrating reset_domain('mathematics')...")
    store2.reset_domain("mathematics")
    math_weights = store2.load_weights("mathematics")
    print(f"  verify after reset: {math_weights.verify:.3f} (back to default)")

    # Clean up demo database
    store2.close()
    demo_db.unlink()
    print(f"\nDemo database cleaned up: {demo_db}")

    print("\n" + "=" * 70)
    print("PERSISTENT WEIGHT STORAGE DEMO COMPLETE")
    print("=" * 70)
    print("""
Key Features Demonstrated:
1. Weights persist to SQLite database (~/.matrixmind/weights.db by default)
2. Learned preferences survive application restarts
3. Per-domain weight learning (math, software, physics, etc.)
4. Feedback history tracking for analytics
5. Statistics API for monitoring learning progress
6. Reset functionality for individual domains or all data

Production Usage:
- Weights are automatically loaded when a reasoning session starts
- Feedback is automatically persisted when recorded via think(action="feedback")
- No manual intervention needed - learning happens transparently
""")


async def demo_weight_decay() -> None:
    """Demonstrate time-based weight decay.

    Shows how weights gradually return to defaults over time,
    allowing the system to adapt when user preferences change.
    """
    from datetime import datetime, timedelta
    from pathlib import Path

    from src.utils.weight_store import DecayConfig, DomainWeights, WeightStore

    print("\n" + "=" * 70)
    print("WEIGHT DECAY DEMO")
    print("Shows how weights decay toward defaults over time")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Decay Configuration
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("DECAY CONFIGURATION")
    print("-" * 70)

    config = DecayConfig(
        decay_rate=0.99,  # 1% decay per day toward default
        threshold_days=7,  # Decay starts after 7 days of inactivity
        enabled=True,
    )

    print(f"Decay rate: {config.decay_rate} (per day)")
    print(f"Threshold: {config.threshold_days} days before decay starts")
    print(f"Enabled: {config.enabled}")

    # Calculate decay factors for different time periods
    print("\nDecay factors by time since last feedback:")
    for days in [0, 7, 14, 30, 60, 90]:
        factor = config.calculate_decay_factor(days)
        pct_remaining = factor * 100
        print(f"  {days:3d} days: {factor:.4f} ({pct_remaining:.1f}% of learned offset remaining)")

    # -------------------------------------------------------------------------
    # Simulating Decay Over Time
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SIMULATING WEIGHT DECAY")
    print("-" * 70)

    # Create weights with learned preferences
    learned_weights = DomainWeights(
        domain="test",
        verify=1.2,  # Default is 0.8, learned +0.4
        continue_default=0.2,  # Default is 0.4, learned -0.2
        finish=0.8,  # Default is 0.5, learned +0.3
        total_feedback=50,
        accepted_count=35,
        rejected_count=15,
        last_updated=datetime.now(),
    )

    print("Original learned weights (just updated):")
    print(f"  verify: {learned_weights.verify:.3f} (default: 0.8, offset: +0.4)")
    print(f"  continue: {learned_weights.continue_default:.3f} (default: 0.4, offset: -0.2)")
    print(f"  finish: {learned_weights.finish:.3f} (default: 0.5, offset: +0.3)")

    # Simulate different time periods
    time_periods = [
        ("Day 0 (now)", 0),
        ("Day 7 (threshold)", 7),
        ("Day 14 (1 week past)", 14),
        ("Day 37 (1 month past)", 37),
        ("Day 67 (2 months past)", 67),
        ("Day 97 (3 months past)", 97),
    ]

    print("\nWeight values over time:")
    print(f"{'Time':<25} {'verify':<12} {'continue':<12} {'finish':<12} {'decay %':<10}")
    print("-" * 70)

    for label, days in time_periods:
        # Create weights with simulated age
        aged_weights = DomainWeights(
            domain="test",
            verify=learned_weights.verify,
            continue_default=learned_weights.continue_default,
            finish=learned_weights.finish,
            total_feedback=learned_weights.total_feedback,
            accepted_count=learned_weights.accepted_count,
            rejected_count=learned_weights.rejected_count,
            last_updated=datetime.now() - timedelta(days=days),
        )

        # Apply decay
        decayed = aged_weights.apply_decay(config)
        factor = config.calculate_decay_factor(days)
        decay_pct = (1 - factor) * 100

        print(
            f"{label:<25} {decayed.verify:<12.3f} {decayed.continue_default:<12.3f} "
            f"{decayed.finish:<12.3f} {decay_pct:>6.1f}%"
        )

    # -------------------------------------------------------------------------
    # Decay Status API
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("DECAY STATUS API")
    print("-" * 70)

    # Create store with decay config
    demo_db = Path("/tmp/matrixmind_decay_demo.db")
    if demo_db.exists():
        demo_db.unlink()

    store = WeightStore(db_path=demo_db, decay_config=config)

    # Add domains with different ages
    domains_data = [
        ("mathematics", 2, {"verify": 1.1}),  # 2 days old - no decay
        ("software", 10, {"verify": 1.3}),  # 10 days old - some decay
        ("physics", 45, {"verify": 1.5}),  # 45 days old - significant decay
    ]

    for domain, days_ago, overrides in domains_data:
        w = DomainWeights(
            domain=domain,
            last_updated=datetime.now() - timedelta(days=days_ago),
            **overrides,
        )
        store.save_weights(w, update_timestamp=False)

    # Get decay status
    status = store.get_decay_status()

    print("Per-domain decay status:")
    for domain, info in status["domains"].items():
        decay_status = "ACTIVE" if info["decay_active"] else "inactive"
        days_until = info["days_until_decay_starts"]
        factor = info["decay_factor"]
        print(f"\n  {domain}:")
        print(f"    Days since update: {info['days_since_update']:.1f}")
        print(f"    Decay status: {decay_status}")
        if not info["decay_active"]:
            print(f"    Days until decay: {days_until:.1f}")
        else:
            print(f"    Decay factor: {factor:.4f}")

    # Show loaded weights (with decay applied)
    print("\n\nLoaded weights (with decay applied):")
    for domain, _, _ in domains_data:
        w = store.load_weights(domain, apply_decay=True)
        raw = store.load_weights(domain, apply_decay=False)
        print(f"  {domain}: verify={w.verify:.3f} (raw: {raw.verify:.3f})")

    # Cleanup
    store.close()
    demo_db.unlink()

    print("\n" + "=" * 70)
    print("WEIGHT DECAY DEMO COMPLETE")
    print("=" * 70)
    print("""
Key Concepts:
1. Decay moves learned weights back toward defaults over time
2. Threshold prevents decay during active usage periods
3. Exponential decay provides smooth, gradual transitions
4. Decay allows system to "forget" outdated preferences
5. Users can adjust decay_rate and threshold_days as needed

Use Cases:
- User preferences change over time
- Different users share the same system
- Recovering from bad feedback data
- Seasonal or project-based preference changes
""")


async def demo_integrated_persistence() -> None:
    """Demonstrate persistence integrated with the reasoning workflow."""
    import tempfile
    from pathlib import Path

    from src.server import think
    from src.tools.unified_reasoner import init_unified_manager
    from src.utils.weight_store import WeightStore, reset_weight_store

    print("\n" + "=" * 70)
    print("INTEGRATED PERSISTENCE DEMO")
    print("Full workflow with automatic weight persistence")
    print("=" * 70)

    # Setup: Use temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_weights.db"
        store = WeightStore(db_path=db_path)

        # Reset and reinitialize manager with our test store
        reset_weight_store()
        await init_unified_manager(weight_store=store, enable_weight_persistence=True)

        # ---------------------------------------------------------------------
        # Run two sessions to show persistence
        # ---------------------------------------------------------------------
        for session_num in [1, 2]:
            print("\n" + "-" * 70)
            print(f"REASONING SESSION {session_num}")
            print("-" * 70)

            # Start session
            import json

            result = json.loads(
                await think.fn(
                    action="start",
                    mode="chain",
                    problem=f"Session {session_num}: Analyze algorithm complexity.",
                    expected_steps=3,
                )
            )
            session_id = result["session_id"]
            print(f"Started session: {session_id}")

            # Get suggestion
            suggestion = json.loads(await think.fn(action="suggest", session_id=session_id))
            print(f"Suggestion: {suggestion['suggested_action']}")

            # Show current weights from suggestion
            learning = suggestion.get("learning", {})
            if learning.get("weights_applied"):
                w = learning["weights_applied"]
                print(
                    f"Loaded weights: verify={w.get('verify', 0):.2f}, continue={w.get('continue_default', 0):.2f}"
                )

            # Record feedback
            if suggestion.get("suggestion_id"):
                # Alternate: accept in session 1, reject in session 2
                outcome = "accepted" if session_num == 1 else "rejected"
                actual = None if outcome == "accepted" else "verify"

                await think.fn(
                    action="feedback",
                    session_id=session_id,
                    suggestion_id=suggestion["suggestion_id"],
                    suggestion_outcome=outcome,
                    actual_action=actual,
                )
                print(f"Recorded feedback: {outcome}" + (f" (actual: {actual})" if actual else ""))

            # Continue and finish
            await think.fn(
                action="continue",
                session_id=session_id,
                thought="Analyzing time complexity of nested loops.",
                confidence=0.85,
            )

            result = json.loads(
                await think.fn(
                    action="finish",
                    session_id=session_id,
                    thought="Complexity is O(n²) due to nested iteration.",
                    confidence=0.9,
                )
            )
            print(f"Session finished: {result['status']}")

        # ---------------------------------------------------------------------
        # Show final statistics
        # ---------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("FINAL DATABASE STATE")
        print("-" * 70)

        stats = store.get_statistics()
        print(f"Total feedback records: {stats['total_feedback']}")
        print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")

        history = store.get_feedback_history()
        print(f"\nFeedback history ({len(history)} records):")
        for rec in history:
            actual = f" -> {rec['actual_action']}" if rec.get("actual_action") else ""
            print(f"  {rec['suggested_action']}: {rec['outcome']}{actual}")

        store.close()

    print("\n" + "=" * 70)
    print("INTEGRATED PERSISTENCE DEMO COMPLETE")
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


async def main_with_persistence() -> None:
    """Run all demonstrations including persistence."""
    parser = argparse.ArgumentParser(
        description="Demonstrate the suggestion feedback loop workflow"
    )
    parser.add_argument(
        "--client",
        action="store_true",
        help="Use MCP client instead of direct server access",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Run weight evolution demo only",
    )
    parser.add_argument(
        "--persistence",
        action="store_true",
        help="Run persistent storage demo only",
    )
    parser.add_argument(
        "--decay",
        action="store_true",
        help="Run weight decay demo only",
    )
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Run integrated persistence demo only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demos",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MatrixMind MCP - Feedback Loop Workflow Examples")
    print("Demonstrating: Suggestions, Feedback, and Weight Learning")
    print("=" * 70)

    if args.weights:
        await demo_weight_evolution()
    elif args.persistence:
        await demo_persistent_weights()
    elif args.decay:
        await demo_weight_decay()
    elif args.integrated:
        await demo_integrated_persistence()
    elif args.client:
        await demo_client_workflow()
    elif args.all:
        await demo_direct_workflow()
        await demo_client_workflow()
        await demo_weight_evolution()
        await demo_persistent_weights()
        await demo_weight_decay()
        await demo_integrated_persistence()
    else:
        await demo_direct_workflow()
        await demo_weight_evolution()

    print("\nAll demos completed successfully!")


if __name__ == "__main__":
    asyncio.run(main_with_persistence())
