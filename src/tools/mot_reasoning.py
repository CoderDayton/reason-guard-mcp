"""Matrix of Thought (MoT) state manager.

Implements a state management tool for tracking matrix-based reasoning.
The calling LLM does all reasoning; this tool tracks the matrix structure,
manages cell filling, and provides synthesis guidance.

Based on paper 2509.03918v2 - Matrix of Thought framework.

Architecture:
    - Tool receives reasoning FROM the calling LLM
    - Tracks matrix cells (rows=strategies, cols=iterations)
    - Manages column synthesis
    - Provides guidance for next cell to fill
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger


class MatrixStatus(Enum):
    """Status of a matrix reasoning session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


# Default reasoning strategies for matrix rows
DEFAULT_STRATEGIES = [
    "direct_factual",  # Row 1: Direct factual analysis
    "logical_inference",  # Row 2: Logical deduction and inference
    "analogical",  # Row 3: Analogical reasoning and comparison
]

STRATEGY_DESCRIPTIONS = {
    "direct_factual": "Analyze facts directly from the context",
    "logical_inference": "Apply logical deduction and inference rules",
    "analogical": "Use analogies and comparisons to similar situations",
    "causal": "Identify cause-effect relationships",
    "counterfactual": "Consider alternative scenarios and outcomes",
}


@dataclass
class MatrixCell:
    """A single cell in the reasoning matrix."""

    row: int
    col: int
    strategy: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "row": self.row,
            "col": self.col,
            "strategy": self.strategy,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
        }


@dataclass
class ColumnSynthesis:
    """Synthesis of a column's reasoning."""

    col: int
    content: str
    contributing_cells: list[tuple[int, int]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "col": self.col,
            "content": self.content,
            "contributing_cells": self.contributing_cells,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MatrixState:
    """State of a matrix reasoning session."""

    session_id: str
    question: str
    context: str
    rows: int
    cols: int
    strategies: list[str]
    cells: dict[tuple[int, int], MatrixCell] = field(default_factory=dict)
    syntheses: dict[int, ColumnSynthesis] = field(default_factory=dict)
    status: MatrixStatus = MatrixStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        # Build matrix representation
        matrix = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cell = self.cells.get((r, c))
                row.append(cell.to_dict() if cell else None)
            matrix.append(row)

        return {
            "session_id": self.session_id,
            "question": self.question,
            "context": self.context[:200] + "..." if len(self.context) > 200 else self.context,
            "rows": self.rows,
            "cols": self.cols,
            "strategies": self.strategies,
            "matrix": matrix,
            "syntheses": {k: v.to_dict() for k, v in self.syntheses.items()},
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "final_answer": self.final_answer,
            "cells_filled": len(self.cells),
            "total_cells": self.rows * self.cols,
            "progress": len(self.cells) / (self.rows * self.cols),
        }


class MatrixOfThoughtManager:
    """Manages Matrix of Thought reasoning sessions.

    This is a STATE MANAGER, not a reasoner. The calling LLM provides
    reasoning content; this tool tracks and organizes the matrix.

    Example usage flow:
        1. Agent calls: start_matrix(question="...", context="...")
        2. Agent fills cells: set_cell(row=0, col=0, thought="...")
        3. After each column: synthesize_column(col=0, synthesis="...")
        4. Agent finalizes: finalize(answer="...")

    Matrix structure:
        - Rows represent different reasoning strategies
        - Columns represent iterative refinement steps
        - Each cell builds on previous column's cell + synthesis

    """

    def __init__(self) -> None:
        """Initialize the matrix manager."""
        self._sessions: dict[str, MatrixState] = {}

    def start_matrix(
        self,
        question: str,
        context: str = "",
        rows: int = 3,
        cols: int = 4,
        strategies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new matrix reasoning session.

        Args:
            question: The question to reason about.
            context: Background context for reasoning.
            rows: Number of strategies (2-5).
            cols: Number of refinement iterations (2-6).
            strategies: Custom strategy names (default: standard 3 strategies).
            metadata: Optional metadata.

        Returns:
            Session info with matrix structure.

        """
        # Validate dimensions
        rows = max(2, min(5, rows))
        cols = max(2, min(6, cols))

        # Set up strategies
        if strategies:
            strategies = strategies[:rows]
            while len(strategies) < rows:
                strategies.append(f"strategy_{len(strategies) + 1}")
        else:
            strategies = DEFAULT_STRATEGIES[:rows]
            while len(strategies) < rows:
                strategies.append(f"strategy_{len(strategies) + 1}")

        session_id = str(uuid.uuid4())[:8]
        state = MatrixState(
            session_id=session_id,
            question=question,
            context=context,
            rows=rows,
            cols=cols,
            strategies=strategies,
            metadata=metadata or {},
        )
        self._sessions[session_id] = state

        logger.debug(f"Started matrix session {session_id}: {rows}x{cols}")

        # Build strategy guidance
        strategy_guide = []
        for i, strat in enumerate(strategies):
            desc = STRATEGY_DESCRIPTIONS.get(strat, f"Apply {strat} reasoning")
            strategy_guide.append(f"Row {i}: {strat} - {desc}")

        return {
            "session_id": session_id,
            "status": "started",
            "question": question,
            "context_length": len(context),
            "matrix_dimensions": {"rows": rows, "cols": cols},
            "strategies": strategies,
            "strategy_guide": strategy_guide,
            "total_cells": rows * cols,
            "instruction": (
                f"Fill the {rows}x{cols} reasoning matrix. "
                f"Start with cell (0,0) using '{strategies[0]}' strategy. "
                f"Call set_cell(row=0, col=0, thought='your reasoning'). "
                f"After completing each column, call synthesize_column()."
            ),
            "next_cell": {"row": 0, "col": 0, "strategy": strategies[0]},
        }

    def set_cell(
        self,
        session_id: str,
        row: int,
        col: int,
        thought: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Fill a cell in the matrix.

        Args:
            session_id: Session to update.
            row: Row index (0-based).
            col: Column index (0-based).
            thought: Reasoning content for this cell.
            confidence: Optional confidence score.

        Returns:
            Updated state and guidance for next cell.

        """
        state = self._get_session(session_id)
        if state.status != MatrixStatus.ACTIVE:
            return {"error": f"Session is {state.status.value}"}

        # Validate position
        if not (0 <= row < state.rows and 0 <= col < state.cols):
            return {
                "error": f"Invalid position ({row}, {col}) for {state.rows}x{state.cols} matrix"
            }

        # Create cell
        cell = MatrixCell(
            row=row,
            col=col,
            strategy=state.strategies[row],
            content=thought,
            confidence=confidence,
        )
        state.cells[(row, col)] = cell
        state.updated_at = datetime.now()

        # Check consistency
        issues = self._check_cell_consistency(state, cell)

        # Determine next action
        next_cell = self._get_next_cell(state)
        col_complete = all((r, col) in state.cells for r in range(state.rows))

        response = {
            "session_id": session_id,
            "cell_set": {"row": row, "col": col, "strategy": state.strategies[row]},
            "cells_filled": len(state.cells),
            "total_cells": state.rows * state.cols,
            "progress": round(len(state.cells) / (state.rows * state.cols), 2),
            "column_complete": col_complete,
        }

        if issues:
            response["consistency_warnings"] = issues

        if col_complete and col not in state.syntheses:
            response["instruction"] = (
                f"Column {col} is complete. Call synthesize_column(col={col}, synthesis='...') "
                f"to combine the {state.rows} perspectives into a unified insight."
            )
            response["pending_synthesis"] = col
        elif next_cell:
            response["next_cell"] = next_cell
            response["instruction"] = (
                f"Continue with cell ({next_cell['row']}, {next_cell['col']}) "
                f"using '{next_cell['strategy']}' strategy."
            )
        else:
            response["instruction"] = (
                "Matrix complete. Call finalize(answer='...') with your final answer."
            )

        logger.debug(f"Set cell ({row}, {col}) in session {session_id}")

        return response

    def synthesize_column(
        self,
        session_id: str,
        col: int,
        synthesis: str,
    ) -> dict[str, Any]:
        """Synthesize a column's reasoning into a unified insight.

        Args:
            session_id: Session to update.
            col: Column to synthesize.
            synthesis: Combined insight from all rows in this column.

        Returns:
            Updated state.

        """
        state = self._get_session(session_id)
        if state.status != MatrixStatus.ACTIVE:
            return {"error": f"Session is {state.status.value}"}

        # Check column is complete
        missing = [r for r in range(state.rows) if (r, col) not in state.cells]
        if missing:
            return {"error": f"Column {col} incomplete. Missing rows: {missing}"}

        # Create synthesis
        contributing = [(r, col) for r in range(state.rows)]
        synth = ColumnSynthesis(
            col=col,
            content=synthesis,
            contributing_cells=contributing,
        )
        state.syntheses[col] = synth
        state.updated_at = datetime.now()

        # Determine next action
        next_cell = self._get_next_cell(state)
        all_cols_synthesized = len(state.syntheses) == state.cols

        response = {
            "session_id": session_id,
            "column_synthesized": col,
            "syntheses_complete": len(state.syntheses),
            "total_columns": state.cols,
        }

        if all_cols_synthesized:
            response["instruction"] = (
                "All columns synthesized. Review the progression and "
                "call finalize(answer='...') with your final answer."
            )
            response["synthesis_summary"] = [
                {
                    "col": c,
                    "synthesis": s.content[:100] + "..." if len(s.content) > 100 else s.content,
                }
                for c, s in sorted(state.syntheses.items())
            ]
        elif next_cell:
            response["next_cell"] = next_cell
            response["instruction"] = (
                f"Continue filling matrix. Next: cell ({next_cell['row']}, {next_cell['col']}) "
                f"with '{next_cell['strategy']}' strategy. "
                f"Use the column {col} synthesis to inform your reasoning."
            )

        logger.debug(f"Synthesized column {col} in session {session_id}")

        return response

    def get_matrix(self, session_id: str) -> dict[str, Any]:
        """Get the current matrix state.

        Args:
            session_id: Session to retrieve.

        Returns:
            Full matrix state.

        """
        state = self._get_session(session_id)
        return state.to_dict()

    def finalize(
        self,
        session_id: str,
        answer: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Finalize the matrix reasoning with an answer.

        Args:
            session_id: Session to finalize.
            answer: The final answer.
            confidence: Optional confidence score.

        Returns:
            Final matrix state with summary.

        """
        state = self._get_session(session_id)

        state.final_answer = answer
        state.status = MatrixStatus.COMPLETED
        state.updated_at = datetime.now()

        # Generate summary
        summary = self._generate_summary(state)

        logger.info(f"Finalized matrix session {session_id}")

        return {
            "session_id": session_id,
            "status": "completed",
            "final_answer": answer,
            "confidence": confidence,
            "summary": summary,
            "matrix": state.to_dict(),
        }

    def abandon(self, session_id: str, reason: str = "") -> dict[str, Any]:
        """Abandon a matrix session.

        Args:
            session_id: Session to abandon.
            reason: Optional reason.

        Returns:
            Final state.

        """
        state = self._get_session(session_id)
        state.status = MatrixStatus.ABANDONED
        state.metadata["abandon_reason"] = reason
        state.updated_at = datetime.now()

        return {
            "session_id": session_id,
            "status": "abandoned",
            "reason": reason,
            "cells_filled": len(state.cells),
        }

    def list_sessions(self) -> dict[str, Any]:
        """List all sessions."""
        sessions = []
        for sid, state in self._sessions.items():
            sessions.append(
                {
                    "session_id": sid,
                    "question": state.question[:50] + "..."
                    if len(state.question) > 50
                    else state.question,
                    "status": state.status.value,
                    "dimensions": f"{state.rows}x{state.cols}",
                    "progress": round(len(state.cells) / (state.rows * state.cols), 2),
                }
            )
        return {"sessions": sessions, "total": len(sessions)}

    def _get_session(self, session_id: str) -> MatrixState:
        """Get session or raise error."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")
        return self._sessions[session_id]

    def _get_next_cell(self, state: MatrixState) -> dict[str, Any] | None:
        """Get the next cell to fill."""
        for col in range(state.cols):
            for row in range(state.rows):
                if (row, col) not in state.cells:
                    return {
                        "row": row,
                        "col": col,
                        "strategy": state.strategies[row],
                    }
        return None

    def _check_cell_consistency(
        self,
        state: MatrixState,
        cell: MatrixCell,
    ) -> list[str]:
        """Check cell for consistency issues."""
        issues = []
        content = cell.content.lower()

        # Check for short content
        if len(cell.content.split()) < 10:
            issues.append("Cell content seems brief. Consider more thorough reasoning.")

        # Check for repetition from previous column
        if cell.col > 0:
            prev_cell = state.cells.get((cell.row, cell.col - 1))
            if prev_cell:
                prev_words = set(prev_cell.content.lower().split())
                curr_words = set(content.split())
                if len(prev_words) > 5 and len(curr_words) > 5:
                    overlap = len(prev_words & curr_words) / len(prev_words | curr_words)
                    if overlap > 0.7:
                        issues.append(
                            "High similarity with previous column. "
                            "Ensure you're building on the synthesis, not repeating."
                        )

        # Check if using assigned strategy
        strategy = state.strategies[cell.row].lower().replace("_", " ")
        strategy_keywords = {
            "direct factual": ["fact", "stated", "according to", "explicitly"],
            "logical inference": ["therefore", "implies", "because", "since", "if"],
            "analogical": ["similar", "like", "compared", "analogy", "reminds"],
            "causal": ["cause", "effect", "result", "leads to", "because of"],
            "counterfactual": ["if", "would", "could have", "alternatively"],
        }
        keywords = strategy_keywords.get(strategy, [])
        if keywords and not any(kw in content for kw in keywords):
            issues.append(
                f"Content may not align with '{state.strategies[cell.row]}' strategy. "
                f"Consider using relevant reasoning patterns."
            )

        return issues

    def _generate_summary(self, state: MatrixState) -> dict[str, Any]:
        """Generate summary of the matrix reasoning."""
        total_words = sum(len(c.content.split()) for c in state.cells.values())
        total_words += sum(len(s.content.split()) for s in state.syntheses.values())

        return {
            "dimensions": f"{state.rows}x{state.cols}",
            "cells_filled": len(state.cells),
            "total_cells": state.rows * state.cols,
            "syntheses": len(state.syntheses),
            "strategies_used": state.strategies,
            "total_words": total_words,
            "duration_seconds": (state.updated_at - state.created_at).total_seconds(),
        }


# Global instance for session persistence
_matrix_manager = MatrixOfThoughtManager()


def get_matrix_manager() -> MatrixOfThoughtManager:
    """Get the global matrix manager instance."""
    return _matrix_manager
