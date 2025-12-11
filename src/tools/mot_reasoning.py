"""Matrix of Thought (MoT) state manager.

Implements a state management tool for tracking matrix-based reasoning.
The calling LLM does all reasoning; this tool tracks the matrix structure,
manages cell filling, and provides synthesis guidance.

Based on paper 2509.03918v2 - Matrix of Thought framework.

Enhancements:
    - Column-Cell Communication: Guides divergent thinking across strategies
    - MPPA integration: Multi-path exploration at planning cells (Row 0)
    - FOBAR verification: Backward verification at finalization

Architecture:
    - Tool receives reasoning FROM the calling LLM
    - Tracks matrix cells (rows=strategies, cols=iterations)
    - Manages column synthesis
    - Provides guidance for next cell to fill
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from src.models.knowledge_graph import KnowledgeGraphExtractor
from src.utils.complexity import ComplexityResult, detect_complexity
from src.utils.scoring import calculate_cell_survival_score
from src.utils.session import SessionManager

# =============================================================================
# MPPA: Multi-Path Plan Aggregation for Row 0 (Planning Cells)
# =============================================================================


@dataclass
class CandidateCell:
    """A candidate reasoning path for multi-path exploration in Row 0."""

    content: str
    survival_score: float
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "survival_score": round(self.survival_score, 3),
            "selected": self.selected,
        }


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
    # MPPA fields for Row 0 planning cells
    alternatives_considered: list[CandidateCell] = field(default_factory=list)
    survival_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "row": self.row,
            "col": self.col,
            "strategy": self.strategy,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
        }
        if self.alternatives_considered:
            result["alternatives_considered"] = len(self.alternatives_considered)
        if self.survival_score is not None:
            result["survival_score"] = round(self.survival_score, 3)
        return result


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


class MatrixOfThoughtManager(SessionManager[MatrixState]):
    """Manager for Matrix of Thought reasoning sessions.

    Implements advanced MoT with:
        - Column-Cell Communication: Horizontal thought propagation
        - MPPA: Multi-Path Plan Aggregation for Row 0 planning
        - FOBAR: Forward-Backward Reasoning verification

    Matrix structure:
        - Rows represent different reasoning strategies
        - Columns represent iterative refinement steps
        - Each cell builds on previous column's cell + synthesis

    """

    def __init__(
        self,
        encoder: Any | None = None,
        kg: Any | None = None,
    ) -> None:
        """Initialize the matrix manager.

        Args:
            encoder: Optional ContextEncoder for semantic scoring.
            kg: Optional KnowledgeGraph for fact alignment scoring.

        """
        super().__init__()
        self._encoder = encoder
        self._kg = kg

    def start_matrix(
        self,
        question: str,
        context: str = "",
        rows: int | str = 3,
        cols: int | str = 4,
        strategies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new matrix reasoning session.

        Args:
            question: The question to reason about.
            context: Background context for reasoning.
            rows: Number of strategies (2-5), or "auto" for adaptive selection.
            cols: Number of refinement iterations (2-6), or "auto" for adaptive.
            strategies: Custom strategy names (default: standard 3 strategies).
            metadata: Optional metadata.

        Returns:
            Session info with matrix structure.

        """
        # Adaptive complexity detection
        complexity_info: ComplexityResult | None = None
        if rows == "auto" or cols == "auto":
            complexity_info = detect_complexity(question, context)
            if rows == "auto":
                rows = complexity_info.recommended_rows
            if cols == "auto":
                cols = complexity_info.recommended_cols
            logger.debug(
                f"Adaptive dimensions: {rows}x{cols} "
                f"(complexity: {complexity_info.complexity_level})"
            )

        # Validate dimensions
        try:
            rows = max(2, min(5, int(rows)))
            cols = max(2, min(6, int(cols)))
        except (ValueError, TypeError):
            return {"error": "Invalid dimensions: rows and cols must be integers 2-5/2-6 or 'auto'"}

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
        self._register_session(session_id, state)

        # Auto-extract entities from question and context into shared KG
        if self._kg is not None:
            try:
                extractor = KnowledgeGraphExtractor()
                extractor.extract(question, existing_graph=self._kg)
                if context:
                    extractor.extract(context, existing_graph=self._kg)
                logger.debug(
                    f"Extracted {self._kg.stats()['entity_count']} entities from question/context"
                )
            except Exception as e:
                logger.warning(f"KG extraction failed (non-fatal): {e}")

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
            "complexity": complexity_info.to_dict() if complexity_info else None,
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
        alternatives: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fill a cell in the matrix with Column-Cell Communication and MPPA.

        For Row 0 (planning cells), supports MPPA multi-path exploration:
        provide multiple alternative thoughts and the best one is selected
        based on survival probability scoring.

        Args:
            session_id: Session to update.
            row: Row index (0-based).
            col: Column index (0-based).
            thought: Reasoning content for this cell (primary candidate).
            confidence: Optional confidence score.
            alternatives: Additional candidate thoughts for Row 0 MPPA exploration.
                         Only used for row=0 cells. The best candidate is selected.

        Returns:
            Updated state and guidance for next cell.

        """
        with self.session(session_id) as state:
            if state.status != MatrixStatus.ACTIVE:
                return {"error": f"Session is {state.status.value}"}

            # Validate position
            if not (0 <= row < state.rows and 0 <= col < state.cols):
                return {
                    "error": f"Invalid position ({row}, {col}) for {state.rows}x{state.cols} matrix"
                }

            # Validate row is within strategies bounds
            if row >= len(state.strategies):
                return {
                    "error": f"Row {row} exceeds available strategies ({len(state.strategies)})"
                }

        # MPPA: Multi-path exploration for Row 0 (planning cells)
        selected_thought = thought
        candidates: list[CandidateCell] = []
        selected_score: float | None = None

        if alternatives and row != 0:
            # R13: Warn when alternatives provided for non-planning rows
            import warnings

            warnings.warn(
                f"alternatives ignored for row {row} (only used for row 0 MPPA planning)",
                stacklevel=2,
            )

        if row == 0 and alternatives:
            # Validate alternatives list
            if not all(isinstance(alt, str) and alt for alt in alternatives):
                return {"error": "All alternatives must be non-empty strings"}

            # Build candidate list: primary + alternatives
            all_thoughts = [thought] + alternatives
            strategy = state.strategies[row]

            for t in all_thoughts:
                score = calculate_cell_survival_score(
                    thought=t,
                    strategy=strategy,
                    context=state.context + " " + state.question,
                    col=col,
                    encoder=self._encoder,
                    kg=self._kg,
                )
                candidates.append(CandidateCell(content=t, survival_score=score))

            # Select best candidate (with random tiebreaker for equal scores)
            candidates.sort(
                key=lambda c: (c.survival_score, random.random()),  # nosec B311
                reverse=True,
            )
            best = candidates[0]
            best.selected = True
            selected_thought = best.content
            selected_score = best.survival_score

            logger.debug(
                f"MPPA: Evaluated {len(candidates)} candidates for cell (0, {col}), "
                f"selected with score {selected_score:.3f}"
            )

        # Create cell
        cell = MatrixCell(
            row=row,
            col=col,
            strategy=state.strategies[row],
            content=selected_thought,
            confidence=confidence,
            alternatives_considered=candidates if candidates else [],
            survival_score=selected_score,
        )
        state.cells[(row, col)] = cell
        state.updated_at = datetime.now()

        # Check consistency
        issues = self._check_cell_consistency(state, cell)

        # Determine next action
        next_cell = self._get_next_cell(state)
        col_complete = all((r, col) in state.cells for r in range(state.rows))

        response: dict[str, Any] = {
            "session_id": session_id,
            "cell_set": {"row": row, "col": col, "strategy": state.strategies[row]},
            "cells_filled": len(state.cells),
            "total_cells": state.rows * state.cols,
            "progress": round(len(state.cells) / (state.rows * state.cols), 2),
            "column_complete": col_complete,
        }

        # MPPA: Include exploration results for Row 0
        if candidates:
            response["mppa"] = {
                "candidates_evaluated": len(candidates),
                "selected_score": round(selected_score, 3) if selected_score else None,
                "all_scores": [round(c.survival_score, 3) for c in candidates],
                "exploration_benefit": (
                    "Multiple reasoning paths evaluated; best candidate selected "
                    "based on strategy alignment and context relevance."
                ),
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

            # Column-Cell Communication: provide anti-pattern guidance
            communication = self._get_cell_communication(state, next_cell["row"], next_cell["col"])
            if communication:
                response["communication"] = communication

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

    def _get_cell_communication(
        self,
        state: MatrixState,
        row: int,
        col: int,
    ) -> dict[str, Any] | None:
        """Get Column-Cell Communication guidance for divergent thinking.

        Based on MTQA paper: provides explicit guidance to avoid patterns
        from previous cells in the same column, forcing strategy divergence.

        Args:
            state: Current matrix state.
            row: Target row for next cell.
            col: Target column for next cell.

        Returns:
            Communication dict with avoid_patterns and build_on guidance.

        """
        communication: dict[str, Any] = {}

        # Collect patterns to avoid from previous rows in same column
        avoid_patterns = []
        for prev_row in range(row):
            prev_cell = state.cells.get((prev_row, col))
            if prev_cell:
                # Extract key phrases (first 80 chars as summary)
                summary = prev_cell.content[:80].strip()
                if len(prev_cell.content) > 80:
                    summary += "..."
                avoid_patterns.append(
                    {
                        "strategy": state.strategies[prev_row],
                        "approach": summary,
                    }
                )

        if avoid_patterns:
            communication["avoid_patterns"] = avoid_patterns
            num_patterns = len(avoid_patterns)
            communication["divergence_guidance"] = (
                f"Previous {num_patterns} strateg{'ies' if num_patterns > 1 else 'y'} "
                f"in this column already covered certain angles. "
                f"Use '{state.strategies[row]}' to explore a DIFFERENT perspective."
            )

        # Build on previous column's synthesis if available
        if col > 0 and (col - 1) in state.syntheses:
            prev_synthesis = state.syntheses[col - 1]
            communication["build_on"] = {
                "previous_synthesis": prev_synthesis.content[:150] + "..."
                if len(prev_synthesis.content) > 150
                else prev_synthesis.content,
                "guidance": "Build on this synthesis while applying your unique strategy.",
            }

        return communication if communication else None

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
        with self.session(session_id) as state:
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
        with self.session(session_id) as state:
            return state.to_dict()

    def finalize(
        self,
        session_id: str,
        answer: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Finalize the matrix reasoning with FOBAR backward verification.

        FOBAR (Forward-Backward Reasoning) verifies the answer by:
        1. Checking answer consistency with column syntheses
        2. Tracing backward through reasoning to verify support
        3. Identifying any contradictions or gaps

        Args:
            session_id: Session to finalize.
            answer: The final answer.
            confidence: Optional confidence score.

        Returns:
            Final matrix state with summary and FOBAR verification results.

        """
        with self.session(session_id) as state:
            # FOBAR backward verification
            fobar_result = self._fobar_verify(state, answer)

            state.final_answer = answer
            state.status = MatrixStatus.COMPLETED
            state.updated_at = datetime.now()

        # Generate summary (read-only, can be outside lock)
        summary = self._generate_summary(state)

        logger.info(f"Finalized matrix session {session_id}")

        result = {
            "session_id": session_id,
            "status": "completed",
            "final_answer": answer,
            "confidence": confidence,
            "summary": summary,
            "matrix": state.to_dict(),
        }

        # Include FOBAR verification results
        if fobar_result:
            result["fobar_verification"] = fobar_result
            # Adjust confidence based on verification
            if confidence and fobar_result.get("verification_score"):
                result["adjusted_confidence"] = round(
                    confidence * fobar_result["verification_score"], 2
                )

        return result

    def _fobar_verify(
        self,
        state: MatrixState,
        answer: str,
    ) -> dict[str, Any]:
        """FOBAR backward verification of the proposed answer.

        Traces backward from the answer through syntheses and cells
        to verify reasoning chain consistency.

        Args:
            state: Current matrix state.
            answer: Proposed final answer.

        Returns:
            Verification results with score and any issues found.

        """
        # Handle empty answer
        if not answer or not answer.strip():
            return {
                "verification_score": 0.0,
                "verification_status": "FAILED",
                "reason": "Empty answer cannot be verified",
            }

        issues: list[str] = []
        supports: list[str] = []
        answer_lower = answer.lower()

        # Extract key terms from answer for matching
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "and",
            "in",
            "for",
            "it",
            "this",
            "that",
        }
        answer_terms = set(answer_lower.split()) - stopwords

        # 1. Check consistency with column syntheses (backward from final)
        synthesis_support = 0
        for col in sorted(state.syntheses.keys(), reverse=True):
            synth = state.syntheses[col]
            synth_lower = synth.content.lower()
            synth_terms = set(synth_lower.split()) - stopwords

            # Check term overlap
            overlap = answer_terms & synth_terms
            if overlap:
                synthesis_support += 1
                supports.append(
                    f"Column {col} synthesis supports answer "
                    f"(shared: {', '.join(list(overlap)[:3])})"
                )
            else:
                # Check for direct mention
                if any(term in synth_lower for term in answer_terms if len(term) > 3):
                    synthesis_support += 0.5  # type: ignore[assignment]
                    supports.append(f"Column {col} synthesis partially supports answer")

        # 2. Check cell-level support (trace reasoning chain)
        cell_support = 0
        contradictions = []

        for (row, col), cell in state.cells.items():
            cell_lower = cell.content.lower()

            # Look for direct support
            if any(term in cell_lower for term in answer_terms if len(term) > 3):
                cell_support += 1

            # Look for contradiction indicators
            contradiction_phrases = [
                "however",
                "but",
                "contrary",
                "opposite",
                "instead",
                "not",
                "never",
                "false",
                "incorrect",
                "wrong",
            ]
            has_contradiction = any(phrase in cell_lower for phrase in contradiction_phrases)

            # If cell has contradiction language AND mentions answer terms, flag it
            if has_contradiction and any(
                term in cell_lower for term in answer_terms if len(term) > 3
            ):
                contradictions.append(
                    f"Cell ({row},{col}) [{state.strategies[row]}] "
                    "may contain contradicting reasoning"
                )

        if contradictions:
            issues.extend(contradictions[:2])  # Limit to top 2

        # 3. Calculate verification score
        total_syntheses = max(len(state.syntheses), 1)
        total_cells = max(len(state.cells), 1)

        synthesis_ratio = synthesis_support / total_syntheses
        cell_ratio = min(cell_support / total_cells, 1.0)  # Cap at 1.0
        contradiction_penalty = len(contradictions) * 0.1

        verification_score = (
            0.6 * synthesis_ratio  # Syntheses are more important
            + 0.4 * cell_ratio  # Cell support
            - contradiction_penalty  # Penalty for contradictions
        )
        verification_score = max(0.0, min(1.0, verification_score))

        # 4. Build result
        result: dict[str, Any] = {
            "verification_score": round(verification_score, 3),
            "synthesis_support": f"{synthesis_support}/{total_syntheses}",
            "cell_support": f"{cell_support}/{total_cells}",
        }

        if supports:
            result["supporting_evidence"] = supports[:3]  # Top 3

        if issues:
            result["potential_issues"] = issues
            result["recommendation"] = (
                "Review the flagged cells for potential contradictions. "
                "Consider whether the reasoning chain fully supports the answer."
            )
        else:
            result["verification_status"] = "PASSED"
            result["note"] = "Backward verification found consistent reasoning chain."

        return result

    def abandon(self, session_id: str, reason: str = "") -> dict[str, Any]:
        """Abandon a matrix session.

        Args:
            session_id: Session to abandon.
            reason: Optional reason.

        Returns:
            Final state.

        """
        with self.session(session_id) as state:
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
