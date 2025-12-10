"""Matrix of Thought (MoT) reasoning tool.

Implements multi-dimensional reasoning combining breadth (multiple strategies)
and depth (iterative refinement) with inter-cell communication.

Based on paper 2509.03918v2 - achieves 7× speedup over RATT with +4.2% F1.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.utils.errors import ReasoningException
from src.utils.retry import retry_with_backoff
from src.utils.schema import ReasoningResult

if TYPE_CHECKING:
    from src.models.knowledge_graph import KnowledgeGraph
    from src.models.llm_client import LLMClientProtocol


class MatrixOfThoughtTool:
    """Matrix of Thought reasoning with configurable breadth and depth.

    The MoT framework organizes reasoning in an m×n matrix where:
    - Rows (m) represent different reasoning strategies/perspectives
    - Columns (n) represent iterative refinement steps
    - Inter-cell communication enables knowledge sharing

    Communication patterns control how cells influence each other:
    - vert&hor-01: Gradual increase in communication weight
    - uniform: Equal communication across all cells
    - none: Independent cells (like standard ToT)

    Optional Knowledge Graph integration extracts entities and relations
    from context to enhance multi-hop reasoning.

    Attributes:
        llm: LLM client for generating thoughts.

    """

    # P2 Optimization: Class-level cache for common weight matrices
    _weight_cache: dict[tuple[int, int, str], np.ndarray] = {}

    def __init__(self, llm_client: LLMClientProtocol) -> None:
        """Initialize MoT tool.

        Args:
            llm_client: LLM client for text generation.

        """
        self.llm = llm_client

    @retry_with_backoff(max_attempts=2, base_delay=1.0)
    def reason(
        self,
        question: str,
        context: str,
        matrix_rows: int = 3,
        matrix_cols: int = 4,
        communication_pattern: str = "vert&hor-01",
        use_knowledge_graph: bool = False,
    ) -> ReasoningResult:
        """Execute Matrix of Thought reasoning.

        Algorithm:
        1. Initialize m×n thought matrix
        2. Optionally extract knowledge graph from context
        3. For each column (depth iteration):
           a. For each row (strategy):
              - Generate thought with communication from previous column
              - Weight previous thought by communication matrix α
           b. Synthesize column into summary node
        4. Extract final answer from last summary node

        Args:
            question: The problem to solve.
            context: Relevant background information.
            matrix_rows: Number of strategies (breadth), 2-5.
            matrix_cols: Number of refinement iterations (depth), 2-5.
            communication_pattern: Weight pattern for inter-cell communication.
                                  Options: "vert&hor-01", "uniform", "none".
            use_knowledge_graph: If True, extract entities and relations from
                                context to enhance multi-hop reasoning.

        Returns:
            ReasoningResult with answer, confidence, and reasoning trace.

        Raises:
            ReasoningException: If reasoning fails due to invalid parameters
                               or LLM errors.

        Example:
            >>> tool = MatrixOfThoughtTool(llm_client)
            >>> result = tool.reason(
            ...     question="Who invented the telephone?",
            ...     context="Alexander Graham Bell was an inventor...",
            ...     matrix_rows=3,
            ...     matrix_cols=4,
            ...     use_knowledge_graph=True
            ... )
            >>> print(result.answer)
            >>> print(result.confidence)  # e.g., 0.85

        """
        # Validate inputs
        if not question or not question.strip():
            raise ReasoningException("Question cannot be empty")

        if not context or not context.strip():
            raise ReasoningException("Context cannot be empty")

        if not 2 <= matrix_rows <= 5:
            raise ReasoningException(f"matrix_rows must be 2-5, got {matrix_rows}")

        if not 2 <= matrix_cols <= 5:
            raise ReasoningException(f"matrix_cols must be 2-5, got {matrix_cols}")

        try:
            # Extract knowledge graph if enabled
            kg_context = ""
            kg_stats: dict[str, int] | None = None
            if use_knowledge_graph:
                kg_context, kg_stats = self._extract_knowledge_graph(context, question)

            # Generate communication weight matrix
            weight_matrix = self._generate_weight_matrix(
                matrix_rows, matrix_cols, communication_pattern
            )

            # Initialize thought matrix
            thought_matrix: list[list[str | None]] = [
                [None for _ in range(matrix_cols)] for _ in range(matrix_rows)
            ]
            summary_nodes: list[str] = []
            reasoning_steps: list[str] = []

            # Column iteration (depth)
            for col in range(matrix_cols):
                column_thoughts: list[str] = []

                # Row iteration (breadth)
                for row in range(matrix_rows):
                    prev_node = thought_matrix[row][col - 1] if col > 0 else None
                    alpha = weight_matrix[row, col - 1] if col > 0 else 0.0

                    # Generate thought node
                    thought = self._generate_thought_node(
                        question=question,
                        context=context,
                        prev_node=prev_node,
                        alpha=alpha,
                        row=row,
                        col=col,
                        total_rows=matrix_rows,
                        total_cols=matrix_cols,
                        kg_context=kg_context,
                    )

                    if thought:
                        thought_matrix[row][col] = thought
                        column_thoughts.append(thought)
                        reasoning_steps.append(
                            f"[R{row + 1}C{col + 1}] {thought[:150]}..."
                            if len(thought) > 150
                            else f"[R{row + 1}C{col + 1}] {thought}"
                        )

                # Synthesize column thoughts
                if column_thoughts:
                    summary = self._synthesize_column(
                        question, column_thoughts, context, col + 1, matrix_cols
                    )
                    summary_nodes.append(summary)

            # Extract final answer
            if summary_nodes:
                final_answer = self._extract_final_answer(question, summary_nodes[-1], context)
            else:
                final_answer = "Unable to generate answer"

            # Calculate confidence based on reasoning coverage
            all_thoughts = [t for row in thought_matrix for t in row if t]
            coverage = len(all_thoughts) / (matrix_rows * matrix_cols)
            confidence = min(0.5 + 0.4 * coverage, 0.95)

            # Build reasoning trace with optional KG stats
            trace: dict[str, object] = {
                "matrix_shape": [matrix_rows, matrix_cols],
                "total_thoughts": len(all_thoughts),
                "summary_iterations": len(summary_nodes),
                "communication_pattern": communication_pattern,
                "knowledge_graph_enabled": use_knowledge_graph,
            }
            if kg_stats:
                trace["knowledge_graph_stats"] = kg_stats

            return ReasoningResult(
                answer=final_answer,
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                tokens_used=self.llm.estimate_tokens("\n".join(reasoning_steps)),
                reasoning_trace=trace,
            )

        except ReasoningException:
            raise
        except Exception as e:
            logger.error(f"MoT reasoning failed: {e}")
            raise ReasoningException(f"Matrix of Thought reasoning failed: {e!s}") from e

    async def reason_async(
        self,
        question: str,
        context: str,
        matrix_rows: int = 3,
        matrix_cols: int = 4,
        communication_pattern: str = "vert&hor-01",
        use_knowledge_graph: bool = False,
    ) -> ReasoningResult:
        """Execute Matrix of Thought reasoning with parallelized row generation.

        P0 Optimization: Rows within each column are generated in parallel using
        asyncio.gather, reducing latency by ~3-4x for typical matrix sizes.

        Algorithm:
        1. Initialize m×n thought matrix
        2. Optionally extract knowledge graph from context
        3. For each column (depth iteration):
           a. PARALLEL: Generate all row thoughts concurrently
           b. Synthesize column into summary node
        4. Extract final answer from last summary node

        Args:
            question: The problem to solve.
            context: Relevant background information.
            matrix_rows: Number of strategies (breadth), 2-5.
            matrix_cols: Number of refinement iterations (depth), 2-5.
            communication_pattern: Weight pattern for inter-cell communication.
            use_knowledge_graph: If True, extract entities and relations from
                                context to enhance multi-hop reasoning.

        Returns:
            ReasoningResult with answer, confidence, and reasoning trace.

        Raises:
            ReasoningException: If reasoning fails.

        """
        # Validate inputs
        if not question or not question.strip():
            raise ReasoningException("Question cannot be empty")

        if not context or not context.strip():
            raise ReasoningException("Context cannot be empty")

        if not 2 <= matrix_rows <= 5:
            raise ReasoningException(f"matrix_rows must be 2-5, got {matrix_rows}")

        if not 2 <= matrix_cols <= 5:
            raise ReasoningException(f"matrix_cols must be 2-5, got {matrix_cols}")

        try:
            # Extract knowledge graph if enabled
            kg_context = ""
            kg_stats: dict[str, int] | None = None
            if use_knowledge_graph:
                kg_context, kg_stats = self._extract_knowledge_graph(context, question)

            # Generate communication weight matrix (cached)
            weight_matrix = self._generate_weight_matrix(
                matrix_rows, matrix_cols, communication_pattern
            )

            # Initialize thought matrix
            thought_matrix: list[list[str | None]] = [
                [None for _ in range(matrix_cols)] for _ in range(matrix_rows)
            ]
            summary_nodes: list[str] = []
            reasoning_steps: list[str] = []

            # Column iteration (depth) - must be sequential due to dependencies
            for col in range(matrix_cols):
                # P0 OPTIMIZATION: Generate all rows in parallel
                tasks = []
                for row in range(matrix_rows):
                    prev_node = thought_matrix[row][col - 1] if col > 0 else None
                    alpha = weight_matrix[row, col - 1] if col > 0 else 0.0

                    tasks.append(
                        self._generate_thought_node_async(
                            question=question,
                            context=context,
                            prev_node=prev_node,
                            alpha=alpha,
                            row=row,
                            col=col,
                            total_rows=matrix_rows,
                            total_cols=matrix_cols,
                            kg_context=kg_context,
                        )
                    )

                # Execute all row tasks in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)

                column_thoughts: list[str] = []
                for row, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Thought generation failed at ({row},{col}): {result}")
                        continue
                    if isinstance(result, str) and result:
                        thought_matrix[row][col] = result
                        column_thoughts.append(result)
                        reasoning_steps.append(
                            f"[R{row + 1}C{col + 1}] {result[:150]}..."
                            if len(result) > 150
                            else f"[R{row + 1}C{col + 1}] {result}"
                        )

                # Synthesize column thoughts (async)
                if column_thoughts:
                    summary = await self._synthesize_column_async(
                        question, column_thoughts, context, col + 1, matrix_cols
                    )
                    summary_nodes.append(summary)

            # Extract final answer (async)
            if summary_nodes:
                final_answer = await self._extract_final_answer_async(
                    question, summary_nodes[-1], context
                )
            else:
                final_answer = "Unable to generate answer"

            # Calculate confidence based on reasoning coverage
            all_thoughts = [t for row in thought_matrix for t in row if t]
            coverage = len(all_thoughts) / (matrix_rows * matrix_cols)
            confidence = min(0.5 + 0.4 * coverage, 0.95)

            # Build reasoning trace with optional KG stats
            trace: dict[str, object] = {
                "matrix_shape": [matrix_rows, matrix_cols],
                "total_thoughts": len(all_thoughts),
                "summary_iterations": len(summary_nodes),
                "communication_pattern": communication_pattern,
                "parallel_execution": True,
                "knowledge_graph_enabled": use_knowledge_graph,
            }
            if kg_stats:
                trace["knowledge_graph_stats"] = kg_stats

            return ReasoningResult(
                answer=final_answer,
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                tokens_used=self.llm.estimate_tokens("\n".join(reasoning_steps)),
                reasoning_trace=trace,
            )

        except ReasoningException:
            raise
        except Exception as e:
            logger.error(f"MoT async reasoning failed: {e}")
            raise ReasoningException(f"Matrix of Thought reasoning failed: {e!s}") from e

    async def _generate_thought_node_async(
        self,
        question: str,
        context: str,
        prev_node: str | None,
        alpha: float,
        row: int,
        col: int,
        total_rows: int,
        total_cols: int,
        kg_context: str = "",
    ) -> str | None:
        """Async version of thought node generation for parallel execution.

        Args:
            question: The problem being solved.
            context: Background information.
            prev_node: Previous thought in same row (if any).
            alpha: Communication weight from previous node.
            row: Current row index.
            col: Current column index.
            total_rows: Total rows in matrix.
            total_cols: Total columns in matrix.
            kg_context: Optional knowledge graph context string.

        Returns:
            Generated thought string, or None if generation failed.

        """
        try:
            # Build strategy guidance based on row
            strategies = [
                "direct factual analysis",
                "logical inference and deduction",
                "analogical reasoning",
                "step-by-step decomposition",
                "critical evaluation",
            ]
            strategy = strategies[row % len(strategies)]

            # Build communication context
            comm_context = ""
            if prev_node and alpha > 0:
                prev_summary = prev_node[:200] if len(prev_node) > 200 else prev_node
                comm_context = (
                    f"\nPrevious reasoning (weight α={alpha:.2f}): {prev_summary}\n"
                    "Build upon or contrast with this approach:"
                )

            prompt = f"""Question: {question}

Context: {context[:800]}{"..." if len(context) > 800 else ""}
{kg_context}
Matrix Position: Strategy {row + 1}/{total_rows}, Iteration {col + 1}/{total_cols}
Strategy Focus: {strategy}
{comm_context}

Generate ONE focused reasoning step (2-3 sentences) using {strategy}:"""

            response = await self.llm.generate_async(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7 if alpha < 0.5 else 0.5,
            )

            return response.strip() if response else None

        except Exception as e:
            logger.warning(f"Failed to generate thought at ({row},{col}): {e}")
            return None

    async def _synthesize_column_async(
        self,
        question: str,
        thoughts: list[str],
        context: str,
        col_num: int,
        total_cols: int,
    ) -> str:
        """Async version of column synthesis.

        Args:
            question: The problem being solved.
            thoughts: List of thoughts from this column.
            context: Background information.
            col_num: Current column number (1-indexed).
            total_cols: Total columns.

        Returns:
            Synthesized summary string.

        """
        try:
            thoughts_text = "\n".join(
                f"- Perspective {i + 1}: {t[:150]}..."
                if len(t) > 150
                else f"- Perspective {i + 1}: {t}"
                for i, t in enumerate(thoughts[:4])
            )

            prompt = f"""Question: {question}

Multiple reasoning perspectives (iteration {col_num}/{total_cols}):
{thoughts_text}

Context: {context[:400]}{"..." if len(context) > 400 else ""}

Synthesize these perspectives into ONE coherent insight (3-4 sentences).
Identify agreements, resolve conflicts, and advance toward the answer:"""

            return await self.llm.generate_async(prompt, max_tokens=400, temperature=0.5)

        except Exception as e:
            logger.warning(f"Column synthesis failed: {e}")
            return "Synthesis pending - multiple perspectives identified"

    async def _extract_final_answer_async(
        self, question: str, final_summary: str, context: str
    ) -> str:
        """Async version of final answer extraction.

        Args:
            question: Original question.
            final_summary: Final column synthesis.
            context: Background information.

        Returns:
            Concise answer string.

        """
        try:
            prompt = f"""Question: {question}

Final reasoning synthesis:
{final_summary}

Context excerpt: {context[:300]}...

Based on the reasoning above, provide a DIRECT, CONCISE answer to the question (1-2 sentences):"""

            return await self.llm.generate_async(prompt, max_tokens=200, temperature=0.3)

        except Exception as e:
            logger.warning(f"Answer extraction failed: {e}")
            # Fall back to first line of summary
            return final_summary.split("\n")[0] if final_summary else "Unable to extract answer"

    def _extract_knowledge_graph(
        self, context: str, question: str
    ) -> tuple[str, dict[str, int] | None]:
        """Extract knowledge graph from context and format for prompts.

        Args:
            context: The context text to extract from.
            question: The question to focus extraction on.

        Returns:
            Tuple of (kg_context_string, kg_stats_dict or None).

        """
        try:
            from src.models.knowledge_graph import KnowledgeGraphExtractor

            extractor = KnowledgeGraphExtractor(self.llm, use_llm=True)
            kg = extractor.extract_for_question(context, question)

            if not kg.relations:
                logger.debug("No relations extracted from knowledge graph")
                return "", None

            # Format relations for prompt injection
            def _format_predicate(pred: object) -> str:
                """Format predicate as string."""
                if isinstance(pred, str):
                    return pred
                return pred.value if hasattr(pred, "value") else str(pred)

            relations_text = "\n".join(
                f"- {r.subject.name} → {_format_predicate(r.predicate)} → {r.object_entity.name}"
                for r in list(kg.relations)[:10]  # Limit to top 10 relations
            )
            kg_context = f"\nKnowledge Graph (extracted entities/relations):\n{relations_text}\n"

            # Build stats
            stats = kg.stats()
            kg_stats = {
                "entities": stats.num_entities,
                "relations": stats.num_relations,
            }

            logger.debug(
                f"Extracted KG with {stats.num_entities} entities, {stats.num_relations} relations"
            )
            return kg_context, kg_stats

        except Exception as e:
            logger.warning(f"Knowledge graph extraction failed: {e}")
            return "", None

    def _format_kg_context(self, kg: KnowledgeGraph) -> str:
        """Format knowledge graph as context string for prompts.

        Args:
            kg: The knowledge graph to format.

        Returns:
            Formatted string of relations for prompt injection.

        """
        if not kg.relations:
            return ""

        def _format_predicate(pred: object) -> str:
            """Format predicate as string."""
            if isinstance(pred, str):
                return pred
            return pred.value if hasattr(pred, "value") else str(pred)

        relations_text = "\n".join(
            f"- {r.subject.name} → {_format_predicate(r.predicate)} → {r.object_entity.name}"
            for r in list(kg.relations)[:10]
        )
        return f"\nKnowledge Graph:\n{relations_text}\n"

    def _generate_thought_node(
        self,
        question: str,
        context: str,
        prev_node: str | None,
        alpha: float,
        row: int,
        col: int,
        total_rows: int,
        total_cols: int,
        kg_context: str = "",
    ) -> str | None:
        """Generate a single thought node in the matrix.

        Args:
            question: The problem being solved.
            context: Background information.
            prev_node: Previous thought in same row (if any).
            alpha: Communication weight from previous node.
            row: Current row index.
            col: Current column index.
            total_rows: Total rows in matrix.
            total_cols: Total columns in matrix.
            kg_context: Optional knowledge graph context string.

        Returns:
            Generated thought string, or None if generation failed.

        """
        try:
            # Build strategy guidance based on row
            strategies = [
                "direct factual analysis",
                "logical inference and deduction",
                "analogical reasoning",
                "step-by-step decomposition",
                "critical evaluation",
            ]
            strategy = strategies[row % len(strategies)]

            # Build communication context
            comm_context = ""
            if prev_node and alpha > 0:
                prev_summary = prev_node[:200] if len(prev_node) > 200 else prev_node
                comm_context = (
                    f"\nPrevious reasoning (weight α={alpha:.2f}): {prev_summary}\n"
                    "Build upon or contrast with this approach:"
                )

            prompt = f"""Question: {question}

Context: {context[:800]}{"..." if len(context) > 800 else ""}
{kg_context}
Matrix Position: Strategy {row + 1}/{total_rows}, Iteration {col + 1}/{total_cols}
Strategy Focus: {strategy}
{comm_context}

Generate ONE focused reasoning step (2-3 sentences) using {strategy}:"""

            response = self.llm.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7 if alpha < 0.5 else 0.5,
            )

            return response.strip() if response else None

        except Exception as e:
            logger.warning(f"Failed to generate thought at ({row},{col}): {e}")
            return None

    def _synthesize_column(
        self,
        question: str,
        thoughts: list[str],
        context: str,
        col_num: int,
        total_cols: int,
    ) -> str:
        """Synthesize thoughts from a column into a summary.

        Args:
            question: The problem being solved.
            thoughts: List of thoughts from this column.
            context: Background information.
            col_num: Current column number (1-indexed).
            total_cols: Total columns.

        Returns:
            Synthesized summary string.

        """
        try:
            thoughts_text = "\n".join(
                f"- Perspective {i + 1}: {t[:150]}..."
                if len(t) > 150
                else f"- Perspective {i + 1}: {t}"
                for i, t in enumerate(thoughts[:4])
            )

            prompt = f"""Question: {question}

Multiple reasoning perspectives (iteration {col_num}/{total_cols}):
{thoughts_text}

Context: {context[:400]}{"..." if len(context) > 400 else ""}

Synthesize these perspectives into ONE coherent insight (3-4 sentences).
Identify agreements, resolve conflicts, and advance toward the answer:"""

            return self.llm.generate(prompt, max_tokens=400, temperature=0.5)

        except Exception as e:
            logger.warning(f"Column synthesis failed: {e}")
            return "Synthesis pending - multiple perspectives identified"

    def _extract_final_answer(self, question: str, final_summary: str, context: str) -> str:
        """Extract concise final answer from summary.

        Args:
            question: Original question.
            final_summary: Final column synthesis.
            context: Background information.

        Returns:
            Concise answer string.

        """
        try:
            prompt = f"""Question: {question}

Final reasoning synthesis:
{final_summary}

Context excerpt: {context[:300]}...

Based on the reasoning above, provide a DIRECT, CONCISE answer to the question (1-2 sentences):"""

            return self.llm.generate(prompt, max_tokens=200, temperature=0.3)

        except Exception as e:
            logger.warning(f"Answer extraction failed: {e}")
            # Fall back to first line of summary
            return final_summary.split("\n")[0] if final_summary else "Unable to extract answer"

    def _generate_weight_matrix(self, rows: int, cols: int, pattern: str) -> np.ndarray:
        """Generate communication weight matrix.

        The weight matrix determines how much influence each cell
        receives from its predecessor in the same row.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            pattern: Communication pattern name.

        Returns:
            NumPy array of shape (rows, cols-1) with weights.

        """
        # P2 Optimization: Cache common weight matrices
        cache_key = (rows, cols, pattern)
        if cache_key in self._weight_cache:
            return self._weight_cache[cache_key]

        matrix = np.zeros((rows, cols - 1))

        if pattern == "vert&hor-01":
            # Paper 2509.03918v2 formula: α = 0.1*(m-i) + 0.1*j
            # - (m-i): Earlier rows (strategies) get more communication weight
            # - j: Later columns (iterations) accumulate more communication
            for i in range(rows):
                for j in range(cols - 1):
                    matrix[i, j] = min(0.1 * (rows - i) + 0.1 * j, 1.0)

        elif pattern == "uniform":
            # Equal communication everywhere
            matrix.fill(0.5)

        elif pattern == "none":
            # No communication (independent cells)
            matrix.fill(0.0)

        else:
            # Default: moderate communication
            matrix.fill(0.3)
            logger.warning(f"Unknown pattern '{pattern}', using default (0.3)")

        # Cache common sizes (2-5 rows/cols)
        if rows <= 5 and cols <= 5:
            self._weight_cache[cache_key] = matrix

        return matrix
