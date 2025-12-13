"""Domain-specific handlers for specialized reasoning validation.

Provides validation and enhancement for different problem domains:
- Math: Equation parsing, step validation, numeric verification
- Code: Syntax checking, trace simulation, complexity analysis
- Logic: Syllogism validation, contradiction detection, proof checking
- Factual: Source tracking, claim verification, temporal analysis

Each handler can:
1. Validate reasoning steps for domain-specific correctness
2. Extract structured information from unstructured text
3. Suggest improvements or identify errors
4. Provide domain-specific confidence adjustments
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationResult(str, Enum):
    """Result of domain validation."""

    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    PARTIAL = "partial"  # Partially valid with issues


@dataclass
class DomainValidation:
    """Result of domain-specific validation."""

    result: ValidationResult
    confidence: float  # 0.0 to 1.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    extracted_info: dict[str, Any] = field(default_factory=dict)
    confidence_adjustment: float = 0.0  # Adjustment to apply to base confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result.value,
            "confidence": round(self.confidence, 3),
            "issues": self.issues,
            "suggestions": self.suggestions,
            "extracted_info": self.extracted_info,
            "confidence_adjustment": round(self.confidence_adjustment, 3),
        }


class DomainHandler(ABC):
    """Abstract base class for domain-specific handlers."""

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Name of the domain this handler covers."""
        ...

    @abstractmethod
    def validate(self, thought: str, context: str) -> DomainValidation:
        """Validate a thought for domain-specific correctness.

        Args:
            thought: The reasoning step to validate.
            context: The problem context and previous steps.

        Returns:
            DomainValidation with results.

        """
        ...

    @abstractmethod
    def extract_structure(self, thought: str) -> dict[str, Any]:
        """Extract domain-specific structure from thought.

        Args:
            thought: The reasoning step to analyze.

        Returns:
            Dictionary with extracted structural information.

        """
        ...


# =============================================================================
# Math Domain Handler
# =============================================================================


@dataclass
class MathEquation:
    """A parsed mathematical equation."""

    raw: str
    lhs: str
    rhs: str
    operator: str  # =, <, >, <=, >=, !=
    variables: set[str] = field(default_factory=set)
    is_valid_syntax: bool = True


class MathHandler(DomainHandler):
    """Handler for mathematical reasoning validation."""

    # Patterns for math content
    EQUATION_PATTERN = re.compile(r"([^=<>!]+)\s*([=<>!]=?|[<>])\s*([^=<>!]+)")
    VARIABLE_PATTERN = re.compile(r"\b([a-zA-Z])\b(?![a-zA-Z])")
    NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
    OPERATION_PATTERN = re.compile(r"[\+\-\*\/\^]")

    # Common math errors
    DIVISION_BY_ZERO_PATTERN = re.compile(r"/\s*0(?![.\d])")
    SQRT_NEGATIVE_PATTERN = re.compile(r"sqrt\s*\(\s*-\d")

    @property
    def domain_name(self) -> str:
        return "math"

    def validate(self, thought: str, context: str) -> DomainValidation:
        """Validate mathematical reasoning."""
        issues: list[str] = []
        suggestions: list[str] = []
        extracted: dict[str, Any] = {}
        confidence = 0.5

        # Extract equations
        equations = self._extract_equations(thought)
        extracted["equations"] = [eq.raw for eq in equations]
        extracted["variables"] = list(set().union(*(eq.variables for eq in equations)))

        # Check for syntax errors
        syntax_errors = [eq for eq in equations if not eq.is_valid_syntax]
        if syntax_errors:
            issues.append(f"Found {len(syntax_errors)} equation(s) with syntax issues")
            confidence -= 0.2

        # Check for division by zero
        if self.DIVISION_BY_ZERO_PATTERN.search(thought):
            issues.append("Potential division by zero detected")
            confidence -= 0.3

        # Check for sqrt of negative
        if self.SQRT_NEGATIVE_PATTERN.search(thought):
            issues.append("Square root of negative number (complex result)")
            suggestions.append("Consider whether complex numbers are intended")
            confidence -= 0.1

        # Check for balanced parentheses
        if not self._check_balanced_parens(thought):
            issues.append("Unbalanced parentheses in expression")
            confidence -= 0.2

        # Validate numeric calculations if present
        calc_validation = self._validate_calculations(thought)
        if calc_validation["errors"]:
            issues.extend(calc_validation["errors"])
            confidence -= 0.2 * len(calc_validation["errors"])

        # Check for step justification
        if not self._has_justification(thought):
            suggestions.append("Consider adding justification for mathematical steps")

        # Determine result
        if issues:
            result = ValidationResult.INVALID if confidence < 0.3 else ValidationResult.PARTIAL
        else:
            result = ValidationResult.VALID
            confidence = min(0.9, confidence + 0.2)

        return DomainValidation(
            result=result,
            confidence=max(0.0, min(1.0, confidence)),
            issues=issues,
            suggestions=suggestions,
            extracted_info=extracted,
            confidence_adjustment=0.1 if result == ValidationResult.VALID else -0.1,
        )

    def extract_structure(self, thought: str) -> dict[str, Any]:
        """Extract mathematical structure from thought."""
        equations = self._extract_equations(thought)
        numbers = self.NUMBER_PATTERN.findall(thought)
        operations = self.OPERATION_PATTERN.findall(thought)

        return {
            "equations": [
                {
                    "raw": eq.raw,
                    "lhs": eq.lhs,
                    "rhs": eq.rhs,
                    "operator": eq.operator,
                    "variables": list(eq.variables),
                }
                for eq in equations
            ],
            "numbers": [float(n) for n in numbers],
            "operations": operations,
            "has_proof_structure": self._has_proof_structure(thought),
        }

    def _extract_equations(self, text: str) -> list[MathEquation]:
        """Extract equations from text."""
        equations: list[MathEquation] = []

        for match in self.EQUATION_PATTERN.finditer(text):
            lhs, op, rhs = match.groups()
            variables = set(self.VARIABLE_PATTERN.findall(lhs + rhs))
            # Exclude common words that look like variables
            variables -= {"a", "I", "A"}

            eq = MathEquation(
                raw=match.group(0).strip(),
                lhs=lhs.strip(),
                rhs=rhs.strip(),
                operator=op,
                variables=variables,
                is_valid_syntax=self._check_balanced_parens(match.group(0)),
            )
            equations.append(eq)

        return equations

    def _check_balanced_parens(self, text: str) -> bool:
        """Check if parentheses are balanced."""
        count = 0
        for char in text:
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
            if count < 0:
                return False
        return count == 0

    def _validate_calculations(self, text: str) -> dict[str, Any]:
        """Validate simple arithmetic calculations."""
        errors: list[str] = []

        # Look for simple arithmetic patterns like "2 + 2 = 5"
        simple_calc_pattern = re.compile(r"(\d+)\s*([\+\-\*])\s*(\d+)\s*=\s*(\d+)")

        for match in simple_calc_pattern.finditer(text):
            a, op, b, result = match.groups()
            a, b, result = int(a), int(b), int(result)

            expected = None
            if op == "+":
                expected = a + b
            elif op == "-":
                expected = a - b
            elif op == "*":
                expected = a * b

            if expected is not None and expected != result:
                errors.append(f"Calculation error: {a} {op} {b} = {result} (expected {expected})")

        return {"errors": errors}

    def _has_justification(self, text: str) -> bool:
        """Check if math steps have justification."""
        justification_words = [
            "because",
            "since",
            "therefore",
            "thus",
            "by",
            "using",
            "applying",
            "from",
            "given",
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in justification_words)

    def _has_proof_structure(self, text: str) -> bool:
        """Check if text has proof structure."""
        proof_words = [
            "assume",
            "suppose",
            "let",
            "given",
            "prove",
            "therefore",
            "thus",
            "hence",
            "qed",
            "∎",
            "contradiction",
            "by induction",
            "base case",
        ]
        text_lower = text.lower()
        return sum(1 for word in proof_words if word in text_lower) >= 2


# =============================================================================
# Code Domain Handler
# =============================================================================


@dataclass
class CodeBlock:
    """A detected code block."""

    language: str
    code: str
    line_count: int
    is_valid_syntax: bool
    syntax_errors: list[str] = field(default_factory=list)


class CodeHandler(DomainHandler):
    """Handler for code-related reasoning validation."""

    # Patterns for code detection
    CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")

    # Python-specific patterns
    PYTHON_INDENT_ERROR = re.compile(r"^\s+\S.*\n\S", re.MULTILINE)

    @property
    def domain_name(self) -> str:
        return "code"

    def validate(self, thought: str, context: str) -> DomainValidation:
        """Validate code-related reasoning."""
        issues: list[str] = []
        suggestions: list[str] = []
        extracted: dict[str, Any] = {}
        confidence = 0.5

        # Extract code blocks
        code_blocks = self._extract_code_blocks(thought)
        extracted["code_blocks"] = len(code_blocks)
        extracted["languages"] = list({cb.language for cb in code_blocks if cb.language})

        # Validate syntax
        for i, block in enumerate(code_blocks):
            if not block.is_valid_syntax:
                issues.extend([f"Block {i + 1}: {err}" for err in block.syntax_errors])
                confidence -= 0.15

        # Check for common code issues in reasoning
        code_issues = self._check_code_reasoning(thought)
        issues.extend(code_issues)
        confidence -= 0.1 * len(code_issues)

        # Check for complexity mentions
        if self._mentions_complexity(thought):
            extracted["discusses_complexity"] = True
            confidence += 0.1

        # Check for edge case consideration
        if self._mentions_edge_cases(thought):
            extracted["considers_edge_cases"] = True
            confidence += 0.1
        else:
            suggestions.append("Consider discussing edge cases")

        # Determine result
        if issues:
            result = ValidationResult.INVALID if confidence < 0.3 else ValidationResult.PARTIAL
        else:
            result = ValidationResult.VALID
            confidence = min(0.9, confidence + 0.2)

        return DomainValidation(
            result=result,
            confidence=max(0.0, min(1.0, confidence)),
            issues=issues,
            suggestions=suggestions,
            extracted_info=extracted,
            confidence_adjustment=0.1 if result == ValidationResult.VALID else -0.1,
        )

    def extract_structure(self, thought: str) -> dict[str, Any]:
        """Extract code structure from thought."""
        code_blocks = self._extract_code_blocks(thought)
        inline_code = self.INLINE_CODE_PATTERN.findall(thought)

        return {
            "code_blocks": [
                {
                    "language": cb.language,
                    "line_count": cb.line_count,
                    "is_valid": cb.is_valid_syntax,
                }
                for cb in code_blocks
            ],
            "inline_code_count": len(inline_code),
            "discusses_complexity": self._mentions_complexity(thought),
            "considers_edge_cases": self._mentions_edge_cases(thought),
        }

    def _extract_code_blocks(self, text: str) -> list[CodeBlock]:
        """Extract code blocks from text."""
        blocks: list[CodeBlock] = []

        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            language = match.group(1).lower() or "unknown"
            code = match.group(2)

            is_valid, errors = self._validate_syntax(code, language)

            blocks.append(
                CodeBlock(
                    language=language,
                    code=code,
                    line_count=len(code.strip().split("\n")),
                    is_valid_syntax=is_valid,
                    syntax_errors=errors,
                )
            )

        return blocks

    def _validate_syntax(self, code: str, language: str) -> tuple[bool, list[str]]:
        """Validate code syntax."""
        errors: list[str] = []

        if language in ("python", "py"):
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append(f"Python syntax error: {e.msg} at line {e.lineno}")
                return False, errors

        # Generic checks
        if not self._check_balanced_brackets(code):
            errors.append("Unbalanced brackets/braces")

        return len(errors) == 0, errors

    def _check_balanced_brackets(self, code: str) -> bool:
        """Check if brackets are balanced."""
        stack: list[str] = []
        pairs = {"(": ")", "[": "]", "{": "}"}

        for char in code:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False

        return len(stack) == 0

    def _check_code_reasoning(self, text: str) -> list[str]:
        """Check for common issues in code reasoning."""
        issues: list[str] = []
        text_lower = text.lower()

        # Check for undefined variable references
        if "undefined" in text_lower and "error" in text_lower:
            issues.append("Mentions undefined variable error")

        # Check for infinite loop mentions without resolution
        if "infinite loop" in text_lower and "fix" not in text_lower:
            issues.append("Mentions infinite loop without resolution")

        return issues

    def _mentions_complexity(self, text: str) -> bool:
        """Check if complexity is discussed."""
        complexity_terms = [
            "o(n)",
            "o(1)",
            "o(log",
            "o(n^",
            "o(n²",
            "time complexity",
            "space complexity",
            "linear",
            "logarithmic",
            "quadratic",
            "exponential",
            "big o",
            "big-o",
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in complexity_terms)

    def _mentions_edge_cases(self, text: str) -> bool:
        """Check if edge cases are considered."""
        edge_case_terms = [
            "edge case",
            "corner case",
            "boundary",
            "empty",
            "null",
            "none",
            "zero",
            "negative",
            "overflow",
            "underflow",
            "special case",
            "exception",
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in edge_case_terms)


# =============================================================================
# Logic Domain Handler
# =============================================================================


@dataclass
class LogicalStatement:
    """A parsed logical statement."""

    raw: str
    statement_type: str  # universal, existential, conditional, negation
    subject: str | None = None
    predicate: str | None = None
    is_negated: bool = False


class LogicHandler(DomainHandler):
    """Handler for logical reasoning validation."""

    # Patterns for logical statements
    UNIVERSAL_PATTERN = re.compile(
        r"\b(all|every|each|any)\s+(\w+)\s+(is|are|have|has)\s+(.+)", re.IGNORECASE
    )
    EXISTENTIAL_PATTERN = re.compile(
        r"\b(some|there exists?|at least one)\s+(\w+)\s+(is|are|have|has)\s+(.+)", re.IGNORECASE
    )
    CONDITIONAL_PATTERN = re.compile(r"\bif\s+(.+?)\s*,?\s*then\s+(.+)", re.IGNORECASE)
    NEGATION_PATTERN = re.compile(r"\b(not|no|none|never|neither)\b", re.IGNORECASE)

    # Logical connectives
    CONNECTIVES = ["and", "or", "not", "if", "then", "iff", "implies", "therefore"]

    @property
    def domain_name(self) -> str:
        return "logic"

    def validate(self, thought: str, context: str) -> DomainValidation:
        """Validate logical reasoning."""
        issues: list[str] = []
        suggestions: list[str] = []
        extracted: dict[str, Any] = {}
        confidence = 0.5

        # Extract logical statements
        statements = self._extract_statements(thought)
        extracted["statement_count"] = len(statements)
        extracted["statement_types"] = [s.statement_type for s in statements]

        # Check for logical structure
        has_structure = self._has_logical_structure(thought)
        extracted["has_logical_structure"] = has_structure
        if has_structure:
            confidence += 0.2

        # Check for common fallacies
        fallacies = self._detect_fallacies(thought)
        if fallacies:
            issues.extend([f"Potential fallacy: {f}" for f in fallacies])
            confidence -= 0.2 * len(fallacies)

        # Check for self-contradiction
        if self._has_self_contradiction(thought):
            issues.append("Potential self-contradiction detected")
            confidence -= 0.3

        # Check for unsupported conclusions
        if self._has_unsupported_conclusion(thought):
            issues.append("Conclusion may not follow from premises")
            suggestions.append("Ensure all conclusions are supported by stated premises")
            confidence -= 0.15

        # Check for quantifier clarity
        if not self._has_clear_quantifiers(thought, statements):
            suggestions.append("Consider clarifying quantifiers (all, some, none)")

        # Determine result
        if issues:
            result = ValidationResult.INVALID if confidence < 0.3 else ValidationResult.PARTIAL
        else:
            result = ValidationResult.VALID
            confidence = min(0.9, confidence + 0.2)

        return DomainValidation(
            result=result,
            confidence=max(0.0, min(1.0, confidence)),
            issues=issues,
            suggestions=suggestions,
            extracted_info=extracted,
            confidence_adjustment=0.15 if result == ValidationResult.VALID else -0.1,
        )

    def extract_structure(self, thought: str) -> dict[str, Any]:
        """Extract logical structure from thought."""
        statements = self._extract_statements(thought)
        connectives_found = [c for c in self.CONNECTIVES if c in thought.lower()]

        return {
            "statements": [
                {
                    "type": s.statement_type,
                    "subject": s.subject,
                    "predicate": s.predicate,
                    "negated": s.is_negated,
                }
                for s in statements
            ],
            "connectives": connectives_found,
            "has_logical_structure": self._has_logical_structure(thought),
            "inference_indicators": self._count_inference_indicators(thought),
        }

    def _extract_statements(self, text: str) -> list[LogicalStatement]:
        """Extract logical statements from text."""
        statements: list[LogicalStatement] = []

        # Universal statements
        for match in self.UNIVERSAL_PATTERN.finditer(text):
            statements.append(
                LogicalStatement(
                    raw=match.group(0),
                    statement_type="universal",
                    subject=match.group(2),
                    predicate=match.group(4),
                )
            )

        # Existential statements
        for match in self.EXISTENTIAL_PATTERN.finditer(text):
            statements.append(
                LogicalStatement(
                    raw=match.group(0),
                    statement_type="existential",
                    subject=match.group(2),
                    predicate=match.group(4),
                )
            )

        # Conditional statements
        for match in self.CONDITIONAL_PATTERN.finditer(text):
            statements.append(
                LogicalStatement(
                    raw=match.group(0),
                    statement_type="conditional",
                    subject=match.group(1),
                    predicate=match.group(2),
                )
            )

        return statements

    def _has_logical_structure(self, text: str) -> bool:
        """Check if text has clear logical structure."""
        structure_indicators = [
            "premise",
            "conclusion",
            "therefore",
            "thus",
            "it follows",
            "hence",
            "consequently",
            "given that",
            "assuming",
            "if and only if",
        ]
        text_lower = text.lower()
        return sum(1 for ind in structure_indicators if ind in text_lower) >= 2

    def _detect_fallacies(self, text: str) -> list[str]:
        """Detect common logical fallacies."""
        fallacies: list[str] = []
        text_lower = text.lower()

        # Ad hominem
        if re.search(r"(he|she|they)\s+(is|are)\s+(stupid|wrong|idiot)", text_lower):
            fallacies.append("Ad hominem attack")

        # Appeal to authority without justification
        if (
            re.search(r"(expert|authority|scientist)\s+said", text_lower)
            and "because" not in text_lower
        ):
            fallacies.append("Appeal to authority without reasoning")

        # False dichotomy
        if re.search(r"either\s+.+\s+or\s+.+\s+(nothing|no other)", text_lower):
            fallacies.append("False dichotomy")

        # Hasty generalization
        if re.search(r"(one|single|this)\s+.+\s+therefore\s+all", text_lower):
            fallacies.append("Hasty generalization")

        return fallacies

    def _has_self_contradiction(self, text: str) -> bool:
        """Check for self-contradictions."""
        text_lower = text.lower()

        # Simple contradiction patterns
        patterns = [
            (r"is true.+is false", r"is false.+is true"),
            (r"always.+never", r"never.+always"),
            (r"all.+none", r"none.+all"),
        ]

        return any(re.search(p1, text_lower) or re.search(p2, text_lower) for p1, p2 in patterns)

    def _has_unsupported_conclusion(self, text: str) -> bool:
        """Check for conclusions without supporting premises."""
        text_lower = text.lower()

        # Has conclusion indicator but no premise indicator
        conclusion_words = ["therefore", "thus", "hence", "so", "conclude"]
        premise_words = ["because", "since", "given", "as", "premise"]

        has_conclusion = any(word in text_lower for word in conclusion_words)
        has_premise = any(word in text_lower for word in premise_words)

        return has_conclusion and not has_premise

    def _has_clear_quantifiers(self, text: str, statements: list[LogicalStatement]) -> bool:
        """Check if quantifiers are clear."""
        # If there are noun phrases without quantifiers, flag it
        if not statements:
            return True  # No statements to check

        vague_patterns = re.findall(r"\b(people|things|stuff|items)\b", text.lower())
        return len(vague_patterns) < 2

    def _count_inference_indicators(self, text: str) -> int:
        """Count inference indicators."""
        indicators = [
            "therefore",
            "thus",
            "hence",
            "so",
            "implies",
            "entails",
            "follows",
            "conclude",
        ]
        text_lower = text.lower()
        return sum(1 for ind in indicators if ind in text_lower)


# =============================================================================
# Factual Domain Handler
# =============================================================================


@dataclass
class FactualClaim:
    """A factual claim extracted from text."""

    raw: str
    claim_type: str  # definition, attribution, temporal, numeric
    subject: str | None = None
    has_source: bool = False
    source: str | None = None


class FactualHandler(DomainHandler):
    """Handler for factual reasoning validation."""

    # Patterns for factual claims
    DEFINITION_PATTERN = re.compile(r"(\w+)\s+(?:is|are|means|refers to)\s+(.+)", re.IGNORECASE)
    ATTRIBUTION_PATTERN = re.compile(r"according to\s+([^,]+),?\s+(.+)", re.IGNORECASE)
    TEMPORAL_PATTERN = re.compile(
        r"in\s+(\d{4})|on\s+(\w+\s+\d+)|(\d{4})-(\d{2})-(\d{2})", re.IGNORECASE
    )

    @property
    def domain_name(self) -> str:
        return "factual"

    def validate(self, thought: str, context: str) -> DomainValidation:
        """Validate factual reasoning."""
        issues: list[str] = []
        suggestions: list[str] = []
        extracted: dict[str, Any] = {}
        confidence = 0.5

        # Extract factual claims
        claims = self._extract_claims(thought)
        extracted["claim_count"] = len(claims)
        extracted["claim_types"] = [c.claim_type for c in claims]

        # Check for source citations
        sourced_claims = [c for c in claims if c.has_source]
        extracted["sourced_claims"] = len(sourced_claims)
        if claims and not sourced_claims:
            suggestions.append("Consider citing sources for factual claims")
            confidence -= 0.1

        # Check for hedging language
        hedging = self._has_appropriate_hedging(thought)
        extracted["has_hedging"] = hedging
        if not hedging and claims:
            suggestions.append("Consider adding hedging for uncertain claims")

        # Check for temporal consistency
        temporal_issues = self._check_temporal_consistency(thought, context)
        if temporal_issues:
            issues.extend(temporal_issues)
            confidence -= 0.15 * len(temporal_issues)

        # Check for verifiable vs opinion
        opinion_ratio = self._opinion_vs_fact_ratio(thought)
        extracted["opinion_ratio"] = opinion_ratio
        if opinion_ratio > 0.5:
            suggestions.append("High proportion of opinion statements; consider adding facts")

        # Determine result
        if issues:
            result = ValidationResult.INVALID if confidence < 0.3 else ValidationResult.PARTIAL
        else:
            result = ValidationResult.VALID
            confidence = min(0.9, confidence + 0.2)

        return DomainValidation(
            result=result,
            confidence=max(0.0, min(1.0, confidence)),
            issues=issues,
            suggestions=suggestions,
            extracted_info=extracted,
            confidence_adjustment=0.05 if result == ValidationResult.VALID else -0.05,
        )

    def extract_structure(self, thought: str) -> dict[str, Any]:
        """Extract factual structure from thought."""
        claims = self._extract_claims(thought)

        return {
            "claims": [
                {
                    "type": c.claim_type,
                    "subject": c.subject,
                    "has_source": c.has_source,
                    "source": c.source,
                }
                for c in claims
            ],
            "dates_mentioned": self._extract_dates(thought),
            "named_entities": self._extract_named_entities(thought),
        }

    def _extract_claims(self, text: str) -> list[FactualClaim]:
        """Extract factual claims from text."""
        claims: list[FactualClaim] = []

        # Definition claims
        for match in self.DEFINITION_PATTERN.finditer(text):
            claims.append(
                FactualClaim(
                    raw=match.group(0),
                    claim_type="definition",
                    subject=match.group(1),
                )
            )

        # Attribution claims
        for match in self.ATTRIBUTION_PATTERN.finditer(text):
            claims.append(
                FactualClaim(
                    raw=match.group(0),
                    claim_type="attribution",
                    has_source=True,
                    source=match.group(1),
                )
            )

        return claims

    def _has_appropriate_hedging(self, text: str) -> bool:
        """Check if text has appropriate hedging for uncertain claims."""
        hedging_words = [
            "may",
            "might",
            "could",
            "possibly",
            "likely",
            "appears",
            "seems",
            "suggests",
            "indicates",
        ]
        certainty_words = [
            "definitely",
            "certainly",
            "absolutely",
            "always",
            "never",
            "proves",
            "confirms",
            "establishes",
        ]

        text_lower = text.lower()
        hedging_count = sum(1 for w in hedging_words if w in text_lower)
        certainty_count = sum(1 for w in certainty_words if w in text_lower)

        # Good hedging if more hedging than absolute certainty
        return hedging_count >= certainty_count

    def _check_temporal_consistency(self, thought: str, context: str) -> list[str]:
        """Check for temporal consistency issues."""
        issues: list[str] = []

        # Extract years
        thought_years = set(re.findall(r"\b(19|20)\d{2}\b", thought))
        set(re.findall(r"\b(19|20)\d{2}\b", context))

        # Check for anachronisms (very basic)
        for year in thought_years:
            year_int = int(year)
            if year_int > 2024:
                issues.append(f"Future date mentioned: {year}")

        return issues

    def _opinion_vs_fact_ratio(self, text: str) -> float:
        """Calculate ratio of opinion to factual language."""
        opinion_words = [
            "i think",
            "i believe",
            "in my opinion",
            "i feel",
            "should",
            "ought",
            "best",
            "worst",
            "beautiful",
            "ugly",
        ]
        fact_words = [
            "according to",
            "research shows",
            "studies indicate",
            "data suggests",
            "evidence",
            "measured",
            "observed",
        ]

        text_lower = text.lower()
        opinion_count = sum(1 for w in opinion_words if w in text_lower)
        fact_count = sum(1 for w in fact_words if w in text_lower)

        total = opinion_count + fact_count
        if total == 0:
            return 0.5  # Neutral

        return opinion_count / total

    def _extract_dates(self, text: str) -> list[str]:
        """Extract dates from text."""
        dates: list[str] = []

        # Year patterns
        dates.extend(re.findall(r"\b(19|20)\d{2}\b", text))

        # Full date patterns
        dates.extend(re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text))

        return dates

    def _extract_named_entities(self, text: str) -> list[str]:
        """Extract named entities (basic pattern matching)."""
        # Simple pattern for capitalized phrases (not at sentence start)
        entities = re.findall(r"(?<=[.!?]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
        # Also find mid-sentence capitalized phrases
        entities.extend(re.findall(r"(?<=[a-z]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text))

        return list(set(entities))


# =============================================================================
# General Domain Handler
# =============================================================================


class GeneralHandler(DomainHandler):
    """Handler for general reasoning validation.

    Applies domain-agnostic heuristics to assess:
    - Coherence: Logical flow and consistency
    - Completeness: Coverage of key aspects
    - Clarity: Clear expression and structure
    - Relevance: Connection to the problem context
    """

    # Coherence indicators
    TRANSITION_WORDS = frozenset(
        [
            "therefore",
            "thus",
            "hence",
            "so",
            "because",
            "since",
            "however",
            "but",
            "although",
            "nevertheless",
            "moreover",
            "furthermore",
            "additionally",
            "first",
            "second",
            "finally",
            "in conclusion",
            "as a result",
            "consequently",
            "specifically",
        ]
    )

    # Completeness indicators
    COMPLETENESS_PHRASES = frozenset(
        [
            "in summary",
            "to conclude",
            "the answer is",
            "in total",
            "altogether",
            "finally",
            "therefore the result",
            "this means",
            "we can conclude",
            "the solution is",
        ]
    )

    # Clarity anti-patterns (vague language)
    VAGUE_PATTERNS = [
        r"\b(something|somehow|somewhat|somewhere)\b",
        r"\b(stuff|things|it)\b(?!\s+(?:is|are|was|were|means|shows))",
        r"\b(maybe|perhaps|possibly|probably)\b.*\b(maybe|perhaps|possibly|probably)\b",
        r"\b(etc|and so on|and stuff)\b",
        r"\b(kind of|sort of|type of)\b(?!\s+\w+\s+(?:is|are))",
    ]

    # Self-reference patterns (good for reasoning)
    SELF_REFERENCE_PATTERNS = [
        r"\b(let me|let's|I will|I'll|we can|we should)\b",
        r"\b(considering|given that|assuming|if we)\b",
        r"\b(this means|this implies|this suggests)\b",
    ]

    # Structure patterns
    STRUCTURED_PATTERNS = [
        r"^\s*\d+[\.\)]\s+",  # Numbered list
        r"^\s*[-•]\s+",  # Bullet list
        r"\b(step \d+|first|second|third|finally)\b",
        r"\b(premise|conclusion|therefore|thus)\b",
    ]

    @property
    def domain_name(self) -> str:
        return "general"

    def validate(self, thought: str, context: str) -> DomainValidation:
        """Validate general reasoning quality."""
        issues: list[str] = []
        suggestions: list[str] = []
        extracted: dict[str, Any] = {}

        # Score components
        coherence_score = self._assess_coherence(thought)
        completeness_score = self._assess_completeness(thought, context)
        clarity_score = self._assess_clarity(thought)
        relevance_score = self._assess_relevance(thought, context)

        extracted["scores"] = {
            "coherence": round(coherence_score, 2),
            "completeness": round(completeness_score, 2),
            "clarity": round(clarity_score, 2),
            "relevance": round(relevance_score, 2),
        }

        # Weighted overall confidence
        confidence = (
            coherence_score * 0.30
            + completeness_score * 0.20
            + clarity_score * 0.25
            + relevance_score * 0.25
        )

        # Generate issues and suggestions based on scores
        if coherence_score < 0.4:
            issues.append("Low coherence: reasoning flow is unclear")
            suggestions.append(
                "Add transition words to connect ideas (therefore, because, however)"
            )
        elif coherence_score < 0.6:
            suggestions.append("Consider strengthening logical connections between statements")

        if completeness_score < 0.4:
            issues.append("Reasoning appears incomplete")
            suggestions.append("Ensure all aspects of the problem are addressed")

        if clarity_score < 0.4:
            issues.append("Expression is vague or unclear")
            suggestions.append("Replace vague terms with specific language")
        elif clarity_score < 0.6:
            suggestions.append("Consider being more specific in your reasoning")

        if relevance_score < 0.4:
            issues.append("Low relevance to the problem context")
            suggestions.append("Ensure reasoning directly addresses the question")

        # Check for specific patterns
        structure_info = self._extract_structure_info(thought)
        extracted["structure"] = structure_info

        if not structure_info["has_structure"]:
            suggestions.append(
                "Consider organizing thoughts with clear structure (numbered steps, bullet points)"
            )

        if structure_info["word_count"] < 20:
            suggestions.append("Response may be too brief; consider elaborating")
        elif structure_info["word_count"] > 500:
            suggestions.append("Response is lengthy; consider being more concise")

        # Determine result
        if confidence >= 0.7:
            result = ValidationResult.VALID
            confidence_adjustment = 0.1
        elif confidence >= 0.5:
            result = ValidationResult.PARTIAL
            confidence_adjustment = 0.0
        elif confidence >= 0.3:
            result = ValidationResult.UNCERTAIN
            confidence_adjustment = -0.05
        else:
            result = ValidationResult.INVALID
            confidence_adjustment = -0.1

        return DomainValidation(
            result=result,
            confidence=max(0.0, min(1.0, confidence)),
            issues=issues,
            suggestions=suggestions,
            extracted_info=extracted,
            confidence_adjustment=confidence_adjustment,
        )

    def extract_structure(self, thought: str) -> dict[str, Any]:
        """Extract structural information from thought."""
        return self._extract_structure_info(thought)

    def _assess_coherence(self, thought: str) -> float:
        """Assess logical coherence of the thought."""
        score = 0.5
        thought_lower = thought.lower()

        # Reward transition words
        transition_count = sum(1 for word in self.TRANSITION_WORDS if word in thought_lower)
        score += min(transition_count * 0.08, 0.3)

        # Reward self-reference patterns (shows reasoning process)
        for pattern in self.SELF_REFERENCE_PATTERNS:
            if re.search(pattern, thought_lower):
                score += 0.05

        # Penalize very short sentences (may indicate choppy reasoning)
        sentences = re.split(r"[.!?]+", thought)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 5:
                score -= 0.15
            elif avg_sentence_length > 8:
                score += 0.1

        # Check for logical flow markers
        if re.search(r"\b(if|when|assuming)\b.*\b(then|therefore|so)\b", thought_lower, re.DOTALL):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _assess_completeness(self, thought: str, context: str) -> float:
        """Assess completeness of reasoning."""
        score = 0.5
        thought_lower = thought.lower()

        # Reward conclusion/summary indicators
        for phrase in self.COMPLETENESS_PHRASES:
            if phrase in thought_lower:
                score += 0.15
                break

        # Check if thought addresses key context terms
        context_words = set(re.findall(r"\b[a-z]{4,}\b", context.lower()))
        thought_words = set(re.findall(r"\b[a-z]{4,}\b", thought_lower))

        # Remove common stopwords
        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "were",
            "will",
            "would",
            "could",
            "should",
        }
        context_words -= stopwords
        thought_words -= stopwords

        if context_words:
            overlap = len(context_words & thought_words) / len(context_words)
            score += overlap * 0.3

        # Penalize if too short relative to context
        if len(thought) < len(context) * 0.1 and len(context) > 100:
            score -= 0.2

        # Reward question answering patterns
        if re.search(r"\b(the answer is|therefore|thus|so the result)\b", thought_lower):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _assess_clarity(self, thought: str) -> float:
        """Assess clarity of expression."""
        score = 0.7  # Start higher, penalize for issues
        thought_lower = thought.lower()

        # Penalize vague patterns
        for pattern in self.VAGUE_PATTERNS:
            matches = len(re.findall(pattern, thought_lower))
            score -= matches * 0.1

        # Reward specific language
        # Numbers indicate specificity
        if re.search(r"\d+", thought):
            score += 0.1

        # Proper nouns indicate specificity
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", thought):
            score += 0.05

        # Quotes indicate specificity
        if '"' in thought or "'" in thought:
            score += 0.05

        # Penalize excessive hedging
        hedge_words = ["maybe", "perhaps", "possibly", "might", "could", "probably"]
        hedge_count = sum(1 for word in hedge_words if word in thought_lower)
        if hedge_count > 3:
            score -= 0.15

        # Penalize run-on sentences (very long without punctuation)
        long_segments = re.findall(r"[^.!?,;:]+", thought)
        for segment in long_segments:
            if len(segment.split()) > 40:
                score -= 0.1
                break

        return max(0.0, min(1.0, score))

    def _assess_relevance(self, thought: str, context: str) -> float:
        """Assess relevance to the problem context."""
        if not context.strip():
            return 0.6  # Neutral when no context provided

        score = 0.4
        thought_lower = thought.lower()
        context_lower = context.lower()

        # Extract key terms from context (nouns, verbs)
        context_terms = set(re.findall(r"\b[a-z]{4,}\b", context_lower))
        thought_terms = set(re.findall(r"\b[a-z]{4,}\b", thought_lower))

        # Remove common words
        common = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "were",
            "will",
            "would",
            "could",
            "should",
            "about",
            "there",
            "which",
            "their",
            "what",
        }
        context_terms -= common
        thought_terms -= common

        if context_terms:
            # Jaccard similarity
            intersection = len(context_terms & thought_terms)
            union = len(context_terms | thought_terms)
            if union > 0:
                score += (intersection / union) * 0.4

        # Reward direct references to context
        if re.search(r"\b(the problem|the question|as stated|given that)\b", thought_lower):
            score += 0.1

        # Reward addressing key question words
        question_words = ["what", "why", "how", "when", "where", "which", "who"]
        for qword in question_words:
            if qword in context_lower and qword in thought_lower:
                score += 0.05

        return max(0.0, min(1.0, score))

    def _extract_structure_info(self, thought: str) -> dict[str, Any]:
        """Extract structural information."""
        sentences = re.split(r"[.!?]+", thought)
        sentences = [s.strip() for s in sentences if s.strip()]

        has_structure = False
        for pattern in self.STRUCTURED_PATTERNS:
            if re.search(pattern, thought, re.MULTILINE | re.IGNORECASE):
                has_structure = True
                break

        return {
            "word_count": len(thought.split()),
            "sentence_count": len(sentences),
            "has_structure": has_structure,
            "has_conclusion": any(p in thought.lower() for p in self.COMPLETENESS_PHRASES),
            "transition_word_count": sum(1 for w in self.TRANSITION_WORDS if w in thought.lower()),
        }


# =============================================================================
# Handler Registry
# =============================================================================


class DomainHandlerRegistry:
    """Registry for domain handlers."""

    def __init__(self) -> None:
        """Initialize with default handlers."""
        self._handlers: dict[str, DomainHandler] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default domain handlers."""
        self.register(MathHandler())
        self.register(CodeHandler())
        self.register(LogicHandler())
        self.register(FactualHandler())
        self.register(GeneralHandler())

    def register(self, handler: DomainHandler) -> None:
        """Register a domain handler."""
        self._handlers[handler.domain_name] = handler

    def get(self, domain: str) -> DomainHandler | None:
        """Get handler for a domain."""
        return self._handlers.get(domain.lower())

    def validate(self, domain: str, thought: str, context: str) -> DomainValidation | None:
        """Validate using the appropriate domain handler.

        Args:
            domain: Domain name (math, code, logic, factual).
            thought: The reasoning step to validate.
            context: The problem context.

        Returns:
            DomainValidation if handler exists, None otherwise.

        """
        handler = self.get(domain)
        if handler:
            return handler.validate(thought, context)
        return None

    def available_domains(self) -> list[str]:
        """Get list of available domain handlers."""
        return list(self._handlers.keys())


# Module-level registry singleton
_registry: DomainHandlerRegistry | None = None


def get_handler_registry() -> DomainHandlerRegistry:
    """Get the domain handler registry singleton."""
    global _registry
    if _registry is None:
        _registry = DomainHandlerRegistry()
    return _registry


def validate_thought(domain: str, thought: str, context: str) -> DomainValidation | None:
    """Convenience function to validate a thought for a domain.

    Args:
        domain: Domain name.
        thought: The reasoning step.
        context: The problem context.

    Returns:
        DomainValidation or None if domain not found.

    """
    return get_handler_registry().validate(domain, thought, context)
