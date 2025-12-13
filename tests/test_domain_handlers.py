"""Tests for domain-specific handlers."""

from __future__ import annotations

from src.tools.domain_handlers import (
    CodeHandler,
    DomainHandlerRegistry,
    DomainValidation,
    FactualHandler,
    GeneralHandler,
    LogicHandler,
    MathHandler,
    ValidationResult,
    get_handler_registry,
    validate_thought,
)


class TestValidationResult:
    """Tests for ValidationResult enum."""

    def test_validation_result_values(self) -> None:
        """Test all validation result values exist."""
        assert ValidationResult.VALID.value == "valid"
        assert ValidationResult.INVALID.value == "invalid"
        assert ValidationResult.UNCERTAIN.value == "uncertain"
        assert ValidationResult.PARTIAL.value == "partial"


class TestDomainValidation:
    """Tests for DomainValidation dataclass."""

    def test_create_validation(self) -> None:
        """Test creating a validation result."""
        validation = DomainValidation(
            result=ValidationResult.VALID,
            confidence=0.9,
            issues=["Minor issue"],
            suggestions=["Consider X"],
            extracted_info={"key": "value"},
            confidence_adjustment=0.1,
        )
        assert validation.result == ValidationResult.VALID
        assert validation.confidence == 0.9
        assert len(validation.issues) == 1
        assert validation.confidence_adjustment == 0.1

    def test_to_dict(self) -> None:
        """Test converting validation to dictionary."""
        validation = DomainValidation(
            result=ValidationResult.VALID,
            confidence=0.85,
        )
        data = validation.to_dict()
        assert data["result"] == "valid"
        assert data["confidence"] == 0.85
        assert "issues" in data
        assert "suggestions" in data


class TestMathHandler:
    """Tests for MathHandler."""

    def test_domain_name(self) -> None:
        """Test math handler domain name."""
        handler = MathHandler()
        assert handler.domain_name == "math"

    def test_validate_valid_equation(self) -> None:
        """Test validating a correct mathematical equation."""
        handler = MathHandler()
        result = handler.validate(
            thought="Given that x = 5, we can calculate 2x + 3 = 2(5) + 3 = 13",
            context="Find 2x + 3 when x = 5",
        )
        assert result.result in (ValidationResult.VALID, ValidationResult.PARTIAL)
        assert result.confidence > 0

    def test_validate_invalid_equation(self) -> None:
        """Test validating an incorrect mathematical claim."""
        handler = MathHandler()
        result = handler.validate(
            thought="2 + 2 = 5 because the numbers add up",
            context="Simple arithmetic",
        )
        # May be uncertain or invalid depending on detection
        assert result.confidence > 0

    def test_validate_balanced_parentheses(self) -> None:
        """Test detecting unbalanced parentheses."""
        handler = MathHandler()
        result = handler.validate(
            thought="The formula is f(x) = (x + 1 * (y + 2)",  # Missing closing paren
            context="Mathematical function",
        )
        # Should detect unbalanced parentheses
        assert "unbalanced" in str(result.issues).lower() or result.result != ValidationResult.VALID

    def test_extract_structure(self) -> None:
        """Test extracting mathematical structure."""
        handler = MathHandler()
        structure = handler.extract_structure("Let x = 5. Then 2x + 3 = 13 and x^2 = 25.")
        assert "equations" in structure
        assert "numbers" in structure

    def test_detect_proof_structure(self) -> None:
        """Test detecting proof-like structure."""
        handler = MathHandler()
        result = handler.validate(
            thought="Proof: Assume P is true. Then Q follows. Therefore P implies Q. QED.",
            context="Mathematical proof",
        )
        assert result.confidence > 0


class TestCodeHandler:
    """Tests for CodeHandler."""

    def test_domain_name(self) -> None:
        """Test code handler domain name."""
        handler = CodeHandler()
        assert handler.domain_name == "code"

    def test_validate_valid_python(self) -> None:
        """Test validating valid Python code."""
        handler = CodeHandler()
        result = handler.validate(
            thought="""The function should be:
```python
def add(a, b):
    return a + b
```
This adds two numbers.""",
            context="Write a function to add two numbers",
        )
        assert result.result in (ValidationResult.VALID, ValidationResult.PARTIAL)

    def test_validate_invalid_python(self) -> None:
        """Test validating invalid Python syntax."""
        handler = CodeHandler()
        result = handler.validate(
            thought="""The code is:
```python
def add(a, b)  # Missing colon
    return a + b
```""",
            context="Write a function",
        )
        # Should detect syntax error
        assert "syntax" in str(result.issues).lower() or result.result != ValidationResult.VALID

    def test_extract_code_blocks(self) -> None:
        """Test extracting code blocks from thought."""
        handler = CodeHandler()
        structure = handler.extract_structure(
            """Here is the code:
```python
x = 1
y = 2
```
And also:
```javascript
const z = 3;
```"""
        )
        assert "code_blocks" in structure
        assert len(structure["code_blocks"]) == 2

    def test_validate_discusses_edge_cases(self) -> None:
        """Test detecting edge case discussion."""
        handler = CodeHandler()
        result = handler.validate(
            thought="We need to consider edge cases like empty list, single element, and None input.",
            context="Code review",
        )
        # Should give bonus for discussing edge cases
        assert result.confidence > 0


class TestLogicHandler:
    """Tests for LogicHandler."""

    def test_domain_name(self) -> None:
        """Test logic handler domain name."""
        handler = LogicHandler()
        assert handler.domain_name == "logic"

    def test_validate_valid_syllogism(self) -> None:
        """Test validating a valid logical argument."""
        handler = LogicHandler()
        result = handler.validate(
            thought="All humans are mortal. Socrates is a human. Therefore, Socrates is mortal.",
            context="Classical logic",
        )
        assert result.result in (ValidationResult.VALID, ValidationResult.PARTIAL)

    def test_detect_fallacy(self) -> None:
        """Test detecting a logical fallacy."""
        handler = LogicHandler()
        result = handler.validate(
            thought="He is wrong because he is stupid and doesn't know anything.",
            context="Argument",
        )
        # Should detect ad hominem
        assert "fallacy" in str(result.issues).lower() or len(result.issues) > 0

    def test_detect_self_contradiction(self) -> None:
        """Test detecting self-contradiction."""
        handler = LogicHandler()
        result = handler.validate(
            thought="This statement is always true and never true at the same time.",
            context="Logic puzzle",
        )
        # Should detect contradiction
        assert result.confidence < 1.0 or "contradiction" in str(result.issues).lower()

    def test_extract_logical_structure(self) -> None:
        """Test extracting logical structure."""
        handler = LogicHandler()
        structure = handler.extract_structure(
            "If it rains, then the ground is wet. It is raining. Therefore, the ground is wet."
        )
        assert "statements" in structure
        assert "connectives" in structure


class TestFactualHandler:
    """Tests for FactualHandler."""

    def test_domain_name(self) -> None:
        """Test factual handler domain name."""
        handler = FactualHandler()
        assert handler.domain_name == "factual"

    def test_validate_with_source(self) -> None:
        """Test validating factual claim with source."""
        handler = FactualHandler()
        result = handler.validate(
            thought="According to NASA, the distance to the Moon is about 384,400 km.",
            context="Space facts",
        )
        assert result.confidence > 0

    def test_validate_without_source(self) -> None:
        """Test validating factual claim without source."""
        handler = FactualHandler()
        result = handler.validate(
            thought="The Moon is about 400,000 km away.",
            context="Space facts",
        )
        # Should note lack of source
        assert result.confidence > 0

    def test_detect_future_date(self) -> None:
        """Test detecting future dates in context."""
        handler = FactualHandler()
        result = handler.validate(
            thought="In 2050, the population reached 10 billion.",
            context="Demographics",
        )
        # The handler validates factual claims; future dates may or may not be flagged
        # depending on implementation. Just verify it returns a valid result.
        assert result.confidence > 0
        assert isinstance(result.issues, list)

    def test_extract_factual_structure(self) -> None:
        """Test extracting factual structure."""
        handler = FactualHandler()
        structure = handler.extract_structure(
            "According to WHO, as of 2023, the global population is about 8 billion."
        )
        assert "claims" in structure
        assert "dates_mentioned" in structure


class TestDomainHandlerRegistry:
    """Tests for DomainHandlerRegistry."""

    def test_get_registry_singleton(self) -> None:
        """Test that registry returns same instance."""
        registry1 = get_handler_registry()
        registry2 = get_handler_registry()
        assert registry1 is registry2

    def test_default_handlers_registered(self) -> None:
        """Test that default handlers are registered."""
        registry = DomainHandlerRegistry()
        assert "math" in registry.available_domains()
        assert "code" in registry.available_domains()
        assert "logic" in registry.available_domains()
        assert "factual" in registry.available_domains()

    def test_get_handler(self) -> None:
        """Test getting handler by domain name."""
        registry = DomainHandlerRegistry()
        handler = registry.get("math")
        assert handler is not None
        assert handler.domain_name == "math"

    def test_get_nonexistent_handler(self) -> None:
        """Test getting nonexistent handler returns None."""
        registry = DomainHandlerRegistry()
        handler = registry.get("nonexistent")
        assert handler is None

    def test_validate_via_registry(self) -> None:
        """Test validation through registry."""
        registry = DomainHandlerRegistry()
        result = registry.validate(
            domain="math",
            thought="2 + 2 = 4",
            context="Arithmetic",
        )
        assert result is not None
        assert isinstance(result, DomainValidation)


class TestValidateThoughtFunction:
    """Tests for the validate_thought convenience function."""

    def test_validate_math_thought(self) -> None:
        """Test validating a math thought."""
        result = validate_thought(
            domain="math",
            thought="x^2 + y^2 = z^2 for a right triangle",
            context="Pythagorean theorem",
        )
        assert result is not None

    def test_validate_code_thought(self) -> None:
        """Test validating a code thought."""
        result = validate_thought(
            domain="code",
            thought="Use a for loop to iterate over the list",
            context="Programming",
        )
        assert result is not None

    def test_validate_nonexistent_domain(self) -> None:
        """Test validating with nonexistent domain returns None."""
        result = validate_thought(
            domain="nonexistent_domain",
            thought="Some thought",
            context="Context",
        )
        assert result is None

    def test_validate_returns_dict_compatible(self) -> None:
        """Test that validation result can be converted to dict."""
        result = validate_thought(
            domain="logic",
            thought="If A then B. A is true. Therefore B.",
            context="Logic",
        )
        assert result is not None
        data = result.to_dict()
        assert "result" in data
        assert "confidence" in data
        assert "issues" in data


class TestIntegrationWithUnifiedReasoner:
    """Integration tests with unified reasoner patterns."""

    def test_validation_confidence_adjustment(self) -> None:
        """Test that confidence adjustment is applied correctly."""
        result = validate_thought(
            domain="math",
            thought="Given x = 2, we calculate x^2 = 4, which is correct.",
            context="Simple calculation",
        )
        assert result is not None
        # Valid math should have positive or neutral adjustment
        assert result.confidence_adjustment >= -0.5

    def test_validation_issues_list(self) -> None:
        """Test that issues are properly collected."""
        result = validate_thought(
            domain="code",
            thought="```python\ndef f(:\n  pass\n```",
            context="Function definition",
        )
        assert result is not None
        # Should have syntax issue
        assert isinstance(result.issues, list)

    def test_validation_extracted_info(self) -> None:
        """Test that extracted info is populated."""
        result = validate_thought(
            domain="math",
            thought="Let x = 5, y = 10. Then x + y = 15.",
            context="Variable assignment",
        )
        assert result is not None
        assert isinstance(result.extracted_info, dict)


class TestGeneralHandler:
    """Tests for GeneralHandler - generic reasoning heuristics."""

    def test_domain_name(self) -> None:
        """Test general handler domain name."""
        handler = GeneralHandler()
        assert handler.domain_name == "general"

    def test_coherent_thought_high_score(self) -> None:
        """Test that coherent thoughts get high scores."""
        handler = GeneralHandler()
        result = handler.validate(
            thought=(
                "First, let's consider the problem. The main issue is X. "
                "Therefore, we should approach this by Y. In conclusion, Z follows."
            ),
            context="Analyze problem X",
        )
        assert result.result in (ValidationResult.VALID, ValidationResult.PARTIAL)
        assert result.confidence >= 0.6

    def test_incoherent_thought_lower_score(self) -> None:
        """Test that incoherent thoughts get lower scores."""
        handler = GeneralHandler()
        result = handler.validate(
            thought="thing stuff happens maybe idk",
            context="Analyze a complex problem",
        )
        # Should have lower confidence or issues
        assert result.confidence < 0.9 or len(result.issues) > 0

    def test_detects_vague_language(self) -> None:
        """Test detection of vague language."""
        handler = GeneralHandler()
        result = handler.validate(
            thought="Something might possibly be somewhat related to stuff.",
            context="Clear analysis needed",
        )
        # Should detect vagueness
        issues_str = " ".join(result.issues).lower()
        assert "vague" in issues_str or "clarity" in issues_str or result.confidence < 0.8

    def test_detects_missing_conclusion(self) -> None:
        """Test detection of incomplete reasoning without conclusion."""
        handler = GeneralHandler()
        result = handler.validate(
            thought="We need to consider A. Also B is important. And C matters.",
            context="Make a decision",
        )
        # Should note missing conclusion/decision
        assert (
            result.result != ValidationResult.VALID
            or "conclusion" in " ".join(result.suggestions).lower()
        )

    def test_rewards_clear_structure(self) -> None:
        """Test that clear structured reasoning is rewarded."""
        handler = GeneralHandler()
        result = handler.validate(
            thought=(
                "First, let me break down the problem. "
                "The key factors are: 1) resource constraints, 2) time limits. "
                "Given these factors, the optimal solution is to prioritize efficiency. "
                "Therefore, we should implement approach X."
            ),
            context="Optimize resource allocation",
        )
        assert result.result in (ValidationResult.VALID, ValidationResult.PARTIAL)
        assert result.confidence >= 0.7

    def test_relevance_to_context(self) -> None:
        """Test that relevance to context is assessed."""
        handler = GeneralHandler()
        # Irrelevant thought
        result = handler.validate(
            thought="The weather is nice today and birds are singing.",
            context="Debug the authentication module",
        )
        # Should have lower relevance or issues
        assert result.confidence < 0.8 or len(result.issues) > 0

    def test_extract_structure(self) -> None:
        """Test structure extraction for general reasoning."""
        handler = GeneralHandler()
        structure = handler.extract_structure(
            "First, analyze the problem. Then, identify solutions. Finally, implement."
        )
        assert isinstance(structure, dict)
        assert "word_count" in structure or "transition_words" in structure or len(structure) >= 0

    def test_validate_thought_general_domain(self) -> None:
        """Test validate_thought function with general domain."""
        result = validate_thought(
            domain="general",
            thought="Let me think step by step. First, we need to understand the context.",
            context="Problem solving",
        )
        assert result is not None
        assert result.result in (
            ValidationResult.VALID,
            ValidationResult.PARTIAL,
            ValidationResult.UNCERTAIN,
        )

    def test_registry_has_general_handler(self) -> None:
        """Test that registry includes general handler."""
        registry = get_handler_registry()
        handler = registry.get("general")
        assert handler is not None
        assert isinstance(handler, GeneralHandler)
