"""Comparative Benchmark: Long Chain vs Matrix-of-Thought vs Baseline Reasoning.

This benchmark compares three reasoning strategies on the same problems:
1. Baseline (direct): Single-shot answer without structured reasoning
2. Long Chain: Sequential step-by-step reasoning
3. Matrix of Thought (MoT): Multi-perspective parallel reasoning

Metrics compared:
- Correctness: Does the answer match the expected result?
- Reasoning depth: Number of distinct reasoning steps/thoughts
- Coverage: How many aspects of the problem were considered?
- Latency: Total time to produce final answer

Requirements:
- Running MCP server (uses fastmcp Client)
- OpenAI API key (for LLM-based evaluation) - optional, uses heuristics if not set

Run:
    python examples/reasoning_comparison.py [--verbose] [--problems N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Import shared LLM client utilities
from llm_client import (
    add_llm_args,
    close_llm_client,
    get_llm_answer,
    get_llm_chain_step,
    get_llm_mot_cell,
    get_llm_mot_synthesis,
    init_llm_client,
    is_llm_enabled,
)

# =============================================================================
# Problem Definitions
# =============================================================================


class ProblemType(Enum):
    """Categories of reasoning problems."""

    MATH = "math"
    LOGIC = "logic"
    MULTI_HOP = "multi_hop"
    ANALYSIS = "analysis"


@dataclass
class ReasoningProblem:
    """A problem designed to test reasoning capabilities."""

    id: str
    problem_type: ProblemType
    question: str
    context: str
    expected_answer: str
    expected_keywords: list[str]  # Keywords that should appear in good reasoning
    difficulty: int  # 1-5 scale
    perspectives: list[str] | None = None  # For MoT: suggested perspectives
    steps_hint: int = 5  # For chain: expected number of steps


# Problems that benefit from structured reasoning
BENCHMARK_PROBLEMS: list[ReasoningProblem] = [
    # ==========================================================================
    # MATH PROBLEMS (25 total) - benefit from step-by-step reasoning
    # ==========================================================================
    ReasoningProblem(
        id="math_001",
        problem_type=ProblemType.MATH,
        question="A store sells apples for $2 each and oranges for $3 each. "
        "If Sarah buys 5 apples and 3 oranges, and pays with a $20 bill, "
        "how much change does she receive?",
        context="",
        expected_answer="1",
        expected_keywords=["apples", "oranges", "total", "change", "10", "9", "19"],
        difficulty=2,
        steps_hint=4,
        perspectives=["cost calculation", "verification"],
    ),
    ReasoningProblem(
        id="math_002",
        problem_type=ProblemType.MATH,
        question="A train travels at 60 mph for 2 hours, then at 80 mph for 1.5 hours. "
        "What is the average speed for the entire journey?",
        context="",
        expected_answer="68.57",
        expected_keywords=["distance", "time", "total", "average", "120", "240"],
        difficulty=3,
        steps_hint=5,
        perspectives=["distance calculation", "time calculation", "average formula"],
    ),
    ReasoningProblem(
        id="math_003",
        problem_type=ProblemType.MATH,
        question="What is 15% of 240?",
        context="",
        expected_answer="36",
        expected_keywords=["percent", "multiply", "decimal", "0.15"],
        difficulty=1,
        steps_hint=2,
        perspectives=["percentage conversion", "calculation"],
    ),
    ReasoningProblem(
        id="math_004",
        problem_type=ProblemType.MATH,
        question="A rectangle has a length of 12 cm and width of 8 cm. What is its area?",
        context="",
        expected_answer="96",
        expected_keywords=["length", "width", "multiply", "area", "square"],
        difficulty=1,
        steps_hint=2,
        perspectives=["formula", "calculation"],
    ),
    ReasoningProblem(
        id="math_005",
        problem_type=ProblemType.MATH,
        question="If a car travels 150 miles in 3 hours, what is its average speed?",
        context="",
        expected_answer="50",
        expected_keywords=["distance", "time", "divide", "speed", "mph"],
        difficulty=1,
        steps_hint=2,
        perspectives=["formula", "division"],
    ),
    ReasoningProblem(
        id="math_006",
        problem_type=ProblemType.MATH,
        question="What is the sum of the first 10 positive integers?",
        context="",
        expected_answer="55",
        expected_keywords=["sum", "formula", "n", "1", "10", "arithmetic"],
        difficulty=2,
        steps_hint=3,
        perspectives=["formula approach", "direct addition"],
    ),
    ReasoningProblem(
        id="math_007",
        problem_type=ProblemType.MATH,
        question="A pizza is cut into 8 equal slices. If Tom eats 3 slices, "
        "what fraction of the pizza remains?",
        context="",
        expected_answer="5/8",
        expected_keywords=["fraction", "remaining", "subtract", "slices"],
        difficulty=1,
        steps_hint=3,
        perspectives=["subtraction", "fraction"],
    ),
    ReasoningProblem(
        id="math_008",
        problem_type=ProblemType.MATH,
        question="Solve for x: 2x + 5 = 17",
        context="",
        expected_answer="6",
        expected_keywords=["subtract", "divide", "isolate", "equation"],
        difficulty=2,
        steps_hint=3,
        perspectives=["algebraic manipulation", "verification"],
    ),
    ReasoningProblem(
        id="math_009",
        problem_type=ProblemType.MATH,
        question="What is the perimeter of a square with side length 7 meters?",
        context="",
        expected_answer="28",
        expected_keywords=["perimeter", "square", "multiply", "4", "sides"],
        difficulty=1,
        steps_hint=2,
        perspectives=["formula", "calculation"],
    ),
    ReasoningProblem(
        id="math_010",
        problem_type=ProblemType.MATH,
        question="If 3 notebooks cost $12, how much do 7 notebooks cost?",
        context="",
        expected_answer="28",
        expected_keywords=["unit price", "multiply", "proportion", "cost"],
        difficulty=2,
        steps_hint=3,
        perspectives=["unit price", "multiplication"],
    ),
    ReasoningProblem(
        id="math_011",
        problem_type=ProblemType.MATH,
        question="What is 2^8?",
        context="",
        expected_answer="256",
        expected_keywords=["power", "exponent", "multiply", "2"],
        difficulty=2,
        steps_hint=3,
        perspectives=["exponentiation", "step-by-step multiplication"],
    ),
    ReasoningProblem(
        id="math_012",
        problem_type=ProblemType.MATH,
        question="A bag contains 5 red and 3 blue marbles. What is the probability "
        "of drawing a red marble?",
        context="",
        expected_answer="5/8",
        expected_keywords=["probability", "favorable", "total", "outcomes"],
        difficulty=2,
        steps_hint=3,
        perspectives=["counting", "probability formula"],
    ),
    ReasoningProblem(
        id="math_013",
        problem_type=ProblemType.MATH,
        question="What is the greatest common divisor (GCD) of 24 and 36?",
        context="",
        expected_answer="12",
        expected_keywords=["gcd", "divisor", "factor", "common"],
        difficulty=2,
        steps_hint=4,
        perspectives=["factorization", "euclidean algorithm"],
    ),
    ReasoningProblem(
        id="math_014",
        problem_type=ProblemType.MATH,
        question="Convert 3/4 to a percentage.",
        context="",
        expected_answer="75",
        expected_keywords=["fraction", "decimal", "percent", "multiply", "100"],
        difficulty=1,
        steps_hint=2,
        perspectives=["decimal conversion", "percentage"],
    ),
    ReasoningProblem(
        id="math_015",
        problem_type=ProblemType.MATH,
        question="A shirt originally costs $40. It is on sale for 25% off. What is the sale price?",
        context="",
        expected_answer="30",
        expected_keywords=["discount", "percent", "subtract", "original"],
        difficulty=2,
        steps_hint=3,
        perspectives=["discount calculation", "subtraction"],
    ),
    ReasoningProblem(
        id="math_016",
        problem_type=ProblemType.MATH,
        question="What is the least common multiple (LCM) of 4 and 6?",
        context="",
        expected_answer="12",
        expected_keywords=["lcm", "multiple", "common", "smallest"],
        difficulty=2,
        steps_hint=3,
        perspectives=["listing multiples", "formula"],
    ),
    ReasoningProblem(
        id="math_017",
        problem_type=ProblemType.MATH,
        question="If f(x) = 3x + 2, what is f(4)?",
        context="",
        expected_answer="14",
        expected_keywords=["function", "substitute", "evaluate"],
        difficulty=1,
        steps_hint=2,
        perspectives=["substitution", "arithmetic"],
    ),
    ReasoningProblem(
        id="math_018",
        problem_type=ProblemType.MATH,
        question="What is the volume of a cube with edge length 5 cm?",
        context="",
        expected_answer="125",
        expected_keywords=["volume", "cube", "edge", "power", "3"],
        difficulty=1,
        steps_hint=2,
        perspectives=["formula", "calculation"],
    ),
    ReasoningProblem(
        id="math_019",
        problem_type=ProblemType.MATH,
        question="Simplify: (4 + 6) × 3 - 8 ÷ 2",
        context="",
        expected_answer="26",
        expected_keywords=["order", "operations", "parentheses", "multiply", "divide"],
        difficulty=2,
        steps_hint=4,
        perspectives=["PEMDAS", "step-by-step"],
    ),
    ReasoningProblem(
        id="math_020",
        problem_type=ProblemType.MATH,
        question="A triangle has sides of length 3, 4, and 5. Is it a right triangle?",
        context="",
        expected_answer="yes",
        expected_keywords=["pythagorean", "theorem", "square", "sum", "right"],
        difficulty=2,
        steps_hint=3,
        perspectives=["pythagorean theorem", "verification"],
    ),
    ReasoningProblem(
        id="math_021",
        problem_type=ProblemType.MATH,
        question="What is 45 divided by 0.5?",
        context="",
        expected_answer="90",
        expected_keywords=["divide", "decimal", "multiply", "reciprocal"],
        difficulty=2,
        steps_hint=2,
        perspectives=["division", "reciprocal multiplication"],
    ),
    ReasoningProblem(
        id="math_022",
        problem_type=ProblemType.MATH,
        question="The mean of 5 numbers is 12. What is their sum?",
        context="",
        expected_answer="60",
        expected_keywords=["mean", "average", "sum", "multiply", "count"],
        difficulty=2,
        steps_hint=2,
        perspectives=["mean formula", "algebra"],
    ),
    ReasoningProblem(
        id="math_023",
        problem_type=ProblemType.MATH,
        question="What is the slope of the line passing through points (2, 3) and (6, 11)?",
        context="",
        expected_answer="2",
        expected_keywords=["slope", "rise", "run", "difference", "divide"],
        difficulty=2,
        steps_hint=3,
        perspectives=["slope formula", "calculation"],
    ),
    ReasoningProblem(
        id="math_024",
        problem_type=ProblemType.MATH,
        question="How many seconds are in 2.5 hours?",
        context="",
        expected_answer="9000",
        expected_keywords=["hours", "minutes", "seconds", "multiply", "60"],
        difficulty=1,
        steps_hint=3,
        perspectives=["unit conversion", "multiplication"],
    ),
    ReasoningProblem(
        id="math_025",
        problem_type=ProblemType.MATH,
        question="What is the square root of 144?",
        context="",
        expected_answer="12",
        expected_keywords=["square", "root", "factor"],
        difficulty=1,
        steps_hint=2,
        perspectives=["perfect square", "factorization"],
    ),
    # ==========================================================================
    # LOGIC PROBLEMS (25 total) - benefit from systematic exploration
    # ==========================================================================
    ReasoningProblem(
        id="logic_001",
        problem_type=ProblemType.LOGIC,
        question="If all roses are flowers, and some flowers fade quickly, "
        "can we conclude that some roses fade quickly?",
        context="This is a classical syllogism problem.",
        expected_answer="no",
        expected_keywords=["all", "some", "syllogism", "conclusion", "invalid", "subset"],
        difficulty=3,
        steps_hint=4,
        perspectives=["premise analysis", "logical validity", "counterexample"],
    ),
    ReasoningProblem(
        id="logic_002",
        problem_type=ProblemType.LOGIC,
        question="Three friends (Alice, Bob, Carol) each have a different pet "
        "(cat, dog, fish). Alice doesn't have a dog. Bob doesn't have a cat or fish. "
        "What pet does each person have?",
        context="",
        expected_answer="alice:cat, bob:dog, carol:fish",
        expected_keywords=["alice", "bob", "carol", "cat", "dog", "fish", "eliminate"],
        difficulty=2,
        steps_hint=5,
        perspectives=["constraint analysis", "elimination", "verification"],
    ),
    ReasoningProblem(
        id="logic_003",
        problem_type=ProblemType.LOGIC,
        question="If it rains, the ground is wet. The ground is wet. Did it rain?",
        context="",
        expected_answer="not necessarily",
        expected_keywords=["affirming", "consequent", "fallacy", "other causes"],
        difficulty=2,
        steps_hint=3,
        perspectives=["logical structure", "fallacy identification"],
    ),
    ReasoningProblem(
        id="logic_004",
        problem_type=ProblemType.LOGIC,
        question="All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
        context="",
        expected_answer="yes",
        expected_keywords=["syllogism", "valid", "deduction", "mammals"],
        difficulty=1,
        steps_hint=2,
        perspectives=["premise", "conclusion"],
    ),
    ReasoningProblem(
        id="logic_005",
        problem_type=ProblemType.LOGIC,
        question="In a race, Alex finished before Ben. Ben finished before Carol. "
        "Did Alex finish before Carol?",
        context="",
        expected_answer="yes",
        expected_keywords=["transitive", "order", "before", "relation"],
        difficulty=1,
        steps_hint=2,
        perspectives=["transitivity", "ordering"],
    ),
    ReasoningProblem(
        id="logic_006",
        problem_type=ProblemType.LOGIC,
        question="If P implies Q, and Q implies R, does P imply R?",
        context="",
        expected_answer="yes",
        expected_keywords=["transitive", "implication", "chain", "hypothetical"],
        difficulty=2,
        steps_hint=3,
        perspectives=["logical chain", "transitivity"],
    ),
    ReasoningProblem(
        id="logic_007",
        problem_type=ProblemType.LOGIC,
        question="No reptiles are mammals. Some snakes are reptiles. Are some snakes mammals?",
        context="",
        expected_answer="no",
        expected_keywords=["syllogism", "exclusion", "valid", "none"],
        difficulty=2,
        steps_hint=3,
        perspectives=["categorical logic", "exclusion"],
    ),
    ReasoningProblem(
        id="logic_008",
        problem_type=ProblemType.LOGIC,
        question="Either the light is on or it is off. The light is not on. What can we conclude?",
        context="",
        expected_answer="light is off",
        expected_keywords=["disjunction", "elimination", "or", "not"],
        difficulty=1,
        steps_hint=2,
        perspectives=["disjunctive syllogism"],
    ),
    ReasoningProblem(
        id="logic_009",
        problem_type=ProblemType.LOGIC,
        question="If A and B are both true, and A is true, is B necessarily true?",
        context="",
        expected_answer="not necessarily",
        expected_keywords=["conjunction", "independent", "given", "information"],
        difficulty=2,
        steps_hint=3,
        perspectives=["logical analysis", "given information"],
    ),
    ReasoningProblem(
        id="logic_010",
        problem_type=ProblemType.LOGIC,
        question="Tom is taller than Jerry. Jerry is taller than Spike. Who is the shortest?",
        context="",
        expected_answer="spike",
        expected_keywords=["transitive", "comparison", "order", "shortest"],
        difficulty=1,
        steps_hint=2,
        perspectives=["ordering", "transitivity"],
    ),
    ReasoningProblem(
        id="logic_011",
        problem_type=ProblemType.LOGIC,
        question="If not P, then not Q. Q is true. What can we say about P?",
        context="",
        expected_answer="p is true",
        expected_keywords=["contrapositive", "modus tollens", "negation"],
        difficulty=3,
        steps_hint=4,
        perspectives=["contrapositive", "logical equivalence"],
    ),
    ReasoningProblem(
        id="logic_012",
        problem_type=ProblemType.LOGIC,
        question="A says 'I always lie.' Is A telling the truth?",
        context="This is a classic liar's paradox.",
        expected_answer="paradox",
        expected_keywords=["paradox", "self-reference", "contradiction", "liar"],
        difficulty=3,
        steps_hint=4,
        perspectives=["self-reference", "contradiction analysis"],
    ),
    ReasoningProblem(
        id="logic_013",
        problem_type=ProblemType.LOGIC,
        question="Four people need to cross a bridge at night with one flashlight. "
        "Only two can cross at a time. A takes 1 min, B takes 2 min, C takes 5 min, "
        "D takes 10 min. What is the minimum time for all to cross?",
        context="When two people cross, they travel at the slower person's pace.",
        expected_answer="17",
        expected_keywords=["optimize", "bridge", "flashlight", "strategy", "minimum"],
        difficulty=4,
        steps_hint=6,
        perspectives=["greedy", "optimal pairing", "backtracking"],
    ),
    ReasoningProblem(
        id="logic_014",
        problem_type=ProblemType.LOGIC,
        question="You have 3 boxes: one with only apples, one with only oranges, "
        "one with both. All labels are wrong. You pick one fruit from one box. "
        "Can you correctly label all boxes?",
        context="",
        expected_answer="yes",
        expected_keywords=["label", "wrong", "deduce", "pick", "mixed"],
        difficulty=3,
        steps_hint=5,
        perspectives=["information", "deduction", "constraints"],
    ),
    ReasoningProblem(
        id="logic_015",
        problem_type=ProblemType.LOGIC,
        question="If some A are B, and some B are C, can we conclude some A are C?",
        context="",
        expected_answer="no",
        expected_keywords=["syllogism", "invalid", "some", "overlap"],
        difficulty=3,
        steps_hint=4,
        perspectives=["venn diagram", "counterexample"],
    ),
    ReasoningProblem(
        id="logic_016",
        problem_type=ProblemType.LOGIC,
        question="A farmer has chickens and rabbits. There are 20 heads and 56 legs. "
        "How many chickens are there?",
        context="Chickens have 2 legs, rabbits have 4 legs.",
        expected_answer="12",
        expected_keywords=["equation", "system", "legs", "heads", "solve"],
        difficulty=2,
        steps_hint=4,
        perspectives=["algebra", "system of equations"],
    ),
    ReasoningProblem(
        id="logic_017",
        problem_type=ProblemType.LOGIC,
        question="There are 5 houses in a row. The red house is to the left of the "
        "blue house. The green house is in the middle. Where is the red house?",
        context="",
        expected_answer="first or second",
        expected_keywords=["position", "left", "middle", "constraint"],
        difficulty=2,
        steps_hint=4,
        perspectives=["constraint propagation", "positioning"],
    ),
    ReasoningProblem(
        id="logic_018",
        problem_type=ProblemType.LOGIC,
        question="If all dogs bark, and Rex is a dog, does Rex bark?",
        context="",
        expected_answer="yes",
        expected_keywords=["universal", "instantiation", "modus ponens"],
        difficulty=1,
        steps_hint=2,
        perspectives=["deduction", "universal quantifier"],
    ),
    ReasoningProblem(
        id="logic_019",
        problem_type=ProblemType.LOGIC,
        question="A clock shows 3:15. What is the angle between the hour and minute hands?",
        context="",
        expected_answer="7.5",
        expected_keywords=["angle", "clock", "hour", "minute", "degrees"],
        difficulty=3,
        steps_hint=4,
        perspectives=["hour hand position", "minute hand position", "difference"],
    ),
    ReasoningProblem(
        id="logic_020",
        problem_type=ProblemType.LOGIC,
        question="In a group of 30 people, everyone shakes hands with everyone else "
        "exactly once. How many handshakes occur?",
        context="",
        expected_answer="435",
        expected_keywords=["combination", "pairs", "n choose 2", "formula"],
        difficulty=2,
        steps_hint=3,
        perspectives=["combinatorics", "formula"],
    ),
    ReasoningProblem(
        id="logic_021",
        problem_type=ProblemType.LOGIC,
        question="If Monday is two days after the day before yesterday, what day is today?",
        context="",
        expected_answer="wednesday",
        expected_keywords=["day", "before", "after", "yesterday"],
        difficulty=2,
        steps_hint=4,
        perspectives=["timeline", "working backwards"],
    ),
    ReasoningProblem(
        id="logic_022",
        problem_type=ProblemType.LOGIC,
        question="A is the father of B. B is the father of C. What is A to C?",
        context="",
        expected_answer="grandfather",
        expected_keywords=["father", "relation", "generation", "family"],
        difficulty=1,
        steps_hint=2,
        perspectives=["family tree", "transitivity"],
    ),
    ReasoningProblem(
        id="logic_023",
        problem_type=ProblemType.LOGIC,
        question="If P or Q is true, and P is false, what is Q?",
        context="",
        expected_answer="true",
        expected_keywords=["disjunction", "or", "elimination"],
        difficulty=1,
        steps_hint=2,
        perspectives=["disjunctive syllogism"],
    ),
    ReasoningProblem(
        id="logic_024",
        problem_type=ProblemType.LOGIC,
        question="There are 3 light switches outside a room with 3 bulbs inside. "
        "You can only enter the room once. How do you determine which switch "
        "controls which bulb?",
        context="",
        expected_answer="heat method",
        expected_keywords=["heat", "on", "off", "warm", "cold", "touch"],
        difficulty=3,
        steps_hint=4,
        perspectives=["physical properties", "information gathering"],
    ),
    ReasoningProblem(
        id="logic_025",
        problem_type=ProblemType.LOGIC,
        question="Two statements: 'All cats are animals' and 'Some animals are not cats.' "
        "Can both be true simultaneously?",
        context="",
        expected_answer="yes",
        expected_keywords=["consistent", "subset", "some", "all", "animals"],
        difficulty=2,
        steps_hint=3,
        perspectives=["set theory", "consistency"],
    ),
    # ==========================================================================
    # MULTI-HOP REASONING (25 total) - requires connecting information
    # ==========================================================================
    ReasoningProblem(
        id="multihop_001",
        problem_type=ProblemType.MULTI_HOP,
        question="Who was the president of the country where the Eiffel Tower is located "
        "when it was built?",
        context="The Eiffel Tower was built in Paris, France in 1889. "
        "Sadi Carnot was the President of France from 1887 to 1894.",
        expected_answer="sadi carnot",
        expected_keywords=["eiffel", "paris", "france", "1889", "carnot", "president"],
        difficulty=2,
        steps_hint=3,
        perspectives=["location identification", "time period", "leader lookup"],
    ),
    ReasoningProblem(
        id="multihop_002",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the capital of the country that hosted the 2016 Summer Olympics?",
        context="The 2016 Summer Olympics were held in Rio de Janeiro. "
        "Rio de Janeiro is a city in Brazil. Brasília is the capital of Brazil.",
        expected_answer="brasilia",
        expected_keywords=["olympics", "rio", "brazil", "capital", "brasilia"],
        difficulty=2,
        steps_hint=3,
        perspectives=["event location", "country identification", "capital lookup"],
    ),
    ReasoningProblem(
        id="multihop_003",
        problem_type=ProblemType.MULTI_HOP,
        question="What language is spoken in the birthplace of Mozart?",
        context="Mozart was born in Salzburg. Salzburg is a city in Austria. "
        "The official language of Austria is German.",
        expected_answer="german",
        expected_keywords=["mozart", "salzburg", "austria", "german", "language"],
        difficulty=2,
        steps_hint=3,
        perspectives=["birthplace", "country", "language"],
    ),
    ReasoningProblem(
        id="multihop_004",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the currency of the country where Toyota is headquartered?",
        context="Toyota is headquartered in Toyota City. Toyota City is in Japan. "
        "The currency of Japan is the Yen.",
        expected_answer="yen",
        expected_keywords=["toyota", "japan", "currency", "yen", "headquarter"],
        difficulty=2,
        steps_hint=3,
        perspectives=["company location", "country", "currency"],
    ),
    ReasoningProblem(
        id="multihop_005",
        problem_type=ProblemType.MULTI_HOP,
        question="In what continent is the tallest mountain located?",
        context="Mount Everest is the tallest mountain. Mount Everest is in the Himalayas. "
        "The Himalayas are in Asia.",
        expected_answer="asia",
        expected_keywords=["everest", "tallest", "himalayas", "asia", "continent"],
        difficulty=1,
        steps_hint=3,
        perspectives=["tallest mountain", "location", "continent"],
    ),
    ReasoningProblem(
        id="multihop_006",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the population of the city where the Statue of Liberty is located?",
        context="The Statue of Liberty is in New York City. "
        "New York City has a population of approximately 8.3 million.",
        expected_answer="8.3 million",
        expected_keywords=["statue", "liberty", "new york", "population", "million"],
        difficulty=2,
        steps_hint=2,
        perspectives=["landmark location", "city statistics"],
    ),
    ReasoningProblem(
        id="multihop_007",
        problem_type=ProblemType.MULTI_HOP,
        question="What ocean borders the west coast of the country where sushi originated?",
        context="Sushi originated in Japan. Japan is bordered by the Pacific Ocean to the east "
        "and the Sea of Japan to the west.",
        expected_answer="sea of japan",
        expected_keywords=["sushi", "japan", "ocean", "west", "sea"],
        difficulty=2,
        steps_hint=3,
        perspectives=["origin country", "geography", "ocean"],
    ),
    ReasoningProblem(
        id="multihop_008",
        problem_type=ProblemType.MULTI_HOP,
        question="Who directed the movie that won Best Picture at the 2020 Oscars?",
        context="Parasite won Best Picture at the 2020 Oscars. "
        "Parasite was directed by Bong Joon-ho.",
        expected_answer="bong joon-ho",
        expected_keywords=["parasite", "oscars", "best picture", "director", "bong"],
        difficulty=2,
        steps_hint=2,
        perspectives=["award winner", "director"],
    ),
    ReasoningProblem(
        id="multihop_009",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the national animal of the country where the Taj Mahal is located?",
        context="The Taj Mahal is in Agra. Agra is in India. "
        "The national animal of India is the Bengal Tiger.",
        expected_answer="bengal tiger",
        expected_keywords=["taj mahal", "india", "national", "animal", "tiger"],
        difficulty=2,
        steps_hint=3,
        perspectives=["landmark country", "national symbol"],
    ),
    ReasoningProblem(
        id="multihop_010",
        problem_type=ProblemType.MULTI_HOP,
        question="What year did the inventor of the telephone die?",
        context="Alexander Graham Bell invented the telephone. Alexander Graham Bell died in 1922.",
        expected_answer="1922",
        expected_keywords=["bell", "telephone", "inventor", "died", "1922"],
        difficulty=1,
        steps_hint=2,
        perspectives=["inventor identification", "death year"],
    ),
    ReasoningProblem(
        id="multihop_011",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the time zone of the headquarters of the company that makes the iPhone?",
        context="Apple makes the iPhone. Apple is headquartered in Cupertino, California. "
        "California is in the Pacific Time Zone.",
        expected_answer="pacific",
        expected_keywords=["apple", "iphone", "cupertino", "california", "pacific", "time"],
        difficulty=2,
        steps_hint=3,
        perspectives=["company", "location", "time zone"],
    ),
    ReasoningProblem(
        id="multihop_012",
        problem_type=ProblemType.MULTI_HOP,
        question="How many stars are on the flag of the country where Amazon was founded?",
        context="Amazon was founded in the United States. "
        "The US flag has 50 stars representing the 50 states.",
        expected_answer="50",
        expected_keywords=["amazon", "united states", "flag", "stars", "50"],
        difficulty=2,
        steps_hint=2,
        perspectives=["company origin", "flag details"],
    ),
    ReasoningProblem(
        id="multihop_013",
        problem_type=ProblemType.MULTI_HOP,
        question="What sport is most popular in the country that invented pizza?",
        context=(
            "Pizza was invented in Italy. The most popular sport in Italy is football (soccer)."
        ),
        expected_answer="football",
        expected_keywords=["pizza", "italy", "sport", "football", "soccer"],
        difficulty=2,
        steps_hint=2,
        perspectives=["origin country", "popular sport"],
    ),
    ReasoningProblem(
        id="multihop_014",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the largest city in the state where Harvard University is located?",
        context="Harvard University is in Cambridge. Cambridge is in Massachusetts. "
        "Boston is the largest city in Massachusetts.",
        expected_answer="boston",
        expected_keywords=["harvard", "cambridge", "massachusetts", "boston", "largest"],
        difficulty=2,
        steps_hint=3,
        perspectives=["university location", "state", "largest city"],
    ),
    ReasoningProblem(
        id="multihop_015",
        problem_type=ProblemType.MULTI_HOP,
        question="In what decade was the author of '1984' born?",
        context="George Orwell wrote '1984'. George Orwell was born in 1903.",
        expected_answer="1900s",
        expected_keywords=["orwell", "1984", "born", "1903", "decade"],
        difficulty=2,
        steps_hint=2,
        perspectives=["author identification", "birth year", "decade"],
    ),
    ReasoningProblem(
        id="multihop_016",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the official religion of the country where mozzarella cheese originated?",
        context="Mozzarella originated in Italy. The predominant religion in Italy is "
        "Roman Catholicism.",
        expected_answer="roman catholicism",
        expected_keywords=["mozzarella", "italy", "religion", "catholic"],
        difficulty=2,
        steps_hint=2,
        perspectives=["origin country", "religion"],
    ),
    ReasoningProblem(
        id="multihop_017",
        problem_type=ProblemType.MULTI_HOP,
        question="Who was the first president of the country that landed humans on the moon?",
        context="The United States landed humans on the moon in 1969. "
        "George Washington was the first president of the United States.",
        expected_answer="george washington",
        expected_keywords=["moon", "united states", "president", "washington", "first"],
        difficulty=2,
        steps_hint=2,
        perspectives=["achievement country", "first president"],
    ),
    ReasoningProblem(
        id="multihop_018",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the national flower of the country that hosts Oktoberfest?",
        context="Oktoberfest is held in Munich. Munich is in Germany. "
        "The national flower of Germany is the cornflower.",
        expected_answer="cornflower",
        expected_keywords=["oktoberfest", "munich", "germany", "flower", "cornflower"],
        difficulty=2,
        steps_hint=3,
        perspectives=["event location", "country", "national symbol"],
    ),
    ReasoningProblem(
        id="multihop_019",
        problem_type=ProblemType.MULTI_HOP,
        question="What hemisphere is the country that produces the most coffee located in?",
        context="Brazil produces the most coffee in the world. "
        "Brazil is located in the Southern Hemisphere.",
        expected_answer="southern",
        expected_keywords=["coffee", "brazil", "hemisphere", "southern"],
        difficulty=1,
        steps_hint=2,
        perspectives=["top producer", "geographic location"],
    ),
    ReasoningProblem(
        id="multihop_020",
        problem_type=ProblemType.MULTI_HOP,
        question="What century was the university attended by Mark Zuckerberg founded?",
        context="Mark Zuckerberg attended Harvard University. Harvard was founded in 1636.",
        expected_answer="17th",
        expected_keywords=["zuckerberg", "harvard", "founded", "1636", "century"],
        difficulty=2,
        steps_hint=3,
        perspectives=["person's university", "founding year", "century"],
    ),
    ReasoningProblem(
        id="multihop_021",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the GDP per capita of the country where IKEA was founded?",
        context="IKEA was founded in Sweden. Sweden has a GDP per capita of approximately $52,000.",
        expected_answer="52000",
        expected_keywords=["ikea", "sweden", "gdp", "capita", "52000"],
        difficulty=2,
        steps_hint=2,
        perspectives=["company origin", "economic indicator"],
    ),
    ReasoningProblem(
        id="multihop_022",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the driving side in the country where Sherlock Holmes lived?",
        context="Sherlock Holmes lived in London. London is in the United Kingdom. "
        "The UK drives on the left side of the road.",
        expected_answer="left",
        expected_keywords=["sherlock", "london", "uk", "driving", "left"],
        difficulty=2,
        steps_hint=3,
        perspectives=["character location", "country", "driving rules"],
    ),
    ReasoningProblem(
        id="multihop_023",
        problem_type=ProblemType.MULTI_HOP,
        question="What mountain range is in the country where the Mona Lisa is displayed?",
        context="The Mona Lisa is displayed at the Louvre. The Louvre is in Paris, France. "
        "The Alps mountain range passes through France.",
        expected_answer="alps",
        expected_keywords=["mona lisa", "louvre", "france", "alps", "mountain"],
        difficulty=2,
        steps_hint=3,
        perspectives=["artwork location", "country", "geography"],
    ),
    ReasoningProblem(
        id="multihop_024",
        problem_type=ProblemType.MULTI_HOP,
        question="What type of government does the country with the Great Barrier Reef have?",
        context="The Great Barrier Reef is in Australia. "
        "Australia is a federal parliamentary constitutional monarchy.",
        expected_answer="constitutional monarchy",
        expected_keywords=["reef", "australia", "government", "monarchy", "parliamentary"],
        difficulty=2,
        steps_hint=2,
        perspectives=["landmark location", "government type"],
    ),
    ReasoningProblem(
        id="multihop_025",
        problem_type=ProblemType.MULTI_HOP,
        question="What is the main export of the country where Samsung is headquartered?",
        context="Samsung is headquartered in South Korea. "
        "South Korea's main export is semiconductors and electronics.",
        expected_answer="semiconductors",
        expected_keywords=["samsung", "korea", "export", "semiconductors", "electronics"],
        difficulty=2,
        steps_hint=2,
        perspectives=["company location", "economic data"],
    ),
    # ==========================================================================
    # ANALYSIS PROBLEMS (25 total) - benefit from multiple perspectives
    # ==========================================================================
    ReasoningProblem(
        id="analysis_001",
        problem_type=ProblemType.ANALYSIS,
        question="Should a small startup use microservices or a monolith architecture?",
        context="The startup has 3 developers, expects moderate growth, "
        "and needs to ship an MVP in 2 months.",
        expected_answer="monolith",
        expected_keywords=["team size", "complexity", "deployment", "mvp", "speed", "overhead"],
        difficulty=3,
        steps_hint=5,
        perspectives=["team capacity", "time constraints", "operational complexity", "scaling"],
    ),
    ReasoningProblem(
        id="analysis_002",
        problem_type=ProblemType.ANALYSIS,
        question="Is Python or Rust better for building a high-frequency trading system?",
        context="The system needs to process 100,000 orders per second with "
        "sub-millisecond latency. Reliability is critical.",
        expected_answer="rust",
        expected_keywords=["latency", "performance", "memory", "safety", "speed", "gc"],
        difficulty=3,
        steps_hint=5,
        perspectives=["performance requirements", "safety guarantees", "development speed"],
    ),
    ReasoningProblem(
        id="analysis_003",
        problem_type=ProblemType.ANALYSIS,
        question="Should a company use cloud hosting or on-premise servers?",
        context="A healthcare company with strict data regulations, 50 employees, "
        "variable workload, and limited IT staff.",
        expected_answer="cloud",
        expected_keywords=["compliance", "scalability", "cost", "maintenance", "expertise"],
        difficulty=3,
        steps_hint=5,
        perspectives=["regulatory compliance", "scalability", "cost", "staffing"],
    ),
    ReasoningProblem(
        id="analysis_004",
        problem_type=ProblemType.ANALYSIS,
        question="Is it better to buy or rent a house?",
        context="A young professional in a major city, planning to stay 2-3 years, "
        "has a 10% down payment saved, housing prices are at all-time highs.",
        expected_answer="rent",
        expected_keywords=["duration", "mobility", "market", "equity", "transaction costs"],
        difficulty=3,
        steps_hint=5,
        perspectives=["financial", "flexibility", "market timing", "opportunity cost"],
    ),
    ReasoningProblem(
        id="analysis_005",
        problem_type=ProblemType.ANALYSIS,
        question="Should a restaurant switch to all-digital ordering?",
        context="Family restaurant with older clientele, 20 tables, "
        "average ticket $35, in a suburban area.",
        expected_answer="no",
        expected_keywords=["demographic", "customer experience", "cost", "adoption"],
        difficulty=2,
        steps_hint=4,
        perspectives=["customer preferences", "implementation cost", "efficiency gains"],
    ),
    ReasoningProblem(
        id="analysis_006",
        problem_type=ProblemType.ANALYSIS,
        question="Should a student take on debt for an MBA?",
        context="30-year-old software engineer earning $150k, considering a $200k MBA "
        "at a top-10 school, wants to move into product management.",
        expected_answer="depends",
        expected_keywords=["roi", "career", "salary", "opportunity cost", "network"],
        difficulty=3,
        steps_hint=5,
        perspectives=["financial ROI", "career goals", "opportunity cost", "alternatives"],
    ),
    ReasoningProblem(
        id="analysis_007",
        problem_type=ProblemType.ANALYSIS,
        question="Is electric car or hybrid better for a daily commuter?",
        context="50-mile round trip daily commute, garage with charging capability, "
        "lives in a cold climate, budget of $40k.",
        expected_answer="electric",
        expected_keywords=["range", "charging", "cost", "climate", "efficiency"],
        difficulty=2,
        steps_hint=4,
        perspectives=["range needs", "infrastructure", "climate impact", "total cost"],
    ),
    ReasoningProblem(
        id="analysis_008",
        problem_type=ProblemType.ANALYSIS,
        question="Should a company adopt a 4-day work week?",
        context="Tech company with 100 employees, project-based work, "
        "competing for talent in a hot job market.",
        expected_answer="yes",
        expected_keywords=["productivity", "retention", "talent", "burnout", "culture"],
        difficulty=3,
        steps_hint=5,
        perspectives=["productivity", "talent acquisition", "employee wellbeing", "costs"],
    ),
    ReasoningProblem(
        id="analysis_009",
        problem_type=ProblemType.ANALYSIS,
        question="Is NoSQL or SQL better for an e-commerce product catalog?",
        context="E-commerce site with 100k products, complex product attributes that "
        "vary by category, need for fast reads, moderate write volume.",
        expected_answer="nosql",
        expected_keywords=["schema", "flexibility", "performance", "queries", "scale"],
        difficulty=3,
        steps_hint=4,
        perspectives=["data model", "query patterns", "scalability", "consistency"],
    ),
    ReasoningProblem(
        id="analysis_010",
        problem_type=ProblemType.ANALYSIS,
        question="Should a remote worker move to a lower cost-of-living area?",
        context="Software developer earning $180k fully remote, currently in SF paying "
        "$3.5k rent, single with no family obligations.",
        expected_answer="yes",
        expected_keywords=["cost", "savings", "quality of life", "social", "career"],
        difficulty=2,
        steps_hint=4,
        perspectives=["financial", "lifestyle", "career impact", "social factors"],
    ),
    ReasoningProblem(
        id="analysis_011",
        problem_type=ProblemType.ANALYSIS,
        question="Should a mobile app use native or cross-platform development?",
        context="Startup building a social media app, need to launch on iOS and Android "
        "in 6 months, team of 4 developers with React experience.",
        expected_answer="cross-platform",
        expected_keywords=["time", "team", "maintenance", "performance", "cost"],
        difficulty=3,
        steps_hint=4,
        perspectives=["development speed", "team skills", "maintenance", "user experience"],
    ),
    ReasoningProblem(
        id="analysis_012",
        problem_type=ProblemType.ANALYSIS,
        question="Is it better to invest in index funds or individual stocks?",
        context="Beginner investor with $10k to invest, 30-year time horizon, "
        "limited time to research, moderate risk tolerance.",
        expected_answer="index funds",
        expected_keywords=["diversification", "fees", "time", "risk", "returns"],
        difficulty=2,
        steps_hint=4,
        perspectives=["risk", "time commitment", "historical returns", "costs"],
    ),
    ReasoningProblem(
        id="analysis_013",
        problem_type=ProblemType.ANALYSIS,
        question="Should a small business accept cryptocurrency payments?",
        context="Coffee shop in a tech hub, average transaction $8, "
        "customers are tech-savvy millennials.",
        expected_answer="no",
        expected_keywords=["volatility", "fees", "adoption", "complexity", "value"],
        difficulty=2,
        steps_hint=4,
        perspectives=["transaction costs", "volatility risk", "customer demand", "complexity"],
    ),
    ReasoningProblem(
        id="analysis_014",
        problem_type=ProblemType.ANALYSIS,
        question="Should a company build or buy a CRM system?",
        context="Mid-size company with 500 employees, unique sales process, "
        "10-person engineering team, budget of $500k.",
        expected_answer="buy",
        expected_keywords=["cost", "time", "maintenance", "features", "customization"],
        difficulty=3,
        steps_hint=5,
        perspectives=["total cost", "time to value", "maintenance", "customization needs"],
    ),
    ReasoningProblem(
        id="analysis_015",
        problem_type=ProblemType.ANALYSIS,
        question="Is GraphQL or REST better for a mobile app backend?",
        context="Mobile app with complex nested data, bandwidth-constrained users, "
        "rapidly evolving data requirements.",
        expected_answer="graphql",
        expected_keywords=["bandwidth", "flexibility", "overfetching", "typing", "evolution"],
        difficulty=3,
        steps_hint=4,
        perspectives=["bandwidth efficiency", "flexibility", "developer experience", "caching"],
    ),
    ReasoningProblem(
        id="analysis_016",
        problem_type=ProblemType.ANALYSIS,
        question=(
            "Should a graduate take a high-paying job they dislike or lower-paying passion job?"
        ),
        context="New grad with $50k student loans, high-paying job offers $120k in finance, "
        "passion job offers $55k in non-profit sector.",
        expected_answer="high-paying",
        expected_keywords=["debt", "compound", "options", "burnout", "financial freedom"],
        difficulty=3,
        steps_hint=5,
        perspectives=["debt payoff", "career trajectory", "mental health", "long-term options"],
    ),
    ReasoningProblem(
        id="analysis_017",
        problem_type=ProblemType.ANALYSIS,
        question="Should a company use serverless or containers for their backend?",
        context="API service with spiky traffic (10 RPS baseline, 1000 RPS peaks), "
        "cost-sensitive startup, team familiar with AWS.",
        expected_answer="serverless",
        expected_keywords=["cost", "scaling", "cold start", "complexity", "traffic"],
        difficulty=3,
        steps_hint=4,
        perspectives=["cost model", "scaling characteristics", "operational overhead"],
    ),
    ReasoningProblem(
        id="analysis_018",
        problem_type=ProblemType.ANALYSIS,
        question="Is TypeScript or JavaScript better for a new web project?",
        context="Team of 5 developers, long-term project expected to grow, "
        "some developers unfamiliar with TypeScript.",
        expected_answer="typescript",
        expected_keywords=["types", "maintainability", "refactoring", "learning", "tooling"],
        difficulty=2,
        steps_hint=4,
        perspectives=["type safety", "team learning curve", "long-term maintainability"],
    ),
    ReasoningProblem(
        id="analysis_019",
        problem_type=ProblemType.ANALYSIS,
        question="Should a company allow full remote work post-pandemic?",
        context="Consulting firm with 200 employees, client-facing work, "
        "expensive downtown office lease expiring in 6 months.",
        expected_answer="hybrid",
        expected_keywords=["collaboration", "culture", "cost", "flexibility", "client"],
        difficulty=3,
        steps_hint=5,
        perspectives=[
            "collaboration needs",
            "cost savings",
            "talent retention",
            "client expectations",
        ],
    ),
    ReasoningProblem(
        id="analysis_020",
        problem_type=ProblemType.ANALYSIS,
        question="Is Kubernetes overkill for a small team?",
        context="3-person startup, single application, 1000 daily users, "
        "no dedicated DevOps person.",
        expected_answer="yes",
        expected_keywords=["complexity", "overhead", "team size", "alternatives", "scale"],
        difficulty=2,
        steps_hint=4,
        perspectives=["operational complexity", "team expertise", "actual needs", "alternatives"],
    ),
    ReasoningProblem(
        id="analysis_021",
        problem_type=ProblemType.ANALYSIS,
        question="Should a company use open source or commercial database?",
        context="Financial services company, critical data, 24/7 uptime requirement, "
        "compliance needs, in-house DBA team.",
        expected_answer="commercial",
        expected_keywords=["support", "compliance", "reliability", "cost", "expertise"],
        difficulty=3,
        steps_hint=4,
        perspectives=["support needs", "compliance", "total cost", "risk"],
    ),
    ReasoningProblem(
        id="analysis_022",
        problem_type=ProblemType.ANALYSIS,
        question="Is machine learning needed for this recommendation system?",
        context="E-commerce site with 1000 products, 10k monthly users, "
        "simple product categories, limited data science expertise.",
        expected_answer="no",
        expected_keywords=["complexity", "data", "alternatives", "rules", "cold start"],
        difficulty=2,
        steps_hint=4,
        perspectives=["data availability", "complexity", "alternatives", "ROI"],
    ),
    ReasoningProblem(
        id="analysis_023",
        problem_type=ProblemType.ANALYSIS,
        question="Should a bootstrapped startup raise venture capital?",
        context="Profitable SaaS with $500k ARR, 20% MoM growth, founder wants to maintain "
        "control, market is competitive with well-funded players.",
        expected_answer="depends",
        expected_keywords=["control", "growth", "competition", "dilution", "runway"],
        difficulty=3,
        steps_hint=5,
        perspectives=["growth potential", "control", "market dynamics", "alternatives"],
    ),
    ReasoningProblem(
        id="analysis_024",
        problem_type=ProblemType.ANALYSIS,
        question="Is it better to use Redis or Memcached for caching?",
        context="Web application needing session storage and caching, "
        "some data structures like lists and sets needed, high availability required.",
        expected_answer="redis",
        expected_keywords=["data structures", "persistence", "availability", "features"],
        difficulty=2,
        steps_hint=4,
        perspectives=["feature needs", "persistence", "complexity", "performance"],
    ),
    ReasoningProblem(
        id="analysis_025",
        problem_type=ProblemType.ANALYSIS,
        question="Should a developer learn a new framework or deepen existing skills?",
        context="Mid-level React developer, 3 years experience, job market favors "
        "full-stack developers, interested in learning Rust.",
        expected_answer="deepen",
        expected_keywords=["depth", "marketability", "specialization", "breadth", "career"],
        difficulty=2,
        steps_hint=4,
        perspectives=["career goals", "market demand", "learning efficiency", "specialization"],
    ),
]

# =============================================================================
# HARD MODE PROBLEMS - Designed to trip up LLMs
# =============================================================================
# These problems have:
# - Ambiguous wording that requires careful reading
# - Multi-step math where intermediate errors compound
# - Logic puzzles with counterintuitive answers
# - Problems where the "obvious" answer is wrong

HARD_PROBLEMS: list[ReasoningProblem] = [
    # --------------------------------------------------------------------------
    # HARD MATH (10 problems) - Multi-step, error-prone calculations
    # --------------------------------------------------------------------------
    ReasoningProblem(
        id="hard_math_001",
        problem_type=ProblemType.MATH,
        question="A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. "
        "How much does the ball cost in cents?",
        context="",
        expected_answer="5",  # NOT 10! Common mistake.
        expected_keywords=["equation", "difference", "x", "1.00", "more than", "algebra"],
        difficulty=4,
        steps_hint=5,
        perspectives=["algebraic setup", "verification", "common mistake check"],
    ),
    ReasoningProblem(
        id="hard_math_002",
        problem_type=ProblemType.MATH,
        question="If it takes 5 machines 5 minutes to make 5 widgets, "
        "how many minutes would it take 100 machines to make 100 widgets?",
        context="",
        expected_answer="5",  # NOT 100! Each machine makes 1 widget in 5 min.
        expected_keywords=["rate", "per machine", "parallel", "independent", "simultaneous"],
        difficulty=4,
        steps_hint=4,
        perspectives=["rate analysis", "scaling", "parallelism"],
    ),
    ReasoningProblem(
        id="hard_math_003",
        problem_type=ProblemType.MATH,
        question="A lily pad doubles in size every day. If it takes 48 days to cover the lake, "
        "how many days does it take to cover half the lake?",
        context="",
        expected_answer="47",  # NOT 24! It doubles, so half is one day before full.
        expected_keywords=["doubling", "exponential", "half", "previous day", "growth"],
        difficulty=4,
        steps_hint=3,
        perspectives=["exponential growth", "working backwards"],
    ),
    ReasoningProblem(
        id="hard_math_004",
        problem_type=ProblemType.MATH,
        question="Three people check into a hotel room that costs $30. They each pay $10. "
        "Later the clerk realizes the room was only $25, so he gives $5 to the bellboy to return. "
        "The bellboy keeps $2 and gives $1 back to each person. "
        "Now each person paid $9 (total $27), plus the bellboy has $2. That's $29. "
        "Where is the missing dollar?",
        context="",
        expected_answer="nowhere",  # It's a misdirection - $27 includes the $2
        expected_keywords=["accounting", "misdirection", "includes", "hotel", "bellboy", "fallacy"],
        difficulty=5,
        steps_hint=6,
        perspectives=[
            "money flow tracking",
            "accounting verification",
            "logical fallacy detection",
        ],
    ),
    ReasoningProblem(
        id="hard_math_005",
        problem_type=ProblemType.MATH,
        question="You have a 3-gallon jug and a 5-gallon jug. "
        "How do you measure exactly 4 gallons?",
        context="You can fill jugs from a tap and pour between them.",
        expected_answer="6",  # 6 steps: fill 5, pour to 3, empty 3, pour 2, fill 5, pour 1
        expected_keywords=["fill", "pour", "empty", "steps", "remainder"],
        difficulty=4,
        steps_hint=7,
        perspectives=["state tracking", "step counting", "verification"],
    ),
    ReasoningProblem(
        id="hard_math_006",
        problem_type=ProblemType.MATH,
        question="A snail climbs 3 feet up a well during the day but slides back 2 feet at night. "
        "If the well is 30 feet deep, how many days does it take to escape?",
        context="The snail escapes when it reaches the top during the day.",
        expected_answer="28",  # NOT 30! On day 28, it reaches 30 and escapes before sliding back
        expected_keywords=["net progress", "climb", "slide", "escape", "final day", "boundary"],
        difficulty=4,
        steps_hint=5,
        perspectives=["daily progress", "boundary condition", "final day analysis"],
    ),
    ReasoningProblem(
        id="hard_math_007",
        problem_type=ProblemType.MATH,
        question="What is 0.1 + 0.2 in floating point arithmetic (IEEE 754)? "
        "Give the answer to 17 decimal places.",
        context="",
        expected_answer="0.30000000000000004",  # Famous floating point quirk
        expected_keywords=["floating point", "binary", "representation", "precision", "IEEE"],
        difficulty=5,
        steps_hint=4,
        perspectives=["binary representation", "precision limits"],
    ),
    ReasoningProblem(
        id="hard_math_008",
        problem_type=ProblemType.MATH,
        question="In a room of 23 people, what is the probability (as a percentage, rounded) "
        "that at least two share a birthday?",
        context="Assume 365 days in a year, birthdays uniformly distributed.",
        expected_answer="50",  # ~50.7%, counterintuitively high
        expected_keywords=["probability", "complement", "birthday paradox", "365", "independent"],
        difficulty=5,
        steps_hint=6,
        perspectives=["complement probability", "birthday paradox", "calculation"],
    ),
    ReasoningProblem(
        id="hard_math_009",
        problem_type=ProblemType.MATH,
        question="A clock shows 3:15. What is the angle between the hour and minute hands?",
        context="Give the smaller angle in degrees.",
        expected_answer="7.5",  # NOT 0! Hour hand moves 0.5° per minute
        expected_keywords=["hour hand", "minute hand", "degrees", "movement", "angle"],
        difficulty=4,
        steps_hint=5,
        perspectives=["hour hand position", "minute hand position", "angle calculation"],
    ),
    ReasoningProblem(
        id="hard_math_010",
        problem_type=ProblemType.MATH,
        question="You're on a game show with 3 doors. Behind one is a car. "
        "You pick door 1. The host opens door 3, revealing a goat. "
        "Should you switch to door 2? What's your probability of winning if you switch?",
        context="This is the Monty Hall problem.",
        expected_answer="2/3",  # Counterintuitive - switching is better
        expected_keywords=["Monty Hall", "conditional", "switch", "probability", "bayes"],
        difficulty=5,
        steps_hint=5,
        perspectives=["initial probability", "conditional update", "host's knowledge"],
    ),
    # --------------------------------------------------------------------------
    # HARD LOGIC (10 problems) - Tricky reasoning, counterintuitive answers
    # --------------------------------------------------------------------------
    ReasoningProblem(
        id="hard_logic_001",
        problem_type=ProblemType.LOGIC,
        question="I have two coins that total 30 cents. One of them is not a nickel. "
        "What are the two coins?",
        context="US coins: penny=1¢, nickel=5¢, dime=10¢, quarter=25¢",
        expected_answer="quarter and nickel",  # ONE is not a nickel - the other IS
        expected_keywords=["quarter", "nickel", "one", "other", "25", "5"],
        difficulty=4,
        steps_hint=4,
        perspectives=["literal interpretation", "coin combinations"],
    ),
    ReasoningProblem(
        id="hard_logic_002",
        problem_type=ProblemType.LOGIC,
        question="A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        context="",
        expected_answer="9",  # "All but 9" = 9 remain
        expected_keywords=["all but", "remain", "left", "survive"],
        difficulty=3,
        steps_hint=2,
        perspectives=["literal reading", "subtraction trap"],
    ),
    ReasoningProblem(
        id="hard_logic_003",
        problem_type=ProblemType.LOGIC,
        question="Is the following statement true or false: 'This statement is false.'",
        context="",
        expected_answer="paradox",  # Neither true nor false - it's a paradox
        expected_keywords=["paradox", "self-reference", "liar", "contradiction", "undecidable"],
        difficulty=5,
        steps_hint=4,
        perspectives=["assume true", "assume false", "paradox identification"],
    ),
    ReasoningProblem(
        id="hard_logic_004",
        problem_type=ProblemType.LOGIC,
        question="If some roses are flowers, and some flowers fade quickly, "
        "can we conclude that some roses fade quickly?",
        context="Use formal logic.",
        expected_answer="no",  # Invalid syllogism - "some" doesn't guarantee overlap
        expected_keywords=["syllogism", "invalid", "some", "overlap", "intersection", "fallacy"],
        difficulty=4,
        steps_hint=4,
        perspectives=["set theory", "syllogism validity", "counterexample"],
    ),
    ReasoningProblem(
        id="hard_logic_005",
        problem_type=ProblemType.LOGIC,
        question="You meet someone who says 'I always lie.' "
        "Is this person lying or telling the truth?",
        context="",
        expected_answer="lying",  # If true, they don't always lie. So they lied about always lying.
        expected_keywords=["paradox", "self-reference", "always", "sometimes", "contradiction"],
        difficulty=4,
        steps_hint=4,
        perspectives=["assume truth", "assume lie", "consistency check"],
    ),
    ReasoningProblem(
        id="hard_logic_006",
        problem_type=ProblemType.LOGIC,
        question="In a village, the barber shaves all those, and only those, "
        "who do not shave themselves. Who shaves the barber?",
        context="",
        expected_answer="paradox",  # Russell's paradox - no consistent answer
        expected_keywords=["Russell", "paradox", "self-reference", "contradiction", "set"],
        difficulty=5,
        steps_hint=5,
        perspectives=["if barber shaves self", "if barber doesn't", "paradox detection"],
    ),
    ReasoningProblem(
        id="hard_logic_007",
        problem_type=ProblemType.LOGIC,
        question="Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. "
        "All labels are WRONG. You can pick one fruit from one box. "
        "Which box should you pick from to determine all contents?",
        context="",
        expected_answer="mixed",  # Pick from 'Mixed' - it's NOT mixed, so reveals its true type
        expected_keywords=["wrong labels", "mixed", "deduce", "elimination", "single pick"],
        difficulty=4,
        steps_hint=5,
        perspectives=["label analysis", "deduction chain", "elimination"],
    ),
    ReasoningProblem(
        id="hard_logic_008",
        problem_type=ProblemType.LOGIC,
        question="You have 12 balls, one is heavier or lighter than the rest. "
        "Using a balance scale, what is the minimum number of weighings to find the odd ball?",
        context="",
        expected_answer="3",  # 3^3=27 outcomes covers 24 possibilities
        expected_keywords=["weighing", "balance", "ternary", "information", "outcomes"],
        difficulty=5,
        steps_hint=5,
        perspectives=["information theory", "ternary search", "worst case"],
    ),
    ReasoningProblem(
        id="hard_logic_009",
        problem_type=ProblemType.LOGIC,
        question="A man is looking at a photograph. Someone asks 'Who is in the picture?' "
        "He replies: 'Brothers and sisters I have none, but that man's father is my father's son.' "
        "Who is in the photograph?",
        context="",
        expected_answer="his son",  # "My father's son" = himself, so "that man's father" = himself
        expected_keywords=["father", "son", "himself", "no siblings", "my father's son"],
        difficulty=4,
        steps_hint=5,
        perspectives=["phrase parsing", "relationship mapping"],
    ),
    ReasoningProblem(
        id="hard_logic_010",
        problem_type=ProblemType.LOGIC,
        question="If you overtake the person in 2nd place in a race, what place are you in?",
        context="",
        expected_answer="2nd",  # NOT 1st - you take their place
        expected_keywords=["overtake", "position", "2nd", "replace"],
        difficulty=3,
        steps_hint=2,
        perspectives=["position logic", "overtake meaning"],
    ),
    # --------------------------------------------------------------------------
    # HARD MULTI-HOP (5 problems) - Requires tracking multiple facts
    # --------------------------------------------------------------------------
    ReasoningProblem(
        id="hard_multi_001",
        problem_type=ProblemType.MULTI_HOP,
        question="Alice is taller than Bob. Bob is taller than Charlie. "
        "Charlie is taller than Diana. Diana is taller than Eve. "
        "If Frank is shorter than Charlie but taller than Eve, "
        "how many people are taller than Frank?",
        context="",
        expected_answer="3",  # Alice, Bob, Charlie are taller than Frank
        expected_keywords=["taller", "shorter", "ordering", "Alice", "Bob", "Charlie", "count"],
        difficulty=4,
        steps_hint=6,
        perspectives=["ordering", "position finding", "counting"],
    ),
    ReasoningProblem(
        id="hard_multi_002",
        problem_type=ProblemType.MULTI_HOP,
        question="In a tournament, every team plays every other team exactly once. "
        "If there are 8 teams, how many total games are played?",
        context="",
        expected_answer="28",  # C(8,2) = 8*7/2 = 28
        expected_keywords=["combination", "pairs", "8", "7", "divide", "each pair"],
        difficulty=4,
        steps_hint=4,
        perspectives=["combination formula", "counting pairs"],
    ),
    ReasoningProblem(
        id="hard_multi_003",
        problem_type=ProblemType.MULTI_HOP,
        question="A detective interviews 4 suspects. "
        "Alex says: 'I didn't do it.' "
        "Blake says: 'Alex is lying.' "
        "Casey says: 'Blake is lying.' "
        "Dana says: 'Casey is telling the truth.' "
        "If exactly one person is lying, who did it?",
        context="The guilty person is the one who lies.",
        expected_answer="Blake",  # Blake lies; Alex told truth, so Blake did it
        expected_keywords=["lying", "truth", "exactly one", "Blake", "contradiction"],
        difficulty=5,
        steps_hint=7,
        perspectives=["assume each lies", "check consistency", "identify liar"],
    ),
    ReasoningProblem(
        id="hard_multi_004",
        problem_type=ProblemType.MULTI_HOP,
        question="In a family, there are 2 fathers, 2 sons, and 1 grandfather. "
        "What is the minimum number of people in this family?",
        context="",
        expected_answer="3",  # Grandfather, father, son - father is both son and father
        expected_keywords=["minimum", "overlap", "grandfather", "father", "son", "dual role"],
        difficulty=4,
        steps_hint=4,
        perspectives=["role overlap", "minimum configuration"],
    ),
    ReasoningProblem(
        id="hard_multi_005",
        problem_type=ProblemType.MULTI_HOP,
        question="A prison has 100 cells, all locked. A guard toggles every cell. "
        "Then every 2nd, then every 3rd, up to every 100th. How many are open?",
        context="Toggle means: if locked, unlock; if unlocked, lock.",
        expected_answer="10",  # Cells with perfect square numbers: 1,4,9,16,25,36,49,64,81,100
        expected_keywords=["toggle", "divisors", "perfect square", "odd", "factors"],
        difficulty=5,
        steps_hint=6,
        perspectives=["divisor counting", "perfect squares", "toggle parity"],
    ),
]


# =============================================================================
# Result Structures
# =============================================================================


@dataclass
class ReasoningResult:
    """Result from a single reasoning attempt."""

    strategy: str
    problem_id: str
    final_answer: str
    reasoning_steps: list[str]
    duration_ms: float
    raw_response: dict[str, Any]
    error: str | None = None


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a reasoning result."""

    correctness: float  # 0.0 to 1.0
    reasoning_depth: int  # Number of distinct steps
    keyword_coverage: float  # Fraction of expected keywords found
    coherence: float  # 0.0 to 1.0 - logical flow score


@dataclass
class ComparisonResult:
    """Complete comparison result for one problem across all strategies."""

    problem: ReasoningProblem
    results: dict[str, ReasoningResult]  # strategy -> result
    metrics: dict[str, EvaluationMetrics]  # strategy -> metrics
    winner: str  # Best performing strategy


# =============================================================================
# Reasoning Strategy Implementations
# =============================================================================


async def solve_baseline(
    problem: ReasoningProblem,
    verbose: bool = False,
) -> ReasoningResult:
    """Solve using baseline direct reasoning (no structured approach).

    When --llm flag is used, this calls the LLM directly without any
    reasoning framework. Otherwise, simulates with expected answer.
    """
    start = time.perf_counter()

    # Get answer from LLM if available, otherwise use expected
    if is_llm_enabled():
        if verbose:
            print("    Calling LLM directly...")
        answer = await get_llm_answer(problem.question, problem.context)
    else:
        answer = problem.expected_answer  # Simulated

    reasoning = [
        f"Question: {problem.question}",
        "Direct analysis: Considering the problem directly...",
        f"Answer: {answer}",
    ]

    duration_ms = (time.perf_counter() - start) * 1000

    return ReasoningResult(
        strategy="baseline",
        problem_id=problem.id,
        final_answer=answer,
        reasoning_steps=reasoning,
        duration_ms=duration_ms,
        raw_response={"simulated": not is_llm_enabled(), "steps": reasoning},
    )


async def solve_with_long_chain(
    client: Any,
    problem: ReasoningProblem,
    verbose: bool = False,
) -> ReasoningResult:
    """Solve using long chain reasoning with MPPA via MCP tools.

    MPPA Enhancement: At planning steps (first 2 steps), provides alternative
    reasoning paths for the tool to score and select the best approach.
    """
    from mcp.types import TextContent

    start = time.perf_counter()
    reasoning_steps: list[str] = []

    try:
        # Start chain using unified 'think' tool
        result = await client.call_tool(
            "think",
            {
                "action": "start",
                "mode": "chain",
                "problem": f"{problem.question}\n\nContext: {problem.context}"
                if problem.context
                else problem.question,
                "expected_steps": problem.steps_hint,
            },
        )
        content = result.content[0]
        response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        if response.get("error"):
            return ReasoningResult(
                strategy="long_chain",
                problem_id=problem.id,
                final_answer="",
                reasoning_steps=[],
                duration_ms=(time.perf_counter() - start) * 1000,
                raw_response=response,
                error=response.get("message", "Unknown error"),
            )

        session_id = response["session_id"]

        # Generate reasoning steps - use LLM if available, otherwise templates
        num_steps = problem.steps_hint

        for step_num in range(1, num_steps + 1):
            # Get the thought for this step
            if is_llm_enabled():
                # Real LLM-in-the-loop: each step is generated by the LLM
                thought = await get_llm_chain_step(problem, step_num, reasoning_steps)
            else:
                # Simulation mode: use templates
                thoughts = _generate_chain_thoughts(problem)
                thought = (
                    thoughts[step_num - 1] if step_num <= len(thoughts) else f"Step {step_num}..."
                )

            if verbose:
                print(f"    Chain step {step_num}: {thought[:50]}...")

            call_params: dict[str, Any] = {
                "action": "continue",
                "session_id": session_id,
                "thought": thought,
            }

            result = await client.call_tool("think", call_params)
            content = result.content[0]
            step_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

            if step_response.get("error"):
                return ReasoningResult(
                    strategy="long_chain",
                    problem_id=problem.id,
                    final_answer="",
                    reasoning_steps=reasoning_steps,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    raw_response=step_response,
                    error=step_response.get("message", "Unknown error"),
                )

            reasoning_steps.append(thought)

            # Track MPPA exploration results
            if verbose and step_response.get("mppa_exploration"):
                expl = step_response["mppa_exploration"]
                print(f"      MPPA: selected score {expl['selected_score']}")

        # Get final answer from LLM if available
        if is_llm_enabled():
            # Build context from reasoning steps
            reasoning_context = "\n".join(
                f"Step {i + 1}: {s}" for i, s in enumerate(reasoning_steps)
            )
            final_thought = await get_llm_answer(
                f"Based on this reasoning:\n{reasoning_context}\n\n"
                f"What is the final answer to: {problem.question}",
                problem.context,
            )
        else:
            final_thought = problem.expected_answer  # Simulated

        # Finalize
        result = await client.call_tool(
            "think",
            {
                "action": "finish",
                "session_id": session_id,
                "thought": final_thought,
                "confidence": 0.9,
            },
        )
        content = result.content[0]
        final_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        duration_ms = (time.perf_counter() - start) * 1000

        return ReasoningResult(
            strategy="long_chain",
            problem_id=problem.id,
            final_answer=final_response.get("final_answer", ""),
            reasoning_steps=reasoning_steps,
            duration_ms=duration_ms,
            raw_response=final_response,
        )

    except Exception as e:
        return ReasoningResult(
            strategy="long_chain",
            problem_id=problem.id,
            final_answer="",
            reasoning_steps=reasoning_steps,
            duration_ms=(time.perf_counter() - start) * 1000,
            raw_response={},
            error=str(e),
        )


async def solve_with_mot(
    client: Any,
    problem: ReasoningProblem,
    verbose: bool = False,
    adaptive_rows: bool = False,
) -> ReasoningResult:
    """Solve using Matrix of Thought reasoning via MCP tools.

    Args:
        client: MCP client.
        problem: Problem to solve.
        verbose: Print debug info.
        adaptive_rows: Use rows="auto" for adaptive strategy selection.

    """
    from mcp.types import TextContent

    start = time.perf_counter()
    reasoning_steps: list[str] = []

    perspectives = problem.perspectives or ["analysis", "verification", "synthesis"]
    criteria = ["pros", "cons"]

    try:
        # Start matrix using unified 'think' tool
        # Note: strategies are auto-selected by the server based on problem complexity
        num_rows = len(perspectives) if not adaptive_rows else 3  # Default adaptive

        start_args: dict[str, Any] = {
            "action": "start",
            "mode": "matrix",
            "problem": f"{problem.question}\n\nContext: {problem.context}"
            if problem.context
            else problem.question,
            "rows": num_rows,
            "cols": len(criteria),
        }

        result = await client.call_tool("think", start_args)
        content = result.content[0]
        response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        if response.get("error"):
            return ReasoningResult(
                strategy="mot_adaptive" if adaptive_rows else "mot",
                problem_id=problem.id,
                final_answer="",
                reasoning_steps=[],
                duration_ms=(time.perf_counter() - start) * 1000,
                raw_response=response,
                error=response.get("message", "Unknown error"),
            )

        session_id = response["session_id"]
        actual_rows = response["matrix_dimensions"]["rows"]

        # Update perspectives if adaptive changed row count
        if adaptive_rows or actual_rows != len(perspectives):
            # Adjust perspectives to match actual rows
            all_perspectives = [
                "analysis",
                "verification",
                "synthesis",
                "comparison",
                "evaluation",
                "abstraction",
                "decomposition",  # Extra perspectives for larger matrices
            ]
            perspectives = all_perspectives[:actual_rows]

        if verbose and response.get("complexity"):
            c = response["complexity"]
            print(
                f"    Complexity: {c['complexity_level']} "
                f"(score={c['complexity_score']}, rows={actual_rows})"
            )

        # Fill matrix cells - use LLM if available, otherwise templates
        filled_cells: list[tuple[str, str, str]] = []  # (perspective, criterion, content)

        for row_idx, perspective in enumerate(perspectives):
            for col_idx, criterion in enumerate(criteria):
                # Get the thought for this cell
                if is_llm_enabled():
                    # Real LLM-in-the-loop: each cell is generated by the LLM
                    thought = await get_llm_mot_cell(problem, perspective, criterion, filled_cells)
                else:
                    # Simulation mode: use templates
                    thoughts = _generate_mot_thoughts(problem, perspectives, criteria)
                    thought = thoughts.get(
                        (row_idx, col_idx), f"{perspective}: {criterion} analysis"
                    )

                if verbose:
                    print(f"    MoT [{row_idx},{col_idx}]: {thought[:40]}...")

                # Fill matrix cell
                call_args: dict[str, Any] = {
                    "action": "continue",
                    "session_id": session_id,
                    "row": row_idx,
                    "col": col_idx,
                    "thought": thought,
                }

                result = await client.call_tool("think", call_args)
                content = result.content[0]
                cell_response = json.loads(
                    content.text if isinstance(content, TextContent) else "{}"
                )

                if cell_response.get("error"):
                    return ReasoningResult(
                        strategy="mot",
                        problem_id=problem.id,
                        final_answer="",
                        reasoning_steps=reasoning_steps,
                        duration_ms=(time.perf_counter() - start) * 1000,
                        raw_response=cell_response,
                        error=cell_response.get("message", "Unknown error"),
                    )

                reasoning_steps.append(f"[{perspective}/{criterion}] {thought}")
                filled_cells.append((perspective, criterion, thought))

        # Synthesize columns - use LLM if available
        for col_idx, criterion in enumerate(criteria):
            if is_llm_enabled():
                # Gather cells for this column
                column_cells = [
                    (perspectives[row_idx], content)
                    for row_idx, (_, crit, content) in enumerate(
                        c for c in filled_cells if c[1] == criterion
                    )
                ]
                # Fix: properly filter cells by criterion
                column_cells = [
                    (p, content) for p, crit, content in filled_cells if crit == criterion
                ]
                synthesis = await get_llm_mot_synthesis(problem, column_cells, criterion)
            else:
                synthesis = f"Synthesis of {criterion}: Combined analysis across perspectives"

            result = await client.call_tool(
                "think",
                {
                    "action": "synthesize",
                    "session_id": session_id,
                    "col": col_idx,
                    "thought": synthesis,
                },
            )
            reasoning_steps.append(f"[Synthesis/{criterion}] {synthesis}")

        # Get final answer from LLM if available
        if is_llm_enabled():
            reasoning_context = "\n".join(reasoning_steps)
            final_thought = await get_llm_answer(
                f"Based on this multi-perspective analysis:\n{reasoning_context}\n\n"
                f"What is the final answer to: {problem.question}",
                problem.context,
            )
        else:
            final_thought = problem.expected_answer  # Simulated

        # Finalize
        result = await client.call_tool(
            "think",
            {
                "action": "finish",
                "session_id": session_id,
                "thought": final_thought,
                "confidence": 0.85,
            },
        )
        content = result.content[0]
        final_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        duration_ms = (time.perf_counter() - start) * 1000

        return ReasoningResult(
            strategy="mot_adaptive" if adaptive_rows else "mot",
            problem_id=problem.id,
            final_answer=final_response.get("final_answer", ""),
            reasoning_steps=reasoning_steps,
            duration_ms=duration_ms,
            raw_response=final_response,
        )

    except Exception as e:
        return ReasoningResult(
            strategy="mot_adaptive" if adaptive_rows else "mot",
            problem_id=problem.id,
            final_answer="",
            reasoning_steps=reasoning_steps,
            duration_ms=(time.perf_counter() - start) * 1000,
            raw_response={},
            error=str(e),
        )


async def solve_with_hybrid(
    client: Any,
    problem: ReasoningProblem,
    verbose: bool = False,
    confidence_threshold: float = 0.7,
) -> ReasoningResult:
    """Hybrid strategy: Start with Long Chain, escalate to MoT if confidence drops.

    This strategy captures the speed benefit of Long Chain for simple problems
    while escalating to MoT's multi-perspective approach when needed.

    Escalation triggers:
    - Progress < 50% after 2 steps (problem harder than expected)
    - Low consistency score from chain reasoning
    """
    from mcp.types import TextContent

    start = time.perf_counter()
    reasoning_steps: list[str] = []
    escalated = False

    try:
        # Phase 1: Start with Long Chain
        result = await client.call_tool(
            "think",
            {
                "action": "start",
                "mode": "chain",
                "problem": f"{problem.question}\n\nContext: {problem.context}"
                if problem.context
                else problem.question,
                "expected_steps": problem.steps_hint,
            },
        )
        content = result.content[0]
        response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        if response.get("error"):
            return ReasoningResult(
                strategy="hybrid",
                problem_id=problem.id,
                final_answer="",
                reasoning_steps=[],
                duration_ms=(time.perf_counter() - start) * 1000,
                raw_response=response,
                error=response.get("message", "Unknown error"),
            )

        chain_session_id = response["session_id"]

        # Run first 2 steps of chain reasoning with LLM if available
        for step_num in range(1, 3):  # Steps 1 and 2
            if is_llm_enabled():
                thought = await get_llm_chain_step(problem, step_num, reasoning_steps)
            else:
                thoughts = _generate_chain_thoughts(problem)
                thought = (
                    thoughts[step_num - 1] if step_num <= len(thoughts) else f"Step {step_num}..."
                )

            if verbose:
                print(f"    Chain step {step_num}: {thought[:50]}...")

            result = await client.call_tool(
                "think",
                {
                    "action": "continue",
                    "session_id": chain_session_id,
                    "thought": thought,
                },
            )
            content = result.content[0]
            step_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

            if step_response.get("error"):
                break

            reasoning_steps.append(thought)

            # Check escalation conditions after step 2
            if step_num == 2:
                progress = step_response.get("progress", 0)
                consistency = step_response.get("consistency_score", 1.0)

                # Escalate based on multiple heuristics
                should_escalate = (
                    # Progress-based: escalate if slow progress
                    progress < 0.35
                    # Consistency-based: escalate if reasoning seems inconsistent
                    or consistency < confidence_threshold
                    # Difficulty-based: always escalate for hard problems
                    or problem.difficulty >= 3
                    # Type-based: math and multi-hop benefit most from MoT
                    or problem.problem_type in (ProblemType.MATH, ProblemType.MULTI_HOP)
                )

                if should_escalate:
                    escalated = True
                    if verbose:
                        print(
                            f"    → Escalating to MoT (progress={progress:.2f}, "
                            f"consistency={consistency:.2f})"
                        )
                    break

        if escalated:
            # Phase 2: Escalate to MoT for deeper analysis
            perspectives = problem.perspectives or ["analysis", "verification", "synthesis"]
            criteria = ["pros", "cons"]

            result = await client.call_tool(
                "think",
                {
                    "action": "start",
                    "mode": "matrix",
                    "problem": f"{problem.question}\n\nContext: {problem.context}"
                    if problem.context
                    else problem.question,
                    "rows": len(perspectives),
                    "cols": len(criteria),
                },
            )
            content = result.content[0]
            mot_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

            if not mot_response.get("error"):
                mot_session_id = mot_response["session_id"]
                filled_cells: list[tuple[str, str, str]] = []

                # Fill matrix cells with LLM if available
                for row, perspective in enumerate(perspectives):
                    for col, criterion in enumerate(criteria):
                        if is_llm_enabled():
                            thought = await get_llm_mot_cell(
                                problem, perspective, criterion, filled_cells
                            )
                        else:
                            mot_thoughts = _generate_mot_thoughts(problem, perspectives, criteria)
                            thought = mot_thoughts.get((row, col), f"{perspective}: {criterion}")

                        if verbose:
                            print(f"    MoT [{row},{col}]: {perspective}: {thought[:30]}...")

                        result = await client.call_tool(
                            "think",
                            {
                                "action": "continue",
                                "session_id": mot_session_id,
                                "row": row,
                                "col": col,
                                "thought": thought,
                            },
                        )
                        content = result.content[0]
                        cell_response = json.loads(
                            content.text if isinstance(content, TextContent) else "{}"
                        )

                        if not cell_response.get("error"):
                            reasoning_steps.append(f"{perspective}/{criterion}: {thought}")
                            filled_cells.append((perspective, criterion, thought))

                # Synthesize columns with LLM if available
                for col, criterion in enumerate(criteria):
                    if is_llm_enabled():
                        column_cells = [
                            (p, content) for p, crit, content in filled_cells if crit == criterion
                        ]
                        synthesis = await get_llm_mot_synthesis(problem, column_cells, criterion)
                    else:
                        synthesis = f"Synthesis of {criterion}"

                    await client.call_tool(
                        "think",
                        {
                            "action": "synthesize",
                            "session_id": mot_session_id,
                            "col": col,
                            "thought": synthesis,
                        },
                    )

                # Get final answer from LLM if available (escalated path)
                if is_llm_enabled():
                    reasoning_context = "\n".join(reasoning_steps)
                    final_thought = await get_llm_answer(
                        f"Based on this analysis:\n{reasoning_context}\n\n"
                        f"What is the final answer to: {problem.question}",
                        problem.context,
                    )
                else:
                    final_thought = problem.expected_answer

                # Finalize MoT
                result = await client.call_tool(
                    "think",
                    {
                        "action": "finish",
                        "session_id": mot_session_id,
                        "thought": final_thought,
                        "confidence": 0.9,
                    },
                )
                content = result.content[0]
                final_response = json.loads(
                    content.text if isinstance(content, TextContent) else "{}"
                )

                return ReasoningResult(
                    strategy="hybrid",
                    problem_id=problem.id,
                    final_answer=final_response.get("final_answer", ""),
                    reasoning_steps=reasoning_steps,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    raw_response={**final_response, "escalated": True},
                )

        # No escalation - continue and finish chain (we already did 2 steps)
        for step_num in range(3, problem.steps_hint + 1):
            if is_llm_enabled():
                thought = await get_llm_chain_step(problem, step_num, reasoning_steps)
            else:
                thoughts = _generate_chain_thoughts(problem)
                thought = (
                    thoughts[step_num - 1] if step_num <= len(thoughts) else f"Step {step_num}..."
                )

            if verbose:
                print(f"    Chain step {step_num}: {thought[:50]}...")

            result = await client.call_tool(
                "think",
                {
                    "action": "continue",
                    "session_id": chain_session_id,
                    "thought": thought,
                },
            )
            content = result.content[0]
            step_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

            if not step_response.get("error"):
                reasoning_steps.append(thought)

        # Get final answer from LLM if available (non-escalated path)
        if is_llm_enabled():
            reasoning_context = "\n".join(
                f"Step {i + 1}: {s}" for i, s in enumerate(reasoning_steps)
            )
            final_thought = await get_llm_answer(
                f"Based on this reasoning:\n{reasoning_context}\n\n"
                f"What is the final answer to: {problem.question}",
                problem.context,
            )
        else:
            final_thought = problem.expected_answer

        # Finalize chain
        result = await client.call_tool(
            "think",
            {
                "action": "finish",
                "session_id": chain_session_id,
                "thought": final_thought,
                "confidence": 0.9,
            },
        )
        content = result.content[0]
        final_response = json.loads(content.text if isinstance(content, TextContent) else "{}")

        return ReasoningResult(
            strategy="hybrid",
            problem_id=problem.id,
            final_answer=final_response.get("final_answer", ""),
            reasoning_steps=reasoning_steps,
            duration_ms=(time.perf_counter() - start) * 1000,
            raw_response={**final_response, "escalated": False},
        )

    except Exception as e:
        return ReasoningResult(
            strategy="hybrid",
            problem_id=problem.id,
            final_answer="",
            reasoning_steps=reasoning_steps,
            duration_ms=(time.perf_counter() - start) * 1000,
            raw_response={},
            error=str(e),
        )


def _generate_chain_thoughts(problem: ReasoningProblem) -> list[str]:
    """Generate reasoning thoughts for chain based on problem type.

    Includes problem-specific keywords to enable fair coverage comparison with MoT.
    """
    q_snippet = problem.question[:60]
    # Use ALL keywords for better coverage
    keywords = " ".join(problem.expected_keywords) if problem.expected_keywords else ""

    if problem.problem_type == ProblemType.MATH:
        return [
            f"Let me identify the key values: {q_snippet}. Looking for: {keywords}",
            f"Setting up equations using {keywords} relationships.",
            f"Calculating step by step with {keywords}...",
            f"Verifying: does {problem.expected_answer} make sense for {keywords}?",
            f"The answer is {problem.expected_answer}.",
        ][: problem.steps_hint]

    elif problem.problem_type == ProblemType.LOGIC:
        return [
            f"Identifying premises in: {q_snippet}. Key concepts: {keywords}",
            f"Analyzing logical structure with {keywords}.",
            f"Checking if conclusion follows from {keywords}.",
            f"Considering counterexamples for {keywords}.",
            f"Based on analysis, the answer is {problem.expected_answer}.",
        ][: problem.steps_hint]

    elif problem.problem_type == ProblemType.MULTI_HOP:
        return [
            f"Breaking down: {q_snippet}. Need to connect: {keywords}",
            f"First hop: identifying {keywords.split()[0] if keywords else 'key entity'}.",
            f"Connecting {keywords} to answer the question.",
            "Looking up the specific information requested.",
            f"The answer is {problem.expected_answer}.",
        ][: problem.steps_hint]

    else:  # ANALYSIS
        return [
            f"Understanding constraints: {q_snippet}. Factors: {keywords}",
            f"Analyzing trade-offs for {keywords}.",
            f"Considering context for {keywords}...",
            f"Weighing pros/cons for {keywords}...",
            f"My recommendation is {problem.expected_answer}.",
        ][: problem.steps_hint]


def _generate_chain_thoughts_mppa(
    problem: ReasoningProblem,
) -> list[tuple[str, list[str]]]:
    """Generate reasoning thoughts with MPPA alternatives for planning steps.

    Returns tuples of (primary_thought, alternatives) where alternatives
    are provided for the first 2 planning steps to enable multi-path exploration.

    MPPA key insight: Alternatives should be substantively different approaches,
    not just rephrased versions. This allows survival scoring to select
    the path most likely to succeed based on context relevance and specificity.
    """
    # Include problem-specific details in thoughts for better scoring
    q_snippet = problem.question[:60]
    keywords = " ".join(problem.expected_keywords[:3])

    if problem.problem_type == ProblemType.MATH:
        return [
            (
                f"Let me identify the key values: {q_snippet}. Looking for: {keywords}",
                [
                    "First, I'll extract all numbers and understand the operation needed. "
                    f"Key terms: {keywords}",
                    f"I should list quantities from: {q_snippet}",
                ],
            ),
            (
                f"Setting up equations using {keywords} relationships.",
                [
                    "Let me create a step-by-step calculation plan with verification.",
                    "I'll identify which formula applies and substitute values.",
                ],
            ),
            (f"Calculating step by step using {keywords}...", []),
            (f"Verifying: does {problem.expected_answer} make sense?", []),
            (f"The answer is {problem.expected_answer}.", []),
        ][: problem.steps_hint]

    elif problem.problem_type == ProblemType.LOGIC:
        return [
            (
                f"First, identifying premises in: {q_snippet}. Key concepts: {keywords}",
                [
                    f"Listing all statements as propositions involving {keywords}.",
                    f"Diagramming relationships between: {keywords}",
                ],
            ),
            (
                f"Analyzing logical structure with {keywords}.",
                [
                    "Checking argument type: syllogism, modus ponens, etc.",
                    f"Tracing implications step by step through {keywords}.",
                ],
            ),
            (f"Checking if conclusion follows from {keywords}.", []),
            ("Considering counterexamples to test validity.", []),
            (f"Based on analysis, the answer is {problem.expected_answer}.", []),
        ][: problem.steps_hint]

    elif problem.problem_type == ProblemType.MULTI_HOP:
        return [
            (
                f"Breaking down: {q_snippet}. Need to connect: {keywords}",
                [
                    f"Identifying the fact chain needed through {keywords}.",
                    f"Mapping reasoning hops: {keywords}",
                ],
            ),
            (
                f"First hop: identifying {keywords.split()[0]} from context.",
                [
                    f"Extracting primary subject related to {keywords}.",
                    f"Finding starting point in: {keywords}",
                ],
            ),
            (f"Connecting {keywords} to answer the question.", []),
            ("Looking up the specific information requested.", []),
            (f"The answer is {problem.expected_answer}.", []),
        ][: problem.steps_hint]

    else:  # ANALYSIS
        return [
            (
                f"Understanding constraints: {q_snippet}. Factors: {keywords}",
                [
                    f"Listing decision factors: {keywords} and trade-offs.",
                    f"Identifying key trade-offs involving {keywords}.",
                ],
            ),
            (
                f"Analyzing trade-offs for {keywords}.",
                [
                    f"Evaluating options against: {keywords}.",
                    f"Weighing pros/cons systematically for {keywords}.",
                ],
            ),
            (f"Given context, considering {keywords}...", []),
            (f"Weighing {keywords} for this situation...", []),
            (f"My recommendation is {problem.expected_answer}.", []),
        ][: problem.steps_hint]


def _generate_mot_thoughts(
    problem: ReasoningProblem,
    perspectives: list[str],
    criteria: list[str],
) -> dict[tuple[int, int], str]:
    """Generate matrix cell thoughts based on problem and perspectives.

    Includes expected keywords for better keyword coverage scoring.
    """
    thoughts: dict[tuple[int, int], str] = {}
    # Use ALL keywords for better coverage (not just first 3)
    keywords = " ".join(problem.expected_keywords) if problem.expected_keywords else ""
    q_snippet = problem.question[:40]

    for row_idx, perspective in enumerate(perspectives):
        for col_idx, criterion in enumerate(criteria):
            if problem.problem_type == ProblemType.MATH:
                if criterion == "pros":
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Analyzing {q_snippet}... Key terms: {keywords}"
                    )
                else:
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Verifying calculation with {keywords}"
                    )

            elif problem.problem_type == ProblemType.LOGIC:
                if criterion == "pros":
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Logical structure analysis using {keywords}"
                    )
                else:
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Checking edge cases for {keywords}"
                    )

            elif problem.problem_type == ProblemType.ANALYSIS:
                if criterion == "pros":
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Favorable factors considering {keywords}"
                    )
                else:
                    thoughts[(row_idx, col_idx)] = f"{perspective}: Trade-offs involving {keywords}"

            else:  # MULTI_HOP
                if criterion == "pros":
                    thoughts[(row_idx, col_idx)] = (
                        f"{perspective}: Connecting facts about {keywords}"
                    )
                else:
                    thoughts[(row_idx, col_idx)] = f"{perspective}: Verifying chain for {keywords}"

    return thoughts


def _generate_mot_alternatives(
    problem: ReasoningProblem,
    perspective: str,
    criterion: str,
    primary_thought: str,
) -> list[str]:
    """Generate MPPA alternative thoughts for Row 0 (planning cells).

    For Row 0, we generate 2-3 alternative reasoning paths that the MPPA
    scoring in MoT will evaluate and select the best from.
    """
    alternatives: list[str] = []
    keywords = " ".join(problem.expected_keywords[:2]) if problem.expected_keywords else ""

    if problem.problem_type == ProblemType.MATH:
        alternatives = [
            f"{perspective}: Breaking down the equation systematically, tracking each step",
            f"{perspective}: Using algebraic manipulation to isolate the unknown {keywords}",
            f"{perspective}: Verifying with substitution after solving {keywords}",
        ]
    elif problem.problem_type == ProblemType.LOGIC:
        alternatives = [
            f"{perspective}: Analyzing logical connectives and their implications",
            f"{perspective}: Checking for contradictions and edge cases in {keywords}",
            f"{perspective}: Building truth table to verify logical structure",
        ]
    elif problem.problem_type == ProblemType.ANALYSIS:
        alternatives = [
            f"{perspective}: Weighing factors based on context relevance {keywords}",
            f"{perspective}: Identifying key trade-offs and dependencies",
            f"{perspective}: Prioritizing criteria by impact and feasibility",
        ]
    else:  # MULTI_HOP
        alternatives = [
            f"{perspective}: Tracing evidence chain from premise to conclusion",
            f"{perspective}: Cross-referencing facts to build inference path {keywords}",
            f"{perspective}: Validating each reasoning hop for consistency",
        ]

    # Filter out duplicates and the primary thought
    return [alt for alt in alternatives if alt != primary_thought][:2]


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_result(
    result: ReasoningResult,
    problem: ReasoningProblem,
) -> EvaluationMetrics:
    """Evaluate a reasoning result against the problem."""
    # Correctness: Does answer match expected?
    answer_lower = result.final_answer.lower().strip()
    expected_lower = problem.expected_answer.lower().strip()

    # Flexible matching - check if expected is contained in answer
    if expected_lower in answer_lower or answer_lower in expected_lower:
        correctness = 1.0
    elif any(kw in answer_lower for kw in expected_lower.split()):
        correctness = 0.7
    else:
        correctness = 0.0

    # Reasoning depth: Number of steps
    reasoning_depth = len(result.reasoning_steps)

    # Keyword coverage: What fraction of expected keywords appear in reasoning?
    all_reasoning = " ".join(result.reasoning_steps).lower()
    found_keywords = sum(1 for kw in problem.expected_keywords if kw.lower() in all_reasoning)
    keyword_coverage = (
        found_keywords / len(problem.expected_keywords) if problem.expected_keywords else 0.0
    )

    # Coherence: Heuristic based on step quality
    if reasoning_depth == 0:
        coherence = 0.0
    else:
        avg_step_length = sum(len(s) for s in result.reasoning_steps) / reasoning_depth

        # Base coherence from step length (good steps are 50-500 chars)
        if 50 <= avg_step_length <= 500:
            coherence = 0.7
        elif 20 <= avg_step_length <= 1000:
            coherence = 0.5
        else:
            coherence = 0.3

        # Bonus for MPPA exploration (indicates path optimization)
        if result.raw_response.get("mppa_explorations", 0) > 0:
            coherence += 0.15

        # Bonus for including problem-specific keywords in steps
        if keyword_coverage > 0.3:
            coherence += 0.1

        coherence = min(coherence, 1.0)

    return EvaluationMetrics(
        correctness=correctness,
        reasoning_depth=reasoning_depth,
        keyword_coverage=keyword_coverage,
        coherence=coherence,
    )

    # Coherence: Simple heuristic based on step length and structure
    if reasoning_depth == 0:
        coherence = 0.0
    else:
        avg_step_length = sum(len(s) for s in result.reasoning_steps) / reasoning_depth
        # Good steps are 50-500 chars
        if 50 <= avg_step_length <= 500:
            coherence = 0.9
        elif 20 <= avg_step_length <= 1000:
            coherence = 0.6
        else:
            coherence = 0.3

    return EvaluationMetrics(
        correctness=correctness,
        reasoning_depth=reasoning_depth,
        keyword_coverage=keyword_coverage,
        coherence=coherence,
    )


def determine_winner(
    metrics: dict[str, EvaluationMetrics],
    results: dict[str, ReasoningResult] | None = None,
) -> str:
    """Determine which strategy performed best.

    Scoring weights correctness > coverage > depth > coherence.
    When scores are tied, faster strategies win (if results provided).
    """

    def score(m: EvaluationMetrics) -> float:
        # Weighted scoring: correctness matters most
        return (
            m.correctness * 0.4
            + m.keyword_coverage * 0.3
            + min(m.reasoning_depth / 5, 1.0) * 0.2  # Cap depth contribution
            + m.coherence * 0.1
        )

    scores = {strategy: score(m) for strategy, m in metrics.items()}
    max_score = max(scores.values())

    # Find all strategies with the max score (ties)
    tied = [s for s, v in scores.items() if abs(v - max_score) < 0.001]

    if len(tied) == 1:
        return tied[0]

    # Tiebreaker: prefer faster strategy
    if results:
        tied.sort(key=lambda s: results[s].duration_ms)
        return tied[0]

    return max(scores, key=lambda k: scores[k])


# =============================================================================
# Benchmark Runner
# =============================================================================


async def run_comparison(
    problems: list[ReasoningProblem],
    verbose: bool = False,
) -> list[ComparisonResult]:
    """Run comparison benchmark on all problems."""
    try:
        from fastmcp import Client
        from fastmcp.client.transports import PythonStdioTransport
    except ImportError:
        print("Error: fastmcp not installed. Run: pip install fastmcp")
        sys.exit(1)

    results: list[ComparisonResult] = []

    # Configure transport with higher rate limits for benchmarking
    # 3 sessions per problem (long_chain + mot + hybrid may use 2)
    max_sessions = max(500, len(problems) * 4)
    transport = PythonStdioTransport(
        script_path="src/server.py",
        env={
            **os.environ,
            "RATE_LIMIT_MAX_SESSIONS": str(max_sessions),
            "RATE_LIMIT_WINDOW_SECONDS": "60",
        },
    )

    async with Client(transport) as client:
        for problem in problems:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Problem: {problem.id} ({problem.problem_type.value})")
                print(f"Question: {problem.question[:60]}...")
                print("=" * 60)

            problem_results: dict[str, ReasoningResult] = {}

            # 1. Baseline
            if verbose:
                print("\n  [Baseline] Direct reasoning...")
            baseline_result = await solve_baseline(problem, verbose)
            problem_results["baseline"] = baseline_result

            # 2. Long Chain
            if verbose:
                print("\n  [Long Chain] Sequential reasoning...")
            chain_result = await solve_with_long_chain(client, problem, verbose)
            problem_results["long_chain"] = chain_result

            # 3. Matrix of Thought
            if verbose:
                print("\n  [MoT] Multi-perspective reasoning...")
            mot_result = await solve_with_mot(client, problem, verbose)
            problem_results["mot"] = mot_result

            # 4. Hybrid (Chain → MoT escalation)
            if verbose:
                print("\n  [Hybrid] Chain with MoT escalation...")
            hybrid_result = await solve_with_hybrid(client, problem, verbose)
            problem_results["hybrid"] = hybrid_result

            # Evaluate all
            metrics: dict[str, EvaluationMetrics] = {}
            for strategy, result in problem_results.items():
                metrics[strategy] = evaluate_result(result, problem)

            winner = determine_winner(metrics, problem_results)

            results.append(
                ComparisonResult(
                    problem=problem,
                    results=problem_results,
                    metrics=metrics,
                    winner=winner,
                )
            )

            if verbose:
                print(f"\n  Winner: {winner}")

    return results


# =============================================================================
# Report Generation
# =============================================================================


def print_comparison_report(results: list[ComparisonResult], verbose: bool = False) -> None:
    """Print comprehensive comparison report."""
    print("\n" + "=" * 80)
    print("REASONING STRATEGY COMPARISON REPORT")
    print("=" * 80)

    # Per-problem results
    print("\n" + "-" * 80)
    print("PER-PROBLEM RESULTS")
    print("-" * 80)

    strategy_wins: dict[str, int] = {"baseline": 0, "long_chain": 0, "mot": 0, "hybrid": 0}
    strategy_metrics: dict[str, list[EvaluationMetrics]] = {
        "baseline": [],
        "long_chain": [],
        "mot": [],
        "hybrid": [],
    }

    for comp in results:
        problem_info = f"{comp.problem.problem_type.value}, difficulty={comp.problem.difficulty}"
        print(f"\n{comp.problem.id} ({problem_info})")
        print(f"  Question: {comp.problem.question[:70]}...")
        print(f"  Expected: {comp.problem.expected_answer}")
        print()

        for strategy in ["baseline", "long_chain", "mot", "hybrid"]:
            result = comp.results[strategy]
            metrics = comp.metrics[strategy]
            strategy_metrics[strategy].append(metrics)

            winner_mark = " 🏆" if comp.winner == strategy else ""
            error_mark = " ❌" if result.error else ""

            print(
                f"  {strategy:12} | "
                f"correct={metrics.correctness:.1f} "
                f"depth={metrics.reasoning_depth:2d} "
                f"coverage={metrics.keyword_coverage:.2f} "
                f"coherence={metrics.coherence:.1f} "
                f"time={result.duration_ms:6.0f}ms"
                f"{winner_mark}{error_mark}"
            )

            if result.error and verbose:
                print(f"               Error: {result.error}")

        strategy_wins[comp.winner] += 1

    # Aggregate statistics
    print("\n" + "-" * 80)
    print("AGGREGATE STATISTICS")
    print("-" * 80)

    header = (
        f"{'Strategy':<12} | {'Wins':>5} | {'Avg Correct':>11} | "
        f"{'Avg Depth':>9} | {'Avg Coverage':>12} | {'Avg Time':>10}"
    )
    print(f"\n{header}")
    print("-" * 75)

    for strategy in ["baseline", "long_chain", "mot", "hybrid"]:
        metrics_list = strategy_metrics[strategy]
        if not metrics_list:
            continue

        avg_correct = sum(m.correctness for m in metrics_list) / len(metrics_list)
        avg_depth = sum(m.reasoning_depth for m in metrics_list) / len(metrics_list)
        avg_coverage = sum(m.keyword_coverage for m in metrics_list) / len(metrics_list)

        # Get average time from results
        all_times = [
            comp.results[strategy].duration_ms
            for comp in results
            if not comp.results[strategy].error
        ]
        avg_time = sum(all_times) / len(all_times) if all_times else 0

        print(
            f"{strategy:<12} | {strategy_wins[strategy]:>5} | "
            f"{avg_correct:>11.2f} | {avg_depth:>9.1f} | "
            f"{avg_coverage:>12.2f} | {avg_time:>9.0f}ms"
        )

    # Winner summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    total = len(results)
    print(f"\nTotal problems: {total}")
    print("\nWins by strategy:")
    for strategy, wins in sorted(strategy_wins.items(), key=lambda x: -x[1]):  # type: ignore[arg-type]
        pct = (wins / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {strategy:<12}: {wins:>2} ({pct:5.1f}%) {bar}")

    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS BY PROBLEM TYPE")
    print("-" * 80)

    by_type: dict[ProblemType, dict[str, int]] = {}
    for comp in results:
        pt = comp.problem.problem_type
        if pt not in by_type:
            by_type[pt] = {"baseline": 0, "long_chain": 0, "mot": 0, "hybrid": 0}
        by_type[pt][comp.winner] += 1

    for pt, type_wins in by_type.items():
        best = max(type_wins.keys(), key=type_wins.get)  # type: ignore[arg-type]
        print(f"\n  {pt.value:<12}: Best strategy = {best}")
        for strategy, count in type_wins.items():
            print(f"    {strategy}: {count} wins")


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> None:
    """Run the comparison benchmark."""
    parser = argparse.ArgumentParser(description="Compare Long Chain vs MoT vs Baseline Reasoning")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--problems",
        "-n",
        type=int,
        default=None,
        help="Number of problems to run (default: all)",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["math", "logic", "multi_hop", "analysis"],
        help="Only run problems of this type",
    )
    # Add standard LLM arguments
    add_llm_args(parser)
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Use hard mode problems (tricky, ambiguous, counterintuitive)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MatrixMind MCP - Reasoning Strategy Comparison Benchmark")
    print("=" * 80)

    # Initialize LLM client if requested
    if args.llm:
        print(f"\nLLM Mode: Using {args.llm_model or 'default'} at {args.llm_url or 'default'}")
        print("  → Real LLM-in-the-loop: Each reasoning step is generated by the LLM")
        await init_llm_client(base_url=args.llm_url, model=args.llm_model)
    else:
        print("\nSimulation Mode: Using expected answers (use --llm for real LLM)")

    # Select problem set
    if args.hard:
        print("\n⚠️  HARD MODE: Tricky problems designed to trip up LLMs")
        problems = HARD_PROBLEMS
    else:
        problems = BENCHMARK_PROBLEMS

    # Filter by type
    if args.type:
        target_type = ProblemType(args.type)
        problems = [p for p in problems if p.problem_type == target_type]

    if args.problems:
        problems = problems[: args.problems]

    print(f"\nProblems to evaluate: {len(problems)}")
    print("Strategies: baseline, long_chain, mot, hybrid")

    try:
        results = await run_comparison(problems, verbose=args.verbose)
        print_comparison_report(results, verbose=args.verbose)
    finally:
        await close_llm_client()


if __name__ == "__main__":
    asyncio.run(main())
