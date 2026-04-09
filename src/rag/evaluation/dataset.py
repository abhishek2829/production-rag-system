"""Golden dataset loader.

A golden dataset is a list of test cases for the RAG system.
Each test case has a question and the expected behavior
(should it answer or refuse? which sources should it cite?).

Think of it like unit tests, but for AI quality:
- Unit tests check: "Does the function return the right value?"
- Golden dataset checks: "Does the AI give the right answer with the right citations?"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalCase:
    """A single test case in the golden dataset.

    Attributes:
        id: Unique identifier for this test case (e.g., "rag_definition").
        question: The question to ask the RAG system.
        expected_behavior: Either "answer" (should answer) or "refuse" (should decline).
        expected_sources: List of filenames that should be cited (e.g., ["rag.md"]).
        must_contain: Keywords that must appear in the answer (case-insensitive).
        notes: Human-readable notes about why this test case exists.
    """

    id: str
    question: str
    expected_behavior: str  # "answer" or "refuse"
    expected_sources: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    notes: str = ""


def load_golden_dataset(path: Path) -> list[EvalCase]:
    """Load a golden dataset from a JSON file.

    Args:
        path: Path to the JSON file containing test cases.

    Returns:
        List of EvalCase objects.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the JSON format is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Golden dataset must be a JSON array of test cases")

    cases = [
        EvalCase(
            id=item["id"],
            question=item["question"],
            expected_behavior=item["expected_behavior"],
            expected_sources=item.get("expected_sources", []),
            must_contain=item.get("must_contain", []),
            notes=item.get("notes", ""),
        )
        for item in data
    ]

    logger.info("Loaded golden dataset: %d test cases from %s", len(cases), path)
    return cases
