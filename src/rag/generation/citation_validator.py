"""Citation validation for RAG responses.

THE PROBLEM:
Even with a good prompt, LLMs sometimes:
1. Make claims without citing any source
2. Cite a source number that doesn't exist (e.g., [Source 8] when we only have 5)
3. Answer questions that the sources don't actually support

This module validates the citations AFTER the LLM generates an answer.
Think of it like spell-check, but for citations.

WHAT WE CHECK:
1. Does every paragraph have at least one [Source N] citation?
2. Are all cited source numbers valid (within range)?
3. What percentage of retrieved chunks were actually cited? (citation coverage)

WHY THIS MATTERS FOR INTERVIEWS:
"How do you prevent hallucination?" is the #1 RAG interview question.
Saying "we validate citations programmatically" is a much stronger answer
than "we use a good prompt."
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rag.models import RAGResponse

logger = logging.getLogger(__name__)

# Pattern to find [Source N] citations in text
_CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")

# Phrases that indicate the model is refusing to answer (which is good!)
_REFUSAL_PHRASES = [
    "i don't have enough information",
    "the provided documents do not contain",
    "i cannot find",
    "no information available",
    "not mentioned in the sources",
    "the sources don't contain",
]


@dataclass
class CitationReport:
    """Report on citation quality for a RAG response.

    Attributes:
        is_valid: Whether the response passes all citation checks.
        is_refusal: Whether the model correctly refused to answer.
        cited_sources: Set of source numbers that were cited (e.g., {1, 3, 5}).
        invalid_sources: Source numbers cited that don't exist.
        uncited_paragraphs: Number of content paragraphs without any citation.
        total_paragraphs: Total number of content paragraphs.
        citation_coverage: Fraction of retrieved chunks that were cited (0.0 to 1.0).
        issues: List of human-readable issue descriptions.
    """

    is_valid: bool
    is_refusal: bool
    cited_sources: set[int]
    invalid_sources: set[int]
    uncited_paragraphs: int
    total_paragraphs: int
    citation_coverage: float
    issues: list[str]


def validate_citations(response: RAGResponse) -> CitationReport:
    """Validate citations in a RAG response.

    Checks that:
    1. The response either cites sources OR is a valid refusal
    2. All cited source numbers are within the valid range
    3. Content paragraphs have citations

    Args:
        response: The RAG response to validate.

    Returns:
        CitationReport with validation results.
    """
    answer = response.answer
    num_sources = len(response.retrieved_chunks)
    issues: list[str] = []

    # Check if this is a refusal (model correctly declined to answer)
    is_refusal = any(phrase in answer.lower() for phrase in _REFUSAL_PHRASES)

    if is_refusal:
        # Refusals are valid — the model correctly said "I don't know"
        return CitationReport(
            is_valid=True,
            is_refusal=True,
            cited_sources=set(),
            invalid_sources=set(),
            uncited_paragraphs=0,
            total_paragraphs=0,
            citation_coverage=0.0,
            issues=[],
        )

    # Find all [Source N] citations in the answer
    cited_numbers = {int(m) for m in _CITATION_PATTERN.findall(answer)}

    # Check for invalid source numbers (e.g., [Source 8] when only 5 sources exist)
    valid_range = set(range(1, num_sources + 1))
    invalid_sources = cited_numbers - valid_range

    if invalid_sources:
        issues.append(
            f"Invalid source numbers cited: {invalid_sources}. "
            f"Valid range: 1-{num_sources}"
        )

    # Check that content paragraphs have citations
    # Split answer into paragraphs and filter out empty lines and headers
    paragraphs = [
        p.strip()
        for p in answer.split("\n")
        if p.strip() and not p.strip().startswith("#")
    ]

    # Count paragraphs without any citation
    uncited = 0
    for para in paragraphs:
        # Skip very short lines (likely headers, bullet markers, etc.)
        if len(para) < 40:
            continue
        if not _CITATION_PATTERN.search(para):
            uncited += 1

    content_paragraphs = [p for p in paragraphs if len(p) >= 40]
    total_paragraphs = len(content_paragraphs)

    if uncited > 0:
        issues.append(
            f"{uncited}/{total_paragraphs} content paragraphs lack citations"
        )

    # No citations at all is a problem (unless it's a refusal, handled above)
    if not cited_numbers:
        issues.append("Answer contains no citations at all")

    # Calculate citation coverage
    # = what fraction of retrieved chunks were actually used?
    citation_coverage = len(cited_numbers & valid_range) / max(num_sources, 1)

    # Determine overall validity
    is_valid = len(issues) == 0

    report = CitationReport(
        is_valid=is_valid,
        is_refusal=False,
        cited_sources=cited_numbers,
        invalid_sources=invalid_sources,
        uncited_paragraphs=uncited,
        total_paragraphs=total_paragraphs,
        citation_coverage=citation_coverage,
        issues=issues,
    )

    if is_valid:
        logger.info(
            "Citation check PASSED: %d sources cited, %.0f%% coverage",
            len(cited_numbers),
            citation_coverage * 100,
        )
    else:
        logger.warning(
            "Citation check FAILED: %s",
            "; ".join(issues),
        )

    return report
