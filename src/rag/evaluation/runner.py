"""Evaluation runner — the grading system for the RAG pipeline.

HOW IT WORKS:
1. Loads all test cases from the golden dataset
2. Runs each question through the RAG pipeline
3. Grades each response on multiple dimensions
4. Calculates overall scores
5. Returns a detailed report

WHAT IT CHECKS FOR EACH QUESTION:

  If expected_behavior is "answer":
    - Did the system actually answer (not refuse)?          → behavior_correct
    - Did it cite the expected source documents?            → source_correct
    - Does the answer contain required keywords?            → content_correct
    - Are the citations valid ([Source N] format)?          → citations_valid

  If expected_behavior is "refuse":
    - Did the system correctly refuse to answer?            → behavior_correct
    - Did it avoid making up an answer?                     → (implicit in refusal)

INTERVIEW QUESTION: "How do you measure RAG quality in production?"
ANSWER: "We maintain a golden dataset of 23 test cases covering normal questions,
refusal scenarios, keyword queries, and cross-document questions. Every code change
runs against this dataset. We measure faithfulness, citation accuracy, and refusal
accuracy. CI blocks merges if scores drop below 85%."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag.evaluation.dataset import EvalCase, load_golden_dataset
from rag.generation.citation_validator import CitationReport, validate_citations
from rag.models import RAGResponse
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """Result of evaluating a single test case.

    Each field is a True/False check. True means the system did the right thing.
    """

    case_id: str
    question: str
    expected_behavior: str

    # Did the system answer when it should answer, or refuse when it should refuse?
    behavior_correct: bool

    # Did it cite the expected source documents?
    source_correct: bool

    # Does the answer contain the required keywords?
    content_correct: bool

    # Are the citations well-formed and valid?
    citations_valid: bool

    # Detailed info for debugging
    actual_answer: str = ""
    cited_sources: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    citation_issues: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0


@dataclass
class EvalReport:
    """Overall evaluation report across all test cases.

    This is what gets checked in CI. If overall_score < threshold,
    the PR is blocked.
    """

    # Individual results
    results: list[CaseResult]

    # Aggregate scores (0.0 to 1.0, where 1.0 = perfect)
    behavior_accuracy: float  # % of cases where answer/refuse was correct
    source_accuracy: float  # % of "answer" cases that cited the right sources
    content_accuracy: float  # % of "answer" cases with all required keywords
    citation_validity: float  # % of "answer" cases with valid citations
    overall_score: float  # weighted average of all metrics

    # Counts
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_latency: float

    def is_passing(self, threshold: float = 0.85) -> bool:
        """Check if the overall score meets the quality threshold.

        Args:
            threshold: Minimum overall score to pass (default 0.85 = 85%).

        Returns:
            True if the system passes the quality bar.
        """
        return self.overall_score >= threshold


def _check_behavior(
    case: EvalCase, response: RAGResponse, citation_report: CitationReport
) -> bool:
    """Check if the system answered or refused correctly."""
    if case.expected_behavior == "refuse":
        # Should have refused — check if it did
        return citation_report.is_refusal
    else:
        # Should have answered — check that it didn't refuse
        return not citation_report.is_refusal


def _check_sources(case: EvalCase, response: RAGResponse) -> tuple[bool, list[str]]:
    """Check if the expected source documents were cited."""
    if case.expected_behavior == "refuse" or not case.expected_sources:
        return True, []

    # Get the filenames of actually cited sources
    cited_files = set()
    for rc in response.citations:
        # Extract just the filename from the full path
        source_name = rc.chunk.source.split("/")[-1]
        cited_files.add(source_name)

    # Check if expected sources are among cited sources
    missing = [s for s in case.expected_sources if s not in cited_files]
    return len(missing) == 0, missing


def _check_content(case: EvalCase, response: RAGResponse) -> tuple[bool, list[str]]:
    """Check if the answer contains required keywords."""
    if case.expected_behavior == "refuse" or not case.must_contain:
        return True, []

    answer_lower = response.answer.lower()
    missing = [kw for kw in case.must_contain if kw.lower() not in answer_lower]
    return len(missing) == 0, missing


def run_evaluation(
    pipeline: RAGPipeline,
    dataset_path: Path,
    session_id: str | None = None,
) -> EvalReport:
    """Run the full evaluation against the golden dataset.

    This is the main function. It:
    1. Loads test cases
    2. Runs each through the pipeline
    3. Grades each response
    4. Calculates aggregate scores
    5. (If traced) Logs everything to Langfuse under one session

    Args:
        pipeline: The RAG pipeline to evaluate.
        dataset_path: Path to the golden dataset JSON file.
        session_id: If provided, groups all eval traces under this session
                    in Langfuse. Only works with TracedRAGPipeline.

    Returns:
        EvalReport with detailed results and aggregate scores.
    """
    cases = load_golden_dataset(dataset_path)
    results: list[CaseResult] = []

    logger.info("Starting evaluation: %d test cases", len(cases))

    for i, case in enumerate(cases, start=1):
        logger.info("[%d/%d] Evaluating: %s", i, len(cases), case.id)

        # Run the question through the pipeline and measure time
        # If pipeline is TracedRAGPipeline, pass session_id and user_id
        # If it's regular RAGPipeline, the extra kwargs are ignored
        start_time = time.time()
        try:
            # Try with session_id (works for TracedRAGPipeline)
            response, citation_report = pipeline.query(
                case.question,
                session_id=session_id,
                user_id="eval_bot",
            )
        except TypeError:
            # Falls back for regular RAGPipeline (doesn't accept session_id)
            response, citation_report = pipeline.query(case.question)
        latency = time.time() - start_time

        # Grade the response
        behavior_ok = _check_behavior(case, response, citation_report)
        source_ok, missing_sources = _check_sources(case, response)
        content_ok, missing_keywords = _check_content(case, response)
        citations_ok = citation_report.is_valid

        result = CaseResult(
            case_id=case.id,
            question=case.question,
            expected_behavior=case.expected_behavior,
            behavior_correct=behavior_ok,
            source_correct=source_ok,
            content_correct=content_ok,
            citations_valid=citations_ok,
            actual_answer=response.answer[:200],  # truncate for report
            cited_sources=[rc.chunk.source.split("/")[-1] for rc in response.citations],
            missing_keywords=missing_keywords,
            citation_issues=citation_report.issues,
            latency_seconds=latency,
        )
        results.append(result)

        status = "PASS" if (behavior_ok and source_ok and content_ok and citations_ok) else "FAIL"
        logger.info(
            "[%d/%d] %s — %s (behavior=%s, source=%s, content=%s, citations=%s, %.1fs)",
            i,
            len(cases),
            case.id,
            status,
            behavior_ok,
            source_ok,
            content_ok,
            citations_ok,
            latency,
        )

    # Calculate aggregate scores
    total = len(results)
    answer_cases = [r for r in results if r.expected_behavior == "answer"]
    refuse_cases = [r for r in results if r.expected_behavior == "refuse"]

    behavior_accuracy = sum(r.behavior_correct for r in results) / max(total, 1)

    source_accuracy = (
        sum(r.source_correct for r in answer_cases) / max(len(answer_cases), 1)
    )

    content_accuracy = (
        sum(r.content_correct for r in answer_cases) / max(len(answer_cases), 1)
    )

    citation_validity = (
        sum(r.citations_valid for r in answer_cases) / max(len(answer_cases), 1)
    )

    # Overall score: weighted average
    # Behavior is most important (can't get anything else right if you
    # answer when you should refuse, or vice versa)
    overall_score = (
        behavior_accuracy * 0.35
        + source_accuracy * 0.25
        + content_accuracy * 0.20
        + citation_validity * 0.20
    )

    passed = sum(
        1
        for r in results
        if r.behavior_correct and r.source_correct and r.content_correct and r.citations_valid
    )

    avg_latency = sum(r.latency_seconds for r in results) / max(total, 1)

    report = EvalReport(
        results=results,
        behavior_accuracy=behavior_accuracy,
        source_accuracy=source_accuracy,
        content_accuracy=content_accuracy,
        citation_validity=citation_validity,
        overall_score=overall_score,
        total_cases=total,
        passed_cases=passed,
        failed_cases=total - passed,
        avg_latency=avg_latency,
    )

    logger.info(
        "Evaluation complete: %d/%d passed (%.0f%%), overall_score=%.2f",
        passed,
        total,
        (passed / max(total, 1)) * 100,
        overall_score,
    )

    return report
