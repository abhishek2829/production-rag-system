"""Tests for evaluation runner logic (without API calls)."""

from rag.evaluation.runner import (
    CaseResult,
    EvalReport,
    _check_behavior,
    _check_content,
    _check_sources,
)
from rag.evaluation.dataset import EvalCase
from rag.generation.citation_validator import CitationReport
from rag.models import Chunk, RAGResponse, RetrievedChunk


def _make_response(
    answer: str, cited_sources: list[str] | None = None
) -> RAGResponse:
    """Helper to create a test RAGResponse."""
    citations = []
    if cited_sources:
        for i, source in enumerate(cited_sources):
            citations.append(
                RetrievedChunk(
                    chunk=Chunk(
                        content=f"Content from {source}",
                        source=f"data/documents/{source}",
                        chunk_index=0,
                    ),
                    score=0.9,
                )
            )
    return RAGResponse(
        answer=answer,
        citations=citations,
        query="test",
        retrieved_chunks=citations,
    )


def _make_citation_report(is_refusal: bool = False) -> CitationReport:
    """Helper to create a test CitationReport."""
    return CitationReport(
        is_valid=True,
        is_refusal=is_refusal,
        cited_sources=set(),
        invalid_sources=set(),
        uncited_paragraphs=0,
        total_paragraphs=0,
        citation_coverage=0.0,
        issues=[],
    )


class TestCheckBehavior:
    def test_answer_case_not_refusal(self):
        case = EvalCase(
            id="t", question="q", expected_behavior="answer"
        )
        report = _make_citation_report(is_refusal=False)
        response = _make_response("Some answer")
        assert _check_behavior(case, response, report) is True

    def test_answer_case_but_refused(self):
        case = EvalCase(
            id="t", question="q", expected_behavior="answer"
        )
        report = _make_citation_report(is_refusal=True)
        response = _make_response("I don't have enough info")
        assert _check_behavior(case, response, report) is False

    def test_refuse_case_and_refused(self):
        case = EvalCase(
            id="t", question="q", expected_behavior="refuse"
        )
        report = _make_citation_report(is_refusal=True)
        response = _make_response("I don't have enough info")
        assert _check_behavior(case, response, report) is True

    def test_refuse_case_but_answered(self):
        case = EvalCase(
            id="t", question="q", expected_behavior="refuse"
        )
        report = _make_citation_report(is_refusal=False)
        response = _make_response("Here is an answer")
        assert _check_behavior(case, response, report) is False


class TestCheckSources:
    def test_correct_source_cited(self):
        case = EvalCase(
            id="t",
            question="q",
            expected_behavior="answer",
            expected_sources=["rag.md"],
        )
        response = _make_response(
            "Answer [Source 1]", cited_sources=["rag.md"]
        )
        ok, missing = _check_sources(case, response)
        assert ok is True
        assert missing == []

    def test_missing_source(self):
        case = EvalCase(
            id="t",
            question="q",
            expected_behavior="answer",
            expected_sources=["rag.md", "vectordb.md"],
        )
        response = _make_response(
            "Answer [Source 1]", cited_sources=["rag.md"]
        )
        ok, missing = _check_sources(case, response)
        assert ok is False
        assert "vectordb.md" in missing

    def test_refuse_case_skips_source_check(self):
        case = EvalCase(
            id="t", question="q", expected_behavior="refuse"
        )
        response = _make_response("I don't know")
        ok, missing = _check_sources(case, response)
        assert ok is True


class TestCheckContent:
    def test_all_keywords_present(self):
        case = EvalCase(
            id="t",
            question="q",
            expected_behavior="answer",
            must_contain=["retrieval", "documents"],
        )
        response = _make_response(
            "Retrieval augmented generation uses documents"
        )
        ok, missing = _check_content(case, response)
        assert ok is True
        assert missing == []

    def test_missing_keyword(self):
        case = EvalCase(
            id="t",
            question="q",
            expected_behavior="answer",
            must_contain=["retrieval", "quantum"],
        )
        response = _make_response("Retrieval augmented generation")
        ok, missing = _check_content(case, response)
        assert ok is False
        assert "quantum" in missing

    def test_case_insensitive(self):
        case = EvalCase(
            id="t",
            question="q",
            expected_behavior="answer",
            must_contain=["RAG"],
        )
        response = _make_response("rag is a technique")
        ok, missing = _check_content(case, response)
        assert ok is True


class TestEvalReport:
    def test_passing_threshold(self):
        report = EvalReport(
            results=[],
            behavior_accuracy=0.9,
            source_accuracy=0.9,
            content_accuracy=0.9,
            citation_validity=0.9,
            overall_score=0.9,
            total_cases=10,
            passed_cases=9,
            failed_cases=1,
            avg_latency=2.0,
        )
        assert report.is_passing(0.85) is True
        assert report.is_passing(0.95) is False
