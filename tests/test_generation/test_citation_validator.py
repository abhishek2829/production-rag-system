"""Tests for citation validation."""

from rag.generation.citation_validator import validate_citations
from rag.models import Chunk, RAGResponse, RetrievedChunk


def _make_response(answer: str, num_chunks: int = 3) -> RAGResponse:
    """Helper to create a RAGResponse for testing."""
    chunks = [
        RetrievedChunk(
            chunk=Chunk(
                content=f"Content for chunk {i}",
                source=f"doc{i}.md",
                chunk_index=0,
            ),
            score=0.9 - i * 0.1,
        )
        for i in range(num_chunks)
    ]
    return RAGResponse(
        answer=answer,
        citations=[],  # We don't use this field in validation
        query="test question",
        retrieved_chunks=chunks,
    )


class TestCitationValidation:
    def test_valid_citations(self):
        response = _make_response(
            "RAG is a technique for grounding LLM responses [Source 1]. "
            "It reduces hallucination by using retrieved documents [Source 2]."
        )
        report = validate_citations(response)
        assert report.is_valid is True
        assert report.cited_sources == {1, 2}
        assert len(report.issues) == 0

    def test_detects_missing_citations(self):
        response = _make_response(
            "RAG is a technique for grounding LLM responses. "
            "It reduces hallucination by using retrieved documents."
        )
        report = validate_citations(response)
        assert report.is_valid is False
        assert "citation" in report.issues[0].lower()

    def test_detects_invalid_source_numbers(self):
        response = _make_response(
            "RAG is a technique [Source 1]. It uses many methods [Source 99].",
            num_chunks=3,
        )
        report = validate_citations(response)
        assert 99 in report.invalid_sources

    def test_recognizes_refusal(self):
        response = _make_response(
            "I don't have enough information in the provided documents "
            "to answer this question."
        )
        report = validate_citations(response)
        assert report.is_refusal is True
        assert report.is_valid is True  # Refusals are valid!

    def test_citation_coverage(self):
        response = _make_response(
            "According to the sources, RAG helps [Source 1]. "
            "It also improves accuracy [Source 2].",
            num_chunks=5,
        )
        report = validate_citations(response)
        # 2 out of 5 chunks were cited = 40% coverage
        assert report.citation_coverage == 2 / 5
