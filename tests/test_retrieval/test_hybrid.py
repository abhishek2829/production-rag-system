"""Tests for hybrid retrieval (Reciprocal Rank Fusion)."""

from rag.models import Chunk, RetrievedChunk
from rag.retrieval.hybrid_retriever import reciprocal_rank_fusion


def _make_rc(source: str, index: int, score: float) -> RetrievedChunk:
    """Helper to create a RetrievedChunk for testing."""
    return RetrievedChunk(
        chunk=Chunk(content=f"Content from {source}", source=source, chunk_index=index),
        score=score,
    )


class TestReciprocalRankFusion:
    def test_chunks_in_both_lists_rank_higher(self):
        """A chunk appearing in BOTH retriever results should rank highest."""
        # Chunk A appears in both lists, B only in vector, C only in BM25
        vector_results = [
            _make_rc("a.md", 0, 0.9),  # Chunk A — rank 1 in vector
            _make_rc("b.md", 0, 0.8),  # Chunk B — rank 2 in vector
        ]
        bm25_results = [
            _make_rc("c.md", 0, 5.0),  # Chunk C — rank 1 in BM25
            _make_rc("a.md", 0, 3.0),  # Chunk A — rank 2 in BM25
        ]

        results = reciprocal_rank_fusion([vector_results, bm25_results], top_k=3)

        # Chunk A should be first because it appeared in BOTH lists
        assert results[0].chunk.source == "a.md"

    def test_returns_correct_number(self):
        list1 = [_make_rc("a.md", 0, 0.9), _make_rc("b.md", 0, 0.8)]
        list2 = [_make_rc("c.md", 0, 5.0), _make_rc("d.md", 0, 3.0)]

        results = reciprocal_rank_fusion([list1, list2], top_k=2)
        assert len(results) == 2

    def test_handles_single_list(self):
        """Should work with just one list (degrades to simple ranking)."""
        results_list = [
            _make_rc("a.md", 0, 0.9),
            _make_rc("b.md", 0, 0.8),
        ]
        results = reciprocal_rank_fusion([results_list], top_k=2)
        assert len(results) == 2

    def test_handles_empty_lists(self):
        results = reciprocal_rank_fusion([[], []], top_k=5)
        assert results == []

    def test_deduplicates_across_lists(self):
        """Same chunk in both lists should appear only once in output."""
        list1 = [_make_rc("a.md", 0, 0.9)]
        list2 = [_make_rc("a.md", 0, 5.0)]

        results = reciprocal_rank_fusion([list1, list2], top_k=5)
        assert len(results) == 1
