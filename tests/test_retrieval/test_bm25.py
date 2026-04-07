"""Tests for BM25 keyword retriever."""

from rag.models import Chunk
from rag.retrieval.bm25_retriever import BM25Retriever, _tokenize


class TestTokenize:
    def test_lowercases_text(self):
        assert _tokenize("Hello WORLD") == ["hello", "world"]

    def test_removes_punctuation(self):
        assert _tokenize("hello, world!") == ["hello", "world"]

    def test_empty_string(self):
        assert _tokenize("") == []


class TestBM25Retriever:
    def _make_chunks(self) -> list[Chunk]:
        """Create test chunks with known content."""
        return [
            Chunk(
                content="ChromaDB uses HNSW indexing for fast similarity search",
                source="vectordb.md",
                chunk_index=0,
            ),
            Chunk(
                content="RAG reduces hallucination by grounding answers in documents",
                source="rag.md",
                chunk_index=0,
            ),
            Chunk(
                content="BM25 is a keyword matching algorithm used in search engines",
                source="search.md",
                chunk_index=0,
            ),
        ]

    def test_finds_exact_keyword(self):
        chunks = self._make_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.search("HNSW", top_k=3)
        # The chunk about HNSW should rank first
        assert len(results) > 0
        assert "HNSW" in results[0].chunk.content

    def test_finds_keyword_case_insensitive(self):
        chunks = self._make_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.search("chromadb", top_k=3)
        assert len(results) > 0
        assert "ChromaDB" in results[0].chunk.content

    def test_returns_empty_for_no_match(self):
        chunks = self._make_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.search("biryani recipe", top_k=3)
        # BM25 should return nothing (or very low scores) for unrelated queries
        # Chunks with zero score are filtered out
        assert len(results) == 0

    def test_respects_top_k(self):
        chunks = self._make_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.search("search indexing algorithm", top_k=1)
        assert len(results) <= 1

    def test_empty_chunks(self):
        retriever = BM25Retriever([])
        results = retriever.search("anything", top_k=5)
        assert results == []

    def test_empty_query(self):
        chunks = self._make_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.search("", top_k=5)
        assert results == []
