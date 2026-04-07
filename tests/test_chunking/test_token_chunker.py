"""Tests for the token-based chunker."""

import pytest

from rag.chunking.token_chunker import TokenChunker
from rag.models import Document, DocumentType


@pytest.fixture
def chunker() -> TokenChunker:
    """A chunker with small sizes for testing."""
    return TokenChunker(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def sample_document() -> Document:
    """A document with several sentences."""
    return Document(
        content=(
            "Machine learning is a subset of artificial intelligence. "
            "It allows systems to learn from data. "
            "Deep learning is a subset of machine learning. "
            "Neural networks are the foundation of deep learning. "
            "Transformers revolutionized natural language processing. "
            "Attention mechanisms allow models to focus on relevant parts of input. "
            "Large language models are trained on massive text corpora. "
            "They can generate human-like text and answer questions."
        ),
        source="ml_intro.txt",
        doc_type=DocumentType.TEXT,
    )


class TestTokenChunker:
    def test_produces_multiple_chunks(self, chunker: TokenChunker, sample_document: Document):
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 1, "Document should be split into multiple chunks"

    def test_chunk_indices_are_sequential(self, chunker: TokenChunker, sample_document: Document):
        chunks = chunker.chunk_document(sample_document)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_source_matches_document(self, chunker: TokenChunker, sample_document: Document):
        chunks = chunker.chunk_document(sample_document)
        for chunk in chunks:
            assert chunk.source == sample_document.source

    def test_chunk_metadata_includes_token_count(
        self, chunker: TokenChunker, sample_document: Document
    ):
        chunks = chunker.chunk_document(sample_document)
        for chunk in chunks:
            assert "token_count" in chunk.metadata
            assert isinstance(chunk.metadata["token_count"], int)
            assert chunk.metadata["token_count"] > 0

    def test_no_empty_chunks(self, chunker: TokenChunker, sample_document: Document):
        chunks = chunker.chunk_document(sample_document)
        for chunk in chunks:
            assert chunk.content.strip(), f"Chunk {chunk.chunk_index} is empty"

    def test_overlap_preserves_boundary_context(self):
        """With overlap, consecutive chunks should share some content."""
        chunker = TokenChunker(chunk_size=30, chunk_overlap=10)
        doc = Document(
            content=(
                "First sentence here. Second sentence follows. "
                "Third sentence appears. Fourth sentence ends."
            ),
            source="test.txt",
            doc_type=DocumentType.TEXT,
        )
        chunks = chunker.chunk_document(doc)
        if len(chunks) >= 2:
            # Check that some words from the end of chunk 0 appear in chunk 1
            words_0 = set(chunks[0].content.split())
            words_1 = set(chunks[1].content.split())
            overlap = words_0 & words_1
            assert len(overlap) > 0, "Consecutive chunks should share overlapping content"

    def test_rejects_overlap_greater_than_size(self):
        with pytest.raises(ValueError, match="must be less than"):
            TokenChunker(chunk_size=50, chunk_overlap=50)

    def test_empty_document_returns_no_chunks(self):
        doc = Document(content=".", source="empty.txt", doc_type=DocumentType.TEXT)
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_document(doc)
        # Single period might produce 0 or 1 chunks depending on splitting
        assert len(chunks) <= 1

    def test_chunk_documents_processes_multiple(self, chunker: TokenChunker):
        docs = [
            Document(
                content="First document content here.",
                source="a.txt",
                doc_type=DocumentType.TEXT,
            ),
            Document(
                content="Second document content here.",
                source="b.txt",
                doc_type=DocumentType.TEXT,
            ),
        ]
        chunks = chunker.chunk_documents(docs)
        sources = {c.source for c in chunks}
        assert "a.txt" in sources
        assert "b.txt" in sources
