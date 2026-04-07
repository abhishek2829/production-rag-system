"""Tests for domain models."""

import pytest

from rag.models import Chunk, Document, DocumentType, RetrievedChunk


class TestDocument:
    def test_creates_valid_document(self):
        doc = Document(
            content="Hello world",
            source="test.txt",
            doc_type=DocumentType.TEXT,
        )
        assert doc.content == "Hello world"
        assert doc.source == "test.txt"
        assert doc.doc_type == DocumentType.TEXT
        assert doc.metadata == {}

    def test_rejects_empty_content(self):
        with pytest.raises(ValueError, match="empty content"):
            Document(content="   ", source="test.txt", doc_type=DocumentType.TEXT)

    def test_is_immutable(self):
        doc = Document(content="Hello", source="test.txt", doc_type=DocumentType.TEXT)
        with pytest.raises(AttributeError):
            doc.content = "Modified"  # type: ignore[misc]


class TestChunk:
    def test_chunk_id_is_deterministic(self):
        chunk = Chunk(content="Hello", source="doc.pdf", chunk_index=3)
        assert chunk.chunk_id == "doc.pdf::chunk_3"

    def test_same_inputs_produce_same_id(self):
        c1 = Chunk(content="Hello", source="doc.pdf", chunk_index=0)
        c2 = Chunk(content="Hello", source="doc.pdf", chunk_index=0)
        assert c1.chunk_id == c2.chunk_id

    def test_different_index_different_id(self):
        c1 = Chunk(content="Hello", source="doc.pdf", chunk_index=0)
        c2 = Chunk(content="Hello", source="doc.pdf", chunk_index=1)
        assert c1.chunk_id != c2.chunk_id


class TestRetrievedChunk:
    def test_wraps_chunk_with_score(self):
        chunk = Chunk(content="Test", source="s.txt", chunk_index=0)
        rc = RetrievedChunk(chunk=chunk, score=0.95)
        assert rc.score == 0.95
        assert rc.chunk.content == "Test"
