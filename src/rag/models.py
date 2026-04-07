"""Domain models for the RAG system.

We define our own Document model rather than coupling to LangChain's Document class.
This gives us control over the schema, makes testing easier, and prevents vendor lock-in.
We convert to/from LangChain Documents only at integration boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DocumentType(StrEnum):
    """Supported document source types."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


@dataclass(frozen=True)
class Document:
    """A source document before chunking.

    Attributes:
        content: The full text content of the document.
        source: Original file path or URL.
        doc_type: The format the document was ingested from.
        metadata: Arbitrary metadata (title, author, page numbers, etc.).
    """

    content: str
    source: str
    doc_type: DocumentType
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError(f"Document from {self.source} has empty content")


@dataclass(frozen=True)
class Chunk:
    """A chunk of a document, ready for embedding.

    Attributes:
        content: The chunk text.
        source: Original document source (file path or URL).
        chunk_index: Position of this chunk within the source document.
        metadata: Inherited + chunk-specific metadata.
    """

    content: str
    source: str
    chunk_index: int
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Deterministic ID for deduplication and tracing."""
        return f"{self.source}::chunk_{self.chunk_index}"


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from retrieval with a relevance score.

    Attributes:
        chunk: The retrieved chunk.
        score: Similarity/relevance score (higher is better).
    """

    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    """The final response from the RAG pipeline.

    Attributes:
        answer: The generated answer text.
        citations: List of chunks that were cited in the answer.
        query: The original user query.
        retrieved_chunks: All chunks that were retrieved (not just cited).
    """

    answer: str
    citations: list[RetrievedChunk]
    query: str
    retrieved_chunks: list[RetrievedChunk]
