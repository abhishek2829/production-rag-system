"""Token-based document chunking.

Uses tiktoken (OpenAI's tokenizer) to ensure chunks are measured in the same
token space as the embedding model. This is critical — character-based chunking
produces inconsistent embedding quality because token counts vary by content.

Design choices:
- Token-based measurement using tiktoken (matches OpenAI embedding model)
- Configurable size and overlap
- Preserves sentence boundaries where possible (no mid-sentence splits)
- Tracks chunk index for tracing back to source
"""

from __future__ import annotations

import logging
import re

import tiktoken

from rag.config import Settings
from rag.models import Chunk, Document

logger = logging.getLogger(__name__)

# Sentence boundary pattern — handles common sentence endings
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving the delimiter."""
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


class TokenChunker:
    """Chunks documents by token count with sentence-boundary awareness.

    Args:
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between consecutive chunks in tokens.
        model_name: Tiktoken encoding model name (should match embedding model).
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        model_name: str = "cl100k_base",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoding = tiktoken.get_encoding(model_name)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._encoding.encode(text))

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into token-counted chunks.

        Strategy:
        1. Split document into sentences
        2. Accumulate sentences until we hit chunk_size tokens
        3. Emit the chunk, then back up by chunk_overlap tokens worth of sentences
        4. Continue until all sentences are consumed

        This preserves sentence boundaries while maintaining consistent token counts.
        """
        sentences = _split_into_sentences(document.content)

        if not sentences:
            logger.warning("Document %s produced no sentences after splitting", document.source)
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If a single sentence exceeds chunk_size, it becomes its own chunk
            if sentence_tokens > self.chunk_size:
                # Flush current buffer first
                if current_sentences:
                    chunks.append(self._make_chunk(current_sentences, document, len(chunks)))
                    current_sentences = []
                    current_tokens = 0

                # Add oversized sentence as its own chunk
                chunks.append(self._make_chunk([sentence], document, len(chunks)))
                continue

            # Would adding this sentence exceed chunk_size?
            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                # Emit current chunk
                chunks.append(self._make_chunk(current_sentences, document, len(chunks)))

                # Calculate overlap: keep trailing sentences that fit within overlap
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_sentences:
            chunks.append(self._make_chunk(current_sentences, document, len(chunks)))

        logger.info(
            "Chunked %s into %d chunks (avg %d tokens/chunk)",
            document.source,
            len(chunks),
            sum(self._count_tokens(c.content) for c in chunks) // max(len(chunks), 1),
        )

        return chunks

    def _make_chunk(self, sentences: list[str], document: Document, index: int) -> Chunk:
        """Create a Chunk from accumulated sentences."""
        content = " ".join(sentences)
        return Chunk(
            content=content,
            source=document.source,
            chunk_index=index,
            metadata={
                **document.metadata,
                "doc_type": document.doc_type.value,
                "token_count": self._count_tokens(content),
            },
        )

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents, returning a flat list of all chunks."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))

        logger.info("Total: %d chunks from %d documents", len(all_chunks), len(documents))
        return all_chunks


def create_chunker(settings: Settings) -> TokenChunker:
    """Factory function to create a chunker from settings."""
    return TokenChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
