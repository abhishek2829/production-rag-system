"""Main RAG pipeline — orchestrates ingestion, chunking, storage, retrieval, and generation.

This is the top-level entry point that wires all components together.
Each component is independently testable, but the pipeline coordinates the flow.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag.chunking.token_chunker import create_chunker
from rag.config import Settings, get_settings
from rag.generation.generator import AnswerGenerator
from rag.ingestion.pipeline import ingest_directory
from rag.models import RAGResponse
from rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest(Path("./data/documents"))
        response = pipeline.query("What is X?")
        print(response.answer)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._chunker = create_chunker(self._settings)
        self._vector_store = VectorStore(self._settings)
        self._generator = AnswerGenerator(self._settings)

    def ingest(self, directory: Path) -> int:
        """Ingest documents from a directory into the vector store.

        Args:
            directory: Path to directory containing documents.

        Returns:
            Number of chunks stored.
        """
        logger.info("Starting ingestion from %s", directory)

        # Step 1: Load documents
        documents = ingest_directory(directory)
        if not documents:
            logger.warning("No documents found to ingest")
            return 0

        # Step 2: Chunk documents
        chunks = self._chunker.chunk_documents(documents)

        # Step 3: Embed and store
        count = self._vector_store.add_chunks(chunks)

        logger.info(
            "Ingestion complete: %d documents → %d chunks stored",
            len(documents),
            count,
        )
        return count

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Query the RAG system.

        Args:
            question: The user's question.
            top_k: Override for number of chunks to retrieve.

        Returns:
            RAGResponse with cited answer and metadata.
        """
        logger.info("Query: %s", question)

        # Step 1: Retrieve relevant chunks
        retrieved = self._vector_store.search(question, top_k=top_k)

        # Step 2: Generate cited answer
        response = self._generator.generate(question, retrieved)

        logger.info(
            "Response: %d chars, %d/%d chunks cited",
            len(response.answer),
            len(response.citations),
            len(response.retrieved_chunks),
        )
        return response

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently in the vector store."""
        return self._vector_store.count
