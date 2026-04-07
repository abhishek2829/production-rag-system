"""Main RAG pipeline — orchestrates the full retrieval and generation flow.

PHASE 1 FLOW (what we had before):
    Query → Vector Search → Top 5 chunks → Generate answer

PHASE 2 FLOW (what we have now):
    Query → Vector Search (top 10) ──┐
                                      ├→ Hybrid Fusion (RRF) → Re-rank → Top 5 → Generate → Validate
    Query → BM25 Search (top 10) ───┘

The key improvements:
1. Two retrieval methods instead of one (catches more relevant chunks)
2. Re-ranking with a cross-encoder (more accurate ranking)
3. Citation validation (catches hallucination programmatically)
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag.chunking.token_chunker import create_chunker
from rag.config import Settings, get_settings
from rag.generation.citation_validator import CitationReport, validate_citations
from rag.generation.generator import AnswerGenerator
from rag.ingestion.pipeline import ingest_directory
from rag.models import Chunk, RAGResponse, RetrievedChunk
from rag.retrieval.bm25_retriever import BM25Retriever
from rag.retrieval.hybrid_retriever import reciprocal_rank_fusion
from rag.retrieval.reranker import Reranker
from rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline with hybrid retrieval and re-ranking.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest(Path("./data/documents"))
        response, report = pipeline.query("What is X?")
        print(response.answer)
        print(report.is_valid)  # True if citations are valid
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._chunker = create_chunker(self._settings)
        self._vector_store = VectorStore(self._settings)
        self._generator = AnswerGenerator(self._settings)

        # Phase 2 components — initialized lazily or conditionally
        self._bm25_retriever: BM25Retriever | None = None
        self._reranker: Reranker | None = None

        # Load re-ranker if enabled
        if self._settings.use_reranker:
            self._reranker = Reranker(self._settings.reranker_model)

        # Track all ingested chunks for BM25 (it needs them in memory)
        self._all_chunks: list[Chunk] = []

    def ingest(self, directory: Path) -> int:
        """Ingest documents from a directory into the vector store.

        Also builds the BM25 index if hybrid retrieval is enabled.

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

        # Step 3: Embed and store in vector DB
        count = self._vector_store.add_chunks(chunks)

        # Step 4: Build BM25 index if hybrid retrieval is enabled
        if self._settings.use_hybrid_retrieval:
            self._all_chunks = chunks
            self._bm25_retriever = BM25Retriever(chunks)
            logger.info("BM25 index built with %d chunks", len(chunks))

        logger.info(
            "Ingestion complete: %d documents → %d chunks stored",
            len(documents),
            count,
        )
        return count

    def query(
        self, question: str, top_k: int | None = None
    ) -> tuple[RAGResponse, CitationReport]:
        """Query the RAG system with hybrid retrieval and re-ranking.

        The full flow:
        1. Retrieve candidates from vector store AND BM25 (if enabled)
        2. Combine results using Reciprocal Rank Fusion
        3. Re-rank with cross-encoder (if enabled)
        4. Generate cited answer with Claude
        5. Validate citations

        Args:
            question: The user's question.
            top_k: Override for final number of chunks.

        Returns:
            Tuple of (RAGResponse, CitationReport).
        """
        final_k = top_k or self._settings.retrieval_top_k
        initial_k = self._settings.retrieval_initial_k

        logger.info("Query: %s", question)

        # --- Step 1: Retrieve candidates ---
        # Get more candidates than we need (initial_k > final_k)
        # because re-ranking will filter down to the best ones
        vector_results = self._vector_store.search(question, top_k=initial_k)

        if self._settings.use_hybrid_retrieval and self._bm25_retriever:
            # Get BM25 results too
            bm25_results = self._bm25_retriever.search(question, top_k=initial_k)

            # --- Step 2: Combine with Reciprocal Rank Fusion ---
            combined = reciprocal_rank_fusion(
                [vector_results, bm25_results],
                top_k=initial_k,
            )
            logger.info(
                "Hybrid retrieval: %d vector + %d BM25 → %d combined",
                len(vector_results),
                len(bm25_results),
                len(combined),
            )
        else:
            combined = vector_results

        # --- Step 3: Re-rank (if enabled) ---
        if self._reranker and combined:
            retrieved = self._reranker.rerank(question, combined, top_k=final_k)
        else:
            retrieved = combined[:final_k]

        # --- Step 4: Generate cited answer ---
        response = self._generator.generate(question, retrieved)

        # --- Step 5: Validate citations ---
        report = validate_citations(response)

        logger.info(
            "Response: %d chars, %d/%d chunks cited, citations_valid=%s",
            len(response.answer),
            len(response.citations),
            len(response.retrieved_chunks),
            report.is_valid,
        )

        if not report.is_valid:
            logger.warning("Citation issues: %s", "; ".join(report.issues))

        return response, report

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently in the vector store."""
        return self._vector_store.count
