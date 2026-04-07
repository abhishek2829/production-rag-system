"""Vector store abstraction over ChromaDB with local embeddings.

Design decisions:
- We use sentence-transformers for LOCAL embeddings — no API key needed,
  no per-request cost, and works offline. This is a real production pattern
  for cost-sensitive deployments.
- We wrap ChromaDB behind our own interface to enable swapping stores later.
- Chunk IDs are deterministic (from chunk_id property) to enable idempotent upserts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from rag.config import Settings
from rag.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store with local sentence-transformer embeddings.

    Handles embedding generation locally (no API calls) and similarity search.
    Uses deterministic chunk IDs for idempotent upserts — re-ingesting
    the same document updates existing chunks instead of duplicating.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # Initialize LOCAL embedding model — no API key needed
        logger.info("Loading embedding model: %s", settings.embedding_model)
        self._embed_model = SentenceTransformer(settings.embedding_model)

        # Initialize ChromaDB with persistent storage
        persist_dir = str(settings.chroma_persist_dir)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Vector store initialized: collection='%s', persist_dir='%s', existing_count=%d",
            settings.chroma_collection_name,
            persist_dir,
            self._collection.count(),
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        """Embed and store chunks in the vector store.

        Embeddings are generated locally using sentence-transformers.
        No API calls, no rate limits, no cost per request.

        Args:
            chunks: List of chunks to embed and store.
            batch_size: Number of chunks to embed per batch.

        Returns:
            Number of chunks added/updated.
        """
        if not chunks:
            return 0

        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            texts = [chunk.content for chunk in batch]
            ids = [chunk.chunk_id for chunk in batch]
            metadatas = [
                {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    **{k: str(v) for k, v in chunk.metadata.items()},
                }
                for chunk in batch
            ]

            # Generate embeddings LOCALLY — no API call
            embeddings = self._embed_model.encode(texts).tolist()

            # Upsert into ChromaDB (idempotent)
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            total_added += len(batch)
            logger.info(
                "Embedded batch %d-%d of %d chunks",
                i + 1,
                min(i + batch_size, len(chunks)),
                len(chunks),
            )

        logger.info("Vector store now contains %d chunks", self._collection.count())
        return total_added

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Search for chunks similar to the query.

        Args:
            query: The search query text.
            top_k: Number of results to return. Defaults to settings.retrieval_top_k.

        Returns:
            List of RetrievedChunk sorted by relevance (highest score first).
        """
        k = top_k or self._settings.retrieval_top_k

        # Embed query locally
        query_embedding = self._embed_model.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []

        if results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],  # type: ignore[index]
                results["distances"][0],  # type: ignore[index]
                strict=True,
            ):
                chunk = Chunk(
                    content=doc,
                    source=metadata.get("source", "unknown"),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ("source", "chunk_index")
                    },
                )
                # Convert cosine distance to similarity: similarity = 1 - distance
                score = 1.0 - distance
                retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        logger.info(
            "Retrieved %d chunks for query: '%s...'", len(retrieved), query[:50]
        )
        return retrieved

    @property
    def count(self) -> int:
        """Number of chunks in the vector store."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete all chunks from the collection. Use with caution."""
        self._client.delete_collection(self._settings.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Vector store reset: all chunks deleted")
