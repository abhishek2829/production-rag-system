"""Cross-encoder re-ranker for improving retrieval precision.

THE PROBLEM WITH INITIAL RETRIEVAL:
Both vector search and BM25 are "bi-encoders" — they encode the query and
each chunk SEPARATELY, then compare the encodings. This is fast (you can
pre-compute chunk encodings) but not very accurate because the query and
chunk never "see" each other.

HOW A CROSS-ENCODER IS DIFFERENT:
A cross-encoder takes the query AND a chunk TOGETHER as input, and outputs
a single relevance score. It's like the difference between:

    Bi-encoder (fast, less accurate):
        "Encode query separately → encode chunk separately → compare"
        Like judging if two puzzle pieces fit by looking at photos of each

    Cross-encoder (slow, more accurate):
        "Look at query AND chunk together → score how well they match"
        Like actually holding two puzzle pieces next to each other

WHY WE USE BOTH:
- Step 1: Bi-encoder (vector search + BM25) quickly finds 10-20 candidates
           from thousands of chunks. Speed matters here.
- Step 2: Cross-encoder carefully re-scores those 10-20 candidates.
           Accuracy matters here, and 10-20 is small enough to be fast.

This two-stage approach is called "retrieve then re-rank" and is the
standard pattern in production search systems (Google uses this too).

INTERVIEW QUESTION: "Why not just use the cross-encoder for everything?"
ANSWER: "Cross-encoders need to process every (query, chunk) pair individually.
With 10,000 chunks, that's 10,000 forward passes through the model — way too
slow for real-time queries. So we use fast retrieval first to narrow down
candidates, then re-rank only the top results."
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from rag.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Default cross-encoder model — small, fast, and effective for re-ranking
# This model was trained specifically for the task of scoring
# (query, passage) pairs for relevance
_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Re-ranks retrieved chunks using a cross-encoder model.

    The cross-encoder reads the query and each chunk together,
    producing a more accurate relevance score than the initial retrieval.

    The model runs LOCALLY on your machine — no API calls, no cost.
    It's ~80MB and takes ~50ms per chunk on a modern CPU.
    """

    def __init__(self, model_name: str = _DEFAULT_RERANKER_MODEL) -> None:
        """Load the cross-encoder model.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
                       Default is ms-marco-MiniLM which is small and fast.
        """
        logger.info("Loading re-ranker model: %s", model_name)
        self._model = CrossEncoder(model_name)
        logger.info("Re-ranker model loaded")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Re-rank retrieved chunks by cross-encoder relevance score.

        Args:
            query: The user's original question.
            chunks: Chunks from initial retrieval (vector + BM25).
            top_k: Number of top results to return after re-ranking.

        Returns:
            Re-ranked list of RetrievedChunk (highest relevance first).
        """
        if not chunks:
            return []

        # Create (query, chunk_text) pairs for the cross-encoder
        # The model needs to see both the question and the chunk together
        pairs = [[query, chunk.chunk.content] for chunk in chunks]

        # Score all pairs — the model returns a relevance score for each
        # Higher score = more relevant to the query
        scores = self._model.predict(pairs)

        # Attach new scores to chunks and sort
        reranked: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores, strict=True):
            reranked.append(
                RetrievedChunk(
                    chunk=chunk.chunk,
                    score=float(score),  # replace old score with re-ranker score
                )
            )

        # Sort by new cross-encoder score (highest first)
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Take top_k
        reranked = reranked[:top_k]

        logger.info(
            "Re-ranked %d chunks → top %d (scores: %.3f to %.3f)",
            len(chunks),
            len(reranked),
            reranked[0].score if reranked else 0,
            reranked[-1].score if reranked else 0,
        )

        return reranked
