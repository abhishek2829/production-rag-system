"""Hybrid retriever — combines vector search and BM25 keyword search.

THE PROBLEM:
- Vector search (embedding-based) understands MEANING but misses exact keywords
- BM25 (keyword-based) finds exact WORDS but doesn't understand meaning

THE SOLUTION:
Combine both results using a technique called Reciprocal Rank Fusion (RRF).

HOW RRF WORKS (simple version):
Imagine two ranked lists:

    Vector search results:       BM25 results:
    1. Chunk A (rank 1)          1. Chunk C (rank 1)
    2. Chunk B (rank 2)          2. Chunk A (rank 2)
    3. Chunk C (rank 3)          3. Chunk D (rank 3)

RRF gives each chunk a score based on its RANK (position), not its raw score.
The formula is: RRF_score = 1 / (k + rank)    where k=60 (a constant)

    Chunk A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325  ← appears in BOTH lists!
    Chunk C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
    Chunk B: 1/(60+2) + 0        = 0.0161                      ← only in vector
    Chunk D: 0        + 1/(60+3) = 0.0159                      ← only in BM25

Final ranking: A > C > B > D

KEY INSIGHT: Chunk A ranks highest because it appeared in BOTH lists.
If both vector search AND keyword search agree a chunk is relevant,
it's probably very relevant. This is the power of hybrid retrieval.

INTERVIEW QUESTION: "Why RRF instead of just averaging scores?"
ANSWER: "Vector scores (0-1 cosine similarity) and BM25 scores (0 to ~25)
are on completely different scales. You can't average them meaningfully.
RRF only uses RANK positions, so the scale doesn't matter."
"""

from __future__ import annotations

import logging
from collections import defaultdict

from rag.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

# RRF constant — 60 is the standard value from the original paper.
# Higher k = less emphasis on top ranks, more democratic scoring.
_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    This is the algorithm that merges vector search and BM25 results
    into a single ranked list. Chunks that appear in multiple lists
    get higher scores.

    Args:
        result_lists: List of ranked result lists (e.g., [vector_results, bm25_results]).
        top_k: Number of results to return after fusion.

    Returns:
        Merged and re-ranked list of RetrievedChunk.
    """
    # Track RRF scores by chunk_id
    # defaultdict(float) means: if a key doesn't exist, it starts at 0.0
    rrf_scores: dict[str, float] = defaultdict(float)

    # Track the actual chunk objects (we need them for the final output)
    chunk_map: dict[str, Chunk] = {}

    for result_list in result_lists:
        for rank, retrieved_chunk in enumerate(result_list, start=1):
            chunk_id = retrieved_chunk.chunk.chunk_id

            # RRF formula: 1 / (k + rank)
            # Rank 1 → 1/61 = 0.0164
            # Rank 2 → 1/62 = 0.0161
            # Rank 5 → 1/65 = 0.0154
            # The scores decrease slowly, so even rank 10 still contributes
            rrf_scores[chunk_id] += 1.0 / (_RRF_K + rank)

            # Keep the chunk object (first occurrence wins)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = retrieved_chunk.chunk

    # Sort by RRF score (highest first) and take top_k
    sorted_chunk_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]  # type: ignore[arg-type]

    results = [
        RetrievedChunk(chunk=chunk_map[chunk_id], score=rrf_scores[chunk_id])
        for chunk_id in sorted_chunk_ids
    ]

    logger.info(
        "RRF fusion: %d unique chunks from %d lists → top %d returned",
        len(rrf_scores),
        len(result_lists),
        len(results),
    )

    return results
