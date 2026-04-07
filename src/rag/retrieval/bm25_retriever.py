"""BM25 keyword-based retriever.

BM25 (Best Match 25) is a classic information retrieval algorithm from the 1990s.
It ranks documents by counting how often search words appear in them, with two
smart adjustments:

1. RARE WORDS MATTER MORE: If you search "HNSW indexing", the word "HNSW" is
   rare (appears in few chunks) so it gets a high score. The word "the" appears
   everywhere, so it gets almost zero score. This is called IDF
   (Inverse Document Frequency).

2. DIMINISHING RETURNS: If a word appears 10 times vs 5 times in a chunk,
   it doesn't get 2x the score — the benefit flattens out. This prevents
   long, repetitive chunks from dominating.

WHY WE NEED THIS:
Vector search finds chunks with similar MEANING but can miss exact keyword matches.
BM25 finds chunks with exact WORDS but doesn't understand meaning.
Together (hybrid retrieval) they cover each other's weaknesses.

INTERVIEW QUESTION: "Why not just use vector search?"
ANSWER: "Vector search can miss exact keyword queries like product names,
error codes, or technical terms. BM25 catches these. Hybrid retrieval
gives us the best of both approaches."
"""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from rag.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer.

    Converts text to lowercase and splits on non-alphanumeric characters.
    This is intentionally simple — in production you might use spaCy or NLTK
    for better tokenization (handling "don't" → "do", "not" etc.).
    But for BM25, simple tokenization works surprisingly well.

    Example:
        "ChromaDB uses HNSW indexing!" → ["chromadb", "uses", "hnsw", "indexing"]
    """
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Retrieves chunks using BM25 keyword matching.

    Unlike the vector store which persists to disk, BM25 builds its index
    in memory from the chunks you give it. This means:
    - You need to rebuild it when chunks change (after re-ingestion)
    - It's fast (no disk I/O, no API calls)
    - It uses minimal memory for typical document collections

    Usage:
        retriever = BM25Retriever(chunks)      # build index from chunks
        results = retriever.search("HNSW", k=5) # find chunks with "HNSW"
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from a list of chunks.

        Args:
            chunks: All chunks in the corpus. BM25 needs to see ALL chunks
                    upfront to calculate word rarity (IDF) scores.
        """
        self._chunks = chunks
        self._bm25: BM25Okapi | None = None

        if chunks:
            # Tokenize every chunk — BM25 works on word lists, not raw text
            tokenized_corpus = [_tokenize(chunk.content) for chunk in chunks]

            # Build the BM25 index
            # BM25Okapi is the most common variant (the "Okapi" part refers to
            # the information retrieval system where it was first implemented)
            self._bm25 = BM25Okapi(tokenized_corpus)

        logger.info("BM25 index built with %d chunks", len(chunks))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search for chunks matching the query keywords.

        Args:
            query: The search query (will be tokenized the same way as chunks).
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by BM25 score (highest first).
        """
        if not self._chunks or not self._bm25:
            return []

        # Tokenize the query the same way we tokenized the chunks
        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores for ALL chunks
        # Each chunk gets a score based on how well it matches the query keywords
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k chunk indices, sorted by score (highest first)
        # argsort returns indices that would sort the array
        # We reverse it ([::-1]) to get highest scores first
        # Then take the first top_k
        ranked_indices = scores.argsort()[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            # Skip chunks with zero score (no keyword overlap at all)
            if score <= 0:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=score))

        logger.info(
            "BM25 retrieved %d chunks for query: '%s...'",
            len(results),
            query[:50],
        )
        return results
