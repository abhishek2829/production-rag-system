"""Populate the Langfuse dashboard with multiple traced queries.

WHY THIS SCRIPT EXISTS:
With just 1-2 traces, the dashboard shows individual data points.
With 20+ traces, it shows TRENDS — latency over time, cost patterns,
quality distributions. That's when observability becomes useful.

This script runs a variety of queries through the traced pipeline,
grouped into sessions, so the dashboard has rich data to display.

Run from project root:
  python scripts/populate_dashboard.py

Then open http://localhost:3000 to see the dashboard with trends.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from rag.observability.traced_pipeline import TracedRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Queries to run — mix of topics, difficulties, and expected behaviors
QUERIES = [
    # --- Should answer (from our documents) ---
    {"q": "What is Retrieval-Augmented Generation?", "user": "abhishek"},
    {"q": "How does RAG reduce hallucination?", "user": "abhishek"},
    {"q": "What are the three stages of a RAG pipeline?", "user": "abhishek"},
    {"q": "What is ChromaDB and when should I use it?", "user": "product_manager"},
    {"q": "When should I use Pinecone instead of ChromaDB?", "user": "product_manager"},
    {"q": "What is cosine similarity?", "user": "new_engineer"},
    {"q": "How does HNSW indexing work?", "user": "new_engineer"},
    {"q": "What is faithfulness in RAG evaluation?", "user": "abhishek"},
    {"q": "How do you evaluate a RAG system?", "user": "team_lead"},
    {"q": "What tools are available for RAG evaluation?", "user": "team_lead"},
    {"q": "What is hybrid retrieval?", "user": "new_engineer"},
    {"q": "What is a cross-encoder re-ranker?", "user": "abhishek"},
    {"q": "What is BM25?", "user": "new_engineer"},
    {"q": "What are vector embeddings?", "user": "product_manager"},
    # --- Should refuse (not in our documents) ---
    {"q": "What is the best recipe for biryani?", "user": "random_user"},
    {"q": "Who is the CEO of Google?", "user": "random_user"},
    {"q": "What will the weather be tomorrow?", "user": "random_user"},
    # --- Single keyword (tests BM25) ---
    {"q": "HNSW", "user": "abhishek"},
    {"q": "BM25", "user": "new_engineer"},
    # --- Cross-document ---
    {"q": "How does faithfulness relate to hallucination?", "user": "team_lead"},
]


def main() -> None:
    print("\n=== Populating Langfuse Dashboard ===\n")
    print(f"Running {len(QUERIES)} queries...\n")

    # Create a session ID based on current timestamp
    # All queries in this run will be grouped under this session
    session_id = f"dashboard_populate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Session: {session_id}\n")

    pipeline = TracedRAGPipeline()

    if pipeline.chunk_count == 0:
        print("No documents ingested. Ingesting...")
        pipeline.ingest(Path("./data/documents"))

    results = []

    for i, item in enumerate(QUERIES, start=1):
        query = item["q"]
        user = item["user"]

        print(f"[{i}/{len(QUERIES)}] {user}: \"{query[:50]}...\"")

        try:
            response, report = pipeline.query(
                query,
                session_id=session_id,
                user_id=user,
            )

            status = "REFUSED" if report.is_refusal else "ANSWERED"
            print(
                f"  → {status} | "
                f"Citations: {len(response.citations)} | "
                f"Coverage: {report.citation_coverage:.0%} | "
                f"Valid: {report.is_valid}"
            )

            results.append({
                "query": query,
                "user": user,
                "status": status,
                "citations": len(response.citations),
                "coverage": report.citation_coverage,
                "valid": report.is_valid,
            })

        except Exception as e:
            print(f"  → ERROR: {e}")
            results.append({"query": query, "user": user, "status": "ERROR"})

        # Small delay to spread traces over time (makes charts look better)
        time.sleep(0.5)

    # Flush all traces
    pipeline.flush_traces()

    # Summary
    answered = sum(1 for r in results if r["status"] == "ANSWERED")
    refused = sum(1 for r in results if r["status"] == "REFUSED")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"\n=== Summary ===")
    print(f"Total queries: {len(results)}")
    print(f"Answered:      {answered}")
    print(f"Refused:       {refused}")
    print(f"Errors:        {errors}")
    print(f"Session:       {session_id}")
    print(f"\nOpen http://localhost:3000 to see the dashboard!")
    print(f"Click 'Sessions' → find '{session_id}' to see all queries grouped.")


if __name__ == "__main__":
    main()
