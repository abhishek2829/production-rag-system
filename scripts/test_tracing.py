"""Quick test script to verify Langfuse tracing works.

Run this from the project root:
  python scripts/test_tracing.py

Then open http://localhost:3000 and check if the trace appears.
"""

import logging
from pathlib import Path

from rag.observability.traced_pipeline import TracedRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

print("\n=== Testing Langfuse Tracing ===\n")

# Create the traced pipeline (connects to Langfuse automatically)
pipeline = TracedRAGPipeline()

# Check if tracing is enabled
print(f"Tracing enabled: {pipeline._tracer.is_enabled}")

if pipeline.chunk_count == 0:
    print("No documents ingested. Ingesting sample documents...")
    pipeline.ingest(Path("./data/documents"))

print(f"\nChunks in store: {pipeline.chunk_count}")

# Run a traced query
print("\n--- Running traced query ---\n")
response, report = pipeline.query("What is RAG and how does it reduce hallucination?")

print(f"\nAnswer: {response.answer[:200]}...")
print(f"Citations: {len(response.citations)}")
print(f"Citation valid: {report.is_valid}")
print(f"Citation coverage: {report.citation_coverage:.0%}")

# Flush traces to make sure they arrive in Langfuse
pipeline.flush_traces()

print("\n=== Done! ===")
print("Open http://localhost:3000 → Traces to see the trace.")
print("Click on it to see each step's timing breakdown.")
