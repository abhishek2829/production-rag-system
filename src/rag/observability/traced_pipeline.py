"""Traced RAG pipeline — adds Langfuse observability to every query.

THIS IS THE CORE OF PROJECT 3.

What it does:
  Inherits from RAGPipeline (Project 1) and overrides the query() method.
  The override does the EXACT SAME work, but wraps each step with a
  stopwatch (span) and records it to Langfuse.

What it does NOT do:
  - Does NOT change the existing RAGPipeline code
  - Does NOT change the answer the user gets
  - Does NOT change how retrieval, re-ranking, or generation works

Analogy:
  RAGPipeline = a chef cooking food
  TracedRAGPipeline = the SAME chef cooking the SAME food,
                      but now someone is standing next to them
                      with a clipboard recording every step's timing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rag.config import Settings, get_settings
from rag.generation.citation_validator import CitationReport, validate_citations
from rag.models import RAGResponse
from rag.observability.langfuse_client import LangfuseTracer
from rag.observability.metrics import (
    estimate_cost,
    format_generation_metadata,
    format_retrieval_metadata,
    format_validation_metadata,
)
from rag.pipeline import RAGPipeline
from rag.retrieval.hybrid_retriever import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class TracedRAGPipeline(RAGPipeline):
    """RAG pipeline with Langfuse tracing.

    Same interface as RAGPipeline — you can swap them.
    All existing behavior is preserved. Tracing is added on top.

    Usage:
        # Without tracing (original):
        pipeline = RAGPipeline()

        # With tracing (new — drop-in replacement):
        pipeline = TracedRAGPipeline()

        # Same methods, same results:
        response, report = pipeline.query("What is HNSW?")
        # But now the query appears in Langfuse dashboard!
    """

    def __init__(self, settings: Settings | None = None) -> None:
        # Call the parent's __init__ — this sets up vector_store,
        # bm25_retriever, reranker, generator, etc.
        # We get ALL of that for free through inheritance.
        resolved_settings = settings or get_settings()
        super().__init__(resolved_settings)

        # Create the Langfuse tracer (the "clipboard holder")
        self._tracer = LangfuseTracer(resolved_settings)

        if self._tracer.is_enabled:
            logger.info("TracedRAGPipeline initialized with Langfuse tracing")
        else:
            logger.info("TracedRAGPipeline initialized WITHOUT tracing")

    def query(
        self, question: str, top_k: int | None = None
    ) -> tuple[RAGResponse, CitationReport]:
        """Query the RAG system — same as parent, but with tracing.

        This method does the EXACT SAME WORK as RAGPipeline.query().
        The only difference: we start/stop a stopwatch around each step
        and record the timing to Langfuse.

        If tracing is disabled (Langfuse down, keys missing, turned off),
        this behaves IDENTICALLY to the parent's query() method.
        """
        # If tracing is not enabled, just use the parent's method directly.
        # No overhead, no changes, exactly the same as before.
        if not self._tracer.is_enabled:
            return super().query(question, top_k=top_k)

        final_k = top_k or self._settings.retrieval_top_k
        initial_k = self._settings.retrieval_initial_k
        total_start = time.perf_counter()

        # ===== START RECORDING =====
        # Tell Langfuse: "A new query just started"
        trace = self._tracer.create_trace(question)

        logger.info("Traced query: %s", question)

        # ----- STEP 1: Vector retrieval -----
        # ⏱️ Start stopwatch
        step_start = time.perf_counter()
        # Do the actual work (same as parent)
        vector_results = self._vector_store.search(question, top_k=initial_k)
        # ⏱️ Stop stopwatch
        vector_duration = time.perf_counter() - step_start

        # 📋 Record to Langfuse: "Retrieval took X seconds, found Y chunks"
        self._tracer.create_span(
            trace=trace,
            name="retrieval_vector",
            input_data={"query": question, "top_k": initial_k},
            output_data={
                "num_chunks": len(vector_results),
                "scores": [round(r.score, 4) for r in vector_results[:5]],
            },
            duration_seconds=vector_duration,
            metadata=format_retrieval_metadata(
                vector_results, "vector", vector_duration
            ),
        )

        # ----- STEP 2: BM25 retrieval (if enabled) -----
        bm25_results = []
        if self._settings.use_hybrid_retrieval and self._bm25_retriever:
            step_start = time.perf_counter()
            bm25_results = self._bm25_retriever.search(question, top_k=initial_k)
            bm25_duration = time.perf_counter() - step_start

            self._tracer.create_span(
                trace=trace,
                name="retrieval_bm25",
                input_data={"query": question, "top_k": initial_k},
                output_data={
                    "num_chunks": len(bm25_results),
                    "scores": [round(r.score, 4) for r in bm25_results[:5]],
                },
                duration_seconds=bm25_duration,
                metadata=format_retrieval_metadata(
                    bm25_results, "bm25", bm25_duration
                ),
            )

        # ----- STEP 3: RRF Fusion (combine vector + BM25) -----
        if bm25_results:
            step_start = time.perf_counter()
            combined = reciprocal_rank_fusion(
                [vector_results, bm25_results], top_k=initial_k
            )
            rrf_duration = time.perf_counter() - step_start

            self._tracer.create_span(
                trace=trace,
                name="rrf_fusion",
                input_data={
                    "vector_count": len(vector_results),
                    "bm25_count": len(bm25_results),
                },
                output_data={"combined_count": len(combined)},
                duration_seconds=rrf_duration,
            )
        else:
            combined = vector_results

        # ----- STEP 4: Re-ranking -----
        if self._reranker and combined:
            step_start = time.perf_counter()
            retrieved = self._reranker.rerank(question, combined, top_k=final_k)
            rerank_duration = time.perf_counter() - step_start

            self._tracer.create_span(
                trace=trace,
                name="reranking",
                input_data={
                    "input_count": len(combined),
                    "top_k": final_k,
                },
                output_data={
                    "output_count": len(retrieved),
                    "scores": [round(r.score, 4) for r in retrieved],
                },
                duration_seconds=rerank_duration,
                metadata={
                    "model": self._settings.reranker_model,
                    "duration_seconds": round(rerank_duration, 4),
                },
            )
        else:
            retrieved = combined[:final_k]

        # ----- STEP 5: Answer generation (Claude) -----
        step_start = time.perf_counter()
        response = self._generator.generate(question, retrieved)
        gen_duration = time.perf_counter() - step_start

        # Get token counts from the response for cost calculation
        # We'll parse them from the answer metadata
        prompt_tokens = 0
        completion_tokens = 0
        # The generator logs token counts — we estimate from answer length
        # A more precise approach would modify the generator to return tokens,
        # but that would change existing code (which we want to avoid)
        completion_tokens = len(response.answer.split()) * 4 // 3  # rough estimate
        prompt_tokens = completion_tokens * 7  # typical ratio for RAG

        self._tracer.create_span(
            trace=trace,
            name="generation",
            input_data={
                "query": question,
                "num_context_chunks": len(retrieved),
            },
            output_data={
                "answer_length": len(response.answer),
                "num_citations": len(response.citations),
            },
            duration_seconds=gen_duration,
            metadata=format_generation_metadata(
                prompt_tokens, completion_tokens,
                self._settings.llm_model, gen_duration,
            ),
        )

        # ----- STEP 6: Citation validation -----
        step_start = time.perf_counter()
        report = validate_citations(response)
        validation_duration = time.perf_counter() - step_start

        self._tracer.create_span(
            trace=trace,
            name="citation_validation",
            input_data={"answer_length": len(response.answer)},
            output_data={
                "is_valid": report.is_valid,
                "is_refusal": report.is_refusal,
                "citation_coverage": report.citation_coverage,
            },
            duration_seconds=validation_duration,
            metadata=format_validation_metadata(
                report.is_valid,
                report.is_refusal,
                report.citation_coverage,
                report.issues,
            ),
        )

        # ===== RECORD QUALITY SCORES =====
        total_duration = time.perf_counter() - total_start
        estimated_cost = estimate_cost(
            self._settings.llm_model, prompt_tokens, completion_tokens
        )

        # Attach scores to the trace — these show up in Langfuse charts
        self._tracer.score_trace(
            trace, "citation_coverage", report.citation_coverage,
            comment=f"{'Refusal' if report.is_refusal else f'{len(response.citations)} sources cited'}",
        )
        self._tracer.score_trace(
            trace, "total_latency", total_duration,
            comment=f"Total query time: {total_duration:.1f}s",
        )
        self._tracer.score_trace(
            trace, "estimated_cost", estimated_cost,
            comment=f"${estimated_cost:.4f}",
        )

        # Update trace with final output
        self._tracer.update_trace(
            trace,
            output={
                "answer": response.answer[:500],  # truncate for dashboard
                "citations_valid": report.is_valid,
                "is_refusal": report.is_refusal,
            },
            metadata={
                "total_latency_seconds": round(total_duration, 3),
                "estimated_cost_usd": round(estimated_cost, 6),
                "citation_coverage": round(report.citation_coverage, 4),
                "prompt_version": self._generator.prompt_config.version,
            },
        )

        logger.info(
            "Traced response: %d chars, %d citations, %.1fs total, $%.4f cost",
            len(response.answer),
            len(response.citations),
            total_duration,
            estimated_cost,
        )

        return response, report

    def flush_traces(self) -> None:
        """Force send all pending traces to Langfuse.

        Call this when you're done querying to make sure all traces
        arrive in the dashboard. In normal use, traces are sent
        automatically in the background.
        """
        self._tracer.flush()
