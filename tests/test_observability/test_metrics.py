"""Tests for observability metrics (no Langfuse needed)."""

from rag.models import Chunk, RetrievedChunk
from rag.observability.metrics import (
    estimate_cost,
    format_generation_metadata,
    format_retrieval_metadata,
    format_validation_metadata,
)


class TestEstimateCost:
    def test_known_model(self):
        """Claude Sonnet: 2500 input + 350 output tokens."""
        cost = estimate_cost("claude-sonnet-4-20250514", 2500, 350)
        # Input: 2500 * 3.00 / 1M = 0.0075
        # Output: 350 * 15.00 / 1M = 0.00525
        # Total: 0.01275
        assert 0.012 < cost < 0.014

    def test_unknown_model_uses_default(self):
        cost = estimate_cost("some-future-model", 1000, 100)
        assert cost > 0

    def test_zero_tokens(self):
        cost = estimate_cost("claude-sonnet-4-20250514", 0, 0)
        assert cost == 0.0

    def test_output_costs_more_than_input(self):
        """Output tokens are 5x more expensive than input tokens."""
        input_only = estimate_cost("claude-sonnet-4-20250514", 1000, 0)
        output_only = estimate_cost("claude-sonnet-4-20250514", 0, 1000)
        assert output_only > input_only


class TestFormatRetrievalMetadata:
    def test_basic_format(self):
        chunks = [
            RetrievedChunk(
                chunk=Chunk(content="test", source="doc.md", chunk_index=0),
                score=0.85,
            )
        ]
        result = format_retrieval_metadata(chunks, "vector", 0.5)
        assert result["search_type"] == "vector"
        assert result["num_chunks"] == 1
        assert result["top_score"] == 0.85
        assert result["duration_seconds"] == 0.5

    def test_empty_chunks(self):
        result = format_retrieval_metadata([], "bm25", 0.01)
        assert result["num_chunks"] == 0
        assert result["top_score"] == 0.0


class TestFormatGenerationMetadata:
    def test_includes_cost(self):
        result = format_generation_metadata(2500, 350, "claude-sonnet-4-20250514", 4.1)
        assert "estimated_cost_usd" in result
        assert result["estimated_cost_usd"] > 0
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["prompt_tokens"] == 2500
        assert result["completion_tokens"] == 350


class TestFormatValidationMetadata:
    def test_valid_response(self):
        result = format_validation_metadata(
            is_valid=True, is_refusal=False,
            citation_coverage=0.4, issues=[],
        )
        assert result["is_valid"] is True
        assert result["citation_coverage"] == 0.4
        assert result["num_issues"] == 0

    def test_refusal(self):
        result = format_validation_metadata(
            is_valid=True, is_refusal=True,
            citation_coverage=0.0, issues=[],
        )
        assert result["is_refusal"] is True

    def test_limits_issues_to_5(self):
        issues = [f"issue {i}" for i in range(10)]
        result = format_validation_metadata(False, False, 0.2, issues)
        assert len(result["issues"]) == 5
