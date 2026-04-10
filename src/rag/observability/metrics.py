"""Metrics calculation for observability.

WHAT THIS FILE DOES:
Two things:
1. Calculates the COST of each Claude API call in dollars
2. Formats data into clean dictionaries for Langfuse spans

WHY COST TRACKING MATTERS:
If your RAG system handles 1000 queries/day and each costs $0.01,
that's $10/day = $300/month. If someone changes the prompt to be
longer, cost might jump to $0.03/query = $900/month.
Without cost tracking, you'd only notice when the bill arrives.
With cost tracking, you see it in the dashboard immediately.

HOW COST IS CALCULATED:
Anthropic charges per token (the units AI models read/write):
  Claude Sonnet input:  $3.00 per 1 million tokens
  Claude Sonnet output: $15.00 per 1 million tokens

Example: A query uses 2500 input tokens + 350 output tokens
  Input cost:  2500 × ($3.00 / 1,000,000) = $0.0075
  Output cost: 350 × ($15.00 / 1,000,000) = $0.00525
  Total cost:  $0.013 (about 1.3 cents)
"""

from __future__ import annotations

from rag.models import RetrievedChunk

# Anthropic pricing per 1 million tokens (as of 2026)
# Source: https://www.anthropic.com/pricing
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
    },
    # Add more models here as needed
    "default": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
    },
}


def estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Estimate the cost of a Claude API call in US dollars.

    Args:
        model: The model name (e.g., "claude-sonnet-4-20250514").
        prompt_tokens: Number of input tokens (the question + context).
        completion_tokens: Number of output tokens (the answer).

    Returns:
        Estimated cost in dollars (e.g., 0.013 = 1.3 cents).
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

    input_cost = prompt_tokens * pricing["input_per_million"] / 1_000_000
    output_cost = completion_tokens * pricing["output_per_million"] / 1_000_000

    return input_cost + output_cost


def format_retrieval_metadata(
    chunks: list[RetrievedChunk],
    search_type: str,
    duration_seconds: float,
) -> dict:
    """Format retrieval results for a Langfuse span.

    This creates a clean dictionary that shows up in the Langfuse
    dashboard when you click on a retrieval span.

    Args:
        chunks: The chunks that were retrieved.
        search_type: "vector", "bm25", or "hybrid".
        duration_seconds: How long retrieval took.

    Returns:
        Dictionary with retrieval metadata.
    """
    return {
        "search_type": search_type,
        "num_chunks": len(chunks),
        "top_score": chunks[0].score if chunks else 0.0,
        "sources": list({c.chunk.source.split("/")[-1] for c in chunks}),
        "duration_seconds": round(duration_seconds, 4),
    }


def format_generation_metadata(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    duration_seconds: float,
) -> dict:
    """Format generation results for a Langfuse span.

    Shows token counts, cost, and timing in the dashboard.
    """
    cost = estimate_cost(model, prompt_tokens, completion_tokens)

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": round(cost, 6),
        "duration_seconds": round(duration_seconds, 4),
    }


def format_validation_metadata(
    is_valid: bool,
    is_refusal: bool,
    citation_coverage: float,
    issues: list[str],
) -> dict:
    """Format citation validation results for a Langfuse span."""
    return {
        "is_valid": is_valid,
        "is_refusal": is_refusal,
        "citation_coverage": round(citation_coverage, 4),
        "num_issues": len(issues),
        "issues": issues[:5],  # Limit to 5 issues to avoid huge metadata
    }
