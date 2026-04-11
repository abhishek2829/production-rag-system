"""CLI for running RAG evaluation.

Provides the rag-eval command that:
1. Runs all golden dataset questions through the pipeline
2. Shows a detailed results table
3. Shows aggregate scores
4. Exits with code 1 if quality is below threshold (for CI gating)

Usage:
  rag-eval                           # Run with defaults
  rag-eval --threshold 0.90          # Require 90% score to pass
  rag-eval --dataset eval/custom.json  # Use a custom dataset
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.evaluation.runner import EvalReport, run_evaluation
from rag.pipeline import RAGPipeline

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def _display_report(report: EvalReport, threshold: float) -> None:
    """Display the evaluation report in a nice format."""
    # Results table
    table = Table(title=f"Evaluation Results ({report.total_cases} cases)")
    table.add_column("ID", style="cyan", max_width=25)
    table.add_column("Expected", max_width=8)
    table.add_column("Behavior", justify="center", max_width=8)
    table.add_column("Sources", justify="center", max_width=8)
    table.add_column("Content", justify="center", max_width=8)
    table.add_column("Citations", justify="center", max_width=8)
    table.add_column("Time", justify="right", max_width=6)
    table.add_column("Status", justify="center", max_width=6)

    for r in report.results:
        all_pass = (
            r.behavior_correct and r.source_correct
            and r.content_correct and r.citations_valid
        )
        table.add_row(
            r.case_id,
            r.expected_behavior,
            "[green]OK[/]" if r.behavior_correct else "[red]FAIL[/]",
            "[green]OK[/]" if r.source_correct else "[red]FAIL[/]",
            "[green]OK[/]" if r.content_correct else "[red]FAIL[/]",
            "[green]OK[/]" if r.citations_valid else "[red]FAIL[/]",
            f"{r.latency_seconds:.1f}s",
            "[green]PASS[/]" if all_pass else "[red]FAIL[/]",
        )

    console.print(table)

    # Show failures detail
    failures = [
        r for r in report.results
        if not (r.behavior_correct and r.source_correct
                and r.content_correct and r.citations_valid)
    ]
    if failures:
        console.print(f"\n[bold red]Failed Cases ({len(failures)}):[/]")
        for r in failures:
            issues = []
            if not r.behavior_correct:
                issues.append(
                    f"Expected '{r.expected_behavior}' but got the opposite"
                )
            if not r.source_correct:
                issues.append(f"Missing source citations")
            if not r.content_correct:
                issues.append(
                    f"Missing keywords: {r.missing_keywords}"
                )
            if not r.citations_valid:
                issues.append(
                    f"Citation issues: {r.citation_issues}"
                )
            console.print(f"  [cyan]{r.case_id}[/]: {'; '.join(issues)}")

    # Scores panel
    passing = report.is_passing(threshold)
    scores_text = (
        f"Behavior accuracy:  {report.behavior_accuracy:.0%}\n"
        f"Source accuracy:     {report.source_accuracy:.0%}\n"
        f"Content accuracy:   {report.content_accuracy:.0%}\n"
        f"Citation validity:  {report.citation_validity:.0%}\n"
        f"{'─' * 30}\n"
        f"Overall score:      {report.overall_score:.0%}\n"
        f"Threshold:          {threshold:.0%}\n"
        f"{'─' * 30}\n"
        f"Passed: {report.passed_cases}/{report.total_cases} | "
        f"Failed: {report.failed_cases}/{report.total_cases}\n"
        f"Avg latency: {report.avg_latency:.1f}s per query"
    )

    color = "green" if passing else "red"
    verdict = "PASSED" if passing else "FAILED"
    console.print(
        Panel(
            scores_text,
            title=f"Quality Gate: [{color}]{verdict}[/{color}]",
            border_style=color,
        )
    )


@click.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    default=Path("eval/golden_dataset.json"),
    help="Path to the golden dataset JSON file",
)
@click.option(
    "--threshold",
    type=float,
    default=0.85,
    help="Minimum overall score to pass (0.0-1.0, default 0.85)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Save detailed results to a JSON file",
)
@click.option(
    "--traced",
    is_flag=True,
    default=False,
    help="Use TracedRAGPipeline to log eval to Langfuse dashboard",
)
def evaluate(dataset: Path, threshold: float, output: Path | None, traced: bool) -> None:
    """Run RAG evaluation against a golden dataset."""
    from datetime import datetime

    mode = "TRACED (Langfuse)" if traced else "standard"
    console.print(
        f"\n[bold blue]Running evaluation against:[/] {dataset}\n"
        f"[bold blue]Quality threshold:[/] {threshold:.0%}\n"
        f"[bold blue]Mode:[/] {mode}\n"
    )

    # Initialize pipeline — traced or standard
    if traced:
        from rag.observability.traced_pipeline import TracedRAGPipeline
        pipeline = TracedRAGPipeline()
    else:
        pipeline = RAGPipeline()

    if pipeline.chunk_count == 0:
        console.print(
            "[red]No documents ingested. Run rag-ingest first.[/]"
        )
        sys.exit(1)

    # Create session ID for Langfuse grouping
    session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if traced else None

    # Run evaluation
    report = run_evaluation(pipeline, dataset, session_id=session_id)

    # Display report
    _display_report(report, threshold)

    # Save results if requested
    if output:
        results_data = {
            "scores": {
                "behavior_accuracy": report.behavior_accuracy,
                "source_accuracy": report.source_accuracy,
                "content_accuracy": report.content_accuracy,
                "citation_validity": report.citation_validity,
                "overall_score": report.overall_score,
            },
            "summary": {
                "total": report.total_cases,
                "passed": report.passed_cases,
                "failed": report.failed_cases,
                "avg_latency": report.avg_latency,
                "threshold": threshold,
                "passing": report.is_passing(threshold),
            },
            "results": [
                {
                    "case_id": r.case_id,
                    "question": r.question,
                    "expected_behavior": r.expected_behavior,
                    "behavior_correct": r.behavior_correct,
                    "source_correct": r.source_correct,
                    "content_correct": r.content_correct,
                    "citations_valid": r.citations_valid,
                    "cited_sources": r.cited_sources,
                    "missing_keywords": r.missing_keywords,
                    "citation_issues": r.citation_issues,
                    "latency_seconds": r.latency_seconds,
                }
                for r in report.results
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results_data, f, indent=2)
        console.print(f"\n[dim]Results saved to: {output}[/]")

    # Flush traces if using traced pipeline
    if traced and hasattr(pipeline, "flush_traces"):
        pipeline.flush_traces()
        if session_id:
            console.print(
                f"\n[dim]Langfuse session: {session_id}[/]\n"
                f"[dim]View at: http://localhost:3000 → Sessions → {session_id}[/]"
            )

    # Exit with error code if failing (for CI)
    if not report.is_passing(threshold):
        sys.exit(1)
