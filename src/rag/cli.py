"""CLI entry points for the RAG system.

Provides two commands:
  rag-ingest — Ingest documents from a directory
  rag-query  — Query the RAG system interactively

Uses Rich for formatted terminal output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.pipeline import RAGPipeline

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


@click.command()
@click.option(
    "--dir",
    "directory",
    type=click.Path(exists=True, path_type=Path),
    default=Path("./data/documents"),
    help="Directory containing documents to ingest",
)
def ingest(directory: Path) -> None:
    """Ingest documents into the RAG vector store."""
    console.print(f"\n[bold blue]Ingesting documents from:[/] {directory}\n")

    pipeline = RAGPipeline()
    count = pipeline.ingest(directory)

    console.print(
        Panel(
            f"[green]Successfully ingested {count} chunks[/]\n"
            f"Total chunks in store: {pipeline.chunk_count}",
            title="Ingestion Complete",
        )
    )


@click.command()
@click.argument("question", required=False)
@click.option("--top-k", type=int, default=None, help="Number of chunks to retrieve")
def query(question: str | None, top_k: int | None) -> None:
    """Query the RAG system. Pass a question or enter interactive mode."""
    pipeline = RAGPipeline()

    if pipeline.chunk_count == 0:
        console.print("[red]No documents ingested yet. Run rag-ingest first.[/]")
        sys.exit(1)

    if question:
        _run_query(pipeline, question, top_k)
    else:
        # Interactive mode
        console.print("[bold]RAG Query System[/] (type 'quit' to exit)\n")
        while True:
            try:
                q = console.input("[bold cyan]Question:[/] ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if q:
                    _run_query(pipeline, q, top_k)
                    console.print()
            except (KeyboardInterrupt, EOFError):
                break


def _run_query(pipeline: RAGPipeline, question: str, top_k: int | None) -> None:
    """Execute a single query and display results."""
    response = pipeline.query(question, top_k=top_k)

    # Display the answer
    console.print(Panel(response.answer, title="Answer", border_style="green"))

    # Display citations
    if response.citations:
        table = Table(title="Citations Used")
        table.add_column("Source", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Preview", max_width=60)

        for rc in response.citations:
            table.add_row(
                rc.chunk.source.split("/")[-1],
                f"{rc.score:.3f}",
                rc.chunk.content[:100] + "...",
            )
        console.print(table)

    # Display stats
    console.print(
        f"\n[dim]Retrieved: {len(response.retrieved_chunks)} chunks | "
        f"Cited: {len(response.citations)} chunks | "
        f"Citation coverage: {len(response.citations)}/{len(response.retrieved_chunks)}[/]"
    )
