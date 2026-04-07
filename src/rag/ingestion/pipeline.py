"""Document ingestion pipeline.

Orchestrates loading documents from a directory, with error handling
that logs failures but continues processing (resilient ingestion).
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag.ingestion.loaders import EXTENSION_LOADER_MAP, load_file
from rag.models import Document

logger = logging.getLogger(__name__)


def ingest_directory(directory: Path) -> list[Document]:
    """Ingest all supported documents from a directory.

    Scans the directory recursively for supported file types.
    Failed documents are logged and skipped — we don't fail the entire
    batch because one PDF is corrupted.

    Args:
        directory: Path to the directory containing documents.

    Returns:
        List of successfully loaded documents.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Expected a directory, got a file: {directory}")

    supported_extensions = set(EXTENSION_LOADER_MAP.keys())
    files = [
        f
        for f in sorted(directory.rglob("*"))
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not files:
        logger.warning("No supported documents found in %s", directory)
        return []

    logger.info("Found %d supported files in %s", len(files), directory)

    documents: list[Document] = []
    failed: list[tuple[Path, str]] = []

    for file_path in files:
        try:
            doc = load_file(file_path)
            documents.append(doc)
            logger.info("Loaded: %s (%d chars)", file_path.name, len(doc.content))
        except Exception as e:
            failed.append((file_path, str(e)))
            logger.error("Failed to load %s: %s", file_path.name, e)

    logger.info(
        "Ingestion complete: %d loaded, %d failed out of %d total",
        len(documents),
        len(failed),
        len(files),
    )

    if failed:
        for path, error in failed:
            logger.warning("  FAILED: %s — %s", path.name, error)

    return documents
