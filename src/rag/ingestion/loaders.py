"""Document loaders for various file formats.

Each loader implements a simple protocol: take a file path, return a Document.
We use the Strategy pattern so adding new formats (e.g., DOCX, CSV) is just
adding a new loader function — no changes to existing code (Open/Closed Principle).
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from rag.models import Document, DocumentType

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> Document:
    """Extract text from a PDF file.

    Uses pypdf which handles most PDFs well. For production systems with
    complex layouts (tables, multi-column), you'd upgrade to unstructured.io
    or a vision model — but pypdf covers 80% of cases.
    """
    reader = PdfReader(path)
    pages: list[str] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
        else:
            logger.warning("Page %d of %s has no extractable text", i + 1, path.name)

    if not pages:
        raise ValueError(f"No text could be extracted from PDF: {path}")

    content = "\n\n".join(pages)

    return Document(
        content=content,
        source=str(path),
        doc_type=DocumentType.PDF,
        metadata={
            "filename": path.name,
            "num_pages": len(reader.pages),
            "pages_with_text": len(pages),
        },
    )


def load_markdown(path: Path) -> Document:
    """Load a Markdown file as-is.

    Markdown is kept raw (not rendered to HTML) because the chunker
    can use heading structure (# ## ###) for intelligent splitting.
    """
    content = path.read_text(encoding="utf-8")

    return Document(
        content=content,
        source=str(path),
        doc_type=DocumentType.MARKDOWN,
        metadata={"filename": path.name},
    )


def load_text(path: Path) -> Document:
    """Load a plain text file."""
    content = path.read_text(encoding="utf-8")

    return Document(
        content=content,
        source=str(path),
        doc_type=DocumentType.TEXT,
        metadata={"filename": path.name},
    )


def load_html_from_url(url: str, timeout: int = 30) -> Document:
    """Fetch and extract text from a web page.

    Uses BeautifulSoup to strip HTML tags and extract meaningful text.
    In production, you'd add: rate limiting, robots.txt checking,
    and a proper HTTP client with retries (httpx + tenacity).
    """
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "RAG-Ingestion/0.1"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content = "\n".join(lines)

    return Document(
        content=content,
        source=url,
        doc_type=DocumentType.HTML,
        metadata={"url": url, "title": soup.title.string if soup.title else ""},
    )


# --- Loader registry ---
# Maps file extensions to their loader functions.
# Adding a new format = adding one entry here + one loader function above.

EXTENSION_LOADER_MAP: dict[str, type[DocumentType]] = {
    ".pdf": DocumentType.PDF,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".txt": DocumentType.TEXT,
    ".text": DocumentType.TEXT,
}

LOADER_FUNCTIONS = {
    DocumentType.PDF: load_pdf,
    DocumentType.MARKDOWN: load_markdown,
    DocumentType.TEXT: load_text,
}


def load_file(path: Path) -> Document:
    """Load a document from a file path, auto-detecting the format.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    doc_type = EXTENSION_LOADER_MAP.get(ext)

    if doc_type is None:
        supported = ", ".join(sorted(EXTENSION_LOADER_MAP.keys()))
        raise ValueError(f"Unsupported file extension '{ext}'. Supported: {supported}")

    loader = LOADER_FUNCTIONS[doc_type]
    logger.info("Loading %s file: %s", doc_type.value, path.name)
    return loader(path)
