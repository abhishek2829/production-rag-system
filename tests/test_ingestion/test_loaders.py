"""Tests for document loaders."""

from pathlib import Path

import pytest

from rag.ingestion.loaders import load_file, load_markdown, load_text
from rag.models import DocumentType


@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file."""
    file = tmp_path / "test.txt"
    file.write_text("Hello, this is a test document.\nIt has two lines.")
    return file


@pytest.fixture
def tmp_markdown_file(tmp_path: Path) -> Path:
    """Create a temporary markdown file."""
    file = tmp_path / "test.md"
    file.write_text("# Heading\n\nSome paragraph content.\n\n## Subheading\n\nMore content.")
    return file


class TestLoadText:
    def test_loads_content(self, tmp_text_file: Path):
        doc = load_text(tmp_text_file)
        assert "Hello, this is a test document" in doc.content
        assert doc.doc_type == DocumentType.TEXT

    def test_source_is_file_path(self, tmp_text_file: Path):
        doc = load_text(tmp_text_file)
        assert doc.source == str(tmp_text_file)

    def test_metadata_includes_filename(self, tmp_text_file: Path):
        doc = load_text(tmp_text_file)
        assert doc.metadata["filename"] == "test.txt"


class TestLoadMarkdown:
    def test_preserves_raw_markdown(self, tmp_markdown_file: Path):
        doc = load_markdown(tmp_markdown_file)
        assert "# Heading" in doc.content
        assert "## Subheading" in doc.content

    def test_doc_type_is_markdown(self, tmp_markdown_file: Path):
        doc = load_markdown(tmp_markdown_file)
        assert doc.doc_type == DocumentType.MARKDOWN


class TestLoadFile:
    def test_auto_detects_text(self, tmp_text_file: Path):
        doc = load_file(tmp_text_file)
        assert doc.doc_type == DocumentType.TEXT

    def test_auto_detects_markdown(self, tmp_markdown_file: Path):
        doc = load_file(tmp_markdown_file)
        assert doc.doc_type == DocumentType.MARKDOWN

    def test_raises_on_unsupported_extension(self, tmp_path: Path):
        file = tmp_path / "data.csv"
        file.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_file(file)

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_file(tmp_path / "nonexistent.txt")
