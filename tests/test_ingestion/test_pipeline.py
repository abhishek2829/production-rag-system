"""Tests for the ingestion pipeline."""

from pathlib import Path

import pytest

from rag.ingestion.pipeline import ingest_directory


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """Create a directory with mixed documents."""
    (tmp_path / "readme.md").write_text("# Project\n\nThis is a test project.")
    (tmp_path / "notes.txt").write_text("Some plain text notes here.")
    (tmp_path / "data.csv").write_text("a,b,c\n1,2,3")  # Unsupported — should be skipped
    return tmp_path


class TestIngestDirectory:
    def test_loads_supported_files(self, docs_dir: Path):
        docs = ingest_directory(docs_dir)
        assert len(docs) == 2  # .md and .txt, not .csv

    def test_skips_unsupported_files(self, docs_dir: Path):
        docs = ingest_directory(docs_dir)
        sources = [d.source for d in docs]
        assert not any("csv" in s for s in sources)

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        docs = ingest_directory(tmp_path)
        assert docs == []

    def test_raises_on_missing_directory(self):
        with pytest.raises(FileNotFoundError):
            ingest_directory(Path("/nonexistent/path"))

    def test_raises_on_file_instead_of_dir(self, tmp_path: Path):
        file = tmp_path / "single.txt"
        file.write_text("hello")
        with pytest.raises(ValueError, match="Expected a directory"):
            ingest_directory(file)

    def test_handles_nested_directories(self, tmp_path: Path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.md").write_text("# Nested document")
        (tmp_path / "top.txt").write_text("Top level document")

        docs = ingest_directory(tmp_path)
        assert len(docs) == 2
