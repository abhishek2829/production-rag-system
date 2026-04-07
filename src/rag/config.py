"""Centralized configuration using Pydantic Settings.

All settings are configurable via environment variables with the RAG_ prefix.
Example: RAG_ANTHROPIC_API_KEY=sk-ant-... RAG_CHUNK_SIZE=600
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Anthropic Claude) ---
    anthropic_api_key: str = Field(description="Anthropic API key for generation")
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model for answer generation",
    )
    llm_temperature: float = Field(
        default=0.0, description="Temperature for generation (0 = deterministic)"
    )
    llm_max_tokens: int = Field(default=1024, description="Max tokens for generated answer")

    # --- Embedding (local sentence-transformers, no API key needed) ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for local embeddings",
    )
    embedding_dimensions: int = Field(
        default=384, description="Embedding vector dimensions (384 for MiniLM)"
    )

    # --- Chunking ---
    chunk_size: int = Field(default=600, description="Target chunk size in tokens")
    chunk_overlap: int = Field(
        default=100, description="Overlap between consecutive chunks in tokens"
    )

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, description="Number of chunks to retrieve")

    # --- Storage ---
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        description="Directory for ChromaDB persistence",
    )
    chroma_collection_name: str = Field(
        default="rag_documents",
        description="ChromaDB collection name",
    )

    # --- Paths ---
    documents_dir: Path = Field(
        default=Path("./data/documents"),
        description="Directory containing source documents to ingest",
    )


def get_settings() -> Settings:
    """Factory function for settings. Allows easy mocking in tests."""
    return Settings()  # type: ignore[call-arg]
