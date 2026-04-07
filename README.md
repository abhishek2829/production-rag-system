# Production RAG System

[![CI](https://github.com/yourusername/production-rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/production-rag-system/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Production-grade Retrieval-Augmented Generation system with hybrid retrieval, citation enforcement, and CI-gated quality evaluation.

## Key Features

- **Token-based chunking** with sentence boundary awareness (tiktoken)
- **Citation-enforced generation** — every claim maps to a [Source N] reference
- **Refusal capability** — declines to answer when sources don't support the query
- **Idempotent ingestion** — deterministic chunk IDs enable safe re-processing
- **Version-controlled prompts** — YAML-based, auditable, Git-tracked
- **Multi-format ingestion** — PDF, Markdown, plain text, web pages

## Architecture

```
Documents → Ingestion → Chunking → Embedding → ChromaDB
                                                    ↓
              Response ← Generator ← Retrieval ←────┘
              (cited)    (GPT-4o)    (vector)
```

> Full architecture docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/production-rag-system.git
cd production-rag-system

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your OpenAI API key

# Ingest documents
rag-ingest --dir ./data/documents

# Query (single question or interactive mode)
rag-query "What is retrieval augmented generation?"
rag-query  # interactive mode
```

## Project Structure

```
src/rag/
├── ingestion/       # Document loaders (PDF, Markdown, Text, HTML)
├── chunking/        # Token-based chunking with overlap
├── retrieval/       # Vector store (ChromaDB) + similarity search
├── generation/      # LLM answer generation with citation enforcement
├── models.py        # Domain models (Document, Chunk, RAGResponse)
├── config.py        # Pydantic Settings (env var configuration)
├── pipeline.py      # End-to-end pipeline orchestration
└── cli.py           # CLI entry points (rag-ingest, rag-query)
```

## Configuration

All settings are configurable via environment variables with the `RAG_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_OPENAI_API_KEY` | (required) | OpenAI API key |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Model for answer generation |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `RAG_CHUNK_SIZE` | `600` | Target chunk size in tokens |
| `RAG_CHUNK_OVERLAP` | `100` | Overlap between chunks in tokens |
| `RAG_RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |

## Testing

```bash
pytest tests/ -v
```

## Roadmap

- [x] **Phase 1**: Core pipeline — ingestion, chunking, embedding, retrieval, cited generation
- [ ] **Phase 2**: Hybrid retrieval (BM25 + vector), cross-encoder re-ranking, citation enforcement rules
- [ ] **Phase 3**: Golden eval dataset, offline faithfulness evaluation, CI quality gates

## Design Decisions

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed rationale on:
- Why ChromaDB over Pinecone/Weaviate
- Why token-based chunking over character-based
- Why we define our own domain models vs. using LangChain's
- Why prompts are YAML files, not hardcoded strings

## License

MIT
