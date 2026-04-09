# Production RAG System

[![CI](https://github.com/abhishek2829/production-rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/abhishek2829/production-rag-system/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Production-grade Retrieval-Augmented Generation system with hybrid retrieval, cross-encoder re-ranking, citation enforcement, and CI-gated quality evaluation.

## Key Features

- **Hybrid retrieval** — combines vector search (semantic) + BM25 (keyword) via Reciprocal Rank Fusion
- **Cross-encoder re-ranking** — ms-marco-MiniLM re-scores candidates for higher precision
- **Citation-enforced generation** — every claim maps to a [Source N] reference
- **Citation validation** — programmatic checks catch uncited claims and invalid references
- **Refusal capability** — declines to answer when sources don't support the query
- **Golden dataset evaluation** — 23-case test suite with automated scoring
- **CI quality gates** — PRs blocked if faithfulness drops below 85%
- **Local embeddings** — sentence-transformers (no API cost for embeddings)
- **Version-controlled prompts** — YAML-based with content hashing for traceability
- **Multi-format ingestion** — PDF, Markdown, plain text, web pages

## Architecture

```
Documents → Ingestion → Chunking → Embedding → ChromaDB
                                                    ↓
                                        ┌── Vector Search (top 10)
                                        │
              Query ────────────────────┤
                                        │
                                        └── BM25 Search (top 10)
                                                    ↓
                                            RRF Fusion → Re-rank → Top 5
                                                    ↓
              Response ← Citation Check ← Claude ←──┘
              (cited)    (validation)     (generation)
```

> Full architecture docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
> Visual learning guide: [docs/VISUAL_GUIDE.md](docs/VISUAL_GUIDE.md)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/abhishek2829/production-rag-system.git
cd production-rag-system

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your Anthropic API key

# Ingest documents
rag-ingest --dir ./data/documents

# Query (single question or interactive mode)
rag-query "What is retrieval augmented generation?"
rag-query  # interactive mode

# Run evaluation against golden dataset
rag-eval --threshold 0.85
```

## Project Structure

```
src/rag/
├── ingestion/          # Document loaders (PDF, Markdown, Text, HTML)
├── chunking/           # Token-based chunking with overlap
├── retrieval/
│   ├── vector_store.py # ChromaDB + local sentence-transformer embeddings
│   ├── bm25_retriever.py  # BM25 keyword search
│   ├── hybrid_retriever.py # Reciprocal Rank Fusion (combines both)
│   └── reranker.py     # Cross-encoder re-ranking
├── generation/
│   ├── generator.py    # Anthropic Claude cited answer generation
│   ├── citation_validator.py # Programmatic citation checking
│   └── prompt_manager.py # Versioned prompt loading with content hashing
├── evaluation/
│   ├── dataset.py      # Golden dataset loader
│   ├── runner.py       # Evaluation runner with scoring
│   └── cli.py          # rag-eval CLI command
├── models.py           # Domain models (Document, Chunk, RAGResponse)
├── config.py           # Pydantic Settings (env var configuration)
├── pipeline.py         # End-to-end pipeline orchestration
└── cli.py              # CLI entry points (rag-ingest, rag-query)
```

## Configuration

All settings are configurable via environment variables with the `RAG_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_ANTHROPIC_API_KEY` | (required) | Anthropic API key for Claude |
| `RAG_LLM_MODEL` | `claude-sonnet-4-20250514` | Anthropic model for generation |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model (no API key) |
| `RAG_CHUNK_SIZE` | `600` | Target chunk size in tokens |
| `RAG_CHUNK_OVERLAP` | `100` | Overlap between chunks in tokens |
| `RAG_RETRIEVAL_TOP_K` | `5` | Final number of chunks after re-ranking |
| `RAG_USE_HYBRID_RETRIEVAL` | `true` | Enable BM25 + vector hybrid search |
| `RAG_USE_RERANKER` | `true` | Enable cross-encoder re-ranking |

## Testing

```bash
# Unit tests (71 tests, no API key needed)
pytest tests/ -v

# Full RAG evaluation (needs API key, runs 23 golden dataset questions)
rag-eval --threshold 0.85
```

## Evaluation

The golden dataset (`eval/golden_dataset.json`) contains 23 test cases:
- 17 questions that should be answered with citations
- 4 questions that should be refused (topics not in documents)
- 1 keyword-only query (tests BM25 retrieval)
- 1 cross-document query (tests multi-source citation)

Scores measured:
- **Behavior accuracy** — answered when should answer, refused when should refuse
- **Source accuracy** — cited the correct source documents
- **Content accuracy** — answer contains expected keywords
- **Citation validity** — all [Source N] references are well-formed

## Completed Phases

- [x] **Phase 1**: Core pipeline — ingestion, chunking, local embedding, retrieval, cited generation
- [x] **Phase 2**: Hybrid retrieval (BM25 + vector), cross-encoder re-ranking, citation validation, prompt versioning
- [x] **Phase 3**: Golden eval dataset, automated scoring, CI quality gates

## Design Decisions

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed rationale on:
- Why ChromaDB over Pinecone/Weaviate
- Why token-based chunking over character-based
- Why local embeddings (sentence-transformers) over OpenAI
- Why Anthropic Claude over OpenAI for generation
- Why we define our own domain models vs. using LangChain's
- Why prompts are YAML files, not hardcoded strings
- Why hybrid retrieval (BM25 + vector) over vector-only

See [docs/VISUAL_GUIDE.md](docs/VISUAL_GUIDE.md) for visual explanations of every component.

## License

MIT
