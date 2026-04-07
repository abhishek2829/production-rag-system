# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐ │
│  │ Ingestion│───▶│ Chunking │───▶│ Embedding │───▶│ ChromaDB │ │
│  │ (loaders)│    │ (token)  │    │ (OpenAI)  │    │ (store)  │ │
│  └──────────┘    └──────────┘    └───────────┘    └────┬─────┘ │
│                                                        │        │
│  ┌──────────┐    ┌───────────┐   ┌───────────┐        │        │
│  │ Response │◀───│ Generator │◀──│ Retrieval │◀───────┘        │
│  │ (cited)  │    │ (GPT-4o) │   │ (vector)  │                  │
│  └──────────┘    └───────────┘   └───────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Design

### Ingestion Layer (`rag/ingestion/`)
- **Strategy pattern**: One loader per file type (PDF, Markdown, Text, HTML)
- **Resilient**: Failed files are logged and skipped, not fatal
- **Extensible**: Add new formats by adding a loader function + registry entry

### Chunking Layer (`rag/chunking/`)
- **Token-based**: Uses tiktoken (same tokenizer as OpenAI) for accurate measurement
- **Sentence-aware**: Splits at sentence boundaries, not mid-word
- **Configurable**: chunk_size=600, overlap=100 (tunable via env vars)

### Retrieval Layer (`rag/retrieval/`)
- **Phase 1**: Pure vector similarity (cosine distance via ChromaDB)
- **Phase 2**: Hybrid retrieval (BM25 + vector) + cross-encoder re-ranking
- **Idempotent**: Deterministic chunk IDs enable safe re-ingestion

### Generation Layer (`rag/generation/`)
- **Citation-enforced**: System prompt mandates [Source N] citations
- **Version-controlled prompts**: YAML files in `configs/`, tracked in Git
- **Refusal-capable**: Declines to answer when sources don't support the query

## Data Flow

1. **Ingest**: Documents → Loaders → `Document` objects
2. **Chunk**: `Document` → TokenChunker → `Chunk` objects with metadata
3. **Embed**: `Chunk` → OpenAI embeddings → ChromaDB (persistent)
4. **Retrieve**: Query → embed → cosine similarity search → `RetrievedChunk`
5. **Generate**: Retrieved chunks + query → LLM with citation prompt → `RAGResponse`

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector DB | ChromaDB | Open-source, self-hostable, good enough for <1M chunks |
| Chunking | Token-based (tiktoken) | Matches embedding model's token space |
| Chunk size | 600 tokens | Sweet spot for text-embedding-3-small |
| Overlap | 100 tokens | Preserves context at chunk boundaries |
| Embeddings | text-embedding-3-small | Cost-effective, high quality, 1536 dims |
| Generation | gpt-4o-mini | Fast, cheap, follows citation instructions well |
| Config | Pydantic Settings | Type-safe, env var support, validation |
| Prompts | YAML files | Version-controlled, auditable, swappable |

## Phase Roadmap

- **Phase 1** ✅: Core pipeline (ingest → chunk → embed → retrieve → generate)
- **Phase 2**: Hybrid retrieval (BM25 + vector), cross-encoder re-ranking, citation enforcement
- **Phase 3**: Golden eval dataset, faithfulness scoring, CI quality gates
