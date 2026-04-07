# Visual Guide: How This RAG System Works

A complete visual walkthrough for understanding every component.

---

## The Big Picture

```
YOU: "What indexing does ChromaDB use?"
 │
 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR RAG SYSTEM                             │
│                                                                 │
│  1. UNDERSTAND ──▶ 2. FIND ──▶ 3. RANK ──▶ 4. ANSWER ──▶ 5. CHECK │
│     (embed)        (retrieve)   (rerank)    (Claude)      (validate)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
 │
 ▼
ANSWER: "ChromaDB uses HNSW indexing [Source 1]..."
        ✅ Citations Valid
```

---

## Part 1: How Documents Get Into The System (Ingestion)

Think of it like preparing a library. Before anyone can search for information,
you need to organize the books.

```
YOUR FILES                     WHAT HAPPENS                    RESULT
──────────                     ────────────                    ──────

┌──────────────┐
│ rag_intro.md │──┐
│ (2718 chars)  │  │
└──────────────┘  │
                  │    ┌──────────┐    ┌──────────┐    ┌──────────┐
┌──────────────┐  ├──▶│  INGEST  │──▶│  CHUNK   │──▶│  EMBED   │
│ vectordb.md  │──┤   │          │    │          │    │          │
│ (3321 chars)  │  │   │ Read the │    │ Split    │    │ Convert  │
└──────────────┘  │   │ files    │    │ into     │    │ to       │
                  │   │          │    │ pieces   │    │ numbers  │
┌──────────────┐  │   └──────────┘    └──────────┘    └────┬─────┘
│ llm_eval.md  │──┘                                        │
│ (3370 chars)  │                                           ▼
└──────────────┘                                    ┌──────────────┐
                                                    │   ChromaDB   │
                                                    │              │
                                                    │ 5 chunks     │
                                                    │ stored with  │
                                                    │ their number │
                                                    │ arrays       │
                                                    └──────────────┘
```

---

## Part 2: What Is Chunking?

Why can't we just store the whole document? Because:

```
PROBLEM: One big document → one vague embedding

  ┌─────────────────────────────────────────────────────┐
  │ This document talks about RAG, vector databases,    │
  │ HNSW indexing, evaluation methods, faithfulness,    │  → [0.12, 0.05, 0.03, ...]
  │ BM25, cosine similarity, CI/CD, golden datasets... │     "This is about... everything?"
  └─────────────────────────────────────────────────────┘     ❌ Vague, diluted meaning


SOLUTION: Small focused chunks → precise embeddings

  ┌────────────────────────────┐
  │ CHUNK 1: RAG is a technique│  → [0.85, 0.72, 0.11, ...]
  │ that reduces hallucination │     "This is about RAG + hallucination"
  │ by grounding in documents  │     ✅ Precise, focused
  └────────────────────────────┘

  ┌────────────────────────────┐
  │ CHUNK 2: ChromaDB uses     │  → [0.05, 0.12, 0.91, ...]
  │ HNSW indexing for fast     │     "This is about ChromaDB + HNSW"
  │ similarity search          │     ✅ Precise, focused
  └────────────────────────────┘
```

### How Overlap Works

```
Without overlap (BAD):
  ┌──────────────────┐ ┌──────────────────┐
  │ ...ChromaDB uses │ │ for fast nearest  │
  │ HNSW indexing    │ │ neighbor search.  │
  └──────────────────┘ └──────────────────┘
  Chunk 1 ends here ↑   ↑ Chunk 2 starts here
  
  ❌ The sentence is CUT IN HALF! Chunk 2 lost the context about "HNSW"


With 100-token overlap (GOOD):
  ┌──────────────────────────┐
  │ ...ChromaDB uses         │
  │ HNSW indexing            │
  └────────────┬─────────────┘
               │ ← overlap zone (shared between chunks)
  ┌────────────┴─────────────┐
  │ HNSW indexing            │
  │ for fast nearest         │
  │ neighbor search.         │
  └──────────────────────────┘
  
  ✅ Both chunks have the full context about HNSW indexing
```

---

## Part 3: What Is Embedding? (The Core Concept)

Embedding = converting text to a list of numbers that captures MEANING.

```
STEP 1: Text goes into the embedding model

  "ChromaDB uses HNSW"  ──▶  ┌─────────────────────┐
                              │  all-MiniLM-L6-v2   │
                              │  (runs on your Mac)  │
                              │                     │
                              │  Reads the text and │
                              │  outputs 384 numbers│
                              └─────────┬───────────┘
                                        │
                                        ▼
STEP 2: Out comes a list of 384 numbers

  [0.23, -0.45, 0.12, 0.78, 0.03, -0.91, ..., 0.67]
   ↑      ↑      ↑     ↑     ↑      ↑          ↑
   These 384 numbers represent the MEANING of the text
   Similar texts → similar numbers
```

### Why 384 Numbers?

Think of describing a person's location:

```
2 numbers (latitude, longitude):
  Can tell you: which city someone is in
  Can't tell:   which floor of which building

384 numbers:
  Can tell you: the precise meaning, topic, sentiment, context
  Like having 384 different "dimensions" of meaning
```

### How Similarity Search Works

```
Your question:  "What indexing does ChromaDB use?"
                          │
                          ▼
              Convert to 384 numbers:
              [0.21, -0.42, 0.15, ...]
                          │
                          ▼
              Compare with EVERY stored chunk:

  Chunk 1 (about RAG):        [0.85, 0.72, 0.11, ...]
  Distance from question: ────────────────────────── 0.73 (far = not similar)

  Chunk 2 (about ChromaDB):   [0.24, -0.40, 0.18, ...]
  Distance from question: ── 0.05 (close = very similar!) ✅

  Chunk 3 (about evaluation): [0.55, 0.33, -0.22, ...]
  Distance from question: ──────────────── 0.52 (medium)

  Winner: Chunk 2! Because its numbers point in the same direction
          as the question's numbers.
```

---

## Part 4: Phase 2 — Why We Need TWO Search Methods

### The Problem With Vector Search Alone

```
CASE 1: Meaning-based query → Vector search works great ✅

  Question: "How do databases find similar items?"
  
  Vector search thinks: "This is about similarity + databases + searching"
  → Finds chunks about vector databases ✅


CASE 2: Keyword query → Vector search struggles ❌

  Question: "HNSW"
  
  Vector search thinks: "Hmm... a single word... not much meaning to work with"
  → Returns random chunks ❌
  
  BM25 thinks: "Let me find every chunk containing the letters H-N-S-W"
  → Finds the exact chunk ✅
```

### How Hybrid Retrieval Works

```
STEP 1: Run BOTH search methods

  Your question: "HNSW indexing in ChromaDB"
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
  ┌──────────────┐    ┌──────────────┐
  │ Vector Search│    │ BM25 Search  │
  │ (by meaning) │    │ (by keywords)│
  └──────┬───────┘    └──────┬───────┘
         │                   │
         ▼                   ▼
  Results:               Results:
  1. Chunk B (0.82)      1. Chunk A (score 8.5)
  2. Chunk A (0.75)      2. Chunk C (score 4.2)
  3. Chunk D (0.71)      3. Chunk B (score 3.1)


STEP 2: Combine with Reciprocal Rank Fusion (RRF)

  The question: "Which chunks appeared in BOTH lists?"
  
  Chunk A: rank 2 in vector + rank 1 in BM25 = appears in BOTH ⭐
  Chunk B: rank 1 in vector + rank 3 in BM25 = appears in BOTH ⭐
  Chunk C: not in vector  + rank 2 in BM25 = only one list
  Chunk D: rank 3 in vector + not in BM25  = only one list
  
  RRF Score calculation (k=60):
  ┌─────────┬────────────────────┬────────────────────┬──────────┐
  │ Chunk   │ Vector score       │ BM25 score         │ TOTAL    │
  │         │ 1/(60 + rank)      │ 1/(60 + rank)      │          │
  ├─────────┼────────────────────┼────────────────────┼──────────┤
  │ Chunk A │ 1/62 = 0.01613     │ 1/61 = 0.01639     │ 0.03252  │ ← WINNER
  │ Chunk B │ 1/61 = 0.01639     │ 1/63 = 0.01587     │ 0.03226  │
  │ Chunk C │ 0                  │ 1/62 = 0.01613     │ 0.01613  │
  │ Chunk D │ 1/63 = 0.01587     │ 0                  │ 0.01587  │
  └─────────┴────────────────────┴────────────────────┴──────────┘
  
  Final ranking: A > B > C > D
  
  KEY INSIGHT: A and B rank highest because BOTH search methods agreed
               they were relevant. Agreement = confidence.


STEP 3: Re-rank the top candidates with a smarter model

  Cross-encoder re-ranker reads question + chunk TOGETHER:
  
  ┌──────────────────────────────────────┐
  │ Question: "HNSW indexing in ChromaDB"│
  │ +                                    │ → Score: 9.2 (very relevant!)
  │ Chunk A: "ChromaDB uses HNSW..."     │
  └──────────────────────────────────────┘
  
  ┌──────────────────────────────────────┐
  │ Question: "HNSW indexing in ChromaDB"│
  │ +                                    │ → Score: 3.1 (less relevant)
  │ Chunk B: "Vector databases store..." │
  └──────────────────────────────────────┘
  
  Why is this more accurate? Because the cross-encoder sees BOTH
  the question and the chunk at the same time. It can understand
  the RELATIONSHIP between them, not just their individual meanings.
```

### Analogy: Hiring Process

```
  STAGE 1 — Resume keyword scan (BM25):
    "Does this resume contain 'Python', 'RAG', 'ChromaDB'?"
    Fast. Scans 10,000 resumes in seconds.
    But misses great candidates who used different words.

  STAGE 2 — AI resume screening (Vector search):
    "Does this resume MEAN similar things to what we're looking for?"
    Catches candidates who wrote "embedding database" instead of "ChromaDB".
    But might miss candidates who literally built ChromaDB.

  STAGE 3 — Combine both shortlists (RRF):
    "Candidates who appeared in BOTH shortlists are probably great."
    Gives us 10-20 strong candidates.

  STAGE 4 — In-depth interview (Cross-encoder re-ranker):
    "Let me actually read each resume alongside the job description."
    Slow (can't do 10,000) but very accurate for 10-20 candidates.
    Picks the final top 5.
```

---

## Part 5: How Answer Generation Works

```
STEP 1: Format the top 5 chunks as numbered sources

  ┌─────────────────────────────────────────────┐
  │ [Source 1] (from: vectordb.md)              │
  │ ChromaDB uses HNSW indexing for fast        │
  │ similarity search. It stores vectors...     │
  │                                             │
  │ [Source 2] (from: rag_intro.md)             │
  │ RAG retrieves relevant documents and uses   │
  │ them as context for generating answers...   │
  │                                             │
  │ [Source 3] (from: llm_eval.md)              │
  │ Faithfulness measures whether the answer    │
  │ is supported by the retrieved context...    │
  │                                             │
  │ ... (Source 4, Source 5)                     │
  └─────────────────────────────────────────────┘


STEP 2: Send to Claude with strict instructions

  ┌─────────────────────────────────────────────┐
  │ SYSTEM PROMPT (the rules):                  │
  │                                             │
  │ "You MUST:                                  │
  │  - ONLY use info from the sources above     │
  │  - CITE with [Source N] for every claim     │
  │  - REFUSE if sources don't have the answer" │
  │                                             │
  │ USER MESSAGE:                               │
  │ "Here are the sources... Now answer:        │
  │  What indexing does ChromaDB use?"           │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼ (sent to Anthropic API)
                     
  ┌─────────────────────────────────────────────┐
  │            CLAUDE'S BRAIN                   │
  │                                             │
  │ "I see Source 1 talks about HNSW indexing   │
  │  in ChromaDB. Source 2 is about RAG, not    │
  │  relevant. Source 3 is about evaluation,    │
  │  not relevant.                              │
  │                                             │
  │  I'll cite Source 1 and ignore the rest."   │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼

STEP 3: Claude's answer comes back

  "ChromaDB uses HNSW (Hierarchical Navigable Small World)
   indexing for fast similarity search [Source 1]."
   
   ✅ Every claim has a [Source N] citation
   ✅ Didn't make up anything not in the sources
   ✅ Ignored irrelevant chunks (Sources 2-5)
```

---

## Part 6: Citation Validation (The Safety Net)

Even good LLMs sometimes break the rules. So we CHECK the answer after it's generated:

```
Claude's answer: "ChromaDB uses HNSW indexing [Source 1]. It was
                  created in 2020 and is used by 50,000 companies."
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │ CITATION VALIDATOR   │
                        │                     │
                        │ Check 1: Does every │
                        │ paragraph have a    │
                        │ [Source N]?          │
                        │                     │
                        │ "created in 2020"   │──▶ ❌ NO CITATION!
                        │ "50,000 companies"  │──▶ ❌ NO CITATION!
                        │                     │
                        │ Check 2: Are source │
                        │ numbers valid?      │
                        │                     │
                        │ [Source 1] exists? ──▶ ✅ Yes
                        │                     │
                        │ VERDICT: ❌ FAILED   │
                        │ "1/2 paragraphs     │
                        │  lack citations"    │
                        └─────────────────────┘

This is what you'll see in the terminal:

  ╭── Citation Check FAILED ──╮
  │ ❌ Citation issues found:  │
  │  - 1/2 content paragraphs │
  │    lack citations          │
  ╰───────────────────────────╯
```

### Refusal Detection (When the system correctly says "I don't know")

```
Question: "What is the best biryani recipe?"

Claude's answer: "I don't have enough information in the provided
                  documents to answer this question."
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │ CITATION VALIDATOR   │
                        │                     │
                        │ Contains refusal    │
                        │ phrase? ────────────▶ ✅ YES!
                        │                     │
                        │ "I don't have       │
                        │  enough information"│
                        │                     │
                        │ VERDICT: ✅ VALID    │
                        │ (correctly refused) │
                        └─────────────────────┘

  ╭── Citation Check ─────────────────────────────╮
  │ ⚠️ Model correctly refused to answer           │
  │ (sources don't support the question)           │
  ╰───────────────────────────────────────────────╯
```

---

## Part 7: Prompt Versioning (Why Prompts Are Code)

```
WEEK 1: prompt version 1.0
  "Be concise but thorough"
  Faithfulness score: 90% ✅
  
WEEK 2: You change the prompt to version 1.1
  "Be very brief, one sentence max"
  Faithfulness score: 65% ❌  ← Quality dropped!

WITHOUT VERSIONING:
  "Why did quality drop?" 🤷 No idea what changed.

WITH VERSIONING:
  Every response logs: {prompt_version: "1.1", hash: "a3f8b2..."}
  You check: "Version 1.1 dropped quality. Let me diff 1.0 vs 1.1"
  You find:  "Oh, 'one sentence max' made it skip citations!"
  You fix:   Roll back to version 1.0

  This is exactly like git blame for code, but for prompts.
```

### Content Hash — Catching Sneaky Changes

```
Version: "1.0"                    Version: "1.0"
Prompt: "Be helpful"              Prompt: "Be helpful and brief"
Hash: abc123                      Hash: def456
                                        ↑
  Someone changed the prompt but forgot to bump the version!
  The HASH catches it because the content changed.
```

---

## Part 8: The Complete Phase 2 Flow (Everything Together)

```
YOU: "What indexing does ChromaDB use?"
 │
 │ STEP 1: Embed your question locally (no API call)
 │ "What indexing does ChromaDB use?" → [0.21, -0.42, 0.15, ...]
 │
 ├──────────────────────┬──────────────────────┐
 │                      │                      │
 ▼                      ▼                      │
┌──────────────┐  ┌──────────────┐             │
│VECTOR SEARCH │  │ BM25 SEARCH  │             │
│              │  │              │             │
│Compare your  │  │Count keyword │             │
│numbers with  │  │matches in    │             │
│stored numbers│  │each chunk    │             │
│              │  │              │             │
│Top 10 by     │  │Top 10 by     │             │
│meaning       │  │keyword match │             │
└──────┬───────┘  └──────┬───────┘             │
       │                 │                     │
       ▼                 ▼                     │
  ┌──────────────────────────┐                 │
  │   RECIPROCAL RANK FUSION │                 │
  │                          │                 │
  │ Chunks in BOTH lists     │                 │
  │ rank highest             │                 │
  │                          │                 │
  │ Output: ~10 candidates   │                 │
  └────────────┬─────────────┘                 │
               │                               │
               ▼                               │
  ┌──────────────────────────┐                 │
  │   CROSS-ENCODER RERANKER │                 │
  │                          │                 │
  │ Reads question + each    │                 │
  │ chunk TOGETHER           │                 │
  │                          │                 │
  │ Picks the best 5         │                 │
  └────────────┬─────────────┘                 │
               │                               │
               ▼                               │
  ┌──────────────────────────┐                 │
  │   CLAUDE (Anthropic API) │                 │
  │                          │                 │
  │ Receives: 5 chunks +     │                 │
  │ your question +          │                 │
  │ strict citation rules    │                 │
  │                          │                 │
  │ Returns: cited answer    │                 │
  └────────────┬─────────────┘                 │
               │                               │
               ▼                               │
  ┌──────────────────────────┐                 │
  │  CITATION VALIDATOR      │                 │
  │                          │                 │
  │ ✅ Every claim cited?     │                 │
  │ ✅ Source numbers valid?   │                 │
  │ ✅ Or valid refusal?       │                 │
  └────────────┬─────────────┘                 │
               │                               │
               ▼                               │
  ╭─────────────────────────────╮              │
  │ ANSWER:                     │              │
  │ "ChromaDB uses HNSW        │              │
  │  indexing [Source 1]..."    │              │
  │                             │              │
  │ Citation Check: ✅ PASSED    │              │
  │ Sources cited: {1}          │              │
  │ Coverage: 20%               │              │
  ╰─────────────────────────────╯
```

---

## File Map: Where Everything Lives

```
production-rag-system/
│
├── src/rag/
│   │
│   ├── ingestion/              ← STEP 1: Read files
│   │   ├── loaders.py          Each file type has its own reader
│   │   └── pipeline.py         Orchestrates reading a whole directory
│   │
│   ├── chunking/               ← STEP 2: Split into pieces
│   │   └── token_chunker.py    Splits by token count, preserves sentences
│   │
│   ├── retrieval/              ← STEP 3: Find relevant chunks
│   │   ├── vector_store.py     ChromaDB + local embeddings (meaning search)
│   │   ├── bm25_retriever.py   BM25 keyword search (exact word matching)
│   │   ├── hybrid_retriever.py Combines vector + BM25 results (RRF)
│   │   └── reranker.py         Cross-encoder (smarter re-scoring)
│   │
│   ├── generation/             ← STEP 4: Generate + validate answer
│   │   ├── generator.py        Sends chunks to Claude, gets cited answer
│   │   ├── citation_validator.py  Checks citations are valid
│   │   └── prompt_manager.py   Loads versioned prompts from YAML
│   │
│   ├── models.py               Domain objects (Document, Chunk, etc.)
│   ├── config.py               All settings via environment variables
│   ├── pipeline.py             Wires everything together
│   └── cli.py                  Terminal commands (rag-ingest, rag-query)
│
├── configs/
│   └── prompts.yaml            Version-controlled prompts (v1.0)
│
├── tests/                      53 tests covering all components
├── docs/
│   ├── ARCHITECTURE.md         Technical architecture
│   └── VISUAL_GUIDE.md         ← YOU ARE HERE
│
└── data/
    ├── documents/              Your source documents go here
    └── chroma_db/              Vector database storage (auto-created)
```

---

## Glossary of Terms Used in This Project

| Term | Plain English | Example |
|------|--------------|---------|
| **Embedding** | Converting text to a list of numbers that capture its meaning | "hello" → [0.23, -0.45, ...] |
| **Vector** | Just a list of numbers | [0.23, -0.45, 0.12] is a 3-dim vector |
| **Dimensions (dim)** | How many numbers in the list | Our model uses 384 dimensions |
| **Cosine Similarity** | How similar two vectors are (by angle) | 1.0 = identical, 0.0 = unrelated |
| **Chunk** | A piece of a document (~600 tokens) | One paragraph about HNSW indexing |
| **Token** | The unit LLMs read (roughly 3/4 of a word) | "hello world" = 2 tokens |
| **ChromaDB** | Database that stores and searches vectors | Like a filing cabinet for embeddings |
| **BM25** | Keyword search algorithm (counts word matches) | "Find chunks containing 'HNSW'" |
| **Hybrid Retrieval** | Using BOTH vector + keyword search | Best of both worlds |
| **RRF** | Method to combine two ranked lists fairly | Chunks in both lists rank highest |
| **Cross-encoder** | Model that reads query + chunk together for accurate scoring | Like a detailed interview |
| **Re-ranking** | Using a smarter model to re-score initial results | Picks best 5 from top 20 |
| **Citation** | [Source N] reference linking a claim to a source | "RAG helps [Source 1]" |
| **Faithfulness** | Whether the answer only uses info from sources | No made-up facts = faithful |
| **Hallucination** | When an LLM confidently states false information | "ChromaDB was made in 1995" (wrong) |
| **PR (Pull Request)** | A request to merge your code changes into the main codebase | "Please review and merge my work" |
| **CI/CD** | Automated testing that runs when you push code | Tests run automatically on every push |
| **Idempotent** | Running the same operation twice gives the same result | Re-ingesting same doc doesn't duplicate |
| **Prompt Versioning** | Tracking which prompt version generated each answer | Like git for prompts |
