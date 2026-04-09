# Project 1 Learnings: Production RAG System

Everything I learned while building this project, explained in simple language.

---

## Table of Contents

1. [What is RAG and Why Does It Exist?](#1-what-is-rag-and-why-does-it-exist)
2. [How Documents Get Prepared (Ingestion + Chunking)](#2-how-documents-get-prepared)
3. [What Are Embeddings?](#3-what-are-embeddings)
4. [How Search Works (Vector Search)](#4-how-search-works)
5. [Why One Search Method Isn't Enough (Hybrid Retrieval)](#5-why-one-search-method-isnt-enough)
6. [What is RRF and Why Not Just Average the Scores?](#6-what-is-rrf)
7. [What is Re-ranking and Why Do We Need It?](#7-what-is-re-ranking)
8. [How Answer Generation Works (Citation Enforcement)](#8-how-answer-generation-works)
9. [What is Citation Validation?](#9-what-is-citation-validation)
10. [What is Prompt Versioning and Why Prompts Are Code?](#10-what-is-prompt-versioning)
11. [What is a Golden Dataset?](#11-what-is-a-golden-dataset)
12. [What is CI and What Are Quality Gates?](#12-what-is-ci-and-quality-gates)
13. [What is a Regression and How Do We Prevent It?](#13-what-is-a-regression)
14. [Project Structure Decisions](#14-project-structure-decisions)
15. [Key Interview Questions and Answers](#15-key-interview-questions)
16. [Tools and Technologies Used](#16-tools-and-technologies)
17. [Numbers and Results](#17-numbers-and-results)

---

## 1. What is RAG and Why Does It Exist?

### The Problem

AI models like ChatGPT and Claude are trained on data up to a certain date. After that
date, they don't know anything new. Also, they don't know anything about YOUR private
documents — your company's internal docs, your personal notes, etc.

Worse, when they don't know something, they don't say "I don't know." Instead, they
**make something up that sounds convincing**. This is called **hallucination**.

Example of hallucination:
```
You: "What is our company's refund policy?"
ChatGPT: "Your company offers a 30-day money-back guarantee..."

But your company actually has a 14-day refund window!
ChatGPT made it up because it has no idea about your company.
```

### The Solution: RAG

RAG = Retrieval-Augmented Generation

In simple terms: **Before asking the AI to answer, first FIND the relevant documents,
then give those documents to the AI and say "answer ONLY from these."**

```
WITHOUT RAG:
  You ask question → AI answers from its training (might hallucinate)

WITH RAG:
  You ask question → System FINDS relevant docs → AI answers ONLY from those docs
```

It's like the difference between:
- Asking someone to answer a history question from memory (might get it wrong)
- Giving them the textbook first and saying "answer from this page only"

### Why I Built This

This is the #1 pattern companies use to build AI products. If you're applying for
AI engineering roles, you MUST know how to build a RAG system. It's like knowing
how to build a REST API — it's fundamental.

---

## 2. How Documents Get Prepared

Before the system can search through documents, it needs to prepare them. This
happens in two steps: **Ingestion** and **Chunking**.

### Ingestion (Reading the files)

Different documents come in different formats — PDFs, Markdown files, plain text,
web pages. Each format needs a different approach to extract the text:

- **PDF**: Use a library (pypdf) to extract text from each page
- **Markdown**: Read the raw text (keep the # headings because they help with structure)
- **Plain text**: Just read the file directly
- **Web pages**: Download the HTML, strip out the navigation/ads/scripts, keep the content

**Why this matters in production**: Real companies have documents in 10+ formats.
A production system needs to handle all of them without crashing. If one PDF is
corrupted, it should skip it and continue with the rest — not crash the entire batch.

### Chunking (Splitting into pieces)

A document might be 50 pages long. We can't feed all 50 pages to the AI for every
question. We need to split it into smaller pieces called "chunks."

**Why not feed the whole document?** Two reasons:

1. **The AI has a limited input size** (called "context window"). You can only fit
   so much text in one request.

2. **Smaller chunks get better search results.** If you have one giant document about
   everything, the search can't tell which PART is relevant. But if you have small
   chunks, each about one specific topic, search finds the exact right piece.

**How big should chunks be?** We use ~600 tokens (roughly 450 words).

- Too small (100 tokens): Each chunk is just one sentence. Not enough context.
  Like giving someone a single sentence from a book and asking them to explain the topic.

- Too big (5000 tokens): Each chunk covers many topics. The search can't tell
  which part is relevant. Like giving someone an entire chapter when they only
  need one paragraph.

- Sweet spot (500-800 tokens): Each chunk covers one concept or section.
  Enough context to be useful, small enough to be focused.

### What is a "Token"?

A token is the unit that AI models read. It's roughly 3/4 of a word.

```
"Hello world" = 2 tokens
"Retrieval-Augmented Generation" = 4 tokens
"I love pizza" = 3 tokens
```

We measure chunks in tokens (not characters or words) because the AI models
think in tokens. A 600-character chunk might be 150 tokens or 800 tokens
depending on the words used. Token counting gives us consistency.

### What is Overlap?

When we split a document into chunks, we might cut a sentence in half.

```
WITHOUT OVERLAP:
  Chunk 1: "...ChromaDB uses HNSW indexing"
  Chunk 2: "for fast nearest neighbor search."
  
  The sentence is CUT! Chunk 2 has no idea what "for fast" refers to.

WITH OVERLAP (100 tokens):
  Chunk 1: "...ChromaDB uses HNSW indexing for fast"
  Chunk 2: "HNSW indexing for fast nearest neighbor search."
  
  Both chunks have the complete thought. The overlapping part appears in both.
```

Overlap = the chunks share some content at their boundaries. This prevents
information loss at the edges.

---

## 3. What Are Embeddings?

This is the most important concept in the entire project.

### The Problem

Computers don't understand text. They understand numbers. When you search
Google for "how to make coffee", Google doesn't compare letters — it converts
your query and every web page into numbers, then compares the numbers.

### Embedding = Converting Text to Numbers That Capture Meaning

An embedding model reads text and outputs a list of numbers (called a "vector")
that represents the MEANING of that text.

```
"RAG reduces hallucination" → [0.23, -0.45, 0.12, 0.78, ..., 0.67]
                                This list has 384 numbers.
                                Together, they represent the MEANING.
```

**The magic**: Similar meanings → similar numbers.

```
"RAG reduces hallucination"      → [0.23, -0.45, 0.12, ...]
"RAG prevents making stuff up"   → [0.21, -0.42, 0.15, ...]  ← SIMILAR numbers!

"Best pizza recipe in town"      → [-0.78, 0.33, -0.91, ...]  ← VERY DIFFERENT numbers
```

### What Does "384 Dimensions" Mean?

Each text gets converted into a list of **384 numbers**. That's it.

Think of it like describing a location:
- GPS uses **2 numbers** (latitude, longitude) → tells you which city
- Embeddings use **384 numbers** → tells you the precise meaning, topic,
  sentiment, context, and nuance of the text

More numbers = more detail captured = better at distinguishing subtle differences.

### What Does "Local Embeddings" Mean?

The embedding model (`all-MiniLM-L6-v2`) runs on YOUR laptop. It's an 80MB
file that converts text to numbers without any internet connection or API call.

- **No cost**: You don't pay per embedding (OpenAI charges for their embeddings)
- **No internet needed**: Works offline
- **Private**: Your documents never leave your machine
- **Trade-off**: Slightly lower quality than OpenAI's model, but good enough for most cases

### What is a Vector?

A vector is just a list of numbers. That's literally it.

```
[0.23, -0.45, 0.12] is a 3-dimensional vector (3 numbers)
[0.23, -0.45, 0.12, ..., 0.67] is a 384-dimensional vector (384 numbers)
```

When people say "vector database" or "vector search", they just mean
"a database that stores lists of numbers and can find similar lists."

---

## 4. How Search Works

### Cosine Similarity (How We Compare Vectors)

When you ask a question, the system converts it to a vector (list of numbers).
Then it needs to find which stored chunks have the most similar vectors.

**Cosine similarity** measures how similar two vectors are by looking at the
**angle** between them:

```
Think of vectors as arrows pointing in a direction:

  Your question's arrow:    →
  Chunk A's arrow:          →    (same direction = SIMILAR! Score: 0.95)
  Chunk B's arrow:          ↗    (slightly different = somewhat similar. Score: 0.70)
  Chunk C's arrow:          ↑    (perpendicular = NOT related. Score: 0.05)
  Chunk D's arrow:          ←    (opposite = OPPOSITE meaning. Score: -0.90)
```

Scores range from -1 to 1:
- **1.0** = identical meaning
- **0.0** = completely unrelated
- **-1.0** = opposite meaning

### ChromaDB (Where We Store Vectors)

ChromaDB is a database specifically designed to store vectors and find similar ones.
Think of it as a filing cabinet where:
- Each drawer contains a chunk of text + its vector (list of numbers)
- When you search, it compares your question's vector against every drawer
- Returns the drawers with the most similar vectors

**Why ChromaDB specifically?**
- Open source (free)
- Runs locally on your laptop (no server needed)
- Good enough for up to ~1 million chunks
- You'd switch to Pinecone or Weaviate for bigger scale

### What is HNSW? (How ChromaDB Searches Fast)

If you have 1 million chunks, comparing your question against ALL of them would
take seconds. HNSW is an algorithm that makes it take milliseconds.

It works like navigating an airport:

```
Layer 3 (Continent): You're looking for a coffee shop in Tokyo.
  First, narrow down: Asia (not Europe, not Americas)

Layer 2 (Country): Now in Asia.
  Narrow down: Japan (not China, not India)

Layer 1 (City): Now in Japan.
  Narrow down: Tokyo (not Osaka, not Kyoto)

Layer 0 (Street): Now in Tokyo.
  Check nearby shops → Found it!
```

Instead of checking every shop in the world (slow), you take smart shortcuts
through layers (fast). HNSW does the same thing with vectors.

**Result**: Instead of checking 1,000,000 vectors, HNSW checks ~100 and still
finds the right answer 95%+ of the time. 1000x faster.

---

## 5. Why One Search Method Isn't Enough

This was a key "aha moment" for me in Phase 2.

### The Problem With Vector Search Alone

Vector search understands MEANING but can miss exact KEYWORDS.

```
EXAMPLE 1 — Vector search works great:

  Question: "How do databases find similar items?"
  
  Vector search thinks: "This is about similarity + databases + searching"
  → Finds chunks about vector databases ✅


EXAMPLE 2 — Vector search struggles:

  Question: "HNSW"
  
  Vector search thinks: "Hmm... one word... not much meaning to work with"
  → Returns random chunks ❌
```

### The Problem With BM25 (Keyword Search) Alone

BM25 finds exact WORDS but doesn't understand MEANING.

```
EXAMPLE 1 — BM25 works great:

  Question: "HNSW"
  
  BM25 counts: "How many chunks contain the letters H-N-S-W?"
  → Finds the exact chunk ✅


EXAMPLE 2 — BM25 struggles:

  Question: "How do databases find similar items?"
  
  BM25 counts: "How many chunks contain 'similar' and 'items'?"
  → Might miss chunks that say "nearest neighbor search" (same meaning, different words) ❌
```

### The Solution: Use BOTH (Hybrid Retrieval)

Run both search methods, then combine their results. Chunks that appear in
BOTH lists are probably very relevant.

```
Your question: "HNSW indexing in ChromaDB"

Vector search finds: Chunk B, Chunk A, Chunk D
BM25 finds:          Chunk A, Chunk C, Chunk B

Chunk A appears in BOTH → probably very relevant! ⭐
Chunk B appears in BOTH → probably very relevant! ⭐
Chunk C only in BM25   → might be relevant
Chunk D only in vector → might be relevant

Final ranking: A, B, C, D
```

### What is BM25?

BM25 (Best Match 25) is a search algorithm from the 1990s. It's simple:

1. **Count how many times the search word appears in each chunk** (more = better)
2. **Rare words score higher** — "HNSW" is rare (appears in few chunks) so it gets
   a high score. "the" appears everywhere so it gets almost zero.
3. **Longer chunks are normalized** — so a short chunk with the word isn't penalized
   compared to a long chunk that mentions it once

No AI, no embeddings, no API calls. Just smart word counting. And it's been powering
search engines (including early Google) for 30 years.

---

## 6. What is RRF?

RRF = Reciprocal Rank Fusion. It's the method we use to COMBINE results from
vector search and BM25 into one ranked list.

### Why Can't We Just Average the Scores?

Vector search scores and BM25 scores are on COMPLETELY DIFFERENT SCALES:

```
Vector search scores: 0.0 to 1.0 (cosine similarity)
  Chunk A: 0.85
  Chunk B: 0.72

BM25 scores: 0 to ~25 (depends on document length, word frequency)
  Chunk A: 3.2
  Chunk C: 18.5
```

If you average these, BM25 scores would dominate because they're larger numbers.
It's like averaging someone's height in centimeters (175) with their weight in
kilograms (70) — the numbers aren't comparable.

### How RRF Works

RRF ignores the actual scores entirely. It only cares about RANK POSITION
(1st, 2nd, 3rd, etc.).

```
Formula: RRF_score = 1 / (60 + rank)

Vector search results:       BM25 results:
  Rank 1: Chunk A              Rank 1: Chunk C
  Rank 2: Chunk B              Rank 2: Chunk A

RRF Scores:
  Chunk A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325  ← in BOTH lists!
  Chunk C: 0        + 1/(60+1) = 0      + 0.0164 = 0.0164   ← only BM25
  Chunk B: 1/(60+2) + 0        = 0.0161 + 0      = 0.0161   ← only vector

Winner: Chunk A (highest combined score because it appeared in BOTH lists)
```

**Key insight**: If both vector search AND keyword search agree a chunk is
relevant, it's almost certainly relevant. Agreement = confidence.

---

## 7. What is Re-ranking?

### The Analogy: Hiring Process

```
STAGE 1 — Resume keyword scan (BM25):
  "Does this resume contain 'Python', 'RAG', 'ChromaDB'?"
  Scans 10,000 resumes in seconds.
  But misses people who used different words.

STAGE 2 — AI resume screening (Vector search):
  "Does this resume MEAN similar things to what we need?"
  Catches people who wrote "embedding database" instead of "ChromaDB."
  But might miss people who literally built ChromaDB.

STAGE 3 — Combine shortlists (RRF):
  "People in BOTH shortlists are probably great."
  Gives us 10-20 strong candidates.

STAGE 4 — Actual interview (Cross-encoder re-ranker):
  Reads each resume next to the job description carefully.
  Picks the best 5.
```

### Bi-encoder vs Cross-encoder

**Bi-encoder** (what vector search uses — fast, less accurate):
```
Monday:  Convert question to numbers → store
Tuesday: Convert chunk to numbers → store

Later: Compare the two lists of numbers.

Problem: The question and chunk never "saw" each other.
It's like judging if a key fits a lock by looking at
PHOTOS of each — separately.
```

**Cross-encoder** (the re-ranker — slow, more accurate):
```
Feed BOTH the question AND the chunk together into the model:

"What indexing does ChromaDB use?" + "ChromaDB uses HNSW indexing..."
                     ↓
              Model reads BOTH together
                     ↓
              Score: 9.2 (very relevant!)

It's like actually TRYING the key in the lock.
Much more accurate because it sees the RELATIONSHIP.
```

### Why Not Use Cross-encoder for Everything?

Speed. With 1,000 chunks:
- Cross-encoder: 1,000 comparisons × 50ms each = 50 seconds ❌
- Our approach: Vector+BM25 find top 10 (0.1 seconds) → Cross-encoder on 10 (0.5 seconds) ✅

---

## 8. How Answer Generation Works

After retrieval gives us the top 5 most relevant chunks, we send them to
Claude (Anthropic's AI) with very strict instructions.

### The Prompt (Our Instructions to Claude)

We tell Claude:
1. "ONLY use information from the provided source chunks"
2. "CITE your sources using [Source N] for every claim"
3. "If the sources don't have the answer, say 'I don't have enough information'"
4. "NEVER make anything up"

### Why This Prevents Hallucination

Without these instructions, Claude would answer from its training data
(which might be wrong or outdated). With these instructions, Claude can
ONLY use the specific documents we gave it.

```
WITHOUT our prompt:
  Q: "What is our refund policy?"
  A: "Most companies offer 30-day refunds..." ← Made up!

WITH our prompt:
  Q: "What is our refund policy?"
  A: "According to the provided documents, your refund policy is
      14 days from purchase [Source 1]." ← From actual document!
```

### What Does [Source N] Mean?

We number each chunk we send to Claude:
```
[Source 1] (from: refund_policy.md) — "Customers may request..."
[Source 2] (from: faq.md) — "Our return window is..."
```

Claude then references these numbers in its answer:
```
"Your refund policy allows returns within 14 days [Source 1]."
```

This lets the user VERIFY where the information came from. They can click
on [Source 1] and read the original document themselves.

---

## 9. What is Citation Validation?

Even with a good prompt, Claude sometimes breaks the rules. It might:
1. Make a claim without citing any source
2. Cite a source number that doesn't exist (e.g., [Source 8] when we only gave 5)
3. Answer a question the sources don't actually cover

So AFTER Claude generates an answer, we run an automated check:

```
Claude's answer: "ChromaDB uses HNSW indexing [Source 1]. It was
                  created in 2020 by a team at Chroma."
                                          ↓
                                   VALIDATION CHECK:
                                          
  ✅ "HNSW indexing [Source 1]" — claim has citation, good
  ❌ "created in 2020" — no [Source N] citation! Where did this come from?
  ❌ "team at Chroma" — no citation! Could be hallucinated!
  
  VERDICT: FAILED — 2 uncited claims detected
```

### Refusal Detection

When Claude correctly says "I don't know," that's actually a GOOD result.

```
Q: "Best biryani recipe?"
A: "I don't have enough information in the provided documents
    to answer this question."
    
VERDICT: ✅ PASSED — model correctly refused to answer
```

This is important because a system that says "I don't know" when it doesn't know
is MORE trustworthy than one that always gives an answer (even when it's making it up).

---

## 10. What is Prompt Versioning?

### Why Prompts Are Code

In traditional software, you write code and version-control it with Git.
If something breaks, you can look at the history and see what changed.

In AI systems, **prompts control behavior just like code does**. Changing one
word in a prompt can completely change the system's behavior:

```
Prompt v1.0: "Be concise but thorough"
  → System gives detailed answers with citations. Faithfulness: 90%

Prompt v1.1: "Be very brief, one sentence max"
  → System gives one-liners and skips citations. Faithfulness: 65%
```

If prompts are just strings buried in Python code, you can't easily:
- See what changed (no diff)
- Roll back to the old version
- Track which prompt version generated which answer

### How We Solved This

We store prompts in a YAML file (`configs/prompts.yaml`) with a version number:

```yaml
version: "1.0"
system_prompt: |
  You are a precise, helpful assistant...
```

Every answer the system generates logs which prompt version was used:
```
Generated answer: prompt_version=1.0, hash=5f12e68604d6
```

### What is Content Hashing?

Even if someone forgets to bump the version number after changing a prompt,
we automatically generate a **hash** (a unique fingerprint) of the prompt content.

```
Prompt: "Be helpful"        → Hash: abc123
Prompt: "Be very helpful"   → Hash: def456  ← Different! Change detected.
```

If the version says "1.0" but the hash changed, we know someone edited the
prompt without updating the version. The hash catches sneaky changes.

---

## 11. What is a Golden Dataset?

A golden dataset is like an **exam paper** for your AI system.

Instead of manually asking questions and eyeballing the answers, you create
a file with questions where you ALREADY KNOW the correct answer:

```json
{
  "question": "What is RAG?",
  "expected_behavior": "answer",        ← should it answer or refuse?
  "expected_sources": ["rag.md"],        ← which document should be cited?
  "must_contain": ["retrieval"],         ← what keywords must appear?
}
```

Then you run ALL 23 questions through the system automatically and check:
- Did it answer when it should answer? (behavior)
- Did it cite the right document? (sources)
- Does the answer contain the expected keywords? (content)
- Are the citations properly formatted? (citations)

### Why 23 Test Cases?

We cover different scenarios:
- **17 normal questions** — "What is RAG?", "What is ChromaDB?", etc.
- **4 refusal questions** — biryani recipe, weather, Python code, CEO name
  (topics NOT in our documents — system should refuse)
- **1 keyword query** — just "HNSW" (tests BM25 keyword search)
- **1 cross-document query** — needs info from TWO different documents

### What Are the Scores?

After running all 23 questions, we calculate:

```
Behavior accuracy:  100%  ← Did it answer/refuse correctly? YES, every time.
Source accuracy:     100%  ← Did it cite the right document? YES, every time.
Content accuracy:   100%  ← Did the answer have expected keywords? YES, every time.
Citation validity:   32%  ← Did every paragraph have a [Source N]? Only sometimes.
─────────────────────────
Overall score:       86%  ← Weighted average of the above.
```

**Why is citation validity low (32%)?** Because our validator was too strict.
It flagged bullet points and numbered lists as "uncited paragraphs." When Claude
writes:

```
RAG has three stages [Source 1]:
1. Indexing              ← validator says: "No citation on this line!"
2. Retrieval             ← validator says: "No citation on this line!"
3. Generation            ← validator says: "No citation on this line!"
```

The answer IS correct and faithful. The citation at the top covers the whole list.
But our validator counts each bullet as a separate paragraph.

**This is a real-world learning**: Your measurement tool can have bugs too.
The system works correctly; the grading rubric was too strict.

---

## 12. What is CI and Quality Gates?

### CI = Continuous Integration

"Continuous" = every time. "Integration" = combining code.

In the old days, developers would work alone for weeks, then try to combine
everyone's code at once. It was chaos — everything would break.

CI means: **Every time someone pushes code, a computer automatically runs
tests to check if anything broke.** This happens within minutes, not weeks.

### How It Works in Practice

```
You push code to GitHub
        ↓
GitHub notices: "New code arrived!"
        ↓
GitHub starts a computer in the cloud (called a "runner")
        ↓
The runner automatically:
  1. Downloads your code
  2. Installs dependencies
  3. Runs the linter (checks code formatting)
  4. Runs all 71 tests
  5. Runs the type checker
  6. (On PRs) Runs the RAG evaluation (23 golden dataset questions)
        ↓
Everything passed? → ✅ Green checkmark
Something failed?  → ❌ Red X — fix it before merging
```

You don't do any of this manually. The machine does it. Every. Single. Time.

### What is a Quality Gate?

A gate is something that blocks you until a condition is met.

```
Airport gate:     Need valid boarding pass → otherwise can't board
Quality gate:     Need 85%+ score          → otherwise can't merge code
```

Our CI quality gate:
- Runs all 23 golden dataset questions through the RAG system
- Calculates the overall score
- If score < 85% → ❌ Your code change is BLOCKED from being merged
- If score ≥ 85% → ✅ Your code change can be merged

This prevents someone from accidentally breaking the system by changing
a prompt, a chunking parameter, or any other setting.

---

## 13. What is a Regression?

### Simple Definition

A **regression** is when something that USED TO WORK stops working because
of a new change.

Think of it like this:

```
Monday:    Your car starts fine. ✅
Tuesday:   Mechanic changes the oil filter.
Wednesday: Car won't start. ❌

The oil filter change REGRESSED (broke) the starting ability.
The car went BACKWARDS — it could start before, now it can't.
That's a regression.
```

### Regression in Software

```
Monday:    RAG system answers questions correctly. Score: 90% ✅
Tuesday:   Developer changes the chunking size from 600 to 200 tokens.
Wednesday: RAG system gives worse answers. Score: 60% ❌

The chunk size change REGRESSED the answer quality.
The system went BACKWARDS — it was better before the change.
```

### How We Prevent Regressions

**Without prevention:**
```
Developer changes chunk size → pushes code → deployed to production
→ Users start getting bad answers → Someone notices 3 days later
→ Scramble to find what broke → Revert the change → 3 days of bad service
```

**With our CI quality gate:**
```
Developer changes chunk size → pushes code → CI automatically runs
→ Golden dataset score drops to 60% → ❌ CI BLOCKS the merge
→ Developer sees: "Quality gate failed. Score 60% < threshold 85%"
→ Developer reverts the change BEFORE it reaches production
→ Users never see bad answers
```

The regression is **caught before it reaches users**. That's the entire point
of automated evaluation and CI quality gates.

---

## 14. Project Structure Decisions

### Why `pyproject.toml` Instead of `setup.py`?

`setup.py` is the old way (pre-2021) of configuring Python projects. `pyproject.toml`
is the modern standard (PEP 621). An interviewer seeing `setup.py` would flag it
as outdated. It's like using Internet Explorer instead of Chrome.

### Why `src/` Layout?

```
project/
├── src/rag/       ← our code is INSIDE src/
├── tests/
└── pyproject.toml
```

This prevents a common Python bug where `import rag` accidentally imports from
the local directory instead of the installed package. Google, Meta, and Stripe
all use this layout.

### Why Our Own Domain Models?

We defined our own `Document`, `Chunk`, `RAGResponse` classes instead of using
LangChain's built-in classes. Why?

- **No vendor lock-in**: If we switch from LangChain to LlamaIndex tomorrow,
  our models stay the same
- **Testability**: We can test our models without importing LangChain
- **Control**: We decide what fields exist, not a third-party library

### Why Pydantic Settings for Configuration?

All settings come from environment variables with the `RAG_` prefix:
```
RAG_ANTHROPIC_API_KEY=sk-ant-...
RAG_CHUNK_SIZE=600
RAG_USE_HYBRID_RETRIEVAL=true
```

Benefits:
- **Type-safe**: If you set `RAG_CHUNK_SIZE=abc` it will error (expects int)
- **Documented**: Each setting has a description
- **12-factor app**: Environment variables are the standard way to configure
  apps in Docker/Kubernetes/production environments
- **No secrets in code**: API keys live in `.env` (gitignored), not in source code

---

## 15. Key Interview Questions

### Q: "How do you prevent hallucination in RAG?"

**Answer**: "Three layers of defense:
1. **Prompt engineering**: System prompt instructs the model to ONLY use
   provided sources and cite with [Source N]
2. **Programmatic validation**: After generation, we parse the response
   and check that every claim has a citation. Uncited paragraphs are flagged.
3. **Refusal capability**: When sources don't support the question, the model
   is trained to say 'I don't have enough information' instead of guessing.
   We validate this with dedicated refusal test cases."

### Q: "Why hybrid retrieval?"

**Answer**: "Vector search understands meaning but misses exact keywords.
If someone searches 'HNSW', vector search doesn't know what that abbreviation
means. BM25 keyword search finds it instantly. By combining both with Reciprocal
Rank Fusion, we get the best of both — semantic understanding AND keyword precision.
Chunks that appear in both lists rank highest because both methods agree they're relevant."

### Q: "Why not just use the cross-encoder for everything?"

**Answer**: "Speed. A cross-encoder reads the query and each chunk together,
which is very accurate but slow — about 50ms per chunk. With 10,000 chunks,
that's 500 seconds. So we use fast methods (vector + BM25) to narrow down to
10-20 candidates, then the cross-encoder carefully re-ranks only those few.
It's like a hiring process: keyword scan → AI screen → actual interview."

### Q: "How do you evaluate RAG quality?"

**Answer**: "We maintain a golden dataset of 23 test cases covering normal questions,
refusal scenarios, keyword queries, and cross-document questions. Every code change
runs against this dataset in CI. We measure behavior accuracy, source accuracy,
content accuracy, and citation validity. CI blocks merges if the overall score
drops below 85%."

### Q: "How do you manage prompts in production?"

**Answer**: "Prompts are version-controlled YAML files, not hardcoded strings.
Each has a version number and a content hash. Every response logs which prompt
version generated it. If quality drops, we can trace it to the exact prompt change.
It's the same principle as Git for code, but applied to prompts."

### Q: "What happens when the system doesn't know the answer?"

**Answer**: "It refuses. We test this explicitly — our golden dataset includes
questions about biryani recipes, weather, and Python code, none of which are in
our documents. The system correctly responds with 'I don't have enough information.'
We validate refusals are correct, not just that they happen."

### Q: "Why ChromaDB? When would you switch?"

**Answer**: "ChromaDB is open source, runs locally, and handles up to ~1 million
vectors. Perfect for development and small-to-medium deployments. I'd switch to
Pinecone or Weaviate when we need: multi-tenant SaaS (many customers sharing
one system), more than 1 million vectors, or managed infrastructure with SLAs."

---

## 16. Tools and Technologies

| Tool | What it does | Why we chose it |
|------|-------------|-----------------|
| **Python 3.11** | Programming language | Modern features (StrEnum, type hints) |
| **ChromaDB** | Vector database | Open source, local, simple |
| **sentence-transformers** | Local embeddings | Free, private, no API needed |
| **all-MiniLM-L6-v2** | Embedding model | Small (80MB), fast, good quality |
| **ms-marco-MiniLM** | Re-ranker model | Trained on search relevance data |
| **Anthropic Claude** | Answer generation | High quality, follows instructions well |
| **rank-bm25** | Keyword search | Simple, proven, fast |
| **tiktoken** | Token counting | Same tokenizer as OpenAI models |
| **Pydantic** | Data validation + config | Type-safe, env var support |
| **Click** | CLI framework | Simple, production-standard |
| **Rich** | Terminal formatting | Beautiful tables and panels |
| **pytest** | Testing framework | Industry standard for Python |
| **ruff** | Linter + formatter | Fastest Python linter, replaces flake8+black |
| **mypy** | Type checker | Catches type errors before runtime |
| **GitHub Actions** | CI/CD | Free for public repos, runs tests automatically |

---

## 17. Numbers and Results

### Codebase Stats
- **20+ Python files** across 6 modules
- **~2,900 lines of code**
- **71 unit tests** (all passing)
- **23 golden dataset test cases**
- **3 Git commits** (one per phase)

### Evaluation Results (First Run)
- **Overall score: 86%** (threshold: 85%) — PASSED
- **Behavior accuracy: 100%** — never answered when it should refuse, never refused when it should answer
- **Source accuracy: 100%** — always cited the correct document
- **Content accuracy: 100%** — all expected keywords present in answers
- **Citation validity: 32%** — validator was too strict on bullet points (not a real quality issue)
- **Average latency: 4.6 seconds** per query
- **4 refusal tests: all passed** — biryani, weather, Python code, Anthropic CEO

### What Each Phase Added

| Phase | What was built | Tests added |
|-------|---------------|-------------|
| Phase 1 | Ingestion, chunking, embedding, vector search, cited generation | 31 |
| Phase 2 | BM25, hybrid retrieval, re-ranking, citation validation, prompt versioning | 22 |
| Phase 3 | Golden eval dataset, evaluation runner, CI quality gates | 18 |

---

## Summary: What I Can Now Build and Explain

After completing this project, I can:

1. **Design** a production RAG system from scratch
2. **Explain** why each component exists and the trade-offs involved
3. **Implement** hybrid retrieval (vector + keyword + re-ranking)
4. **Build** citation enforcement to prevent hallucination
5. **Create** automated evaluation with golden datasets
6. **Set up** CI quality gates that prevent regressions
7. **Version-control** prompts like code
8. **Choose** the right tools for each part (and explain when to switch)

This project is live at: https://github.com/abhishek2829/production-rag-system
