# Project 3 Learnings: RAG Monitoring & Observability

Everything I learned while building this project, explained in simple language.

---

## Table of Contents

1. [What Is Observability and Why Does It Matter?](#1-what-is-observability)
2. [What Is Langfuse?](#2-what-is-langfuse)
3. [What Are Traces and Spans?](#3-traces-and-spans)
4. [What Is Docker and Why Did We Need It?](#4-what-is-docker)
5. [The Wrapper Pattern (How We Added Tracing Without Breaking Anything)](#5-the-wrapper-pattern)
6. [What Is Graceful Degradation?](#6-graceful-degradation)
7. [What Is Async Sending?](#7-async-sending)
8. [Sessions and User IDs](#8-sessions-and-user-ids)
9. [Cost Tracking (How Much Does Each Query Cost?)](#9-cost-tracking)
10. [The Dashboard (What Each View Shows)](#10-the-dashboard)
11. [Traced Eval (CI Regression Gates + Observability)](#11-traced-eval)
12. [Enterprise Lifecycle (What I Did Differently This Time)](#12-enterprise-lifecycle)
13. [Library Version Mismatch (A Real-World Debugging Story)](#13-library-version-mismatch)
14. [Interview Questions and Answers](#14-interview-questions)
15. [Numbers and Results](#15-numbers-and-results)

---

## 1. What Is Observability?

### The Problem

Imagine you own a restaurant. Customers order food, eat it, and leave. Some days
business is good, some days it's bad. But you have NO IDEA:
- How long does each dish take to cook?
- Which dish is most expensive to make?
- Are customers satisfied with the quality?
- Is the kitchen getting slower over time?

You're flying blind. You only know something is wrong when customers complain
or stop coming.

### The Solution: Observability

Observability = putting CAMERAS and SENSORS in your kitchen so you can see
exactly what's happening at every step.

```
WITHOUT OBSERVABILITY:
  Customer orders → Kitchen cooks → Food arrives → ???
  "The food was bad." → Why? → No idea.

WITH OBSERVABILITY:
  Customer orders → Kitchen cooks → EVERY STEP RECORDED → Food arrives
  "The food was bad." → Check the recording → 
  "Ah, the chef used expired ingredients at step 3."
```

### Applied to Our RAG System

```
WITHOUT OBSERVABILITY (Projects 1-2):
  User asks question → RAG answers → "The answer was wrong." → Why? → 🤷

WITH OBSERVABILITY (Project 3):
  User asks question → EVERY STEP RECORDED:
    Retrieval: found 5 chunks in 0.5s, top score 0.85
    Re-ranking: reordered chunks in 0.4s, best score 9.2
    Generation: Claude took 4.1s, used 2500 tokens, cost $0.01
    Validation: 2 citations, coverage 40%
  → "The answer was wrong." → Check Langfuse → 
  "Retrieval found wrong chunks. The question used keywords 
   not in our documents."
```

---

## 2. What Is Langfuse?

Langfuse is an **open-source observability platform for AI systems**. Think of
it as "Datadog for AI" — but specifically designed for LLM applications.

```
WHAT LANGFUSE GIVES YOU:
  1. A DASHBOARD (web page at localhost:3000)
     → Shows charts, tables, trends
  
  2. TRACE STORAGE (saves every query's details)
     → Click any query to see its full breakdown
  
  3. SCORE TRACKING (quality metrics over time)
     → See if quality is improving or degrading
  
  4. SESSION GROUPING (batch related queries)
     → See all eval questions from one run together
  
  5. USER FILTERING (who asked what)
     → Filter by "show me only the product manager's queries"
```

### Why Self-Hosted (Docker) Instead of Cloud?

```
CLOUD LANGFUSE:
  - Quick to set up (just sign up)
  - Your trace data goes to Langfuse's servers
  - Free tier: 50,000 traces/month

SELF-HOSTED (what we chose):
  - Runs on YOUR machine via Docker
  - Your data NEVER leaves your laptop
  - Unlimited traces (no monthly limit)
  - Production-realistic (companies self-host in production)
  - More work to set up (Docker + 6 containers)
```

We chose self-hosted because it's production-realistic and demonstrates
that you can run infrastructure, not just write code.

---

## 3. Traces and Spans

### What Is a Trace?

A trace = the COMPLETE journey of ONE query, from start to finish.

```
TRACE: "What is HNSW?" (total: 5.0 seconds)
```

It's like a receipt from a restaurant — one receipt per order.

### What Is a Span?

A span = ONE STEP within that journey.

```
TRACE: "What is HNSW?" (total: 5.0 seconds)
  │
  ├── SPAN: retrieval_vector   (0.5 seconds)
  ├── SPAN: retrieval_bm25     (0.01 seconds)
  ├── SPAN: rrf_fusion         (0.002 seconds)
  ├── SPAN: reranking          (0.4 seconds)
  ├── SPAN: generation         (4.1 seconds)  ← slowest step!
  └── SPAN: citation_validation (0.001 seconds)
```

It's like the line items on a receipt — each item with its price (time).

### Why Spans Matter

Without spans: "The query took 5 seconds." → Which step is slow? → 🤷

With spans: "The query took 5 seconds. Generation took 4.1s (82% of total).
If we want faster queries, we should optimize the Claude call."

---

## 4. What Is Docker?

### The Problem

Langfuse needs 6 programs running simultaneously:
1. Langfuse web server (the dashboard)
2. Langfuse worker (processes traces in background)
3. PostgreSQL (database for settings and users)
4. ClickHouse (database for trace analytics)
5. Redis (message queue between web and worker)
6. MinIO (file storage for large traces)

Installing and configuring all 6 separately would be a nightmare.

### The Solution: Docker

Docker lets you run programs in isolated "containers." Think of containers
as lightweight virtual computers, each running one program.

```
WITHOUT DOCKER:
  Install PostgreSQL on your Mac... configure it...
  Install ClickHouse... configure it differently...
  Install Redis... configure it...
  Install MinIO... configure it...
  Install Langfuse... point it at all 4 databases...
  Something breaks? Good luck figuring out which one.

WITH DOCKER:
  docker compose up -d
  → All 6 programs start automatically, pre-configured, working together.
  → Stop everything: docker compose down
  → That's it.
```

### Key Docker Concepts

```
IMAGE:       A blueprint for a program (like a recipe)
             "postgres:17" = the PostgreSQL v17 blueprint

CONTAINER:   A running instance of an image (like a dish made from the recipe)
             You can have multiple containers from the same image

VOLUME:      Persistent storage (data survives container restarts)
             Without volumes, data disappears when containers stop

DOCKER COMPOSE: A file (docker-compose.yml) that defines multiple containers
                and how they connect to each other
```

### What docker-compose.yml Does

```yaml
services:
  langfuse-web:        # Container 1: Dashboard on port 3000
    image: langfuse/langfuse:3
    ports: 3000:3000
    depends_on: postgres, clickhouse, redis, minio

  postgres:            # Container 2: Settings database
    image: postgres:17
    volumes: langfuse_postgres_data  # Data survives restarts

  clickhouse:          # Container 3: Analytics database
    image: clickhouse/clickhouse-server

  redis:               # Container 4: Message queue
    image: redis:7

  minio:               # Container 5: File storage
    image: minio/minio
```

Each "service" becomes a container. They can talk to each other by name
(langfuse-web can reach postgres at "postgres:5432").

---

## 5. The Wrapper Pattern

### The Problem

We needed to add tracing to the RAG pipeline. But the existing code
(71 tests, working perfectly) should NOT be changed.

### The Solution: Inheritance (Wrapper Pattern)

```python
# EXISTING (untouched):
class RAGPipeline:
    def query(question):
        # does retrieval, re-ranking, generation, validation
        return response, report

# NEW (added on top):
class TracedRAGPipeline(RAGPipeline):   # ← inherits from original
    def query(question):
        # does the SAME work
        # but wraps each step with a stopwatch
        # and sends timing data to Langfuse
        return response, report  # ← same result
```

### Why This Is Better Than Modifying Existing Code

```
OPTION: Modify RAGPipeline directly
  Risk: Break existing code, break 71 tests
  Undo: Hard to remove tracing later
  Testing: Must retest everything

OPTION: Inherit and override (what we did)
  Risk: Zero — original code untouched
  Undo: Just use RAGPipeline instead of TracedRAGPipeline
  Testing: Original 71 tests still pass automatically
```

### The Interview Name for This

This is called the **Open/Closed Principle** — one of the SOLID principles:
- **Open** for extension (we CAN add new behavior)
- **Closed** for modification (we DON'T change existing code)

---

## 6. What Is Graceful Degradation?

If Langfuse crashes (Docker stops, network issue), what happens to the
RAG system?

```
BAD DESIGN (no graceful degradation):
  Langfuse is down → RAG system crashes → User gets an error
  "Sorry, the system is unavailable."
  
  The MONITORING tool broke the PRODUCT. That's terrible.

GOOD DESIGN (graceful degradation — what we built):
  Langfuse is down → RAG system detects it → Skips tracing → Works normally
  User gets their answer as usual.
  The only thing missing: the trace doesn't appear in the dashboard.
  
  The PRODUCT keeps working even when MONITORING fails.
```

### How We Implemented It

```python
def create_trace(query):
    try:
        trace = langfuse.start_observation(...)
        return trace
    except Exception:
        # Langfuse is down? Just log a warning and continue.
        logger.warning("Langfuse unreachable, skipping trace")
        return None  # ← returns None, doesn't crash
```

Every method in LangfuseTracer follows this pattern: try, and if it fails,
log a warning and continue. The RAG system never sees the error.

---

## 7. What Is Async Sending?

When our code records a trace, it doesn't wait for Langfuse to confirm
receipt before continuing. It sends the data in the BACKGROUND.

```
SYNCHRONOUS (slow — not what we do):
  User asks question
  → Retrieval (0.5s)
  → Send retrieval trace to Langfuse... wait for confirmation (0.2s) ← ADDED DELAY
  → Re-ranking (0.4s)
  → Send re-ranking trace to Langfuse... wait (0.2s) ← ADDED DELAY
  → Generation (4.1s)
  → Send generation trace to Langfuse... wait (0.2s) ← ADDED DELAY
  Total: 5.0s + 0.6s = 5.6s ← 12% slower!

ASYNCHRONOUS (fast — what we do):
  User asks question
  → Retrieval (0.5s) + queue trace in background
  → Re-ranking (0.4s) + queue trace in background
  → Generation (4.1s) + queue trace in background
  → Return answer to user (total: 5.0s ← NO added delay)
  
  Meanwhile, in the background, traces are sent to Langfuse.
  User doesn't wait for this.
```

The Langfuse SDK handles this automatically — it batches traces and
sends them every few seconds in a background thread.

---

## 8. Sessions and User IDs

### Sessions = Grouping Related Queries

```
WITHOUT SESSIONS:
  Langfuse shows 43 traces, all mixed together:
    "What is RAG?"
    "Best biryani recipe?"
    "What is HNSW?"
    ... hard to find related queries ...

WITH SESSIONS:
  Session "eval_20260411_161423":
    → 23 eval queries grouped together
  
  Session "dashboard_populate_20260411_153828":
    → 20 populate queries grouped together
  
  Click a session → see all queries from that run.
  Like email threads vs flat inbox.
```

### User IDs = Who Asked What

```
user_id="abhishek"        → your manual queries
user_id="eval_bot"        → automated eval queries
user_id="product_manager" → simulated PM queries
user_id="new_engineer"    → simulated new hire queries

Filter dashboard: "Show me only eval_bot queries"
→ See only automated evaluation results
```

---

## 9. Cost Tracking

### How We Calculate Cost

Anthropic charges per token (the units AI reads/writes):

```
Claude Sonnet pricing:
  Input tokens:  $3.00 per 1 million tokens
  Output tokens: $15.00 per 1 million tokens

Example query:
  Input:  2500 tokens (question + 5 chunks of context)
  Output: 350 tokens (Claude's answer)
  
  Input cost:  2500 × ($3.00 / 1,000,000) = $0.0075
  Output cost: 350 × ($15.00 / 1,000,000) = $0.00525
  Total: $0.013 (about 1.3 cents per query)
```

### Why Cost Tracking Matters

```
Without tracking:
  "How much does our AI system cost?"
  "No idea. We'll find out when the bill arrives."

With tracking:
  "How much does our AI system cost?"
  "Average $0.01 per query. At 1000 queries/day = $10/day = $300/month.
   If we switch to a cheaper model, we could cut that to $100/month."
```

---

## 10. The Dashboard

### What Each Section Shows

```
HOME PAGE:
  Traces: 25 total traces tracked (how many queries ran)
  Scores: citation_coverage (avg 0.4), estimated_cost (avg $0.01), 
          total_latency (avg 6.8s)
  Traces by time: chart showing query volume over time

TRACING PAGE:
  List of every individual query with step breakdown
  Click any trace → see retrieval, re-ranking, generation, validation
  Each span shows: input, output, duration, metadata

SESSIONS PAGE:
  Grouped queries (eval runs, batch operations)
  Click a session → see all queries from that batch

SCORES PAGE:
  Aggregated quality metrics over time
  citation_coverage trend, cost trend, latency trend

USERS PAGE:
  Queries per user (abhishek, eval_bot, product_manager, etc.)
```

---

## 11. Traced Eval

### What --traced Does

```
WITHOUT --traced (standard eval):
  rag-eval --threshold 0.85
  → Runs 23 questions
  → Shows results in terminal
  → Results disappear when terminal closes

WITH --traced (new in Project 3):
  rag-eval --threshold 0.85 --traced
  → Runs 23 questions
  → Shows results in terminal
  → ALSO logs every question to Langfuse as a session
  → Results visible in dashboard FOREVER
  → Can compare eval scores from last week vs this week
```

### Why This Matters for CI

In CI (GitHub Actions), the eval runs automatically on every code change.
With --traced, each CI eval run creates a Langfuse session:

```
eval_20260410_120000 → score: 86% ✅
eval_20260411_120000 → score: 86% ✅
eval_20260412_120000 → score: 72% ❌ ← what happened?
→ Click the session → see which questions failed
→ Compare with previous session → find the regression
```

---

## 12. Enterprise Lifecycle

This was the first project where we wrote proper enterprise documents
BEFORE coding. Here's what each document is:

```
SoW (Statement of Work):
  "What are we building, why, and how do we know it's done?"
  Written FIRST. Sets the boundaries.

PRD (Product Requirements Document):
  "What exactly should it do? Who are the users?"
  Written SECOND. Detailed feature list with priorities.
  Includes user stories for BOTH engineers AND non-technical users.

HLD (High Level Design):
  "What are the big components and how do they connect?"
  The floor plan. Shows RAG system → Tracing Wrapper → Langfuse.

LLD (Low Level Design):
  "What are the classes, functions, and data flows?"
  The wiring diagram. Shows exact class names, method signatures.

Then CODE. Then TEST. Then DEPLOY.
```

### Key Learning

Writing the SoW and PRD BEFORE coding saved time because:
- We knew exactly what was in scope (and what wasn't)
- We caught an architecture issue BEFORE writing code (wrapper vs modify)
- The user stories reminded us to build for non-technical users too
- Success criteria gave us a clear "done" definition

---

## 13. Library Version Mismatch

### What Happened

I wrote code using `langfuse.trace()` — a method from Langfuse v2.
But we installed Langfuse v4, where the method was renamed to
`langfuse.start_observation()`.

```
MY CODE:     langfuse.trace(name="rag-query")
ERROR:       'Langfuse' object has no attribute 'trace'
TRANSLATION: "This method doesn't exist anymore."
```

### How I Fixed It

```
Step 1: Check what version is installed → 4.2.0
Step 2: List available methods → found 'start_observation'
Step 3: Check its parameters → name, input, metadata, as_type
Step 4: Rewrite code to use start_observation()
Step 5: Hit another error: end() doesn't accept 'output'
Step 6: Check end() signature → only accepts end_time
Step 7: Found update() method → that's where output goes
Step 8: Fixed: call update(output=...) then end()
```

### The Learning

Libraries change their APIs between major versions. Your code that
worked yesterday might break with a new version. The skill is:
1. Read the error message carefully
2. Check what methods exist in the new version
3. Update your code to match

This happens ALL THE TIME in production. It's not a bug in your code —
it's the library that changed.

---

## 14. Interview Questions and Answers

### Q: "How do you monitor AI systems in production?"

**Answer:** "I built a self-hosted Langfuse observability system that traces
every RAG query end-to-end. Each query creates a trace with 6 spans — retrieval,
BM25, fusion, re-ranking, generation, and citation validation. I track three
key scores: citation coverage (quality), estimated cost, and total latency.
The dashboard shows trends over time, and I can filter by user, session, or
time range to investigate issues."

### Q: "What happens if your monitoring system goes down?"

**Answer:** "Graceful degradation. The RAG system keeps working normally —
users still get answers. Only tracing is skipped. I implemented this with
try/except around every Langfuse call. The monitoring tool should NEVER
break the product it's monitoring."

### Q: "How do you prevent quality regressions?"

**Answer:** "Three layers: First, 87 unit tests run on every push. Second,
a golden dataset of 23 questions runs as a CI quality gate — if the overall
score drops below 85%, the PR is blocked. Third, with the --traced flag,
every eval run is logged to Langfuse as a session, so I can compare eval
scores over time and see exactly when and where quality degraded."

### Q: "How did you add observability without breaking existing code?"

**Answer:** "I used the Open/Closed Principle — inheritance instead of
modification. TracedRAGPipeline inherits from RAGPipeline and overrides
only the query() method. The original 71 tests still pass because the
original code is completely untouched. You can swap between traced and
untraced pipelines with one line change."

### Q: "What is the difference between a trace and a span?"

**Answer:** "A trace is the complete journey of one query from start to
finish — like a receipt. A span is one step within that journey — like a
line item on the receipt. My RAG system creates 6 spans per trace:
retrieval, BM25, fusion, re-ranking, generation, and validation. This
lets me see exactly which step is slow or problematic."

### Q: "How do you track AI costs?"

**Answer:** "I calculate estimated cost per query based on Anthropic's
pricing: $3 per million input tokens, $15 per million output tokens.
Each query logs its cost as a Langfuse score. The dashboard shows average
cost and total cost over time. At $0.01 per query and 1000 queries/day,
that's $300/month — visible in the dashboard, not a surprise on the bill."

---

## 15. Numbers and Results

### What We Built

| Component | What it does |
|-----------|-------------|
| LangfuseTracer | Connects to Langfuse, creates traces/spans/scores |
| TracedRAGPipeline | Wraps RAG pipeline with timing for every step |
| metrics.py | Calculates cost, formats metadata for dashboard |
| populate_dashboard.py | Runs 20 queries to fill dashboard with trend data |
| DASHBOARD_GUIDE.md | How to use the dashboard (for engineers, PMs, leads) |
| --traced flag on rag-eval | Logs eval results to Langfuse sessions |

### Test Results

| Metric | Value |
|--------|-------|
| Total tests | 87 (all passing) |
| New tests added | 16 (observability module) |
| Existing tests broken | 0 (zero regressions) |
| Eval score | 86% (same as before — we didn't change the pipeline) |

### Dashboard Data

| Metric | Value |
|--------|-------|
| Total traces in Langfuse | 25+ (from test + populate + eval) |
| Average citation coverage | 0.4 (40%) |
| Average estimated cost | $0.01 per query |
| Average total latency | 5-7 seconds per query |
| Eval sessions logged | 1 (eval_20260411_161423) |

### Infrastructure

| Component | Technology | Port |
|-----------|-----------|------|
| Langfuse Dashboard | langfuse/langfuse:3 | 3000 |
| Langfuse Worker | langfuse/langfuse-worker:3 | 3030 |
| PostgreSQL | postgres:17 | 5432 |
| ClickHouse | clickhouse-server | 8123 |
| Redis | redis:7 | 6379 |
| MinIO | minio | 9090 |

### Enterprise Documents Created

| Document | Purpose |
|----------|---------|
| SOW.md | What we're building, scope, success criteria, risks |
| PRD.md | User stories (8), functional requirements (19), NFRs (6) |
| ARCHITECTURE_HLD.md | System overview, component diagram |
| ARCHITECTURE_LLD.md | Class design, data flow, error handling strategy |
| DASHBOARD_GUIDE.md | How to use the dashboard for different user types |

---

## Summary: What I Can Now Build and Explain

After completing this project, I can:

1. **Set up** self-hosted observability with Langfuse + Docker
2. **Instrument** an AI pipeline with end-to-end tracing (traces + spans)
3. **Track** latency, cost, and quality metrics per query
4. **Build** dashboards that serve both technical and non-technical users
5. **Add** observability without breaking existing code (Open/Closed Principle)
6. **Handle** infrastructure failures gracefully (degradation, not crashes)
7. **Connect** evaluation to observability (traced eval runs)
8. **Write** enterprise planning documents (SoW, PRD, HLD, LLD)
9. **Explain** the difference between traces, spans, and scores
10. **Debug** library version mismatches (a real-world production skill)

This project is live at: https://github.com/abhishek2829/production-rag-system
