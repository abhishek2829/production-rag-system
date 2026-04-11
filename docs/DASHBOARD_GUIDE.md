# Langfuse Dashboard Guide

How to use the Langfuse dashboard to monitor your RAG system.

Dashboard URL: **http://localhost:3000**

---

## For Engineers: Debugging a Specific Query

### Step 1: Go to Tracing

Click **"Tracing"** in the left sidebar. You'll see a list of all queries.

### Step 2: Click on a Trace

Each row is one query. Click on it to see the full breakdown:

```
rag-query (total: 6.2s)
  ├── retrieval_vector (0.5s)    ← How long vector search took
  ├── retrieval_bm25 (0.01s)     ← How long keyword search took
  ├── rrf_fusion (0.002s)        ← How long combining results took
  ├── reranking (0.4s)           ← How long re-ranking took
  ├── generation (5.2s)          ← How long Claude took (usually the slowest)
  └── citation_validation (0.001s) ← How long citation check took
```

### Step 3: Identify the Bottleneck

Look at which step takes the most time. In most cases:
- **Generation (Claude)** is the slowest → consider shorter prompts or faster model
- **Reranking** is second → consider re-ranking fewer chunks
- **Retrieval** is fast → probably not the bottleneck

---

## For Engineers: Comparing Across Queries

### Scores View

Click **"Scores"** in the left sidebar. You'll see aggregated metrics:

| Score Name | What It Means | Healthy Range |
|-----------|--------------|---------------|
| citation_coverage | What % of retrieved chunks were actually cited | 20-60% is normal |
| estimated_cost | Cost in USD per query | $0.01-0.03 is typical |
| total_latency | Total query time in seconds | Under 10s is good |

### Sessions View

Click **"Sessions"** to see grouped queries. Each session is a batch run
(like an eval run or a populate script run). Click a session to see all
queries in that batch.

---

## For Product Managers: Is the System Healthy?

### Quick Health Check (30 seconds)

1. Open http://localhost:3000
2. Look at the **Home** page
3. Check these 3 things:

```
✅ HEALTHY:
  - Traces chart shows consistent activity (queries are flowing)
  - Average score for citation_coverage > 0.3 (answers cite sources)
  - No spike in total_latency (system isn't slowing down)

❌ SOMETHING IS WRONG:
  - Traces chart drops to zero (system stopped receiving queries)
  - citation_coverage drops suddenly (answers lost quality)
  - total_latency spikes (something is slow)
```

---

## For Team Leads: Quality Trends

### How to Check if Quality is Improving or Declining

1. Go to **Scores** in the sidebar
2. Look at **citation_coverage** over time
3. The chart shows the trend:

```
Score going UP ↗  = quality is improving (more claims are cited)
Score is FLAT →   = quality is stable (good)
Score going DOWN ↘ = quality is degrading (investigate!)
```

### What to Do If Quality Drops

1. Check the **Tracing** view for recent queries
2. Filter by queries with low citation_coverage scores
3. Click on a low-scoring trace
4. Look at the **generation** span → what did Claude say?
5. Look at the **retrieval** span → did it find the right documents?
6. Common causes:
   - Prompt was changed → check prompt version in trace metadata
   - New documents were added that confused retrieval
   - Claude model was updated by Anthropic

---

## For Business Stakeholders: Cost and Usage

### Monthly Cost Estimate

1. Go to **Home** page
2. Look at **Scores** → **estimated_cost**
3. Multiply average cost by expected monthly queries:

```
Average cost per query: $0.01
Expected queries per month: 10,000
Estimated monthly cost: $0.01 × 10,000 = $100/month
```

### Usage Tracking

1. Go to **Home** page
2. Look at **Traces by time** chart
3. Shows how many queries per day/week
4. Rising trend = more adoption
5. Flat or declining = users may have stopped using the system

---

## Key Dashboard Sections

| Section | Where | Who Uses It | What It Shows |
|---------|-------|-------------|---------------|
| Home | Left sidebar → Home | Everyone | Overview: trace count, scores, trends |
| Tracing | Left sidebar → Tracing | Engineers | Individual query details with step breakdown |
| Sessions | Left sidebar → Sessions | Engineers | Grouped queries (eval runs, batch operations) |
| Scores | Left sidebar → Scores | Everyone | Aggregated quality metrics over time |
| Users | Left sidebar → Users | Product managers | Queries per user |
