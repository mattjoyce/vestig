# Vestig Recall — Detailed Explainer

This document explains how recall works in Vestig, from query embedding to scoring, including TraceRank and temporal weighting.

## 1) Goal of Recall
Given a user query, return the most relevant **memories** (not events or edges), formatted for agent context insertion.

Recall is designed to be:
- **Relevant** (semantic similarity)
- **Stable** (avoid duplicates)
- **Temporal-aware** (freshness and reinforcement)
- **Explainable** (inspectable scores)

---

## 2) Retrieval Entry Point
The CLI uses:
- `vestig memory recall` → formatted context blocks
- `vestig memory recall --explain` → formatted blocks with per-result explanations

`memory recall` accepts either:
- plain text queries
- conversation JSON (array or JSONL), which is summarized into a focused query when configured

---

## 3) Baseline Similarity (M1)
### Steps
1. Embed the query using the configured embedding model.
2. Use FalkorDB native vector search to retrieve candidate memories.
3. Rank by cosine similarity (with optional TraceRank multiplier).

### Cosine similarity
Given embeddings `a` and `b`:
```
cosine_similarity = dot(a, b) / (||a|| * ||b||)
```

This is the baseline score used across all recall.

---

## 4) Temporal Reinforcement (TraceRank — M3)
TraceRank adds a temporal multiplier to the base similarity.

### Inputs
- **Event history** (`Event` nodes): ADD / REINFORCE events linked via `AFFECTS`.
- **Temporal parameters** (from config):
  - `tau_days` — decay constant
  - `cooldown_hours` — burst discount window
  - `burst_discount` — penalty for rapid repeat events
  - `k` — overall multiplier strength

### Recency decay
Each reinforcement event is weighted by recency:
```
w_recency = exp(-Δt / τ)
```
Where `Δt` is time since event and `τ` is `tau_days`.

### Burst discount
Events too close together are discounted:
```
if Δt < cooldown_hours:
    w_burst = burst_discount
else:
    w_burst = 1.0
```

### TraceRank score
Sum weighted event contributions:
```
trace = Σ (w_recency * w_burst)
```

### Final multiplier
TraceRank becomes a multiplier on similarity:
```
trace_multiplier = 1 + k * log1p(trace)
```

---

## 5) Recall Path (M5: Chunk Expansion)
`memory recall` uses chunk expansion:
1. Search for relevant **SUMMARY** nodes via vector search.
2. For each summary, hop to its chunk and retrieve all memories in that chunk.
3. Re-rank the expanded set by similarity and apply TraceRank (if enabled).

This keeps recall grounded in chunk-level provenance while preserving semantic ordering.

---

## 6) Temporal Fields (Bi-temporal Context)
Each memory has temporal fields used for contextual understanding:
- `t_valid`: when the fact became true
- `t_invalid`: when it stopped being true (if known)
- `t_created`: when we learned it
- `t_expired`: when it was deprecated
- `temporal_stability`: static/dynamic/unknown

These fields are not yet fully integrated into scoring, but are used for:
- filtering expired memories
- future temporal ranking (M5+)

---

## 7) Graph Features (M4/M5)
Graph edges are used directly during recall:
- chunk expansion via `SUMMARY` → `Chunk` → `Memory`
- provenance links via `PRODUCED` and `CONTAINS`

Entity-based retrieval exists in `search_memories()` but is not wired into
`memory recall` yet.

---

## 8) Output Formatting
### `memory recall`
- Returns text blocks separated by `---`.
- Each memory includes content and created timestamp.
- Add `--explain` to include an explanation section per result (scores + TraceRank details).

---

## 9) Key Files
- `src/vestig/core/retrieval.py` — similarity search + formatting
- `src/vestig/core/tracerank.py` — TraceRank implementation
- `src/vestig/core/db_falkordb.py` — FalkorDB persistence
