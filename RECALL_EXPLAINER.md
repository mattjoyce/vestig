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
- `vestig memory search` → raw ranked list
- `vestig memory recall` → formatted context blocks

Both routes call `search_memories()` with:
- `query` (string)
- `limit`
- storage + embedding engine
- optional TraceRank config

---

## 3) Baseline Similarity (M1)
### Steps
1. Embed the query using the configured embedding model.
2. Compare query embedding against every stored memory embedding.
3. Rank by cosine similarity.

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
- **Event history** (`memory_events`): ADD / REINFORCE events.
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

## 5) Combined Recall Score
For each memory:
```
recall_score = cosine_similarity * trace_multiplier
```

In M3/M4, this is the primary scoring model.

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

## 7) Graph Features (M4)
Graph is not used directly in baseline recall yet, but supports:
- entity inspection
- edge-based explainability
- future graph expansion (M5)

---

## 8) Output Formatting
### `memory search`
- Returns a ranked list with similarity scores.

### `memory recall`
- Returns text blocks separated by `---`.
- Each memory includes content and created timestamp.

---

## 9) Key Files
- `src/vestig/core/retrieval.py` — similarity search + formatting
- `src/vestig/core/tracerank.py` — TraceRank implementation
- `src/vestig/core/storage.py` — memory/event persistence

