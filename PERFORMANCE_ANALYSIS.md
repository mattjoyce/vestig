# Performance Analysis - The 19-Second Mystery Solved

Date: 2025-12-31

## Summary

**Problem:** Queries take 15-20 seconds
**Root Cause:** Embedding model loads fresh on every query (19 seconds)
**Actual query time:** <500ms

---

## The Smoking Gun

### Measured Times:
```
Total subprocess time:  19,500ms (100%)
  ├─ Model loading:     19,041ms  (98%)  ← THE BOTTLENECK
  └─ Actual query:         459ms   (2%)
```

### Query Breakdown (459ms):
```
1. Embedding generation    93ms  (21%)
2. Load memories         293ms  (65%) ← Biggest query component
3. Semantic scoring       42ms   (9%)
4. TraceRank total        21ms   (5%)
   ├─ Events               6ms
   ├─ Edges               12ms
   └─ Compute              1ms
5. Sort & slice            0ms   (0%)
```

---

## Root Cause

**File:** `src/vestig/core/embeddings.py:28`

```python
class EmbeddingEngine:
    def __init__(self, model_name: str, ...):
        # This loads the model EVERY TIME
        self.model = SentenceTransformer(model_name, ...)
```

**Every vestig command:**
1. Starts Python process
2. Loads SentenceTransformer model (BAAI/bge-m3) - **19 seconds**
3. Runs actual query - **459ms**
4. Exits

---

## Solutions

### Option 1: Model Caching (Quick Fix)
Cache the loaded model in memory/disk so subsequent loads are faster.

**Pros:** Easy to implement
**Cons:** Still slow on first load

### Option 2: Persistent Server (Best)
Run vestig as a server that keeps the model in memory.

```
vestig serve --config config.yaml  # Start server once
vestig query "..."                  # Fast queries (no reload)
```

**Pros:**
- First query: 19.5s (model load)
- Subsequent queries: <500ms ✓
- Best UX for interactive use

**Cons:** More complex architecture

### Option 3: Pre-load in Shell Session
Keep a Python REPL open with model loaded.

**Pros:** Simple for dev/testing
**Cons:** Not production-ready

---

## Recommendations

### Immediate (for testing):
Use persistent Python session to avoid reloading:
```python
# Load once
from vestig.core.config import load_config
from vestig.core.cli import build_runtime
config = load_config('config.yaml')
storage, embedding, events, tracerank = build_runtime(config)

# Query many times (fast)
from vestig.core.retrieval import search_memories
results = search_memories(query="...", storage=storage, ...)
```

### Short-term:
Implement `vestig serve` command with HTTP API or gRPC

### Long-term:
Consider embedding API services:
- OpenAI embeddings (API call, no local model)
- Dedicated embedding service (e.g., txtai, Jina)

---

## Updated Performance Priorities

~~1. Fix slow queries~~ ✓ **SOLVED - queries are fast!**

**New priorities:**
1. **Implement model caching or server mode** (eliminate 19s reload)
2. Optimize "load memories" (293ms, 65% of query time)
3. Re-evaluate TraceRank strategy (separate issue)

---

## TraceRank Impact on Performance

TraceRank overhead: **21ms (5% of query time)**

**Breakdown:**
- Event fetches: 6ms
- Edge fetches: 12ms
- Computation: 1ms

**Conclusion:** TraceRank is NOT a performance bottleneck.
The problem is it **hurts accuracy**, not speed.

---

## Next Steps

1. ✅ **Add --timing flag** (done)
2. ✅ **Identify bottleneck** (done - model loading)
3. ⏭️ **Implement caching or server mode**
4. ⏭️ **Optimize SQLite memory loading** (293ms)
5. ⏭️ **Return to TraceRank redesign** (clustering, summary nodes)
