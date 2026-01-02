# TraceRank Tuning Comparison

## Problem Identified

TraceRank was boosting memories with **high graph connectivity** (many entities/edges), but this isn't necessarily a quality indicator. A memory with many entities could be less relevant than a focused memory with fewer entities.

## Test Configurations

Using **embeddinggemma** (best performing model) with existing embeddings to isolate the retrieval tuning impact:

### 1. TraceRank Full (Baseline)
**Config:** `config-embeddinggemma.yaml`
- TraceRank enabled: `k = 0.35`
- Graph connectivity boost: `graph_k = 0.15` (default)
- **What it tests:** Current production configuration with all TraceRank features

### 2. TraceRank WITHOUT Graph Connectivity Boost
**Config:** `config-embeddinggemma-no-graph.yaml`
- TraceRank enabled: `k = 0.35`
- Graph connectivity disabled: `graph_connectivity_enabled = false`
- **What it tests:** Temporal + access patterns, but no entity connection bias

### 3. Pure Embedding Quality (No TraceRank)
**Config:** `config-embeddinggemma-no-tracerank.yaml`
- TraceRank disabled: `k = 0.0`
- **What it tests:** Pure semantic similarity without any temporal/graph signals

## Running the Tests

```bash
./test_retrieval_configs.sh
```

This will:
- Re-run all 20 QA tests with each configuration
- Test both `search` and `recall` methods
- Save results to `test/retrieval_comparison_<timestamp>/`
- **No embedding regeneration needed** - uses existing databases

## Expected Insights

1. **Config 1 vs 2:** Impact of graph connectivity boost
   - If Config 2 is better → confirms graph connectivity was hurting quality
   - Shows whether entity-rich memories were wrongly boosted

2. **Config 2 vs 3:** Value of temporal/access signals
   - If Config 2 is better → TraceRank (without graph) adds value
   - Shows if recency + access patterns improve retrieval

3. **Overall:** Optimal TraceRank configuration
   - Best balance between semantic similarity and temporal signals
   - Whether to keep graph connectivity (with different weighting) or remove it

## Analysis

After tests complete, run:
```bash
python3 analyze_embedding_results.py ./test/retrieval_comparison_<timestamp>/
```

This will generate a comparison report showing:
- Accuracy differences across configurations
- Performance impact (if any)
- Category-level breakdown

## Cost

**Zero cost** - all embeddings already exist, we're just re-running queries (20 questions × 2 methods × 3 configs = 120 queries total, ~2 minutes runtime)
