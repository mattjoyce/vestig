# M5 Implementation Plan: Advanced Retrieval (Excluding MemRank)

**Date**: 2025-12-29
**Milestone**: M5 - Advanced Retrieval
**Goal**: Make recall feel intentionally smart, not accidentally lucky

---

## Executive Summary

M5 transforms Vestig's retrieval from simple brute-force cosine similarity into an intelligent, multi-factor ranking system. This milestone implements three core capabilities:

1. **Hypothetical Query Generation (HyDE)** - Bridge the query-memory gap
2. **Hybrid Start Node Selection** - Multi-signal entry point selection
3. **Probabilistic Graph Traversal** - Human-like memory activation spread

**Important**: MemRank (PageRank-based graph centrality) is **DEFERRED** per user directive. The system will use TraceRank (already implemented in M3) as the graph-based scoring component.

---

## Current State (M1-M4 Complete)

### What We Have
- ✅ M1: Basic memory storage, retrieval, embeddings
- ✅ M2: Quality firewall (hygiene, deduplication, substance filter)
- ✅ M3: Bi-temporal tracking, TraceRank (reinforcement + graph connectivity + temporal confidence)
- ✅ M4: Entity extraction, graph layer (entities, MENTIONS/RELATED edges)

### Current Retrieval Pipeline (`retrieval.py`)
```python
# Current flow (M1-M3):
1. Embed query
2. Compute cosine similarity against all memories (brute-force)
3. Apply TraceRank multiplier (if enabled):
   - Reinforcement boost (from events)
   - Graph connectivity boost (from inbound edges)
   - Temporal confidence (decay for dynamic facts)
4. Sort by final_score = semantic_score × tracerank_multiplier
5. Return top-K results
```

### Gaps for M5
- ❌ No hypothetical query generation
- ❌ No hybrid start node selection
- ❌ No probabilistic graph traversal
- ❌ No explainability (reason traces)
- ❌ Simple top-K ranking (no multi-stage retrieval)

---

## M5 Architecture Overview

### New Retrieval Pipeline (M5)

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Query Processing                              │
│ - Embed user query                                      │
│ - Extract query metadata (keywords, temporal hints)     │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│ Stage 2: Candidate Expansion (NEW)                     │
│ - Match against hypothetical queries (HyDE)             │
│ - Match against memory content                          │
│ - Match against memory metadata (keywords, entities)    │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│ Stage 3: Hybrid Start Node Selection (NEW)             │
│ - Score candidates via:                                 │
│   • Semantic similarity (query ↔ content/hypotheticals) │
│   • Metadata match (keywords, temporal language)        │
│   • Recency signal                                      │
│ - Combined: α_meta × S_meta + (1-α_meta) × S_query      │
│ - Select top-K start nodes (K=3)                        │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│ Stage 4: Probabilistic Graph Traversal (NEW)           │
│ - Start from selected nodes (signal_strength=1.0)       │
│ - For each node:                                        │
│   • Get neighbors via RELATED edges                     │
│   • Compute transition probability:                     │
│     p = edge_weight × temporal_decay × (1+boost) ×      │
│         signal_strength × exploration_factor            │
│   • Random selection based on p                         │
│   • Decay signal: new_signal = signal × edge_weight    │
│   • Recurse until max_nodes reached                     │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│ Stage 5: Final Ranking & Explainability                │
│ - Score all traversed memories:                         │
│   final_score = semantic × tracerank × path_score       │
│ - Generate reason trace for each memory:               │
│   • "Direct semantic match (0.87)"                      │
│   • "Via entity: PROJECT:vestig (2 hops)"              │
│   • "Reinforced 3x, graph connectivity +0.15"          │
│ - Return ranked results with reasons                    │
└─────────────────────────────────────────────────────────┘
```

---

## Feature 1: Hypothetical Query Generation (HyDE)

### Purpose
Bridge the semantic gap between user queries and memory content by generating questions that each memory would answer.

### Strategy
**Selective, gated generation** to avoid noise:
- Only generate for `problem-solution` and `learning-failure` trigger types
- Max 2 queries per memory
- Skip for routine feedback/observations
- Generated during commitment pipeline, stored in memory metadata

### Implementation

#### 1.1 Data Model Changes
**File**: `src/vestig/core/models.py`

```python
class MemoryNode:
    # ... existing fields ...

    # NEW: Hypothetical queries (M5)
    hypothetical_queries: list[str] | None = None  # Max 2 questions
    hypothetical_embeddings: list[list[float]] | None = None  # Embeddings for queries
```

#### 1.2 Generation Logic
**File**: `src/vestig/core/commitment.py`

Add new function after entity extraction:
```python
def _generate_hypothetical_queries(
    content: str,
    trigger: str | None,
    config: dict
) -> tuple[list[str], list[list[float]]]:
    """
    Generate 1-2 hypothetical queries this memory would answer.

    Gated by:
    - config.m5.hyde.enabled
    - Trigger type in allowed_triggers (problem-solution, learning-failure)

    Returns:
        (queries, query_embeddings)
    """
```

Call from `commit_memory()` after entity extraction.

#### 1.3 Prompt Template
**File**: `src/vestig/core/prompts.yaml`

```yaml
generate_hypothetical_queries: |
  Given this memory, generate 1-2 specific questions that this memory would answer.
  Make them natural and actionable, as a user would ask.

  Memory content: {{content}}
  Trigger: {{trigger}}

  Guidelines:
  - Focus on the core problem/solution or learning
  - Use natural language (not overly formal)
  - Be specific (not generic "how do I...")

  Output format (JSON):
  {
    "queries": ["Question 1?", "Question 2?"]
  }
```

#### 1.4 Storage Schema
**File**: `src/vestig/core/storage.py`

Add columns to `memories` table (migration):
```sql
ALTER TABLE memories ADD COLUMN hypothetical_queries TEXT;  -- JSON array
ALTER TABLE memories ADD COLUMN hypothetical_embeddings BLOB;  -- Serialized embeddings
```

#### 1.5 Configuration
**File**: `config.yaml`

```yaml
m5:
  hyde:
    enabled: true
    allowed_triggers:
      - problem-solution
      - learning-failure
    max_queries: 2
    model: claude-sonnet-4.5
```

---

## Feature 2: Hybrid Start Node Selection

### Purpose
Select the best entry points for graph traversal by combining multiple relevance signals.

### Scoring Components

#### 2.1 Semantic Score (S_query)
Match query against:
- Memory content embedding
- Hypothetical query embeddings (if available)

```python
def compute_semantic_score(query_emb, memory):
    # Content similarity
    content_sim = cosine(query_emb, memory.content_embedding)

    # Hypothetical query similarity (if available)
    if memory.hypothetical_embeddings:
        hyp_sims = [cosine(query_emb, hyp_emb)
                    for hyp_emb in memory.hypothetical_embeddings]
        max_hyp_sim = max(hyp_sims)

        # Combine: favor hypothetical matches (they're query-shaped)
        return 0.4 * content_sim + 0.6 * max_hyp_sim

    return content_sim
```

#### 2.2 Metadata Score (S_meta)
Match query against metadata features:
- **Keywords**: Jaccard similarity between query keywords and memory keywords
- **Temporal language**: Naturalized time delta ("last week", "yesterday")
- **Entity overlap**: Shared entities between query and memory

```python
def compute_metadata_score(query_text, memory):
    # Keyword overlap
    query_keywords = extract_keywords(query_text)
    memory_keywords = memory.metadata.get("keywords", [])
    keyword_sim = jaccard(query_keywords, memory_keywords)

    # Temporal language (if query contains time references)
    temporal_sim = compute_temporal_match(query_text, memory)

    # Entity overlap (if query mentions entities)
    entity_sim = compute_entity_match(query_text, memory)

    # Weighted combination
    return 0.5 * keyword_sim + 0.3 * temporal_sim + 0.2 * entity_sim
```

#### 2.3 Combined Score
```python
def compute_hybrid_score(query_emb, query_text, memory, config):
    s_query = compute_semantic_score(query_emb, memory)
    s_meta = compute_metadata_score(query_text, memory)

    alpha_meta = config.m5.start_node.alpha_meta  # Default: 0.6

    return alpha_meta * s_meta + (1 - alpha_meta) * s_query
```

#### 2.4 Start Node Selection
**File**: `src/vestig/core/retrieval.py`

```python
def select_start_nodes(
    query: str,
    query_embedding: list[float],
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    config: dict,
    top_k: int = 3
) -> list[MemoryNode]:
    """
    Select top-K start nodes using hybrid scoring.

    Returns:
        List of MemoryNode objects to start graph traversal from
    """
    all_memories = storage.get_active_memories()

    scored = []
    for memory in all_memories:
        score = compute_hybrid_score(query_embedding, query, memory, config)
        scored.append((memory, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return top-K nodes
    return [memory for memory, score in scored[:top_k]]
```

---

## Feature 3: Probabilistic Graph Traversal

### Purpose
Model human-like memory activation spread through the knowledge graph.

### Algorithm
Depth-first search with probabilistic edge selection and signal decay.

### Implementation

#### 3.1 Core Traversal Logic
**File**: `src/vestig/core/retrieval.py`

```python
def probabilistic_recall(
    start_nodes: list[MemoryNode],
    storage: MemoryStorage,
    event_storage: MemoryEventStorage,
    tracerank_config: TraceRankConfig,
    config: dict,
    max_nodes: int = 5
) -> list[tuple[MemoryNode, float, dict]]:
    """
    Probabilistic graph traversal from start nodes.

    Args:
        start_nodes: Entry point memories from hybrid selection
        storage: Memory storage
        event_storage: Event storage for TraceRank
        tracerank_config: TraceRank configuration
        config: M5 configuration
        max_nodes: Maximum memories to recall

    Returns:
        List of (MemoryNode, final_score, reason_trace) tuples
    """
    recalled = []
    visited = set()

    def traverse(current: MemoryNode, signal_strength: float, path: list[str]):
        if len(recalled) >= max_nodes:
            return

        if current.id in visited:
            return

        visited.add(current.id)

        # Compute final score for this node
        final_score = compute_final_score(
            current, signal_strength, event_storage,
            storage, tracerank_config, config
        )

        # Generate reason trace
        reason = generate_reason_trace(current, path, signal_strength, final_score)

        recalled.append((current, final_score, reason))

        # Get neighbors via RELATED edges
        edges = storage.get_edges_from_memory(current.id, edge_type="RELATED")

        if not edges:
            return

        # Compute transition probabilities
        neighbors = []
        for edge in edges:
            neighbor = storage.get_memory(edge.to_node)
            if not neighbor or neighbor.id in visited:
                continue

            # Probability = edge × decay × boost × signal × exploration
            p_select = compute_transition_probability(
                edge, neighbor, signal_strength,
                event_storage, tracerank_config, config
            )

            neighbors.append((neighbor, edge, p_select))

        if not neighbors:
            return

        # Select neighbors probabilistically
        selected = probabilistic_selection(neighbors, config)

        for neighbor, edge, p in selected:
            # Signal decays as it propagates
            new_signal = signal_strength * edge.weight
            new_path = path + [f"{edge.edge_type}:{neighbor.id}"]

            traverse(neighbor, new_signal, new_path)

    # Start traversal from each start node
    for start_node in start_nodes:
        traverse(start_node, signal_strength=1.0, path=["START"])

    # Sort by final score
    recalled.sort(key=lambda x: x[1], reverse=True)

    return recalled[:max_nodes]
```

#### 3.2 Transition Probability
```python
def compute_transition_probability(
    edge: EdgeNode,
    neighbor: MemoryNode,
    signal_strength: float,
    event_storage: MemoryEventStorage,
    tracerank_config: TraceRankConfig,
    config: dict
) -> float:
    """
    Compute probability of traversing to neighbor.

    p = edge_weight × temporal_decay × (1 + boost) × signal × μ

    Where:
    - edge_weight: RELATED edge weight (0-1)
    - temporal_decay: Age-based decay from TraceRank
    - boost: TraceRank reinforcement boost
    - signal: Current signal strength (decays with hops)
    - μ: Exploration factor (config, default 2.0)
    """
    # Get TraceRank multiplier
    events = event_storage.get_reinforcement_events(neighbor.id)
    inbound_edges = storage.get_edges_to_memory(neighbor.id)

    tracerank_mult = compute_enhanced_multiplier(
        memory_id=neighbor.id,
        temporal_stability=neighbor.temporal_stability,
        t_valid=neighbor.t_valid or neighbor.created_at,
        inbound_edge_count=len(inbound_edges),
        reinforcement_events=events,
        config=tracerank_config
    )

    # Transition probability
    p = (
        edge.weight *
        tracerank_mult *
        signal_strength *
        config.m5.traversal.exploration_factor  # μ, default 2.0
    )

    return min(p, 1.0)  # Cap at 1.0
```

#### 3.3 Probabilistic Selection
```python
def probabilistic_selection(
    neighbors: list[tuple[MemoryNode, EdgeNode, float]],
    config: dict
) -> list[tuple[MemoryNode, EdgeNode, float]]:
    """
    Select neighbors to traverse based on probabilities.

    Strategy:
    - Random selection weighted by probability
    - May select 0, 1, or multiple neighbors (stochastic)
    """
    import random

    selected = []
    for neighbor, edge, p in neighbors:
        if random.random() < p:
            selected.append((neighbor, edge, p))

    return selected
```

---

## Feature 4: Explainability (Reason Traces)

### Purpose
Make retrieval inspectable - explain why each memory was surfaced.

### Implementation

#### 4.1 Reason Trace Generation
```python
def generate_reason_trace(
    memory: MemoryNode,
    path: list[str],
    signal_strength: float,
    final_score: float
) -> dict:
    """
    Generate human-readable explanation for why memory was retrieved.

    Returns:
        {
            "retrieval_method": "direct_match" | "graph_traversal",
            "semantic_score": 0.87,
            "path": ["START", "RELATED:mem_abc", "RELATED:mem_xyz"],
            "hops": 2,
            "signal_strength": 0.64,
            "tracerank_boost": 1.42,
            "final_score": 1.23,
            "explanation": "Direct semantic match (0.87). Reinforced 3x..."
        }
    """
    hops = len(path) - 1  # Exclude "START"

    if hops == 0:
        method = "direct_match"
        explanation = f"Direct semantic match (score={final_score:.2f})"
    else:
        method = "graph_traversal"
        explanation = f"Reached via {hops}-hop traversal (signal={signal_strength:.2f})"

    # Add TraceRank details if boosted
    # Add graph connectivity details if relevant

    return {
        "retrieval_method": method,
        "path": path,
        "hops": hops,
        "signal_strength": signal_strength,
        "final_score": final_score,
        "explanation": explanation
    }
```

#### 4.2 Updated Recall Output Format
Extend `format_recall_results()` to include reason traces:

```
[mem_abc123] (score=0.87, method=direct_match)
Reason: Direct semantic match. Reinforced 3x, graph connectivity +0.15
Content: ...

---

[mem_xyz789] (score=0.64, method=graph_traversal, hops=2)
Reason: Reached via 2-hop traversal through PROJECT:vestig. Signal strength 0.64
Content: ...
```

---

## Implementation Roadmap

### Phase 1: Hypothetical Query Generation (HyDE)
**Estimated effort**: 2-3 hours

1. Update `models.py` - Add hypothetical fields to MemoryNode
2. Update `storage.py` - Schema migration for new columns
3. Add prompt to `prompts.yaml` - HyDE generation template
4. Implement generation in `commitment.py` - Selective, gated logic
5. Update commitment pipeline - Call HyDE after entity extraction
6. Add config section - `m5.hyde` in config.yaml
7. Test with sample memories - Verify queries are generated and stored

**Acceptance**:
- Problem-solution memories have 1-2 hypothetical queries
- Other trigger types have no queries (gated correctly)
- Queries are semantically relevant to memory content

---

### Phase 2: Hybrid Start Node Selection
**Estimated effort**: 3-4 hours

1. Implement semantic scoring - Content + hypothetical query matching
2. Implement metadata scoring - Keywords + temporal + entities
3. Implement combined scoring - Weighted combination
4. Add `select_start_nodes()` function - Top-K selection
5. Update config - `m5.start_node` section
6. Test scoring components - Verify hybrid scores make sense
7. Integration test - Select start nodes from test corpus

**Acceptance**:
- Hybrid scoring combines semantic + metadata signals
- Top-K start nodes are semantically and temporally relevant
- Config can tune alpha_meta weight

---

### Phase 3: Probabilistic Graph Traversal
**Estimated effort**: 4-5 hours

1. Implement `compute_transition_probability()` - Edge selection logic
2. Implement `probabilistic_selection()` - Random walk helper
3. Implement `probabilistic_recall()` - Main traversal algorithm
4. Add signal strength tracking - Decay with hops
5. Add visited set - Prevent cycles
6. Update config - `m5.traversal` section
7. Test traversal on graph - Verify stochastic behavior
8. Integration test - End-to-end with start nodes

**Acceptance**:
- Traversal explores graph probabilistically (not deterministic)
- Signal decays with hops (distant memories lower score)
- Max nodes limit is respected
- No infinite loops (visited set works)

---

### Phase 4: Explainability & Integration
**Estimated effort**: 2-3 hours

1. Implement `generate_reason_trace()` - Explanation generation
2. Update `format_recall_results()` - Include reason traces
3. Add CLI flag `--explain` - Toggle explainability output
4. Update main retrieval function - Switch between simple and advanced
5. Integration test - Full M5 pipeline end-to-end
6. Performance test - Verify acceptable latency (<2s for 1000 memories)

**Acceptance**:
- Each retrieved memory has a reason trace
- Reason traces are human-readable
- User can inspect "why this memory surfaced"
- No silent, uninspectable magic

---

### Phase 5: Testing & Validation
**Estimated effort**: 3-4 hours

1. Create test corpus - 50-100 memories with known relationships
2. Define test queries - 10-15 queries with expected results
3. Baseline test - Simple retrieval results
4. M5 test - Advanced retrieval results
5. Compare results - M5 should beat simple top-K
6. Document findings - Acceptance criteria validation
7. Create smoke test script - `tests/test_m5_smoke.sh`

**Acceptance**:
- For test queries, M5 retrieval beats simple top-K
- Improvements are quantifiable (precision, recall, relevance)
- No regressions in basic functionality

---

## Configuration Schema (M5 Additions)

```yaml
# M5: Advanced Retrieval (excluding MemRank)
m5:
  # Feature toggles
  enabled: true  # Master switch for all M5 features

  # Hypothetical query generation (HyDE)
  hyde:
    enabled: true
    allowed_triggers:
      - problem-solution
      - learning-failure
    max_queries: 2
    model: claude-sonnet-4.5

  # Hybrid start node selection
  start_node:
    top_k: 3             # Number of start nodes
    alpha_meta: 0.6      # Weight: metadata vs semantic (0-1)

    # Metadata scoring
    metadata:
      keyword_weight: 0.5
      temporal_weight: 0.3
      entity_weight: 0.2

  # Probabilistic graph traversal
  traversal:
    enabled: true
    max_nodes: 5              # Max memories to recall
    exploration_factor: 2.0   # μ - higher = more exploration
    signal_decay: 0.8         # Per-hop decay (not used, edge.weight handles this)

  # Explainability
  explainability:
    enabled: true
    include_path: true        # Show traversal path
    include_scores: true      # Show score breakdown
```

---

## Success Criteria (M5 Acceptance)

### 1. Hypothetical Queries Work
- ✅ Problem-solution memories have 1-2 queries
- ✅ Queries are semantically relevant
- ✅ Queries improve retrieval (match user intent)
- ✅ Gating works (other triggers have no queries)

### 2. Hybrid Start Nodes Work
- ✅ Start nodes combine semantic + metadata signals
- ✅ Start nodes are better than pure semantic
- ✅ Config can tune weights

### 3. Probabilistic Traversal Works
- ✅ Traversal explores graph (not just direct matches)
- ✅ Stochastic behavior (runs vary)
- ✅ Signal decay works (distant memories lower score)
- ✅ No infinite loops

### 4. Explainability Works
- ✅ Each memory has a reason trace
- ✅ Reason traces are human-readable
- ✅ User can understand "why retrieved"

### 5. M5 Beats Baseline
- ✅ For test queries, M5 > simple top-K
- ✅ Improvements are measurable
- ✅ No regressions

### 6. System Remains Observable
- ✅ No silent magic (all decisions inspectable)
- ✅ Config flags work (can disable features)
- ✅ Performance acceptable (<2s for 1000 memories)

---

## Key Design Decisions

### 1. Defer MemRank (Per User Directive)
**Decision**: Do not implement PageRank-based MemRank in M5.

**Rationale**: User explicitly deferred MemRank. TraceRank (already implemented) provides graph-based scoring via:
- Reinforcement boost (event-based)
- Graph connectivity boost (inbound edge count)
- Temporal confidence (decay for dynamic facts)

**Trade-off**: MemRank would provide centrality scoring (important nodes stay retrievable). Without it, we rely on TraceRank's simpler connectivity boost. Acceptable for M5; can add MemRank in M6 if needed.

---

### 2. Selective HyDE (Not Universal)
**Decision**: Only generate hypothetical queries for problem-solution and learning-failure triggers.

**Rationale**:
- Reduce LLM cost and latency
- Avoid noise (hypotheticals for routine observations aren't helpful)
- Focus on use cases where query-memory gap is largest

**Trade-off**: May miss benefits for other trigger types. Can expand later based on usage.

---

### 3. Probabilistic vs Deterministic Traversal
**Decision**: Use probabilistic graph traversal (random walk) instead of deterministic ranking.

**Rationale**:
- Models human memory activation (not deterministic)
- Explores diverse paths (serendipity)
- Prevents over-reliance on top-scoring edges

**Trade-off**: Non-deterministic results (runs vary). Can be confusing for users. Mitigated by explainability (reason traces).

---

### 4. Signal Strength Decay
**Decision**: Signal decays with hops via edge weights (not separate decay factor).

**Rationale**:
- Simpler model (edge.weight already encodes relationship strength)
- Avoids tuning yet another parameter
- RELATED edges already have similarity-based weights

**Trade-off**: Less control over decay rate. Acceptable for M5; can add explicit decay factor later if needed.

---

### 5. Explainability as First-Class Feature
**Decision**: Generate reason traces for all retrieved memories, not just on demand.

**Rationale**:
- Core to M5 acceptance criteria ("no silent magic")
- Helps users understand and trust the system
- Essential for debugging and tuning

**Trade-off**: Adds slight overhead. Acceptable given small corpus size (<10K memories).

---

## Files to Modify

### Core Logic
1. `src/vestig/core/models.py` - Add hypothetical fields to MemoryNode
2. `src/vestig/core/storage.py` - Schema migration, new queries
3. `src/vestig/core/commitment.py` - HyDE generation logic
4. `src/vestig/core/retrieval.py` - Hybrid scoring, traversal, explainability
5. `src/vestig/core/prompts.yaml` - HyDE prompt template

### Configuration
6. `config.yaml` - Add `m5` section with all config

### CLI
7. `src/vestig/core/cli.py` - Update recall command, add --explain flag

### Testing
8. `tests/test_m5_smoke.sh` - Smoke test script
9. `tests/test_m5_retrieval.py` - Integration test

---

## Out of Scope (Deferred to M6)

- ❌ MemRank (PageRank-based graph centrality)
- ❌ Working set (recency-based saliency)
- ❌ Lateral thinking (serendipitous associations)
- ❌ Daydream mode (creative synthesis)
- ❌ User entity & core summary
- ❌ Multi-factor recall scoring formula (SPEC.md full version)

These features require M5 foundation (graph traversal, hybrid scoring) and will build on top in M6.

---

## Risk Mitigation

### Risk 1: Hypothetical queries are low quality
**Mitigation**:
- Test with sample memories
- Iterate on prompt template
- Can disable via config if not helpful

### Risk 2: Probabilistic traversal is too random
**Mitigation**:
- Tune exploration_factor (μ)
- Add minimum score threshold
- Can fall back to deterministic top-K

### Risk 3: Performance degrades with large graphs
**Mitigation**:
- Limit max_nodes (5)
- Add early termination
- Profile and optimize hot paths
- Can add graph indexing later

### Risk 4: Explainability is verbose/confusing
**Mitigation**:
- Start with simple explanations
- Iterate based on user feedback
- Add --explain flag (opt-in)

---

## Testing Strategy

### Unit Tests
- `test_hybrid_scoring()` - Verify hybrid score calculation
- `test_transition_probability()` - Verify edge selection
- `test_reason_trace()` - Verify explanation generation

### Integration Tests
- `test_hyde_generation()` - End-to-end HyDE
- `test_start_node_selection()` - Top-K selection
- `test_probabilistic_traversal()` - Graph exploration
- `test_m5_retrieval()` - Full pipeline

### Smoke Test
- `tests/test_m5_smoke.sh` - CLI-based end-to-end test

### Acceptance Test
- Test corpus (50-100 memories)
- Test queries (10-15 queries)
- Baseline vs M5 comparison
- Document improvements

---

## Documentation Updates

1. Update `README.md` - Mark M5 as complete
2. Create `M5_Completion_Report.md` - Detailed milestone report
3. Update `SPEC.md` - Reflect M5 implementation status
4. Add `docs/m5_retrieval_guide.md` - User guide for M5 features

---

## Effort Estimate

- Phase 1 (HyDE): 2-3 hours
- Phase 2 (Hybrid selection): 3-4 hours
- Phase 3 (Traversal): 4-5 hours
- Phase 4 (Explainability): 2-3 hours
- Phase 5 (Testing): 3-4 hours

**Total**: 14-19 hours (2-3 days)

---

## Summary

M5 transforms Vestig from simple semantic search into an intelligent, multi-factor retrieval system. By implementing HyDE, hybrid start node selection, and probabilistic graph traversal, we make recall feel **intentionally smart, not accidentally lucky**.

**Key achievements**:
- ✅ Query-memory gap bridged (hypothetical queries)
- ✅ Multi-signal entry points (hybrid scoring)
- ✅ Human-like activation spread (probabilistic traversal)
- ✅ Fully inspectable (reason traces)
- ✅ No silent magic (all decisions explainable)

**Deferred**: MemRank (PageRank-based centrality) per user directive. TraceRank provides sufficient graph-based scoring for M5.

**Next**: M6 will add cognitive features (working set, lateral thinking, daydream) building on M5 foundation.
