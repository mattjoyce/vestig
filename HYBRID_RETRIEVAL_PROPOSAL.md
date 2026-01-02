# Hybrid Retrieval Proposal: Semantic + Entity-Based Context Expansion

## Executive Summary

Enhance retrieval with entity-based signals to enable **context expansion saliency** for agent use cases. Combine semantic similarity (existing) with entity matching (new) to retrieve comprehensive relevant context, not just semantically similar memories.

**Key insight:** This is not a Q&A system seeking "the answer." It's a context expansion system seeking "all relevant context" around salient entities and concepts.

## Problem Statement

### Current Limitation: Semantic-Only Retrieval

**Existing flow:**
```
Query/Context → Embed → Compare to all memory embeddings → Top-K by cosine similarity
```

**Example scenario:**
```
Agent context: "Working with John Sutherland on AI Adoption Survey"

Current retrieval (semantic only):
→ Returns memories with similar wording ("AI survey", "adoption study")
→ Misses structural knowledge:
  - John Sutherland is CIO, Executive Sponsor
  - Survey has 6 sections, 113 responses
  - Related to AI governance project
  - Barriers identified: resource allocation, skill gaps
  - Next actions agreed with steering committee
```

**The gap:** Semantic similarity doesn't capture **entity relationships**. Memories that mention "John Sutherland" are relevant even if they don't contain "AI Adoption Survey."

### Use Case: Context Expansion Saliency

**Not this (Q&A):**
```
Query: "What is John's role?"
Goal: Find THE answer → "CIO"
```

**But this (Context Expansion):**
```
Agent context: "Working with John on AI project"
Goal: Find ALL relevant context →
  - John's role and responsibilities
  - Projects John is involved in
  - Recent interactions with John
  - Related stakeholders and teams
  - Decisions and commitments made
  - Current status and next actions
```

**Retrieval as scaffolding:** Provide the agent with comprehensive context to understand the situation and help effectively.

## Proposed Architecture

### Three Retrieval Paths

#### Path 1: Query/Memory Similarity (Existing)

**Direct semantic matching:**
```python
def semantic_retrieval(query: str) -> list[(memory_id, semantic_score)]:
    query_embedding = embed_text(query)

    scores = []
    for memory in all_memories:
        similarity = cosine_similarity(query_embedding, memory.embedding)
        scores.append((memory.id, similarity))

    return scores
```

**Strengths:**
- Captures conceptual similarity
- Works for abstract queries
- No entity extraction required

**Weaknesses:**
- Misses structural relationships
- Can't leverage graph knowledge
- Surface-level matching only

#### Path 2: Query/Entity Similarity (New)

**Entity-based context expansion:**
```python
def entity_retrieval(query: str) -> list[(memory_id, entity_score)]:
    # Extract entities from query
    query_entities = extract_entities_from_text(query, model)

    if not query_entities:
        return []  # Fall back to semantic only

    # Match to database entities via semantic similarity
    matched_entities = []
    for q_ent_name, q_ent_type, q_conf, q_evidence in query_entities:
        q_ent_embedding = embed_text(q_ent_name.lower())

        # Find similar entities in DB
        for db_entity in get_all_entities(entity_type=q_ent_type):
            if db_entity.embedding:
                db_embedding = json.loads(db_entity.embedding)
                similarity = cosine_similarity(q_ent_embedding, db_embedding)

                if similarity >= ENTITY_MATCH_THRESHOLD:  # e.g., 0.7
                    matched_entities.append((db_entity.id, similarity))

    # Get memories linked to matched entities (via MENTIONS edges)
    memory_scores = {}
    for entity_id, entity_similarity in matched_entities:
        edges = get_edges_to_entity(entity_id, edge_type="MENTIONS")

        for edge in edges:
            memory_id = edge.from_node

            # Score = entity_similarity × edge_confidence
            score = entity_similarity * (edge.confidence or 1.0)

            # Keep max score if memory mentions multiple matched entities
            memory_scores[memory_id] = max(
                memory_scores.get(memory_id, 0.0),
                score
            )

    return list(memory_scores.items())
```

**Strengths:**
- Leverages graph structure (entities, MENTIONS edges)
- Finds memories about the same entities even if wording differs
- Provides context expansion around salient entities
- Handles entity variations (embeddings match "Heidi" to "Heidi Health")

**Weaknesses:**
- Requires entity extraction (LLM call, ~200ms)
- Only works if query contains extractable entities
- Entity matching might produce false positives

#### Path 3: Best of Both (Hybrid)

**Combine semantic and entity signals:**
```python
def hybrid_retrieval(
    query: str,
    entity_weight: float = 0.5,  # Configurable balance
    limit: int = 5
) -> list[(memory, final_score)]:

    # Get scores from both paths
    semantic_scores = semantic_retrieval(query)  # dict[memory_id → score]
    entity_scores = entity_retrieval(query)      # dict[memory_id → score]

    # Combine scores
    all_memory_ids = set(semantic_scores.keys()) | set(entity_scores.keys())

    combined = []
    for memory_id in all_memory_ids:
        sem_score = semantic_scores.get(memory_id, 0.0)
        ent_score = entity_scores.get(memory_id, 0.0)

        # Weighted combination
        combined_score = (
            (1 - entity_weight) * sem_score +
            entity_weight * ent_score
        )

        combined.append((memory_id, combined_score))

    # Sort by combined score
    combined.sort(key=lambda x: x[1], reverse=True)

    # Apply TraceRank boosting (existing)
    final_results = []
    for memory_id, combined_score in combined[:limit * 2]:  # Over-fetch for TraceRank
        memory = get_memory(memory_id)

        # Apply TraceRank multiplier
        tracerank_mult = compute_enhanced_multiplier(memory, ...)
        final_score = combined_score * tracerank_mult

        final_results.append((memory, final_score))

    # Re-sort after TraceRank and return top-K
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results[:limit]
```

**Combination strategies (alternatives):**

1. **Weighted sum** (proposed above):
   ```python
   score = α × semantic + β × entity
   # α + β = 1, configurable weights
   ```

2. **Max** (best of either):
   ```python
   score = max(semantic, entity)
   # Use whichever path found it most relevant
   ```

3. **Multiply** (both must agree):
   ```python
   score = semantic × entity
   # Penalizes if either path gives low score
   ```

4. **Boosted semantic** (entity as amplifier):
   ```python
   score = semantic × (1 + entity_boost)
   # Entity match amplifies semantic score
   ```

**Recommendation:** Start with **weighted sum** (most flexible, tunable).

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Query / Agent Context                     │
│          "Working with John on AI Adoption Survey"           │
└────────────────┬──────────────────┬─────────────────────────┘
                 │                  │
                 ▼                  ▼
        ┌────────────────┐  ┌──────────────────┐
        │  Path 1:       │  │  Path 2:         │
        │  Embed Query   │  │  Extract         │
        │                │  │  Entities        │
        └───────┬────────┘  └────────┬─────────┘
                │                    │
                │                    ├─→ ["John Sutherland", "AI Adoption Survey"]
                │                    │
                ▼                    ▼
        ┌────────────────┐  ┌──────────────────┐
        │  Compare to    │  │  Match to DB     │
        │  Memory        │  │  Entities        │
        │  Embeddings    │  │  (via embed)     │
        └───────┬────────┘  └────────┬─────────┘
                │                    │
                │                    ├─→ [ent_abc (0.95), ent_xyz (0.88)]
                │                    │
                ▼                    ▼
        ┌────────────────┐  ┌──────────────────┐
        │  Semantic      │  │  Get Memories    │
        │  Scores        │  │  via MENTIONS    │
        │  mem_1: 0.82   │  │  edges           │
        │  mem_2: 0.76   │  │                  │
        └───────┬────────┘  └────────┬─────────┘
                │                    │
                │                    ├─→ Entity Scores
                │                    │   mem_3: 0.91
                │                    │   mem_1: 0.77
                │                    │
                └────────┬───────────┘
                         │
                         ▼
                ┌────────────────────┐
                │  Combine Scores    │
                │  (weighted sum)    │
                │                    │
                │  mem_1: 0.795      │
                │  mem_3: 0.455      │
                │  mem_2: 0.380      │
                └────────┬───────────┘
                         │
                         ▼
                ┌────────────────────┐
                │  Apply TraceRank   │
                │  (reinforcement +  │
                │   graph + decay)   │
                └────────┬───────────┘
                         │
                         ▼
                ┌────────────────────┐
                │  Top-K Results     │
                │  with explanation  │
                └────────────────────┘
```

## Implementation Details

### 1. Entity Matching

**Function signature:**
```python
def match_query_entities_to_db(
    query_entities: list[tuple[str, str, float, str]],  # (name, type, conf, evidence)
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    similarity_threshold: float = 0.7,
) -> list[tuple[str, str, float]]:  # (db_entity_id, query_entity_name, similarity)
    """
    Match query entities to database entities using semantic similarity.

    Args:
        query_entities: Entities extracted from query
        storage: Storage for entity lookup
        embedding_engine: For embedding query entity names
        similarity_threshold: Minimum similarity to consider a match (default: 0.7)

    Returns:
        List of (db_entity_id, query_entity_name, similarity_score) tuples
    """
```

**Algorithm:**
1. For each query entity:
   - Embed the entity name (lowercase for consistency)
   - Get all DB entities of the same type
   - Compute cosine similarity between query embedding and each DB entity embedding
   - Keep matches above threshold
   - Return best match (highest similarity)

**Optimizations:**
- Cache DB entity embeddings in memory (avoid repeated JSON parsing)
- Early exit if exact norm_key match found
- Vectorize similarity computation (batch cosine similarities)

### 2. Memory Retrieval via Entities

**Function signature:**
```python
def retrieve_memories_by_entities(
    matched_entities: list[tuple[str, float]],  # (entity_id, match_score)
    storage: MemoryStorage,
    include_expired: bool = False,
) -> dict[str, float]:  # memory_id → entity_score
    """
    Retrieve memories that mention matched entities.

    Args:
        matched_entities: DB entities that matched query entities
        storage: Storage for edge lookup
        include_expired: Include expired memories (default: False)

    Returns:
        Dict mapping memory_id to entity-based score
    """
```

**Algorithm:**
1. For each matched entity:
   - Get all MENTIONS edges pointing to this entity (from_node → memory)
   - For each edge:
     - Compute score = entity_match_score × edge_confidence
     - If memory already seen (mentions multiple matched entities), keep max score
2. Return dict of memory_id → best_entity_score

**Edge query:**
```sql
SELECT from_node, confidence
FROM edges
WHERE to_node = ?
  AND edge_type = 'MENTIONS'
  AND (t_expired IS NULL OR ? = 1)
```

### 3. Score Combination

**Function signature:**
```python
def combine_scores(
    semantic_scores: dict[str, float],  # memory_id → semantic_score
    entity_scores: dict[str, float],    # memory_id → entity_score
    entity_weight: float = 0.5,
    combination_mode: str = "weighted_sum",  # "weighted_sum" | "max" | "multiply"
) -> dict[str, float]:  # memory_id → combined_score
    """
    Combine semantic and entity scores using specified strategy.
    """
```

**Weighted sum implementation:**
```python
if combination_mode == "weighted_sum":
    for memory_id in all_memory_ids:
        sem = semantic_scores.get(memory_id, 0.0)
        ent = entity_scores.get(memory_id, 0.0)
        combined[memory_id] = (1 - entity_weight) * sem + entity_weight * ent
```

**Score normalization (if needed):**
- Semantic scores: Already 0-1 (cosine similarity)
- Entity scores: Also 0-1 (entity similarity × edge confidence)
- No normalization needed if both in same range

**Configuration:**
```yaml
retrieval:
  entity_path:
    enabled: true
    entity_weight: 0.5          # Balance between semantic and entity
    similarity_threshold: 0.7   # Minimum entity match similarity
    combination_mode: "weighted_sum"  # Strategy for combining scores
```

### 4. Integration with Existing Retrieval

**Modify `search_memories()` in retrieval.py:**
```python
def search_memories(
    query: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    limit: int = 5,
    event_storage: MemoryEventStorage | None = None,
    tracerank_config: TraceRankConfig | None = None,
    include_expired: bool = False,
    show_timing: bool = False,
    # New parameters for hybrid retrieval
    entity_config: dict | None = None,  # Entity path configuration
    model: str | None = None,           # For entity extraction
) -> list[tuple[MemoryNode, float]]:
    """
    Search memories with optional entity-based augmentation.
    """

    # Path 1: Semantic similarity (existing code)
    semantic_scores = {...}

    # Path 2: Entity-based retrieval (new)
    entity_scores = {}
    if entity_config and entity_config.get("enabled", False) and model:
        # Extract entities from query
        query_entities = extract_entities_from_text(query, model)

        if query_entities:
            # Match to DB entities
            matched = match_query_entities_to_db(query_entities, storage, embedding_engine, ...)

            # Get memories via MENTIONS edges
            entity_scores = retrieve_memories_by_entities(matched, storage, ...)

    # Path 3: Combine scores
    if entity_scores:
        combined_scores = combine_scores(semantic_scores, entity_scores, ...)
    else:
        combined_scores = semantic_scores

    # Apply TraceRank (existing code, but on combined scores)
    # ... existing TraceRank logic ...

    # Sort and return top-K
    # ... existing sorting logic ...
```

**Backward compatibility:**
- If `entity_config` is None or disabled, behaves exactly like before
- Existing tests pass without changes
- Feature-flagged rollout

### 5. Explainability

**Extend result metadata to include entity matches:**
```python
@dataclass
class RetrievalResult:
    memory: MemoryNode
    final_score: float
    semantic_score: float
    entity_score: float
    tracerank_multiplier: float
    matched_entities: list[tuple[str, str, float]]  # (entity_name, entity_type, similarity)
```

**Enhanced explanation format:**
```
[META] (score=0.82, age=3d, stability=static)
Semantic: 0.65 | Entity: 0.87 (matched: John Sutherland [PERSON, 0.95], AI Adoption Survey [PROJECT, 0.79]) | TraceRank: 1.15x (2x reinforced, 3 conn)
[MEMORY]
<content>
```

**CLI flag:**
```bash
vestig memory recall "working with John" --explain
# Shows entity matches and score breakdown
```

## Configuration

**Add to config.yaml:**
```yaml
retrieval:
  # Semantic search (existing)
  limit: 5

  # Entity-based retrieval (new)
  entity_path:
    enabled: true                    # Feature flag
    entity_weight: 0.5               # Weight for entity scores (0.0-1.0)
    similarity_threshold: 0.7        # Min similarity for entity matching
    combination_mode: "weighted_sum" # weighted_sum | max | multiply
    max_entity_matches_per_query: 10 # Limit matched entities

  # TraceRank (existing)
  tracerank:
    enabled: true
    # ... existing TraceRank config ...
```

## Performance Considerations

### Expected Latency Breakdown

**Pure semantic (baseline):**
```
Embed query:          50-100ms
Load memories:        10-50ms
Compute similarities: 50-200ms (depends on corpus size)
TraceRank:            100-300ms
Total:                210-650ms
```

**With entity path:**
```
Extract entities:     150-300ms (LLM call)
Embed entity names:   20-50ms (per entity, ~3 entities avg)
Match to DB entities: 50-150ms (similarity computation)
Edge queries:         20-100ms (depends on entity count)
Combine scores:       10-20ms
TraceRank:            100-300ms
Total:                350-920ms
```

**Target: <1s total retrieval time**

### Optimization Strategies

**1. Parallel execution:**
```python
# Run semantic and entity paths in parallel
with ThreadPoolExecutor() as executor:
    semantic_future = executor.submit(semantic_retrieval, query)
    entity_future = executor.submit(entity_retrieval, query)

    semantic_scores = semantic_future.result()
    entity_scores = entity_future.result()
```
**Savings: ~200-300ms** (entity extraction overlaps with semantic search)

**2. Entity embedding cache:**
```python
# Cache DB entity embeddings in memory (avoid JSON parsing)
_ENTITY_EMBEDDING_CACHE = {}  # entity_id → embedding vector

def get_entity_embedding(entity_id: str, storage: MemoryStorage) -> list[float]:
    if entity_id not in _ENTITY_EMBEDDING_CACHE:
        entity = storage.get_entity(entity_id)
        _ENTITY_EMBEDDING_CACHE[entity_id] = json.loads(entity.embedding)
    return _ENTITY_EMBEDDING_CACHE[entity_id]
```
**Savings: ~20-50ms** (per retrieval call)

**3. Vectorized similarity computation:**
```python
# Compute all entity similarities at once using numpy
query_embedding = np.array(embed_text(query_entity_name))
db_embeddings = np.array([get_entity_embedding(e.id) for e in db_entities])
similarities = np.dot(db_embeddings, query_embedding)  # Batch cosine
```
**Savings: ~30-80ms** (vs sequential)

**4. Early exit on exact match:**
```python
# If exact norm_key match, skip semantic entity matching
norm_key = compute_norm_key(query_entity_name, query_entity_type)
exact_match = storage.find_entity_by_norm_key(norm_key)
if exact_match:
    return [(exact_match.id, query_entity_name, 1.0)]  # Perfect match
```
**Savings: ~50-100ms** (when applicable)

**With optimizations: 250-600ms total** ✅ Under 1s target

## Testing Strategy

### 1. Entity Matching Quality

**Test cases:**
```python
test_cases = [
    # Exact match
    ("John Sutherland", "John Sutherland", 1.0),

    # Case variation
    ("john sutherland", "John Sutherland", 1.0),

    # Abbreviation
    ("Heidi", "Heidi Health AI Scribe", 0.85),

    # Partial name
    ("Matt", "Matt Joyce", 0.88),

    # Different entity (should NOT match)
    ("John Smith", "John Sutherland", 0.45),  # Below threshold
]

for query_name, db_name, expected_similarity in test_cases:
    actual = compute_similarity(query_name, db_name)
    assert abs(actual - expected_similarity) < 0.1
```

### 2. Contextual Retrieval Scenarios

**Not Q&A style:**
```python
# BAD: Q&A test
query = "What is John's role?"
expected_answer = "CIO"
# Don't do this!
```

**Good: Context expansion test:**
```python
context = "Working with John on AI project"
results = hybrid_retrieval(context, limit=10)

# Check for context coverage
expected_entities = ["John Sutherland", "AI Adoption Survey"]
expected_topics = ["governance", "barriers", "steering committee"]

assert any(ent in result.matched_entities for ent in expected_entities)
assert any(topic in result.memory.content for topic in expected_topics)
assert len(results) >= 5  # Comprehensive context
```

**Metrics:**
- **Coverage:** Did we retrieve all relevant memories?
- **Relevance:** Are the retrieved memories actually useful?
- **Diversity:** Do we cover different aspects (people, projects, decisions)?

### 3. Comparison Tests

**Baseline vs Hybrid:**
```python
# Same query, different methods
query = "Working with John on AI governance"

semantic_only = search_memories(query, entity_config=None)
hybrid = search_memories(query, entity_config={"enabled": True})

# Measure:
# - Overlap: How many memories in both result sets?
# - Unique to hybrid: What does entity path add?
# - Ranking changes: How do scores differ?
```

**Expected outcome:**
- Hybrid retrieves more entity-related context
- Memories mentioning "John Sutherland" rank higher
- Related projects and decisions surface

### 4. Performance Benchmarks

```python
def benchmark_retrieval(method, query, iterations=10):
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        results = method(query)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    return {
        "mean": np.mean(timings),
        "p50": np.percentile(timings, 50),
        "p95": np.percentile(timings, 95),
        "p99": np.percentile(timings, 99),
    }

# Compare
semantic_perf = benchmark_retrieval(semantic_only, query)
hybrid_perf = benchmark_retrieval(hybrid_retrieval, query)

assert hybrid_perf["p95"] < 1000  # <1s at 95th percentile
```

## Success Metrics

### Functional Goals

- ✅ Entity matching works consistently (>90% accuracy for exact/fuzzy)
- ✅ Memories retrieved via entities are relevant (manual review)
- ✅ Hybrid retrieval provides more comprehensive context than semantic alone
- ✅ Results are explainable (can show matched entities and score breakdown)

### Performance Goals

- ✅ Total retrieval latency <1s (p95)
- ✅ Entity extraction <300ms (p95)
- ✅ Entity matching <150ms (p95)
- ✅ Edge queries <100ms (p95)

### Quality Goals

- ✅ Coverage: Retrieve >80% of relevant memories (manual test set)
- ✅ Precision: <20% irrelevant memories in top-10 results
- ✅ Entity match accuracy: >90% correct matches (test cases)
- ✅ No regressions: Pure semantic queries perform same as before

## Implementation Plan

### Phase 1: Core Entity Matching (2-3 hours)

**Tasks:**
1. Implement `match_query_entities_to_db()` function
2. Implement `retrieve_memories_by_entities()` function
3. Add unit tests for entity matching
4. Test with sample queries

**Deliverables:**
- `entity_matching.py` module
- Unit tests passing
- Manual verification with known entities

### Phase 2: Score Combination (1-2 hours)

**Tasks:**
1. Implement `combine_scores()` with multiple strategies
2. Add configuration schema for entity path
3. Update config.yaml with entity settings
4. Test different combination modes

**Deliverables:**
- Score combination function with tests
- Config validation
- Comparison of strategies (weighted_sum vs max vs multiply)

### Phase 3: Integration (2-3 hours)

**Tasks:**
1. Modify `search_memories()` to support entity path
2. Add parallel execution of semantic + entity paths
3. Preserve backward compatibility (feature flag)
4. Add timing instrumentation

**Deliverables:**
- Updated `retrieval.py`
- Existing tests pass unchanged
- New tests for hybrid path

### Phase 4: Explainability (1-2 hours)

**Tasks:**
1. Extend result metadata to include entity matches
2. Update `format_recall_results_with_explanation()`
3. Add `--explain` flag to CLI
4. Document explanation format

**Deliverables:**
- Enhanced explanation output
- CLI flag working
- User documentation

### Phase 5: Testing & Optimization (3-4 hours)

**Tasks:**
1. Create contextual retrieval test harness
2. Compare hybrid vs semantic-only on test queries
3. Profile and optimize (caching, vectorization)
4. Tune threshold and weight defaults

**Deliverables:**
- Test harness with coverage/precision metrics
- Performance benchmarks
- Optimized implementation (<1s p95)
- Recommended config values

**Total estimated effort: 10-14 hours**

## Trade-offs & Risks

### Trade-offs

**Added complexity:**
- More code to maintain (entity matching, score combination)
- More configuration options (weights, thresholds)
- More failure modes (entity extraction, matching errors)

**Mitigation:**
- Keep entity path simple (no multi-hop traversal yet)
- Feature flag allows disabling if problematic
- Clear documentation and tests

**Performance overhead:**
- Entity extraction adds ~200-300ms
- Entity matching adds ~100-200ms

**Mitigation:**
- Run in parallel with semantic path (saves ~200ms)
- Cache entity embeddings
- Early exit on exact matches

**Score calibration:**
- Semantic and entity scores might have different distributions
- Weighted combination might need tuning per corpus

**Mitigation:**
- Make weights configurable
- Provide comparison tools to tune
- Document recommended values

### Risks

**Low risk:**
- Using proven components (entity extraction, embeddings, edge queries)
- Feature-flagged (can disable if issues)
- Backward compatible (no schema changes)

**Medium risk:**
- Entity matching quality (false positives/negatives)
- Performance on large corpora (many entities, many edges)
- Score combination might not improve results

**Mitigations:**
- Start with high similarity threshold (0.7) to reduce false positives
- Profile with realistic corpus sizes (10K+ memories)
- A/B test hybrid vs semantic-only with real queries
- Provide tuning tools and guidance

**High risk (avoided):**
- ❌ Hypothetical entity generation (HyDe Ent) - NOT implementing (too much variability)
- ❌ Complex graph traversal - NOT implementing yet (premature)
- ❌ Automatic entity consolidation - NOT implementing (could corrupt data)

## Future Enhancements

**Once proven valuable:**

1. **Multi-hop entity traversal:**
   - Follow entity→entity relationships (RELATED edges)
   - Expand context to related entities (e.g., project members, related systems)

2. **Entity disambiguation:**
   - When "John" matches multiple people, use context to disambiguate
   - Graph distance to other matched entities

3. **Entity-based re-ranking:**
   - Use entity co-occurrence patterns to boost relevant memories
   - "Memories mentioning these entities together are often relevant"

4. **Temporal entity filtering:**
   - Only match entities valid at query time (if temporal context provided)
   - "Who was CTO in 2023?" → filter entities by t_valid

5. **Hypothetical entities (HyDe Ent):**
   - If entity extraction returns nothing, generate hypothetical entities
   - Only if data shows value (not building speculatively)

## Recommendation

**Proceed with implementation.**

**Why now:**
- Foundation is solid (M4 complete, entity embeddings ready)
- Use case is clear (context expansion for agents)
- Architecture is simple (two paths + combination)
- Low risk (feature-flagged, backward compatible)
- High value (enables entity-based contextual retrieval)

**Why this approach:**
- Avoids premature complexity (no HyDe Ent, no multi-hop traversal)
- Uses existing infrastructure (entity extraction, embeddings, edges)
- Explainable and tunable (weights, thresholds, combination modes)
- Measurable (can compare hybrid vs semantic quantitatively)

**Success looks like:**
- Agent queries like "working with John" retrieve comprehensive context
- Entity-based path adds value beyond pure semantic matching
- Performance is acceptable (<1s total retrieval)
- Results are explainable (can show why each memory surfaced)

**This is M5 territory:** Hybrid retrieval that feels intentionally smart, leveraging the graph structure we've earned through M4.
