# Two-Pass Extraction Architecture Proposal

## Executive Summary

Separate memory extraction from entity extraction to enable consistent entity recognition across both ingestion (memories) and retrieval (queries). This architectural change allows the same entity extraction logic to process user queries, enabling entity-based contextual retrieval through the M4 graph layer.

## Problem Statement

### Current Architecture Limitations

**Single-pass extraction:**
```
Conversation → extract_memories_v4 → {memory content + entities}
```

**Issues:**
1. **No query entity extraction:** Cannot extract entities from user queries to leverage M4 graph
2. **Inconsistent entity recognition:** No guarantee query entities match memory entities
3. **Inflexible:** Cannot re-extract entities without re-ingesting all memories
4. **Mixed concerns:** Memory extraction and entity recognition in single complex prompt
5. **Query-time gap:** Entity graph exists but no way to access it via query entities

### Use Case Requirements

**Primary use case:** Contextual retrieval for agents (not Q&A)

**Example scenario:**
```
Agent context: "I'm working on the Heidi Health AI Scribe pilot with John Sutherland"

Desired flow:
1. Extract entities from context: ["Heidi Health AI Scribe", "John Sutherland"]
2. Match to database entities (fuzzy/exact)
3. Find all memories mentioning these entities
4. Expand via M4 graph (related entities, related memories)
5. Return comprehensive context to agent
```

**Current gap:** Step 1 (extract entities from query/context) doesn't exist

## Proposed Architecture

### Two-Pass Extraction Flow

**Ingestion Pipeline:**
```
Document/Conversation
    ↓
[PASS 1] extract_memories_simple(chunk)
    → Memories: {content, confidence, temporal_stability}
    ↓
[PASS 2] extract_entities(memory.content)
    → Entities: {name, type, confidence, evidence}
    → Create MENTIONS edges: memory → entity
```

**Query Pipeline:**
```
User Query / Agent Context
    ↓
extract_entities(query)
    → Query entities: {name, type, confidence}
    ↓
fuzzy_match_entities(query_entities, db_entities)
    → Matched entities with similarity scores
    ↓
find_memories_by_mentions(matched_entities)
    → Memories with MENTIONS edges to matched entities
    ↓
[OPTIONAL] graph_expansion(memories, entities)
    → Related memories via RELATED edges
    → Related entities via entity similarity
    ↓
Return contextualized memory set
```

### Component Design

#### 1. Memory Extraction Prompt (Simplified)

**Name:** `extract_memories_simple`

**Purpose:** Extract factual memories with temporal classification only

**Input:** Conversation chunk / document chunk

**Output:**
```json
{
  "memories": [
    {
      "content": "self-contained memory with context",
      "confidence": 0.0-1.0,
      "rationale": "why this matters",
      "temporal_stability": "static|dynamic|ephemeral|unknown"
    }
  ]
}
```

**Guidelines:**
- One clear fact per memory
- Self-contained with sufficient context
- Temporal stability classification
- NO entity extraction (delegated to pass 2)

**Benefits:**
- Simpler prompt = faster, cheaper LLM calls
- Focus on content quality
- Easier to tune memory extraction independently

#### 2. Entity Extraction Prompt (Specialized)

**Name:** `extract_entities`

**Purpose:** Extract named entities from ANY text (reusable)

**Input:** Any text string (memory content, query, document excerpt)

**Output:**
```json
{
  "entities": [
    {
      "name": "canonical entity name",
      "type": "PERSON|ORG|SYSTEM|PROJECT|TOOL|PLACE|SKILL|FILE|CONCEPT",
      "confidence": 0.0-1.0,
      "evidence": "text span supporting extraction",
      "normalization_notes": "why this canonical form"
    }
  ]
}
```

**Guidelines:**
- Use v4 tiered hierarchy (Tier 1/2/3)
- Apply all quality gates (specificity, discriminability, retrievability, consistency)
- Normalize to canonical forms
- Same rules for memory content AND queries

**Benefits:**
- Consistent entity extraction everywhere
- Reusable for ingestion, queries, ad-hoc entity recognition
- Can tune entity quality independently
- Can use different models (fast vs accurate)

#### 3. Entity Matching System

**Purpose:** Match query entities to database entities (fuzzy + exact)

**Matching strategies:**

**Level 1: Exact match**
```python
def exact_match(query_entity: Entity) -> list[Entity]:
    """Match on canonical_name and type"""
    return db.query(
        "SELECT * FROM entities
         WHERE canonical_name = ? AND entity_type = ?",
        query_entity.name, query_entity.type
    )
```

**Level 2: Normalized match**
```python
def normalized_match(query_entity: Entity) -> list[Entity]:
    """Match on norm_key (type:normalized_name)"""
    norm_key = normalize_entity_key(query_entity)
    return db.query(
        "SELECT * FROM entities WHERE norm_key = ?",
        norm_key
    )
```

**Level 3: Fuzzy match (string similarity)**
```python
def fuzzy_match(query_entity: Entity, threshold=0.85) -> list[Entity]:
    """Match on string similarity within same type"""
    candidates = db.query(
        "SELECT * FROM entities WHERE entity_type = ?",
        query_entity.type
    )
    matches = []
    for candidate in candidates:
        similarity = levenshtein_ratio(
            query_entity.name.lower(),
            candidate.canonical_name.lower()
        )
        if similarity >= threshold:
            matches.append((candidate, similarity))
    return sorted(matches, key=lambda x: x[1], reverse=True)
```

**Level 4: Semantic match (requires entity embeddings)**
```python
def semantic_match(query_entity: Entity, threshold=0.85) -> list[Entity]:
    """Match on entity embedding similarity"""
    query_embedding = embed_entity(query_entity.name)
    similar = db.vector_search(
        query_embedding,
        filter=f"entity_type = '{query_entity.type}'",
        limit=5
    )
    return [(e, score) for e, score in similar if score >= threshold]
```

**Matching pipeline:**
```python
def match_query_entities(query_entities: list[Entity]) -> dict:
    """Match query entities using cascading strategy"""
    results = {}
    for qe in query_entities:
        # Try exact first
        exact = exact_match(qe)
        if exact:
            results[qe] = {"matches": exact, "strategy": "exact"}
            continue

        # Try normalized
        normalized = normalized_match(qe)
        if normalized:
            results[qe] = {"matches": normalized, "strategy": "normalized"}
            continue

        # Try fuzzy
        fuzzy = fuzzy_match(qe, threshold=0.85)
        if fuzzy:
            results[qe] = {"matches": fuzzy, "strategy": "fuzzy"}
            continue

        # No match
        results[qe] = {"matches": [], "strategy": "none"}

    return results
```

#### 4. Memory Retrieval via Entities

**Purpose:** Find memories linked to matched entities

```python
def retrieve_by_entities(
    matched_entities: list[Entity],
    expand_graph: bool = True,
    include_related: bool = True
) -> list[Memory]:
    """Retrieve memories via entity mentions and optional graph expansion"""

    # Find direct mentions
    memory_ids = set()
    for entity in matched_entities:
        edges = db.query(
            "SELECT memory_id FROM edges
             WHERE edge_type = 'MENTIONS'
             AND target_id = ?",
            entity.id
        )
        memory_ids.update(e.memory_id for e in edges)

    if expand_graph:
        # Find related entities
        related_entities = set()
        for entity in matched_entities:
            related = db.query(
                "SELECT target_id FROM edges
                 WHERE edge_type = 'RELATED'
                 AND source_id = ?
                 AND confidence >= 0.75",
                entity.id
            )
            related_entities.update(r.target_id for r in related)

        # Find memories mentioning related entities
        for rel_entity_id in related_entities:
            edges = db.query(
                "SELECT memory_id FROM edges
                 WHERE edge_type = 'MENTIONS'
                 AND target_id = ?",
                rel_entity_id
            )
            memory_ids.update(e.memory_id for e in edges)

    # Retrieve memories
    memories = db.query(
        f"SELECT * FROM memories
         WHERE id IN ({','.join('?' * len(memory_ids))})",
        *memory_ids
    )

    return memories
```

## Benefits

### 1. Consistent Entity Extraction

**Same prompt for memories and queries:**
- "Heidi Health" in query → same normalization as in memories
- "Matt" → "Matt Joyce" (both contexts)
- Quality gates applied uniformly

**Example:**
```
Query: "Update on Heidi Health pilot"
Extract: ["Heidi Health" (SYSTEM)]
Normalize: "Heidi Health AI Scribe"
Match: Exact match to DB entity
Retrieve: All memories mentioning "Heidi Health AI Scribe"
```

### 2. Entity-Based Contextual Retrieval

**Enable the primary use case:**
```
Agent context: "Working with John Sutherland on AI Adoption Survey"
→ Extract: ["John Sutherland" (PERSON), "AI Adoption Survey" (PROJECT)]
→ Match: Exact matches in DB
→ Retrieve:
    - John Sutherland's role (CIO, Executive Sponsor)
    - Survey structure (6 sections)
    - Survey results (113 leadership responses)
    - Related projects (AI governance)
    - Barriers identified
    - Next actions
→ Agent has full context to help user
```

### 3. Flexibility & Maintainability

**Re-extract entities without re-ingesting:**
```bash
# Update entity prompt (v5 with better CONCEPT filtering)
# Re-run entity extraction on existing memories
vestig admin re-extract-entities --prompt extract_entities_v5
```

**Different models per task:**
- Memory extraction: Fast model (Cerebras) for throughput
- Entity extraction: Accurate model (Haiku) for quality
- Query entity extraction: Fast model (low latency) or cached

**Independent tuning:**
- Tune memory extraction for content quality
- Tune entity extraction for entity quality
- No interference between concerns

### 4. Performance Optimization

**Batch entity extraction:**
```python
# Extract entities from 100 memories in one LLM call
batch = memories[:100]
batch_text = "\n---\n".join(m.content for m in batch)
entities_batch = extract_entities(batch_text)
```

**Caching:**
```python
# Cache entity extractions per memory
cache_key = f"entities:{memory.id}:{entity_prompt_version}"
if cached := cache.get(cache_key):
    return cached
entities = extract_entities(memory.content)
cache.set(cache_key, entities, ttl=None)  # Permanent
```

**Query-time speed:**
```python
# Query entity extraction is fast (single short text)
# Fuzzy matching is local (no LLM call)
# Total latency: ~100-500ms (vs current pure semantic: ~2-3s)
```

## Trade-offs

### Costs

**Additional LLM call per ingestion:**
- Before: 1 call (extract_memories_v4)
- After: 2 calls (extract_memories_simple + extract_entities)
- Mitigation: Batch entity extraction, use faster model

**Two prompts to maintain:**
- Memory extraction prompt
- Entity extraction prompt
- Mitigation: Simpler prompts are easier to maintain than one complex prompt

### Complexity

**Additional pipeline stage:**
- More code paths, more potential failure points
- Mitigation: Clear separation makes debugging easier

**Entity matching logic:**
- Fuzzy matching, normalization, disambiguation
- Mitigation: Start simple (exact match), add sophistication as needed

### Context Loss

**Entities extracted from isolated memories:**
- Less context than extracting from full conversation
- Mitigation: Include surrounding memories or metadata in extraction

## Implementation Plan

### Phase 1: Core Prompts (1-2 hours)

**Tasks:**
1. Create `extract_memories_simple` prompt
   - Copy extract_memories_v4
   - Remove entity extraction guidelines
   - Remove entity output schema
   - Test on sample conversation

2. Create `extract_entities` prompt
   - Extract entity guidelines from v4
   - Make text-agnostic (works on any input)
   - Add normalization notes field
   - Test on sample memories and queries

3. Add to prompts.yaml
4. Create test configs

**Deliverables:**
- `prompts.yaml` with new prompts
- `test/config-two-pass.yaml`
- Sample test showing entity extraction from query

### Phase 2: Pipeline Changes (2-3 hours)

**Tasks:**
1. Modify ingestion pipeline in `ingestion.py`
   - First pass: extract memories only
   - Second pass: extract entities from memories
   - Create MENTIONS edges
   - Handle batching

2. Add entity extraction function in new `entity_extraction.py`
   - Reusable entity extractor
   - Handles both memory content and queries
   - Implements batching
   - Implements caching

3. Update CLI commands
   - `memory add` uses two-pass
   - New: `memory extract-entities` (re-run on existing memories)

**Deliverables:**
- Modified `ingestion.py`
- New `entity_extraction.py`
- Updated CLI

### Phase 3: Query Entity Matching (3-4 hours)

**Tasks:**
1. Implement entity matching in new `entity_matching.py`
   - Exact match
   - Normalized match
   - Fuzzy match (string similarity)
   - Cascading match pipeline

2. Implement retrieval by entities in `retrieval.py`
   - Find memories by entity mentions
   - Optional graph expansion
   - Combine with semantic search (hybrid)

3. Add CLI command for entity-based retrieval
   - `memory recall-by-entities "working on Heidi Health"`
   - Extracts entities from query
   - Matches and retrieves

**Deliverables:**
- New `entity_matching.py`
- Updated `retrieval.py`
- New CLI command

### Phase 4: Testing & Validation (2-3 hours)

**Tasks:**
1. Create test harness for contextual retrieval
   - Not Q&A style (find THE answer)
   - Contextual style (find ALL context)
   - Coverage and relevance metrics

2. Test entity matching quality
   - Exact match accuracy
   - Fuzzy match quality
   - False positives/negatives

3. Compare retrieval methods
   - Pure semantic (baseline)
   - Pure entity-based
   - Hybrid (semantic + entity)

**Deliverables:**
- Contextual test harness
- Entity matching benchmarks
- Retrieval comparison report

### Phase 5: Optimization (2-3 hours)

**Tasks:**
1. Implement batching for entity extraction
2. Implement caching
3. Performance profiling
4. Rate limiting integration

**Deliverables:**
- Optimized pipeline
- Performance benchmarks

## Schema Changes

**None required!** M4 already has:
- `entities` table with canonical_name and norm_key
- `edges` table with MENTIONS and RELATED types
- Entity types and confidence

**Optional additions:**
```sql
-- Cache table for entity extractions
CREATE TABLE entity_extraction_cache (
    memory_id TEXT PRIMARY KEY,
    prompt_version TEXT NOT NULL,
    entities TEXT NOT NULL,  -- JSON
    extracted_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

-- Index for faster fuzzy matching
CREATE INDEX idx_entities_name_lower
ON entities(LOWER(canonical_name));
```

## Migration Path

### Stage 1: Parallel Implementation (Week 1)
- Implement two-pass extraction as opt-in
- Config flag: `ingestion.two_pass_extraction: true`
- Test on small dataset
- Compare entity quality vs v4

### Stage 2: Validation (Week 2)
- Create new test database with two-pass
- Compare entity distributions
- Test query entity extraction
- Measure retrieval quality

### Stage 3: Rollout (Week 3)
- If validation successful, make default
- Update all configs to use two-pass
- Document migration for users
- Provide re-extraction script

### Stage 4: Deprecation (Week 4)
- Remove single-pass option if two-pass is superior
- Archive old prompts as legacy

## Success Metrics

### Entity Quality
- [ ] CONCEPT entities reduced by >80% (only domain-specific)
- [ ] Zero generic entities ("storage", "legal", dates)
- [ ] >95% entity normalization accuracy ("Matt" → "Matt Joyce")
- [ ] Entity duplication rate <5%

### Query Entity Extraction
- [ ] Entity extraction from queries works consistently
- [ ] >90% of query entities match database entities (exact or fuzzy)
- [ ] Query entity extraction latency <200ms

### Contextual Retrieval
- [ ] Entity-based retrieval returns comprehensive context
- [ ] Coverage: >80% of relevant memories retrieved
- [ ] Precision: <20% irrelevant memories in results
- [ ] Hybrid (semantic + entity) outperforms pure semantic

### Performance
- [ ] Ingestion throughput: ≥80% of single-pass (with batching)
- [ ] Query-time entity matching: <100ms
- [ ] Total retrieval latency: <1s (entity + semantic + expansion)

## Future Enhancements

### 1. Entity Embeddings

**Purpose:** Semantic entity matching (not just string similarity)

**Approach:**
- Use small embedding model (all-minilm 384d) for entities
- Embed entity canonical names
- Semantic similarity search for matching
- Auto-merge high-similarity entities

**Benefits:**
- "Heidi" matches "Heidi Health" semantically
- Handles abbreviations, variants better than fuzzy string match
- Enables entity consolidation via similarity clustering

### 2. Entity Disambiguation

**Purpose:** Handle ambiguous entity references

**Example:**
- "John" → could be "John Sutherland" or "John Smith"
- Context: "Working with John on AI Survey"
- Disambiguation: Check which John is linked to "AI Survey" project

**Approach:**
- Use context entities to disambiguate
- Graph distance to related entities
- Most likely entity given context

### 3. Entity Consolidation Pipeline

**Purpose:** Periodic merging of duplicate/similar entities

**Approach:**
```python
# Daily/weekly consolidation
similar_pairs = find_similar_entities(threshold=0.90)
for pair in similar_pairs:
    if same_type(pair) and high_overlap(pair.mentions):
        propose_merge(pair)  # Human review or auto-merge
```

### 4. Multi-Modal Entity Extraction

**Purpose:** Extract entities from images, audio, video

**Example:**
- Screenshot → OCR → extract entities from text
- Audio transcript → extract entities
- Image with labels → extract entities from labels

### 5. Temporal Entity Tracking

**Purpose:** Track entity changes over time

**Example:**
- "Matt Joyce" has role "CTO" (2023-2024)
- "Matt Joyce" has role "VP Engineering" (2024-present)
- Query: "Who was CTO in 2023?" → "Matt Joyce"

## Open Questions

1. **Batching strategy:** How many memories per entity extraction batch?
   - Trade-off: Larger batches = more efficient, but harder to parse output
   - Suggestion: 50-100 memories per batch

2. **Model selection:** Which model for entity extraction?
   - Memory extraction: Cerebras (fast, cheap)
   - Entity extraction: Haiku (accurate) or Cerebras (fast)?
   - Need to test quality vs speed

3. **Fuzzy match threshold:** What similarity threshold for fuzzy matching?
   - Too low: False positives (wrong entities matched)
   - Too high: False negatives (miss valid matches)
   - Suggestion: 0.85 default, configurable

4. **Graph expansion depth:** How many hops in entity graph?
   - 1-hop: Direct mentions only
   - 2-hop: Related entities + their mentions
   - Config: max_expand_via_entities: 3 (current M4 default)

5. **Hybrid retrieval:** How to combine entity-based + semantic?
   - Pure entity: Fast but might miss relevant memories
   - Pure semantic: Comprehensive but slower
   - Hybrid: Union, intersection, or weighted combination?
   - Suggestion: Union with ranking boost for entity matches

## Recommendation

**Proceed with implementation** in phases:

**Immediate (Phase 1-2):** Core prompts and pipeline
- Low risk, high value
- Enables entity-based retrieval
- Can validate approach quickly

**Near-term (Phase 3-4):** Entity matching and testing
- Prove value with contextual retrieval tests
- Measure improvement over pure semantic

**Future (Phase 5+):** Optimization and enhancements
- Entity embeddings for better matching
- Entity consolidation for cleanup
- Advanced features (disambiguation, temporal tracking)

**Expected timeline:** 2-3 weeks for core implementation and validation

**Expected impact:**
- Entity-based contextual retrieval becomes viable
- Query entities enable M4 graph utilization
- Foundation for advanced entity-centric features
