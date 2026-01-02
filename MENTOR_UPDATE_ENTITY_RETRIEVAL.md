# Mentor Update: Entity-Based Retrieval (M4→M5 Transition)

## Current Status: M4 Complete, Entering M5 Territory

### What We've Earned

**M4 Graph Layer (Complete):**
- ✅ Entity extraction working (two-pass architecture)
- ✅ Entity embeddings implemented (semantic entity matching)
- ✅ Graph structure in place (entities table, edges table, MENTIONS relationships)
- ✅ Entity deduplication via norm_key
- ✅ CLI tools for entity management (`vestig entity regen-embeddings`)

**M3 TraceRank (Stable):**
- ✅ Bi-temporal tracking (event time vs transaction time)
- ✅ Temporal stability classification (static/dynamic/ephemeral)
- ✅ Enhanced TraceRank with reinforcement + graph connectivity
- ✅ Decay mechanics for dynamic facts

**Foundation (Solid):**
- ✅ Memory ingestion with quality gates
- ✅ Semantic search via embeddings
- ✅ Event logging for all lifecycle changes
- ✅ Schema stability maintained

### What We're Building Toward

**M5 Territory: Agent-Grade Retrieval**

The graph exists. The entities are extracted. The embeddings are ready. Now we need to **use** them for retrieval.

**Key insight:** This is not a Q&A system. It's a **context expansion saliency system**. When an agent says "I'm working on the Heidi Health pilot with John Sutherland," we need to:
1. Understand what's salient (Heidi Health, John Sutherland)
2. Expand context around those entities
3. Return comprehensive relevant memories

This is retrieval as **contextual scaffolding**, not answer finding.

### The Gap We're Addressing

**Current retrieval:** Pure semantic similarity
```
Query → Embed → Compare to all memory embeddings → Top-K
```

**Problem:** Misses structural knowledge in the graph
- Memories that mention the same entities
- Related entities that provide context
- Graph connectivity that indicates importance

**Solution:** Hybrid retrieval (semantic + entity-based)

### Architectural Direction

**Three retrieval paths:**

**Path 1: Query/Memory Similarity (existing)**
```
Query → Embed → Cosine vs Memory Embeddings → Semantic scores
```

**Path 2: Query/Entity Similarity (new)**
```
Query → Extract Entities → Match to DB Entities (via embeddings)
      → Follow MENTIONS edges → Get linked Memories → Entity scores
```

**Path 3: Best of Both (hybrid)**
```
Combine semantic scores + entity scores → Re-rank → Apply TraceRank → Top-K
```

**Why this is M5 territory:**
- "Hybrid start node selection" ← entity matching gives us entity-based start nodes
- "Probabilistic traversal" ← following MENTIONS edges is graph traversal
- "Advanced retrieval beats simple top-K" ← combining signals improves results

### Earned Complexity Checkpoint

**Have we earned this?**

✅ **Yes.** Reasoning:
1. Foundation is stable (M1-M3 working)
2. Graph layer is implemented (M4 complete)
3. Entity embeddings are ready (just shipped)
4. Use case is clear (contextual retrieval for agents)
5. Architecture is simple (two paths + combination, no speculative generation)

**Are we staying disciplined?**

✅ **Yes.** We're NOT building:
- Hypothetical entity generation (HyDe Ent) - might add noise
- Complex graph traversal (save for later if needed)
- Speculative features (working set, daydream)
- Clever magic that can't be explained

We're building:
- Two clear retrieval paths (semantic, entity)
- Simple combination strategy (weighted sum or max)
- Explainable results (can show why each memory surfaced)
- Configurable weights (user can tune the balance)

### What's Next

**Immediate task:** Write proposal for hybrid retrieval architecture

**Then:** Implement in phases
1. Entity matching (query entities → DB entities via embeddings)
2. Memory retrieval via MENTIONS edges
3. Score combination (semantic + entity)
4. Integration with existing TraceRank
5. Testing with contextual retrieval scenarios (not Q&A)

**Success criteria:**
- Entity-based retrieval returns comprehensive context
- Hybrid retrieval outperforms pure semantic for entity-rich queries
- Results are explainable (can show semantic score + entity score + matched entities)
- Performance acceptable (<1s total retrieval time)

### Boundaries & Constraints

**In scope:**
- Query entity extraction (reuse existing `extract_entities`)
- Entity matching via embeddings (semantic similarity)
- Memory retrieval via MENTIONS edges
- Hybrid scoring (semantic + entity weights)
- TraceRank integration

**Out of scope (for now):**
- Hypothetical entity generation
- Multi-hop graph traversal (beyond direct MENTIONS)
- Entity disambiguation (unless trivial)
- Complex ranking models
- Real-time entity consolidation

**Interfaces to preserve:**
- CLI commands stay stable (`vestig memory search`, `vestig memory recall`)
- Schema remains unchanged (using existing tables)
- Config structure backward compatible
- Output format consistent (with optional metadata)

### Risk Assessment

**Low risk:**
- Using existing entity extraction infrastructure
- Simple combination of two signals
- Can be feature-flagged (`retrieval.use_entity_path: true`)
- No schema changes required

**Medium risk:**
- Performance (need to profile entity matching + edge queries)
- Score calibration (semantic and entity scores might have different scales)
- False positives (wrong entity matches could retrieve irrelevant memories)

**Mitigations:**
- Start with high similarity threshold (0.7+) for entity matching
- Make weights configurable in config.yaml
- Add timing instrumentation (like existing `--timing` flag)
- Provide explain mode (`--explain` shows matched entities)

### Observability Plan

**Users should be able to answer:**
1. "Why did this memory surface?" → Show semantic score, entity score, matched entities
2. "Which path contributed more?" → Show score breakdown
3. "What entities were matched?" → List query entities → DB entities
4. "Is it working better?" → Compare retrieval quality vs pure semantic

**Implementation:**
- Extend `format_recall_results_with_explanation()` to include entity matches
- Add `--explain` flag to show entity matching details
- Log timing breakdown (entity extraction, matching, edge queries, scoring)
- Provide config to disable entity path if not helpful

### Alignment with Progressive Maturation

**M5 Goal:** "Make recall feel intentionally smart, not accidentally lucky"

**This work delivers:**
- Intentional use of graph structure (not just embeddings)
- Explainable results (can trace why each memory surfaced)
- Hybrid approach (multiple signals = more robust)
- Observable behavior (can inspect entity matches)

**We're building the "smallest end-to-end loop" that proves:**
- Entities extracted from queries match entities in memories
- Entity-based retrieval returns relevant context
- Hybrid approach improves over pure semantic
- System remains explainable and tunable

Once proven, we can deepen:
- Add multi-hop traversal if needed
- Tune scoring weights based on usage
- Add entity disambiguation if valuable
- Explore hypothetical entities if gap exists

### Next Document: Proposal

The proposal will detail:
- Concrete architecture (functions, data flow)
- Entity matching strategy (semantic similarity thresholds)
- Score combination approaches (weighted sum, max, multiply)
- Implementation plan (files, changes, testing)
- Success metrics (contextual retrieval quality)
- Performance targets (<1s retrieval, <100ms entity matching)

**Core design principle:** Context expansion saliency, not answer finding.
