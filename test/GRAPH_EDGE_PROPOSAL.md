# Graph Edge Architecture Proposal

## Current State (Mixed FK + Edges)

**FK Pointers (to be removed):**
- `entity.chunk_id` → points to chunk where entity was extracted
- `summary.chunk_id` → points to chunk being summarized
- `memory.chunk_id` → points to chunk where memory was extracted

**Existing Edges:**
- Summary -(SUMMARIZES)→ Memory (summary covers these memories)
- Memory -(MENTIONS)→ Entity (memory mentions entity - 2nd class, from synthesized facts)
- Memory -(RELATED)→ Memory (semantic relationships)

## Proposed Pure Graph Model

### Hub-and-Spoke Edges (Chunk as Central Hub)

**Chunk -(CONTAINS)→ Memory**
- Direction: Chunk → Memory
- Semantics: "This chunk contains these extracted memories"
- Replaces: `memory.chunk_id` FK
- Traversal: Given chunk, find all memories extracted from it
- Inverse traversal: Given memory, find source chunk via inbound CONTAINS edge

**Chunk -(LINKED)→ Entity**
- Direction: Chunk → Entity
- Semantics: "This chunk text mentions this entity" (1st class - from original text)
- Replaces: `entity.chunk_id` FK (partially)
- Note: Same entity can have LINKED edges from multiple chunks
- Traversal: Given chunk, find entities mentioned in original text

**Chunk -(SUMMARIZED_BY)→ Summary**
- Direction: Chunk → Summary
- Semantics: "This chunk is summarized by this summary node"
- Replaces: `summary.chunk_id` FK
- Traversal: Given chunk, find its summary
- Note: Typically 1:1 relationship (one summary per chunk)

### Secondary Edges (Between Extracted Nodes)

**Memory -(MENTIONS)→ Entity** (existing)
- Direction: Memory → Entity
- Semantics: "This memory mentions this entity" (2nd class - from synthesized facts)
- Used for: Entity-based retrieval from memories
- Note: Different from LINKED (which is from original chunk text)

**Summary -(SUMMARIZES)→ Memory** (existing)
- Direction: Summary → Memory
- Semantics: "This summary covers these memories"
- Used for: Understanding which memories contributed to summary

**Memory -(RELATED)→ Memory** (existing)
- Direction: Memory → Memory
- Semantics: "These memories are semantically related"
- Used for: Memory clustering and co-retrieval

## Edge Type Summary

| Edge Type | From | To | Class | Replaces FK | Semantics |
|-----------|------|-----|-------|-------------|-----------|
| CONTAINS | Chunk | Memory | Hub | memory.chunk_id | Chunk contains extracted memories |
| LINKED | Chunk | Entity | Hub | entity.chunk_id | Chunk text mentions entity (1st class) |
| SUMMARIZED_BY | Chunk | Summary | Hub | summary.chunk_id | Chunk has this summary |
| MENTIONS | Memory | Entity | 2nd | - | Memory mentions entity (synthesized) |
| SUMMARIZES | Summary | Memory | 2nd | - | Summary covers memories |
| RELATED | Memory | Memory | 2nd | - | Memories are related |

## Retrieval Patterns

### Pattern 1: Recall via Chunk Expansion
```
Query → (similarity) → Summary
Summary ← (SUMMARIZED_BY) → Chunk
Chunk -(CONTAINS)→ Memory*
Return: Expanded memories
```

### Pattern 2: Entity-Based Retrieval
```
Query → (extract entities) → [Entity, Entity, ...]
Entity ← (LINKED) ← Chunk -(CONTAINS)→ Memory*
Entity ← (MENTIONS) ← Memory
Return: Memories mentioning entities
```

### Pattern 3: Chunk Context Retrieval
```
Memory ← (CONTAINS) ← Chunk
Chunk -(LINKED)→ Entity* (chunk context entities)
Chunk -(SUMMARIZED_BY)→ Summary (chunk summary)
Return: Full chunk context
```

## Migration Path

### Phase 1: Add Edges Alongside FKs
1. Create CONTAINS edges from chunk_id in memories table
2. Create LINKED edges from chunk_id in entities table
3. Create SUMMARIZED_BY edges from chunk_id in summaries table
4. Verify edge creation matches FK relationships

### Phase 2: Update Code to Use Edges
1. Update retrieval.py to traverse CONTAINS edges instead of FK lookup
2. Update storage methods to use edge traversal
3. Update ingestion to create edges instead of setting FK
4. Add helper methods: get_chunk_for_memory(), get_entities_for_chunk(), etc.

### Phase 3: Remove FK Columns
1. Verify all code uses edge traversal
2. Drop chunk_id column from memories, entities, summaries tables
3. Update MemoryNode, EntityNode dataclasses to remove chunk_id field

## Open Questions

1. **Edge direction consistency:** Should we standardize on "hub → spoke" or allow bidirectional?
2. **SUMMARIZES direction:** Keep Summary -(SUMMARIZES)→ Memory or change to Memory ← (SUMMARIZED_BY) ← Summary for consistency?
3. **Edge naming:** CONTAINS vs EXTRACTED_FROM? LINKED vs MENTIONS_IN_TEXT?
4. **Multiplicity:** Enforce 1:1 for Chunk-Summary relationship?
5. **FalkorDB compatibility:** Any specific edge naming conventions we should follow?

## Recommendation

Proceed with the proposed edge names (CONTAINS, LINKED, SUMMARIZED_BY) as they:
- Are semantically clear
- Follow "hub → spoke" direction from Chunk
- Distinguish 1st class (LINKED - original text) from 2nd class (MENTIONS - synthesized)
- Support all current retrieval patterns
- Prepare for eventual FalkorDB migration
