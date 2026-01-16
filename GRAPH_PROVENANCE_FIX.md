# Graph Provenance & Summary Linking Fix

## Problem Statement

Current implementation stores relationships as properties (IDs in JSON/fields) instead of graph edges, making Cypher queries difficult and violating graph database best practices.

### Current Issues:

1. **Memory IDs stored in Summary metadata** (line 573, ingestion.py):
   ```python
   "summarized_ids": memory_ids  # ← JSON array in metadata
   ```

2. **Source ID stored as property** (line 594, ingestion.py):
   ```python
   source_id=source_id  # ← Property, not edge
   ```

3. **Chunk ID stored as property** (line 593, ingestion.py):
   ```python
   chunk_id=chunk_id  # ← Property, not edge
   ```

4. **Only ONE edge created** (lines 604-607, ingestion.py):
   ```python
   # Only creates Chunk → Summary edge
   (Chunk)-[:SUMMARIZED_BY]->(Summary)
   ```

5. **No PRODUCED edges created anywhere**:
   - `commit_memory()` doesn't create (Source)-[:PRODUCED]->(Memory)
   - `commit_summary()` doesn't create (Source)-[:PRODUCED]->(Summary)
   - `store_memory()` doesn't create any edges

## Correct Graph Model (from schema_falkor.cypher)

### Source Provenance (Phase 2):
```cypher
# Primary provenance - always present
(Source)-[:PRODUCED]->(Memory)
(Source)-[:PRODUCED]->(Summary)

# Chunked content - dual linking
(Source)-[:HAS_CHUNK]->(Chunk)
(Chunk)-[:CONTAINS]->(Memory)
(Chunk)-[:CONTAINS]->(Summary)  # If summary is per-chunk
```

### Summary Relationships:
```cypher
# Summary links to what it summarizes
(Summary)-[:SUMMARIZES]->(Memory)  # For each memory

# Chunk points to its summary
(Chunk)-[:SUMMARIZED_BY]->(Summary)  # ✓ Already implemented

# Summary might also have entity mentions
(Summary)-[:MENTIONS]->(Entity)
```

### Complete Example (chunked document):
```cypher
(Source{type:'file', path:'doc.pdf'})
  -[:HAS_CHUNK]->(Chunk{id:'chunk_1'})
    -[:CONTAINS]->(Memory{id:'mem_1'})
    -[:CONTAINS]->(Memory{id:'mem_2'})
    -[:SUMMARIZED_BY]->(Summary{id:'mem_sum_1', kind:'SUMMARY'})
      -[:SUMMARIZES]->(Memory{id:'mem_1'})
      -[:SUMMARIZES]->(Memory{id:'mem_2'})

# AND primary provenance links:
(Source)-[:PRODUCED]->(Memory{id:'mem_1'})
(Source)-[:PRODUCED]->(Memory{id:'mem_2'})
(Source)-[:PRODUCED]->(Summary{id:'mem_sum_1'})
```

## Benefits of Edge-based Model:

1. **Natural Cypher traversal**:
   ```cypher
   # Find all memories from a source
   MATCH (s:Source)-[:PRODUCED]->(m:Memory)
   WHERE s.id = $source_id
   RETURN m

   # Find what a summary summarizes
   MATCH (sum:Memory{kind:'SUMMARY'})-[:SUMMARIZES]->(m:Memory)
   WHERE sum.id = $summary_id
   RETURN m

   # Find summary for a chunk
   MATCH (c:Chunk)-[:SUMMARIZED_BY]->(sum:Memory)
   WHERE c.id = $chunk_id
   RETURN sum
   ```

2. **Proper graph queries** - no JSON parsing in Cypher
3. **Index efficiency** - edges can be indexed by type
4. **Graph algorithms** - PageRank, centrality, etc. work naturally
5. **Temporal queries** - edges have t_valid/t_invalid timestamps

## Implementation Plan

### Phase 1: Add Edge Creation (Backward Compatible)

Keep properties temporarily, add edges alongside.

#### 1.1 Update `commit_memory()` (commitment.py)
```python
def commit_memory(...):
    # After storing memory (line 311)
    stored_id = storage.store_memory(node)

    # NEW: Create provenance edge
    if source_id:
        edge = EdgeNode.create(
            from_node=source_id,
            to_node=stored_id,
            edge_type="PRODUCED",
            weight=1.0,
        )
        storage.store_edge(edge)

    # NEW: Create chunk containment edge
    if chunk_id:
        edge = EdgeNode.create(
            from_node=chunk_id,
            to_node=stored_id,
            edge_type="CONTAINS",
            weight=1.0,
        )
        storage.store_edge(edge)
```

#### 1.2 Update `commit_summary()` (ingestion.py)

```python
def commit_summary(...):
    # After storing summary (line 600)
    storage.store_memory(summary_node, kind="SUMMARY")

    # NEW: Create Source → Summary edge (primary provenance)
    if source_id:
        edge = EdgeNode.create(
            from_node=source_id,
            to_node=summary_id,
            edge_type="PRODUCED",
            weight=1.0,
        )
        storage.store_edge(edge)

    # NEW: Create Summary → Memory edges (SUMMARIZES)
    for memory_id in memory_ids:
        edge = EdgeNode.create(
            from_node=summary_id,
            to_node=memory_id,
            edge_type="SUMMARIZES",
            weight=1.0,
        )
        storage.store_edge(edge)

    # NEW: Create Chunk → Summary edge (CONTAINS, positional metadata)
    if chunk_id:
        edge = EdgeNode.create(
            from_node=chunk_id,
            to_node=summary_id,
            edge_type="CONTAINS",
            weight=1.0,
        )
        storage.store_edge(edge)

    # KEEP existing SUMMARIZED_BY edge (line 604-607)
    # This provides reverse lookup: Chunk → Summary
```

### Phase 2: Update Query Methods

#### 2.1 Add edge-based retrieval methods (db_falkordb.py)

```python
def get_memories_for_source(self, source_id: str) -> list[MemoryNode]:
    """Get all memories produced by a source."""
    result = self._graph.ro_query(
        """
        MATCH (s:Source {id: $source_id})-[:PRODUCED]->(m:Memory)
        WHERE m.kind = 'MEMORY'
        RETURN m.id, m.content, ...
        ORDER BY m.created_at DESC
        """,
        {"source_id": source_id},
    )
    return [self._row_to_memory(row) for row in result.result_set]

def get_memories_for_chunk(self, chunk_id: str) -> list[MemoryNode]:
    """Get all memories in a chunk."""
    result = self._graph.ro_query(
        """
        MATCH (c:Chunk {id: $chunk_id})-[:CONTAINS]->(m:Memory)
        WHERE m.kind = 'MEMORY'
        RETURN m.id, m.content, ...
        """,
        {"chunk_id": chunk_id},
    )
    return [self._row_to_memory(row) for row in result.result_set]

def get_summary_for_chunk_via_edge(self, chunk_id: str) -> MemoryNode | None:
    """Get summary for a chunk via SUMMARIZED_BY edge."""
    result = self._graph.ro_query(
        """
        MATCH (c:Chunk {id: $chunk_id})-[:SUMMARIZED_BY]->(m:Memory)
        WHERE m.kind = 'SUMMARY'
        RETURN m.id, m.content, ...
        LIMIT 1
        """,
        {"chunk_id": chunk_id},
    )
    if not result.result_set:
        return None
    return self._row_to_memory(result.result_set[0])

def get_memories_summarized_by(self, summary_id: str) -> list[MemoryNode]:
    """Get memories that a summary summarizes."""
    result = self._graph.ro_query(
        """
        MATCH (sum:Memory {id: $summary_id})-[:SUMMARIZES]->(m:Memory)
        RETURN m.id, m.content, ...
        """,
        {"summary_id": summary_id},
    )
    return [self._row_to_memory(row) for row in result.result_set]
```

#### 2.2 Update existing methods to use edges

Replace:
```python
# OLD: Query by chunk_id property
def get_summary_for_chunk(self, chunk_id: str) -> MemoryNode | None:
    result = self._graph.ro_query(
        "MATCH (m:Memory {chunk_id: $chunk_id, kind: 'SUMMARY'}) ...",
        {"chunk_id": chunk_id},
    )
```

With:
```python
# NEW: Query via SUMMARIZED_BY edge
def get_summary_for_chunk(self, chunk_id: str) -> MemoryNode | None:
    return self.get_summary_for_chunk_via_edge(chunk_id)
```

### Phase 3: Remove Property-based Storage

After migration script runs and all edges exist:

1. **Remove from MemoryNode**:
   - `chunk_id` property (use CONTAINS edge)
   - `source_id` property (use PRODUCED edge)

2. **Remove from Summary metadata**:
   - `"summarized_ids"` array (use SUMMARIZES edges)

3. **Update schema**:
   - Remove chunk_id, source_id from Memory node creation
   - Update documentation

### Phase 4: Migration Script

**File:** `src/vestig/tools/migrate_to_edge_based_provenance.py`

```python
def migrate_provenance_to_edges(storage: FalkorDBDatabase):
    """Migrate property-based relationships to edges."""

    # 1. Migrate chunk_id properties to CONTAINS edges
    memories = storage.get_all_memories()
    for memory in memories:
        if memory.chunk_id:
            # Check if edge already exists
            existing = storage.get_edges_from_node(
                memory.chunk_id, edge_type="CONTAINS"
            )
            if not any(e.to_node == memory.id for e in existing):
                edge = EdgeNode.create(
                    from_node=memory.chunk_id,
                    to_node=memory.id,
                    edge_type="CONTAINS",
                    weight=1.0,
                )
                storage.store_edge(edge)

    # 2. Migrate source_id properties to PRODUCED edges
    for memory in memories:
        if memory.source_id:
            existing = storage.get_edges_from_node(
                memory.source_id, edge_type="PRODUCED"
            )
            if not any(e.to_node == memory.id for e in existing):
                edge = EdgeNode.create(
                    from_node=memory.source_id,
                    to_node=memory.id,
                    edge_type="PRODUCED",
                    weight=1.0,
                )
                storage.store_edge(edge)

    # 3. Migrate summarized_ids from metadata to SUMMARIZES edges
    summaries = [m for m in memories if m.kind == "SUMMARY"]
    for summary in summaries:
        summarized_ids = summary.metadata.get("summarized_ids", [])
        for memory_id in summarized_ids:
            existing = storage.get_edges_from_node(
                summary.id, edge_type="SUMMARIZES"
            )
            if not any(e.to_node == memory_id for e in existing):
                edge = EdgeNode.create(
                    from_node=summary.id,
                    to_node=memory_id,
                    edge_type="SUMMARIZES",
                    weight=1.0,
                )
                storage.store_edge(edge)

    print(f"Migration complete:")
    print(f"  - Processed {len(memories)} memories")
    print(f"  - Processed {len(summaries)} summaries")
```

## Testing Strategy

### Component Tests

**File:** `tests/test_graph_provenance.py`

```python
def test_memory_produced_by_source(storage):
    """Test PRODUCED edge from Source to Memory."""
    # Create source
    source = SourceNode.create(...)
    storage.store_source(source)

    # Commit memory with source_id
    outcome = commit_memory(
        content="test",
        storage=storage,
        embedding_engine=embedding_engine,
        source_id=source.id,
    )

    # Verify PRODUCED edge exists
    memories = storage.get_memories_for_source(source.id)
    assert len(memories) == 1
    assert memories[0].id == outcome.memory_id

def test_summary_summarizes_memories(storage):
    """Test SUMMARIZES edges from Summary to Memories."""
    # Create memories
    mem_ids = [...]

    # Create summary
    summary_id = commit_summary(
        summary_result=...,
        memory_ids=mem_ids,
        storage=storage,
        ...
    )

    # Verify SUMMARIZES edges exist
    summarized = storage.get_memories_summarized_by(summary_id)
    assert len(summarized) == len(mem_ids)
    assert set(m.id for m in summarized) == set(mem_ids)

def test_chunk_contains_memory(storage):
    """Test CONTAINS edge from Chunk to Memory."""
    # Create chunk
    chunk = ChunkNode.create(...)
    storage.store_chunk(chunk)

    # Commit memory with chunk_id
    outcome = commit_memory(
        content="test",
        storage=storage,
        embedding_engine=embedding_engine,
        chunk_id=chunk.id,
    )

    # Verify CONTAINS edge exists
    memories = storage.get_memories_for_chunk(chunk.id)
    assert len(memories) == 1
    assert memories[0].id == outcome.memory_id
```

## Success Criteria

- [ ] All memories have PRODUCED edge to Source
- [ ] All summaries have PRODUCED edge to Source
- [ ] All summaries have SUMMARIZES edges to their memories
- [ ] All chunked memories have CONTAINS edge from Chunk
- [ ] Chunk → Summary SUMMARIZED_BY edge exists
- [ ] Query methods use edge traversal, not property lookups
- [ ] No IDs stored in JSON metadata
- [ ] Migration script successfully converts existing data
- [ ] All tests pass with edge-based model

## Timeline Estimate

- Phase 1 (Add edge creation): 2-3 hours
- Phase 2 (Update query methods): 2-3 hours
- Phase 3 (Remove properties): 1 hour
- Phase 4 (Migration script): 2-3 hours
- Testing: 2 hours
- **Total:** ~10-12 hours

## References

- `src/vestig/core/schema_falkor.cypher` - Intended graph schema
- Phase 2 Source abstraction (Issue #3) - Established PRODUCED edge pattern
- FalkorDB edge traversal documentation
