# Vestig Architecture: Source-Centric Knowledge Graph with Dual Linking

**Version:** 2.0 (Phase 2 Complete)
**Date:** 2026-01-11
**Status:** Production (SQLite + FalkorDB backends)

## Executive Summary

Vestig implements a **source-centric graph architecture** with **dual linking** for knowledge management, where unified source nodes provide primary provenance for all content origins (files, agent contributions, legacy data), while optional chunk nodes provide positional metadata. This design prioritizes **provenance**, **temporal tracking**, and **multi-resolution querying** over traditional entity-centric approaches.

**Key Innovations:**
1. **SOURCE as primary provenance** - Unified tracking for file, agentic, and legacy content
2. **Dual linking model** - Memories link to both Source (primary provenance) AND Chunk (positional metadata)
3. **Chunk as optional metadata** - Location pointers within sources, not required intermediary
4. **Multi-backend support** - SQLite (relational) and FalkorDB (graph-native) implementations

---

## Table of Contents

1. [Overview](#overview)
2. [Current State (SQLite)](#current-state-sqlite)
3. [Target State (Neo4j)](#target-state-neo4j)
4. [Graph Model](#graph-model)
5. [Architectural Rationale](#architectural-rationale)
6. [Comparison to Other Approaches](#comparison-to-other-approaches)
7. [Migration Path](#migration-path)
8. [Benefits & Trade-offs](#benefits--trade-offs)

---

## Overview

### Design Philosophy

1. **Provenance-First**: Every knowledge artifact must be traceable to its exact source
2. **Temporal-Aware**: Track both when facts were true (event time) and when we learned them (transaction time)
3. **Multi-Resolution**: Support querying at multiple levels of abstraction from the same source
4. **Re-extraction Friendly**: Enable reprocessing without re-ingesting source documents

### Core Principles (Phase 2 Updated)

- **Source as Primary Provenance**: All content originates from unified Source nodes (file/agentic/legacy)
- **Dual Linking**: Memories link to Source (always) AND Chunk (when chunked)
- **Chunk as Optional Metadata**: Location pointers within sources, not required provenance chain
- **Unified Provenance Model**: Files, agent contributions, and legacy data share common source abstraction
- **Normalized Graph**: Avoid redundancy; use relationships over denormalized properties
- **Bi-temporal Model**: Track t_valid (event time) and t_created (transaction time)
- **Quality Firewall**: Hygiene checks and deduplication prevent low-quality data

---

## Phase 2: Source Abstraction (Current State)

### Implementation

**Storage:** Dual-backend architecture
- **SQLite**: Relational tables with foreign keys
- **FalkorDB**: Graph-native with Cypher queries

**Provenance Model:** Source → Memory (always) AND Source → Chunk → Memory (when chunked)

**Source Types:**
- `file`: Document ingestion (path-based)
- `agentic`: AI agent contributions (agent name: claude-code, codex, goose, etc.)
- `legacy`: Backfilled orphans from housekeeping

### Schema (Phase 2 - SQLite)

```sql
-- Sources table (NEW in Phase 2)
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,        -- 'file' | 'agentic' | 'legacy'
    created_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    source_hash TEXT,
    metadata TEXT,
    -- Type-specific fields (nullable)
    path TEXT,                        -- For 'file'
    agent TEXT,                       -- For 'agentic'
    session_id TEXT                   -- Optional session tracking
);

-- Chunks table (UPDATED in Phase 2)
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,          -- Changed from file_id
    start INTEGER NOT NULL,
    length INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

-- Memories table (UPDATED in Phase 2)
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_embedding TEXT NOT NULL,
    content_hash TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL,
    -- Bi-temporal fields
    t_valid TEXT,
    t_invalid TEXT,
    t_created TEXT,
    t_expired TEXT,
    temporal_stability TEXT DEFAULT 'unknown',
    last_seen_at TEXT,
    reinforce_count INTEGER DEFAULT 0,
    kind TEXT DEFAULT 'MEMORY',       -- 'MEMORY' | 'SUMMARY'
    -- Phase 2: Dual linking
    chunk_id TEXT,                    -- Optional positional metadata
    source_id TEXT                    -- Primary provenance (ALWAYS set)
);

-- Entities table
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    norm_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    embedding TEXT,
    expired_at TEXT,
    merged_into TEXT,
    chunk_id TEXT                     -- Optional chunk link
);

-- Edges table
CREATE TABLE edges (
    edge_id TEXT PRIMARY KEY,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    edge_type TEXT NOT NULL,          -- MENTIONS | RELATED | SUMMARIZED_BY | CONTAINS
    weight REAL NOT NULL,
    confidence REAL,
    evidence TEXT,
    -- Bi-temporal fields
    t_valid TEXT,
    t_invalid TEXT,
    t_created TEXT,
    t_expired TEXT
);

-- Files table (DEPRECATED - kept for backward compatibility)
CREATE TABLE files (
    file_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    file_hash TEXT,
    metadata TEXT
);
```

### Phase 2 Provenance Model (Dual Linking)

**Key Concept:** Memories have TWO provenance links:

1. **Primary: source_id** → Always points to Source node (required)
2. **Secondary: chunk_id** → Points to Chunk node (optional, for chunked content)

**File Ingestion Example:**
```
Source{type='file', path='/path/to/doc.md'}
    └──→ Chunk{sequence=1, start=0, length=5000}
            └──→ Memory{content='...', source_id, chunk_id}
                    ↑
                    └── Both source_id AND chunk_id are set
```

**Agentic Content Example:**
```
Source{type='agentic', agent='claude-code', session_id='abc123'}
    └──→ Memory{content='...', source_id, chunk_id=NULL}
            ↑
            └── Only source_id is set (no chunking)
```

**Legacy/Orphan Example:**
```
Source{type='legacy', agent='unknown'}
    └──→ Memory{content='...', source_id, chunk_id=NULL}
            ↑
            └── Created by housekeeping backfill
```

**Benefits:**
✅ All content has provenance (no orphans)
✅ Session-level tracking for agentic sources
✅ Trust signals (file vs agentic vs legacy)
✅ Positional metadata when needed (chunks)
✅ Backward compatible (Files table deprecated but kept)

---

## FalkorDB Backend (Graph-Native Implementation)

### Status: Production

**Implemented:** January 2026 (Phase 2)
**Why:** Native graph traversal, Cypher query language, better relationship modeling, scalable querying

FalkorDB provides a graph-native backend alternative to SQLite, using edges for provenance relationships rather than foreign key properties.

### Node Types (Phase 2)

#### 1. SOURCE Node (Primary Provenance)

```cypher
(:Source {
  id: "source_<uuid>",
  source_type: "file",            // 'file' | 'agentic' | 'legacy'
  created_at: "2024-01-15T10:30:00Z",
  ingested_at: "2024-01-20T15:00:00Z",
  source_hash: "sha256_...",
  metadata: {...},
  // Type-specific fields (nullable)
  path: "/absolute/path/to/document.md",  // For 'file'
  agent: NULL,                              // For 'agentic': 'claude-code', 'codex', etc.
  session_id: NULL                          // Optional session tracking
})
```

**Purpose:** Unified provenance for all content origins. Replaces FILE nodes with support for multiple source types.

**Example - Agentic Source:**
```cypher
(:Source {
  id: "source_<uuid>",
  source_type: "agentic",
  agent: "claude-code",
  session_id: "abc123",
  created_at: "2026-01-11T14:30:00Z",
  ingested_at: "2026-01-11T14:30:00Z",
  metadata: {conversation_id: "xyz", turn: 42}
})
```

#### 2. CHUNK Node (Optional Positional Metadata)

```cypher
(:Chunk {
  id: "chunk_<uuid>",
  source_id: "source_<uuid>",  // Phase 2: references Source, not File
  start: 0,                     // Character position in source
  length: 3933,                 // Length in characters
  sequence: 1,                  // Position in document (1-indexed)
  created_at: "2024-01-20T15:00:00Z"
})
```

**Purpose:** Location pointers within sources. Provides positional metadata for chunked content (primarily for files). Optional - not all memories have chunks (e.g., agentic sources).

#### 3. MEMORY Node

```cypher
(:MEMORY {
  memory_id: "mem_<uuid>",
  content: "Matt Joyce developed Vestig as a chunk-centric knowledge graph system",
  content_hash: "sha256_...",
  created_at: "2024-01-20T15:00:00Z",
  -- Bi-temporal fields
  t_valid: "2024-01-15T00:00:00Z",      // When fact became true (event time)
  t_invalid: NULL,                       // When fact stopped being true
  t_created: "2024-01-20T15:00:00Z",    // When we learned it (transaction time)
  t_expired: NULL,                       // When deprecated/superseded
  temporal_stability: "static",          // "static" | "dynamic" | "ephemeral" | "unknown"
  -- Reinforcement tracking
  last_seen_at: "2024-01-25T10:00:00Z",
  reinforce_count: 3
})
```

**Purpose:** Granular facts with temporal metadata. Supports bi-temporal querying ("what did we know on date X?").

#### 4. ENTITY Node

```cypher
(:ENTITY {
  entity_id: "ent_<uuid>",
  entity_type: "PERSON",        // PERSON | ORG | SYSTEM | PROJECT | TOOL | etc.
  canonical_name: "Matt Joyce",
  norm_key: "person:matt joyce", // Deduplication key (type:normalized_name)
  created_at: "2024-01-20T15:00:00Z",
  embedding: [0.123, -0.456, ...],  // Optional: for semantic entity matching
  expired_at: NULL,
  merged_into: NULL              // If entity was merged into canonical
})
```

**Purpose:** Canonical entities extracted from chunks. Deduplicated via norm_key.

#### 5. SUMMARY Node

```cypher
(:SUMMARY {
  summary_id: "sum_<uuid>",
  content: "This chunk discusses the architecture of Vestig, a chunk-centric knowledge graph...",
  summary_type: "chunk",         // "chunk" | "document" | "multi-document"
  created_at: "2024-01-20T15:05:00Z",
  model: "gpt-5.1",
  prompt_version: "summary_v2"
})
```

**Purpose:** High-level overviews at chunk or document level. Enables fast gist retrieval.

### Edge Types

#### FILE → CHUNK: CONTAINS

```cypher
(FILE)-[:CONTAINS {
  sequence: 1,           // Chunk position in file
  extraction_date: "2024-01-20T15:00:00Z"
}]->(CHUNK)
```

**Purpose:** Link files to their constituent chunks, preserving document structure.

#### CHUNK → CHUNK: NEXT

```cypher
(CHUNK {sequence: 1})-[:NEXT]->(CHUNK {sequence: 2})
```

**Purpose:** Sequential linking for document flow navigation. Enables "show context before/after this chunk" queries.

#### CHUNK → MEMORY: EXTRACTED

```cypher
(CHUNK)-[:EXTRACTED {
  extraction_date: "2024-01-20T15:00:00Z",
  model: "gpt-5.1",
  prompt_version: "extract_memories_v4",
  confidence: 0.92
}]->(MEMORY)
```

**Purpose:** Track which chunk each memory was extracted from. Enables provenance queries.

#### CHUNK → ENTITY: EXTRACTED

```cypher
(CHUNK)-[:EXTRACTED {
  extraction_date: "2024-01-20T15:00:00Z",
  model: "gpt-5-nano",
  prompt_version: "extract_entities_v1",
  confidence: 0.85
}]->(ENTITY)
```

**Purpose:** Track entity extraction from original chunk text (not from memories). Ensures high-quality entity extraction.

#### CHUNK → SUMMARY: SUMMARIZED_BY

```cypher
(CHUNK)-[:SUMMARIZED_BY {
  created_at: "2024-01-20T15:05:00Z",
  model: "gpt-5.1",
  prompt_version: "summary_v2"
}]->(SUMMARY)
```

**Purpose:** Link chunks to their summaries for fast overview retrieval.

#### MEMORY → ENTITY: MENTIONS

```cypher
(MEMORY)-[:MENTIONS {
  confidence: 0.87,
  evidence: "Found in original chunk text via substring match",
  extraction_source: "chunk",  // "chunk" or "memory"
  t_valid: "2024-01-20T15:00:00Z",
  t_created: "2024-01-20T15:00:00Z"
}]->(ENTITY)
```

**Purpose:** Link memories to entities they mention. Created via substring matching after chunk-level entity extraction.

#### MEMORY → MEMORY: RELATED

```cypher
(MEMORY)-[:RELATED {
  similarity: 0.78,
  evidence: "semantic_similarity=0.780",
  t_valid: "2024-01-20T15:00:00Z"
}]->(MEMORY)
```

**Purpose:** Semantic similarity edges for memory graph traversal.

---

## Graph Model

### Visual Representation

```
┌─────────────┐
│    FILE     │
│  (document) │
└──────┬──────┘
       │ [:CONTAINS]
       ↓
┌─────────────┐      [:NEXT]      ┌─────────────┐
│   CHUNK₁    │ ──────────────→   │   CHUNK₂    │
│ (start:0)   │                   │ (start:3533)│
└──────┬──────┘                   └──────┬──────┘
       │                                  │
       ├─[:EXTRACTED]────→ ┌──────────┐  │
       │                   │  MEMORY  │  │
       │                   └────┬─────┘  │
       │                        │[:MENTIONS]
       │                        ↓        │
       ├─[:EXTRACTED]────→ ┌──────────┐ │
       │                   │  ENTITY  │←┘
       │                   └──────────┘
       │
       └─[:SUMMARIZED_BY]→ ┌──────────┐
                           │  SUMMARY │
                           └──────────┘
```

### Example Query Patterns

#### 1. Provenance Query: "Where did this memory come from?"

```cypher
MATCH (m:MEMORY {memory_id: "mem_abc123"})
      <-[:EXTRACTED]-(c:CHUNK)
      <-[:CONTAINS]-(f:FILE)
RETURN f.path, c.start, c.length, c.text
```

#### 2. Entity-Centric Query: "Show all memories mentioning Matt Joyce"

```cypher
MATCH (e:ENTITY {canonical_name: "Matt Joyce"})
      <-[:MENTIONS]-(m:MEMORY)
      <-[:EXTRACTED]-(c:CHUNK)
RETURN m.content, m.t_valid, c.start, c.length
ORDER BY m.t_valid DESC
```

#### 3. Multi-Resolution Query: "Get entity, summary, and memories from chunk"

```cypher
MATCH (c:CHUNK {chunk_id: "chunk_xyz"})
OPTIONAL MATCH (c)-[:EXTRACTED]->(e:ENTITY)
OPTIONAL MATCH (c)-[:SUMMARIZED_BY]->(s:SUMMARY)
OPTIONAL MATCH (c)-[:EXTRACTED]->(m:MEMORY)
RETURN c, collect(e) as entities, s, collect(m) as memories
```

#### 4. Temporal Query: "What did we know about Vestig on 2024-01-15?"

```cypher
MATCH (m:MEMORY)-[:MENTIONS]->(e:ENTITY {canonical_name: "Vestig"})
WHERE m.t_valid <= datetime("2024-01-15T23:59:59Z")
  AND (m.t_invalid IS NULL OR m.t_invalid > datetime("2024-01-15T23:59:59Z"))
  AND m.t_created <= datetime("2024-01-15T23:59:59Z")
RETURN m.content, m.t_valid, m.temporal_stability
ORDER BY m.t_valid DESC
```

#### 5. Context Navigation: "Show chunk context (before/after)"

```cypher
MATCH (c:CHUNK {chunk_id: "chunk_xyz"})
OPTIONAL MATCH (prev:CHUNK)-[:NEXT]->(c)
OPTIONAL MATCH (c)-[:NEXT]->(next:CHUNK)
RETURN prev, c, next
```

#### 6. Document Structure: "Show all chunks from this file in order"

```cypher
MATCH (f:FILE {path: "/path/to/doc.md"})-[:CONTAINS]->(c:CHUNK)
RETURN c.chunk_id, c.sequence, c.start, c.length
ORDER BY c.sequence
```

---

## Architectural Rationale

### Why Chunk-Centric vs. Entity-Centric?

Most knowledge graph systems (GraphRAG, LlamaIndex, Neo4j+LangChain) use **entity-centric** architectures where entities are hub nodes. Vestig chooses **chunk-centric** for the following reasons:

#### 1. **Provenance is Fundamental**

**Problem:** In entity-centric systems, provenance is weak. You know an entity was mentioned, but not exactly where.

**Solution:** Chunks are anchored to exact source locations (`file:start:length`). Every memory, entity, and summary traces back to precise text.

**Benefit:**
- "Show me the exact text where this entity was extracted"
- "Re-extract entities from this chunk using updated prompt"
- "Debug why this memory was created"

#### 2. **Multi-Resolution Querying**

**Problem:** Entity-centric systems force you to choose: extract entities OR extract facts/summaries.

**Solution:** Chunks support multiple abstraction levels from the same source:
- **Entities**: What's mentioned? (nouns, concepts)
- **Summaries**: What's the gist? (high-level overview)
- **Memories**: What are the details? (granular facts with temporal data)

**Benefit:**
- Fast overview: Query summaries
- Detailed analysis: Query memories
- Entity tracking: Query entities
- All from the same chunk with shared provenance

#### 3. **Temporal Tracking**

**Problem:** Most knowledge graphs ignore time. Facts are timeless, but reality changes.

**Solution:** Memories carry bi-temporal metadata:
- `t_valid`: When the fact became true (event time)
- `t_created`: When we learned it (transaction time)
- `temporal_stability`: Classification (static, dynamic, ephemeral)

**Benefit:**
- "What did we know on date X?" (as-of queries)
- "When did this fact become true?" (event time)
- "Track opinion changes over time" (temporal stability)

Chunks anchor these temporal memories to source documents, enabling "show me the document version from which this temporal fact was extracted."

#### 4. **Re-extraction Without Re-ingestion**

**Problem:** When extraction prompts improve, entity-centric systems require full document re-ingestion.

**Solution:** Chunks store references to source text. Can re-extract entities or memories from chunks without re-reading files.

**Benefit:**
- "Re-extract entities from all chunks using new ontology"
- "Re-run memory extraction with improved prompt"
- "Partial updates without full pipeline re-run"

#### 5. **Document Structure Preservation**

**Problem:** Entity-centric systems lose document narrative flow.

**Solution:** Chunks link sequentially (`:NEXT` edges), preserving document structure.

**Benefit:**
- "Show context before and after this chunk"
- "Navigate document flow: previous/next chunk"
- "Understand narrative arc, not just isolated facts"

### Why FILE as Separate Node?

**Alternative:** Store file_path as property on CHUNK.

**Chosen:** FILE as separate node linked via `[:CONTAINS]` edge.

**Rationale:**

1. **Normalization**: File path stored once, not repeated across all chunks
2. **File-Level Metadata**: Track creation date, author, tags, ingestion version
3. **File-Centric Queries**: "Show all files ingested last week"
4. **File Lifecycle**: Handle renames, moves, re-ingestion without updating all chunks
5. **Graph-Native Modeling**: FILE is logically a separate entity with its own lifecycle

**Trade-off:** One extra hop in queries (`MEMORY → CHUNK → FILE`), but negligible in Neo4j and benefits outweigh cost.

### Why Both Summaries AND Memories?

**Question:** Why not just summaries (faster) or just memories (detailed)?

**Answer:** Different use cases require different granularity.

**Summaries (Chunk-level):**
- **Use Case**: Fast overview, gist retrieval, exploratory queries
- **Example**: "What's this chunk about?"
- **Trade-off**: Less detail, but faster retrieval

**Memories (Granular):**
- **Use Case**: Precise facts, temporal tracking, detailed analysis
- **Example**: "When did Matt Joyce start Vestig?" (needs temporal precision)
- **Trade-off**: More nodes, but supports bi-temporal queries

**Hybrid Retrieval:**
1. Vector search → find relevant chunks
2. Return summaries for quick scan
3. User drills down → retrieve memories for details
4. Expand via entities → find related chunks

### Why Bi-temporal Model?

**Event Time (t_valid):** When the fact became true in reality
**Transaction Time (t_created):** When we learned about it

**Example:**
- Matt Joyce started Vestig on 2024-01-15 (event time: t_valid)
- We learned this on 2024-01-20 from a document (transaction time: t_created)

**Queries Enabled:**
- **As-of queries**: "What did we know on 2024-01-18?" (filter by t_created)
- **Event-time queries**: "What was true on 2024-01-18?" (filter by t_valid)
- **Audit trail**: "When did we first learn about this fact?" (t_created)

**Inspiration:** Bi-temporal databases (Datomic, temporal SQL extensions), knowledge graph versioning research.

---

## Comparison to Other Approaches

| Feature | GraphRAG (MS) | LlamaIndex KG | Neo4j+Lang | Diffbot | **Vestig** |
|---------|---------------|---------------|------------|---------|------------|
| **Hub Node** | Community | Entity | Entity | Entity | **Chunk** |
| **Provenance** | Weak | Medium | Medium | Strong | **Strong** |
| **Temporal** | No | No | No | Some | **Bi-temporal** |
| **Granularity** | Communities | Triplets | Chunks | Facts | **Multi-level** |
| **Doc Flow** | Lost | Lost | Lost | Lost | **Preserved** |
| **Re-extraction** | Full re-ingest | Full re-ingest | Difficult | N/A | **Chunk-level** |
| **File Metadata** | No | No | Some | Some | **Full** |
| **Multi-resolution** | No | No | No | No | **Yes** |

### Unique Value Proposition

**Vestig is the only system that combines:**
1. Chunk-centric hub architecture (provenance-first)
2. Bi-temporal tracking (event time vs. transaction time)
3. Multi-resolution querying (entity/summary/memory from same chunk)
4. Document structure preservation (sequential chunk linking)
5. Re-extraction without re-ingestion (chunk-level reprocessing)

**Closest comparisons:**
- **Provenance**: Similar to Diffbot's evidence linking
- **Temporal**: Similar to bi-temporal databases (Datomic)
- **Chunk-centric**: Similar to HippoRAG's memory encoding, but more comprehensive

**Research-grade architecture** suitable for:
- Knowledge work requiring audit trails
- Temporal reasoning ("what changed when?")
- Provenance-critical applications (research, legal, medical)
- Evolving knowledge bases (opinions, predictions, time-sensitive facts)

---

## Migration Path

### Phase 1: Current (SQLite with String References)

**Status:** ✅ Implemented (as of 2025-01-04)

```python
# Chunk reference stored in memory metadata
memory.metadata = {
    "source": "document_ingest",
    "chunk_ref": "/path/to/file.md:0:3933"
}
```

**Benefits:**
- No schema changes required
- Functional provenance tracking
- Neo4j-migration-ready (parseable strings)

**Limitations:**
- No CHUNK or FILE nodes
- Limited querying (string matching only)
- No chunk-to-chunk linking

### Phase 2: Add CHUNK Nodes to SQLite (Optional Intermediate)

**Status:** ⏸️ Deferred (waiting for Neo4j migration)

**If needed, could add:**

```sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    start_pos INTEGER NOT NULL,
    length INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    chunk_text TEXT,  -- Optional: store for re-extraction
    created_at TEXT NOT NULL,
    UNIQUE(file_path, start_pos)
);

CREATE TABLE files (
    file_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    modified_at TEXT,
    ingested_at TEXT NOT NULL
);
```

**Benefits:**
- Start building chunk-centric graph in SQLite
- Cleaner queries (JOIN vs. string parsing)
- Easier Neo4j migration

**Cons:**
- Schema changes to existing database
- Migration needed for existing chunk_ref strings
- Additional complexity before Neo4j

### Phase 3: Neo4j Migration (Target)

**Status:** 🎯 Planned (Q1 2025)

**Migration Script (Conceptual):**

```python
def migrate_sqlite_to_neo4j(sqlite_db, neo4j_session):
    """
    Migrate SQLite data to Neo4j graph model.
    """
    # Step 1: Create FILE nodes from unique chunk_refs
    chunk_refs = sqlite_db.execute("""
        SELECT DISTINCT json_extract(metadata, '$.chunk_ref')
        FROM memories
        WHERE json_extract(metadata, '$.chunk_ref') IS NOT NULL
    """).fetchall()

    file_nodes = {}
    chunk_nodes = {}

    for (chunk_ref,) in chunk_refs:
        file_path, start, length = parse_chunk_ref(chunk_ref)

        # Create or get FILE node
        if file_path not in file_nodes:
            file_id = neo4j_session.run("""
                MERGE (f:FILE {path: $path})
                ON CREATE SET f.file_id = randomUUID(),
                             f.ingested_at = datetime()
                RETURN f.file_id
            """, path=file_path).single()["file_id"]
            file_nodes[file_path] = file_id

        # Create CHUNK node
        chunk_id = f"chunk_{uuid.uuid4()}"
        neo4j_session.run("""
            CREATE (c:CHUNK {
                chunk_id: $chunk_id,
                start: $start,
                length: $length
            })
        """, chunk_id=chunk_id, start=start, length=length)

        chunk_nodes[chunk_ref] = chunk_id

        # Link FILE → CHUNK
        neo4j_session.run("""
            MATCH (f:FILE {file_id: $file_id})
            MATCH (c:CHUNK {chunk_id: $chunk_id})
            CREATE (f)-[:CONTAINS]->(c)
        """, file_id=file_nodes[file_path], chunk_id=chunk_id)

    # Step 2: Create CHUNK → CHUNK :NEXT edges (sequential linking)
    # Group chunks by file, sort by start position, link sequentially
    for file_path in file_nodes.keys():
        file_chunks = [(ref, chunk_nodes[ref]) for ref in chunk_nodes
                       if ref.startswith(file_path)]
        # Sort by start position
        file_chunks.sort(key=lambda x: int(x[0].split(':')[1]))

        # Create :NEXT edges
        for i in range(len(file_chunks) - 1):
            neo4j_session.run("""
                MATCH (c1:CHUNK {chunk_id: $chunk_id1})
                MATCH (c2:CHUNK {chunk_id: $chunk_id2})
                CREATE (c1)-[:NEXT]->(c2)
            """, chunk_id1=file_chunks[i][1], chunk_id2=file_chunks[i+1][1])

    # Step 3: Migrate MEMORY nodes
    memories = sqlite_db.execute("SELECT * FROM memories").fetchall()
    for memory in memories:
        chunk_ref = json.loads(memory['metadata']).get('chunk_ref')
        chunk_id = chunk_nodes.get(chunk_ref)

        neo4j_session.run("""
            CREATE (m:MEMORY {
                memory_id: $memory_id,
                content: $content,
                content_hash: $content_hash,
                created_at: $created_at,
                t_valid: $t_valid,
                t_created: $t_created,
                temporal_stability: $temporal_stability
            })
        """, memory_id=memory['id'], content=memory['content'], ...)

        # Link CHUNK → MEMORY
        if chunk_id:
            neo4j_session.run("""
                MATCH (c:CHUNK {chunk_id: $chunk_id})
                MATCH (m:MEMORY {memory_id: $memory_id})
                CREATE (c)-[:EXTRACTED]->(m)
            """, chunk_id=chunk_id, memory_id=memory['id'])

    # Step 4: Migrate ENTITY nodes (same pattern)
    # Step 5: Migrate EDGE relationships (MENTIONS, RELATED, etc.)
    # Step 6: Create indexes for performance
    neo4j_session.run("CREATE INDEX file_path FOR (f:FILE) ON (f.path)")
    neo4j_session.run("CREATE INDEX memory_id FOR (m:MEMORY) ON (m.memory_id)")
    neo4j_session.run("CREATE INDEX entity_name FOR (e:ENTITY) ON (e.canonical_name)")
```

**Post-migration:**
- Verify data integrity (count nodes/edges)
- Performance test common queries
- Create indexes on frequently queried properties
- Deprecate SQLite database or keep as archival backup

---

## Benefits & Trade-offs

### Benefits

#### 1. Full Provenance Chain
```
User asks: "Where did this fact come from?"
Answer: MEMORY → CHUNK → FILE
Result: "Extracted from file X, characters 1000-4000, on line 45"
```

#### 2. Temporal Reasoning
```
User asks: "What did we know about AI safety on 2024-01-15?"
Query: Filter memories by t_created <= 2024-01-15
Result: As-of snapshot of knowledge at that date
```

#### 3. Multi-Resolution Retrieval
```
User asks: "Tell me about Vestig"
Step 1: Find entity "Vestig" → get chunks
Step 2: Fast scan: Read chunk summaries
Step 3: Details: Retrieve memories from relevant chunks
Step 4: Expand: Follow entity links to related chunks
```

#### 4. Re-extraction Efficiency
```
Scenario: Improved entity extraction prompt
Action: Re-extract entities from chunks (not full re-ingestion)
Benefit: Faster iteration on extraction quality
```

#### 5. Document Context Navigation
```
User reads memory from chunk 5
Action: Show context → retrieve chunks 4, 5, 6
Benefit: Understand narrative flow, not isolated facts
```

### Trade-offs

#### 1. Graph Complexity
- **More nodes/edges** than entity-centric (chunks + entities + memories + summaries)
- **Mitigation**: Neo4j handles millions of nodes efficiently; complexity is managed via indexes

#### 2. Query Complexity
- **Multi-hop traversals** may be slower than direct lookups
- **Mitigation**: Neo4j optimized for graph traversal; use indexes on hub nodes (CHUNK)

#### 3. Storage Overhead
- **Chunk text storage** (optional) increases database size
- **Mitigation**: Store chunk_ref only; re-read from file when needed

#### 4. Chunk Boundary Issues
- **Semantic units may span chunks** (e.g., sentence split across chunks)
- **Mitigation**: Overlap (400 chars) + sequential linking helps preserve context

#### 5. Learning Curve
- **Chunk-centric is unconventional** compared to entity-centric RAG
- **Mitigation**: Clear documentation (this file), query pattern examples

### When to Use This Architecture

**Good fit:**
- Knowledge work requiring audit trails (research, legal, compliance)
- Temporal reasoning ("what changed when?")
- Provenance-critical applications (medical, scientific)
- Evolving knowledge bases (opinions, predictions, time-sensitive facts)
- Large document collections with version tracking needs

**Poor fit:**
- Simple entity extraction without provenance needs
- Purely semantic search (vector DB sufficient)
- Static knowledge (Wikipedia-like, no temporal changes)
- Real-time streaming data (graph writes may be bottleneck)

---

## Conclusion

Vestig's chunk-centric architecture represents a novel approach to knowledge graph design, prioritizing **provenance**, **temporal tracking**, and **multi-resolution querying** over traditional entity-centric models.

**Key Innovation:** CHUNK as hub node enables traceability from any knowledge artifact (memory, entity, summary) back to exact source locations while supporting multiple abstraction levels.

**Target State:** Neo4j graph database with FILE → CHUNK → {MEMORY, ENTITY, SUMMARY} structure, bi-temporal tracking, and sequential chunk linking.

**Current Status:** SQLite implementation with string-based chunk references (migration-ready).

**Next Steps:**
1. Continue building on SQLite with chunk_ref strings
2. Plan Neo4j migration (Q1 2025)
3. Develop migration scripts and test queries
4. Evaluate performance at scale (millions of nodes)

---

**Document Version:** 1.0
**Author:** Matt Joyce (with Claude Sonnet 4.5)
**Last Updated:** 2025-01-04
