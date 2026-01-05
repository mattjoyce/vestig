# Chunk-Centric Knowledge Graph Architecture

**Version:** 1.0  
**Target:** SQLite3 Implementation  
**Status:** Initial Build

---

## Core Concept

Build a **chunk-centric knowledge graph** where document chunks act as hub nodes connecting source files to extracted knowledge (memories, entities, summaries).

**Key Principle:** Everything traces back to exact locations in source documents through chunks.

---

## The Hub-and-Spoke Model

```
                    FILE
                     │
                     ↓
                ┌─────────┐
                │  CHUNK  │ ← Hub Node (source location pointer)
                └─────────┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
    MEMORY       ENTITY       SUMMARY
  (facts)      (entities)   (overview)
```

**CHUNK is the hub** - all extracted artifacts connect through it, not to each other.

---

## Node Types

### 1. FILE
Represents a source document.

**Properties:**
- `file_id` - Unique identifier
- `path` - Absolute file path
- `created_at` - When file was created
- `ingested_at` - When file was processed

**Purpose:** Track source documents and file-level metadata.

---

### 2. CHUNK (Hub Node)
A pointer to a specific location in a file. Does not store the text itself.

**Properties:**
- `chunk_id` - Unique identifier
- `file_id` - Which file this chunk belongs to
- `start` - Character position in file where chunk starts
- `length` - Number of characters in chunk
- `sequence` - Position in document (1st chunk, 2nd chunk, etc.)

**Purpose:** Central connection point that preserves exact source location. Enables retrieval of original text when needed.

**Critical:** Chunk text is NOT stored in the database - only the pointer (file + start + length).

---

### 3. MEMORY
Extracted facts or statements from a chunk.

**Properties:**
- `memory_id` - Unique identifier
- `content` - The actual fact/statement text
- `embedding` - Vector representation (for semantic search)
- `chunk_id` - Which chunk this was extracted from

**Purpose:** Granular, searchable facts with full provenance back to source.

---

### 4. ENTITY
Extracted entities (people, organizations, concepts) from a chunk.

**Properties:**
- `entity_id` - Unique identifier
- `name` - Entity name (e.g., "Matt Joyce", "AI governance")
- `type` - Entity category (PERSON, ORG, CONCEPT, etc.)
- `embedding` - Vector representation (for semantic matching)
- `chunk_id` - Which chunk this was extracted from

**Purpose:** Enable entity-based connections between chunks. Same entity appearing in different chunks creates implicit links.

---

### 5. SUMMARY
High-level overview of a chunk's content.

**Properties:**
- `summary_id` - Unique identifier
- `content` - Summary text
- `embedding` - Vector representation (for semantic search)
- `chunk_id` - Which chunk this summarizes

**Purpose:** Fast gist retrieval without reading full chunk text. Primary entry point for vector search.

---

## Relationships

All relationships flow **through CHUNK**. There are no direct connections between MEMORY, ENTITY, and SUMMARY nodes.

### Direct Relationships (Stored in Database)

```
FILE → CHUNK       (file contains chunks)
CHUNK → MEMORY     (chunk has extracted memories)
CHUNK → ENTITY     (chunk mentions entities)
CHUNK → SUMMARY    (chunk has summary)
```

### Implied Relationships (Derived via Traversal)

```
SUMMARY → ENTITY   (traverse: SUMMARY → CHUNK → ENTITY)
MEMORY → ENTITY    (traverse: MEMORY → CHUNK → ENTITY)
CHUNK → CHUNK      (via shared entities: CHUNK → ENTITY ← CHUNK)
```

---

## SQLite3 Schema Overview

**Tables:**
- `files` - Source documents
- `chunks` - Location pointers (file_id, start, length)
- `memories` - Extracted facts with embeddings
- `entities` - Extracted entities with embeddings
- `summaries` - Chunk overviews with embeddings

**Key Design Decisions:**

1. **No chunk text in database** - Store only `(file_id, start, length)` pointer
2. **All embeddings stored** - On memories, entities, and summaries for vector search
3. **Hub constraint** - All foreign keys point to `chunk_id`, not to each other
4. **Simple schema** - Start minimal, add complexity only if needed

---

## Data Flow

### Ingestion Pipeline

```
1. Load document → Create FILE record
                     ↓
2. Split document → Create CHUNK records (pointers only!)
                     ↓
3. For each chunk:
   a. Extract entities → Create ENTITY records → Link to CHUNK
   b. Generate summary → Create SUMMARY record → Link to CHUNK
   c. Extract memories → Create MEMORY records → Link to CHUNK
   d. Generate embeddings for all extracted content
```

### Retrieval Modes

**Recall Mode** - Find specific facts with full source context:
```
Query → Search SUMMARY embeddings → Get CHUNK pointer → 
Get MEMORYs from that CHUNK → Retrieve original text from file
```

**Expand Mode** - Discover related information:
```
Query → Search SUMMARY embeddings → Get CHUNK → 
Get ENTITYs from that CHUNK → Find other CHUNKs with same entities →
Get MEMORYs from those related CHUNKs
```

---

## Why This Architecture?

### 1. **Full Provenance**
Every fact traces back to exact character position in source file:
```
MEMORY → CHUNK → FILE (path:start:length)
```

### 2. **Storage Efficiency**
Chunk text stored once in original file, not duplicated in database. Only pointers stored.

### 3. **Flexible Retrieval**
- Vector search on summaries (fast, broad semantic match)
- Vector search on entities (concept-level fuzzy matching)
- Vector search on memories (precise fact matching)
- Graph traversal for entity-based expansion

### 4. **Re-extraction Friendly**
Improve extraction prompts and re-process without re-ingesting files. Chunk pointers remain stable.

### 5. **Multi-Resolution**
Query at different granularities:
- Quick scan: Read summaries
- Detailed: Read memories
- Full context: Retrieve original chunk text from file

---

## Implementation Phases

### Phase 1: Core Tables (SQLite3)
- Create tables: files, chunks, memories, entities, summaries
- Implement chunk pointer system (file_id + start + length)
- Basic insert and retrieval functions

### Phase 2: Vector Search
- Add embedding columns
- Implement vector similarity search (using sqlite-vec or similar)
- Test semantic retrieval

### Phase 3: Graph Traversal
- Implement entity-based chunk expansion
- Test recall and expand retrieval modes
- Measure retrieval quality

### Phase 4: Evaluation
- Test with 2WikiMultiHopQA dataset or real documents
- Measure: precision, recall, storage efficiency
- Decide: Continue with SQLite3 or migrate to FalkorDB?

---

## Success Criteria

**The architecture is successful if:**

1. ✅ Can trace any fact back to exact source location
2. ✅ Storage footprint is reasonable (no chunk text duplication)
3. ✅ Retrieval finds relevant context across documents
4. ✅ Entity-based expansion discovers non-obvious connections
5. ✅ Re-extraction works without breaking existing data

---

## Future Migration Path

If SQLite3 implementation proves successful, migrate to **FalkorDB** for:
- Native graph traversal (vs. JOIN operations)
- Better performance at scale
- Built-in vector search optimization
- More natural graph query language

**Migration preserves the same hub-and-spoke model** - just moves from relational tables to graph nodes/edges.

---

## Key Constraints

**Remember:**
1. **Chunk text is NOT stored** - only pointers (file_id, start, length)
2. **All artifacts connect through CHUNK** - no sibling relationships
3. **Embeddings on everything** - memories, entities, summaries all get vectors
4. **Hub model is sacred** - don't shortcut by linking memories directly to entities

---

## Questions for Implementation

1. **Chunking strategy:** Fixed size? Semantic boundaries? Overlap?
2. **Entity extraction:** LLM-based or NER model?
3. **Vector embeddings:** Which model? (OpenAI, local, etc.)
4. **Vector search:** sqlite-vec, external service, or custom implementation?
5. **Deduplication:** How to handle same entity appearing multiple times?

---

**Document Author:** Matt Joyce  
**Date:** 2025-01-05  
**Purpose:** Developer guide for initial SQLite3 implementation
