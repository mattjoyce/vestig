# Chunk-Centric Knowledge Graph Architecture

**Version:** 1.1  
**Target:** FalkorDB Implementation  
**Status:** Production Baseline (M4)

---

## Core Concept

Build a **chunk-centric knowledge graph** where chunks act as hub nodes connecting sources to extracted knowledge (memories, entities, summaries).

**Key Principle:** Everything traces back to provenance via Source nodes, with optional positional context via Chunk nodes.

---

## The Hub-and-Spoke Model

```
                 SOURCE (type=file|agentic|legacy)
                             │
                             │ PRODUCED
                             ▼
                          MEMORY (kind=MEMORY)
                             │
                             │ MENTIONS
                             ▼
                           ENTITY

             SOURCE ──HAS_CHUNK──▶ CHUNK ──CONTAINS──▶ MEMORY
                                   │
                                   │ LINKED
                                   ▼
                                 ENTITY

             CHUNK ──SUMMARIZED_BY──▶ SUMMARY (Memory kind=SUMMARY)
             SUMMARY ──SUMMARIZES──▶ MEMORY
```

**CHUNK is the positional hub** - all extracted artifacts can be traced to source locations through it.

---

## Node Types

### 1. SOURCE
Represents a content origin.

**Properties:**
- `id` - Unique identifier
- `source_type` - `file` | `agentic` | `legacy`
- `path` (file sources)
- `agent`, `session_id` (agentic sources)
- `created_at`, `ingested_at`, `source_hash`, `metadata`

**Purpose:** Unified provenance for all content origins.

---

### 2. CHUNK (Positional Hub)
A pointer to a specific location in a source.

**Properties:**
- `id`
- `source_id` - Which Source this chunk belongs to
- `start` - Character position
- `length` - Characters in chunk
- `sequence` - Position in source

**Purpose:** Preserve exact source location without storing raw chunk text.

---

### 3. MEMORY
Extracted facts or statements.

**Properties:**
- `id`
- `content`
- `content_embedding`
- `content_hash`
- `created_at`
- temporal fields (`t_valid`, `t_created`, `t_expired`, `temporal_stability`, etc.)
- `metadata` (source tags, etc.)

**Purpose:** Granular, searchable facts with full provenance.

---

### 4. SUMMARY (Memory kind=SUMMARY)
Summaries are stored as Memory nodes with `kind="SUMMARY"`.

**Purpose:** Fast gist retrieval and chunk-level expansion.

---

### 5. ENTITY
Canonical entities extracted from memories or chunks.

**Properties:**
- `id`, `entity_type`, `canonical_name`, `norm_key`
- `created_at`, `embedding`, `expired_at`, `merged_into`

**Purpose:** Link memories and chunks via shared entities.

---

### 6. EVENT
Lifecycle and audit trail events.

**Properties:**
- `id`, `memory_id`, `event_type`, `occurred_at`, `source`, `actor`, `payload`

**Purpose:** TraceRank and observability.

---

## Relationships

**Provenance**
- `Source -[:PRODUCED]-> Memory`
- `Source -[:HAS_CHUNK]-> Chunk`
- `Chunk -[:CONTAINS]-> Memory`

**Entity graph**
- `Memory -[:MENTIONS]-> Entity`
- `Chunk -[:LINKED]-> Entity`
- `Memory -[:RELATED]-> Memory`

**Summaries**
- `Summary -[:SUMMARIZES]-> Memory`
- `Chunk -[:SUMMARIZED_BY]-> Summary`

**Events**
- `Event -[:AFFECTS]-> Memory`

---

## Data Flow

### Ingestion Pipeline

```
1. Load artifact → Create SOURCE node
2. Chunk text → Create CHUNK nodes
3. Extract memories → Create MEMORY nodes
4. Link provenance → PRODUCED + CONTAINS edges
5. Extract entities → MENTIONS (Memory) + LINKED (Chunk)
6. Generate summaries (>=2 memories per chunk)
7. Store events → Event nodes with AFFECTS edges
```

### Retrieval Mode (Recall)

```
Query → Search SUMMARY embeddings → Expand to CHUNK → Pull all MEMORY in chunk → Re-rank by similarity + TraceRank
```

---

## Why This Architecture?

1. **Full Provenance**
   `MEMORY → CHUNK → SOURCE` preserves exact source location and origin.

2. **Storage Efficiency**
   Chunk text stays in the original artifact, not duplicated in the graph.

3. **Flexible Retrieval**
   Summary-first retrieval keeps recall fast and expandable.

4. **Graph-Native**
   FalkorDB supports vector search + traversal in one store.

---

## Future Evolution

- Expand entity-based retrieval in recall
- Add richer traversal and graph scoring (M5+)
- Continue hardening provenance and observability
