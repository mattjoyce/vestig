# Vestig Architecture (FalkorDB-Only)

**Version:** 2.2  
**Status:** Production baseline (M4), M5 in progress  
**Last Updated:** 2026-01-17

---

## Purpose
Vestig is a local-first memory system for agents. It ingests artifacts, extracts structured memories, and stores them in a graph for retrieval and provenance.

Key priorities:
- **Earn complexity**: ship minimal end-to-end slices, then deepen.
- **Stable interfaces**: CLI contract + schema are guarded.
- **Content hygiene**: quality gates at the boundary.
- **Observability**: behavior is inspectable via CLI and graph queries.

---

## Current Scope (M4)
- FalkorDB graph storage with vector search
- Source abstraction (file/agentic/legacy)
- Chunk-based provenance
- Entity extraction and graph edges
- Summary generation per chunk
- TraceRank temporal scoring

M5 work is in progress: advanced retrieval, scoring, and traversal behaviors.

---

## Graph Model

### Node Types

**Source**
- `source_type`: `file` | `agentic` | `legacy`
- `path` (file), `agent` + `session_id` (agentic)
- `created_at`, `ingested_at`, `source_hash`, `metadata`

**Chunk**
- `source_id`, `start`, `length`, `sequence`, `created_at`

**Memory**
- `id`, `content`, `content_embedding`, `content_hash`, `created_at`
- `metadata` (source, tags)
- Temporal fields: `t_valid`, `t_invalid`, `t_created`, `t_expired`, `temporal_stability`
- Reinforcement fields: `last_seen_at`, `reinforce_count`
- `kind`: `MEMORY` (default) or `SUMMARY`

**Entity**
- `entity_type`, `canonical_name`, `norm_key`
- `embedding`, `created_at`, `expired_at`, `merged_into`

**Event**
- `event_type` (ADD, REINFORCE_EXACT, REINFORCE_NEAR, DEPRECATE, SUMMARY_CREATED)
- `occurred_at`, `source`, `actor`, `artifact_ref`, `payload`

**File (deprecated)**
- Retained only for migration/backward compatibility. New writes use `Source`.

### Edge Types

**Provenance**
- `Source -[:PRODUCED]-> Memory` (and summaries)
- `Source -[:HAS_CHUNK]-> Chunk`
- `Chunk -[:CONTAINS]-> Memory`

**Entities**
- `Memory -[:MENTIONS]-> Entity`
- `Chunk -[:LINKED]-> Entity`
- `Memory -[:RELATED]-> Memory`

**Summaries**
- `Summary -[:SUMMARIZES]-> Memory`
- `Chunk -[:SUMMARIZED_BY]-> Summary`
- `Chunk -[:CONTAINS]-> Summary`

**Events**
- `Event -[:AFFECTS]-> Memory`

Schema constraints and indexes are initialized in `src/vestig/core/db_falkordb.py`. The reference schema is in `src/vestig/core/schema_falkor.cypher`.

---

## Provenance Model (Phase 2)

**Dual linking** keeps provenance explicit and positional metadata optional:
- All content is linked to a `Source` via `PRODUCED`.
- Chunked content adds a positional link via `Chunk` and `CONTAINS`.

Examples:
- File ingest: `Source(file)` → `Chunk` → `Memory` (plus `Source` → `Memory`)
- Agentic add: `Source(agentic)` → `Memory` (no chunk)
- Orphan backfill: `Source(legacy)` → `Memory`

### Design Decisions (Issue #6)

1. **Chunk remains a first-class node** - Enables entity→chunk linking and clean separation of provenance (Source) from position (Chunk).

2. **Only file sources support chunking** - Agentic and legacy sources are atomic; long conversations should be separate Sources rather than chunked.

3. **Edge naming: PRODUCED** - Source → Memory edges use `PRODUCED` (agent/system produced this memory).

4. **Summary generation is per-source-type:**
   - `file`: Yes (summarize document chunks)
   - `agentic`: No (agent messages are already atomic)
   - `legacy`: No (orphans are usually atomic)

5. **Future: Direct entity extraction from Chunks** - Extract entities from raw Chunk text (0 LLM hops) for higher trust. Currently entities are extracted from Memory nodes (1 LLM hop).

---

## Ingestion Pipeline

1. **Normalize input** (plain text or Claude session JSONL).
2. **Chunk** content by character count (configurable overlap).
3. **Extract memories** via LLM prompts.
4. **Commit** with hygiene + dedupe:
   - exact hash dedupe
   - near-duplicate reinforcement
5. **Create provenance edges** (`PRODUCED`, `CONTAINS`).
6. **Extract entities** (LLM-based) and write `MENTIONS` + `LINKED` edges.
7. **Generate summaries** per chunk (when >=2 memories) and write `SUMMARIZES` + `SUMMARIZED_BY`.
8. **Log events** (`Event` nodes with `AFFECTS`).

Key files:
- `src/vestig/core/ingestion.py`
- `src/vestig/core/commitment.py`
- `src/vestig/core/ingest_sources.py`

---

## Retrieval Pipeline

### search_memories()
- Native vector search via FalkorDB
- Optional hybrid entity path (entity extraction + matching)
- TraceRank temporal multiplier

### memory recall (CLI)
- Uses `recall_with_chunk_expansion()`:
  1. search summaries (kind=SUMMARY)
  2. expand to their chunks
  3. re-rank all memories by similarity
  4. apply TraceRank

Key files:
- `src/vestig/core/retrieval.py`
- `src/vestig/core/tracerank.py`

---

## Temporal Model (M3)

Each memory is bi-temporal:
- `t_valid`, `t_invalid` (event time)
- `t_created`, `t_expired` (transaction time)
- `temporal_stability` (static/dynamic/ephemeral/unknown)

TraceRank uses reinforcement events + graph connectivity to apply a decay-aware multiplier.

---

## Observability

- `vestig memory show` exposes temporal fields and embeddings
- `vestig entity list/show` surfaces graph nodes
- `vestig edge list/show` inspects relationships
- `vestig housekeeping report/orphans` inspects graph health

---

## Configuration

Key config sections:
- `embedding`: model, provider, dimension, normalization
- `storage.falkordb`: host, port, graph_name
- `ingestion`: model, chunk size/overlap, confidence thresholds
- `m3`: event logging + TraceRank settings
- `m4`: entity types, extraction config, edge creation

---

## Out of Scope (M5+)

- Service/daemon mode and external integrations
- Multi-hop traversal and MemRank-like scoring
- Background job scheduling

---

**Author:** Matt Joyce (with Claude Sonnet 4.5)
