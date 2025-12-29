# Vestig Ingestion Pipeline — Detailed Explainer

This document explains how Vestig ingests artifacts, extracts memories, and stores nodes/edges in the database. It focuses on the end-to-end flow from input file to memory + entity graph.

## 1) High-Level Flow (Happy Path)
1. **Read artifact** (`ingest_document`) from a file path.
2. **Normalize and filter** input text based on format (plain text or Claude session JSONL).
3. **Chunk** normalized text into manageable segments.
4. **Extract memories** with the LLM, producing structured memory objects with entities and temporal hints.
5. **Commit memories**: apply hygiene + dedupe, store in SQLite, create events, extract entities, and create graph edges.

---

## 2) Input Normalization
### Supported formats
- `plain` — raw text.
- `claude-session` — JSONL exports from Claude Code sessions.
- `auto` — detects format based on extension + content.

### Normalization behavior
- **Plain text**: uses the raw text as-is.
- **Claude session JSONL**:
  - Filters by role (default `user`, `assistant`).
  - Keeps only textual content (default `type: text`).
  - Drops `thinking` and tool calls by default.

### Temporal hints (per-chunk)
For JSONL ingestion, each chunk gets a `t_valid` derived from the **earliest event timestamp in that chunk**.
- Timestamps are normalized to a canonical UTC ISO-8601 format: `YYYY-MM-DDTHH:MM:SS.sss+00:00`.
- Temporal stability is initially `unknown` unless overridden by LLM classification.

---

## 3) Chunking
### Why chunking exists
LLM extraction must stay within model context windows and avoid overloading the prompt.

### Chunking strategy
- **Default**: character-based chunking with overlap.
- **Chunk size**: configured in `config.yaml` (`ingestion.chunk_size`).
- **Overlap**: configured in `config.yaml` (`ingestion.chunk_overlap`).

For Claude session JSONL, chunking is done on **message blocks** rather than raw text, preserving message boundaries.

---

## 4) Memory Extraction (LLM)
### Prompt
The LLM receives each chunk and is asked to extract:
- **Memory content** (self-contained fact/insight)
- **Confidence**
- **Rationale**
- **Entities** (typed)
- **Temporal stability** (`static`, `dynamic`, or `unknown`)

### Output schema
Each extracted memory becomes an `ExtractedMemory` with:
- `content`, `confidence`, `rationale`
- `entities` (name, type, confidence, evidence)
- `t_valid_hint` (from chunk temporal hints)
- `temporal_stability_hint` (LLM classification, with parser fallback)

---

## 5) Commit Pipeline (Storage + Graph)
The commit pipeline applies quality gates and stores nodes in SQLite.

### 5.1 Hygiene (Quality Firewall)
Before storing anything:
- Minimum length
- Maximum length
- Whitespace normalization
- Simple non-substantive rejection

### 5.2 Dedupe
Two layers:
- **Exact duplicate**: content hash match (sha256). If matched, no new memory is inserted.
- **Near duplicate**: semantic similarity vs existing memories. If above threshold, it is treated as reinforcement.

### 5.3 Memory node creation
`MemoryNode.create()` sets:
- `id` (mem_*)
- `content` (normalized)
- `embedding`
- `content_hash`
- `metadata` (source, tags)
- **Temporal fields**:
  - `t_valid`: from temporal hint if available, else now
  - `t_created`: now
  - `t_invalid`: null
  - `t_expired`: null
  - `temporal_stability`: from LLM or parser

### 5.4 Event logging (M3)
When event storage is enabled:
- ADD / REINFORCE events are stored in `memory_events`.
- Reinforcement updates `reinforce_count` and `last_seen_at`.

---

## 6) Entity Extraction + Graph Edges (M4)
### Entity extraction
If enabled:
- Entities are extracted (or passed in as pre-extracted).
- Entities are stored as **canonical EntityNodes** with dedupe via `norm_key`.

### Edge creation
- **MENTIONS** edges: Memory → Entity
- **RELATED** edges: Memory → Memory (semantic similarity)

Edges include:
- weight (e.g., similarity)
- confidence
- evidence (short string)
- temporal fields (`t_valid`, `t_created`, `t_expired`)

---

## 7) What gets stored
Tables created in SQLite:
- `memories` (MemoryNode)
- `entities` (EntityNode)
- `edges` (EdgeNode)
- `memory_events` (EventNode)

---

## 8) Key Files
- `src/vestig/core/ingestion.py` — main ingest pipeline
- `src/vestig/core/ingest_sources.py` — format-specific parsing
- `src/vestig/core/commitment.py` — hygiene, dedupe, node creation
- `src/vestig/core/storage.py` — SQLite schema + persistence

