-- Vestig Schema v2.0
-- Milestone: Phase 2 (Source Abstraction)
-- Last Updated: 2026-01-11
-- DO NOT MODIFY: This is the sovereign interface for Vestig's data model
--
-- This file defines the complete schema for fresh Vestig database creation.
-- Existing databases are migrated using additive migrations in storage.py.
--
-- Schema Evolution:
-- M1: Base memories table with content and embeddings
-- M2: Added content_hash for deduplication
-- M3: Added bi-temporal fields (t_valid, t_invalid, t_created, t_expired) and events table
-- M4: Added graph layer (entities, edges) and kind discriminator for MEMORY/SUMMARY nodes
-- M5: Added files/chunks tables (chunk-centric hub architecture)
-- Phase 2: Source abstraction - unified provenance for all content origins

-- =============================================================================
-- Phase 2: Source Provenance Layer
-- =============================================================================

-- sources: Unified provenance for all content origins
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,          -- Unique identifier (source_<uuid>)
    source_type TEXT NOT NULL,           -- 'file' | 'agentic' | 'legacy'
    created_at TEXT NOT NULL,            -- ISO 8601 timestamp (source creation time)
    ingested_at TEXT NOT NULL,           -- ISO 8601 timestamp (when processed)
    source_hash TEXT,                    -- SHA256 of content (for change detection)
    metadata TEXT,                       -- JSON-serialized type-specific metadata

    -- Type-specific fields (nullable)
    path TEXT,                           -- For 'file': absolute file path
    agent TEXT,                          -- For 'agentic': agent name (claude-code, codex, etc.)
    session_id TEXT                      -- Optional: session tracking across types
);

-- =============================================================================
-- M5 Hub Layer: Files and Chunks (Backward Compatibility)
-- =============================================================================

-- files: DEPRECATED - use sources table for new code
-- Kept for backward compatibility during migration
CREATE TABLE files (
    file_id TEXT PRIMARY KEY,            -- Unique identifier (file_<uuid>)
    path TEXT NOT NULL,                  -- Absolute file path
    created_at TEXT NOT NULL,            -- ISO 8601 timestamp (file creation/modification time)
    ingested_at TEXT NOT NULL,           -- ISO 8601 timestamp (when file was processed)
    file_hash TEXT,                      -- SHA256 of file content (for change detection)
    metadata TEXT                        -- JSON-serialized file metadata (format, size, etc.)
);

-- chunks: Location pointers within sources (optional positional metadata)
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,           -- Unique identifier (chunk_<uuid>)
    source_id TEXT NOT NULL,             -- Foreign key to sources table (was file_id)
    start INTEGER NOT NULL,              -- Character position in source where chunk starts
    length INTEGER NOT NULL,             -- Number of characters in chunk
    sequence INTEGER NOT NULL,           -- Position in document (1st chunk, 2nd chunk, etc.)
    created_at TEXT NOT NULL,            -- ISO 8601 timestamp (when chunk was created)
    FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

-- =============================================================================
-- Core Tables: Memories and Events
-- =============================================================================

-- memories: Core memory nodes with bi-temporal tracking
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_embedding TEXT NOT NULL,  -- JSON-serialized embedding vector
    created_at TEXT NOT NULL,         -- ISO 8601 timestamp (when memory was created)
    metadata TEXT NOT NULL,           -- JSON-serialized metadata dict

    -- M2: Deduplication
    content_hash TEXT,                -- SHA256 of normalized content

    -- M3: Bi-temporal fields (event time)
    t_valid TEXT,                     -- When fact became true (event time)
    t_invalid TEXT,                   -- When fact stopped being true (event time)
    t_created TEXT,                   -- When we learned about it (transaction time)
    t_expired TEXT,                   -- When deprecated/superseded
    temporal_stability TEXT DEFAULT 'unknown',  -- 'static' | 'dynamic' | 'unknown'

    -- M3: Reinforcement tracking (cached from events)
    last_seen_at TEXT,                -- Most recent reinforcement timestamp
    reinforce_count INTEGER DEFAULT 0, -- Total reinforcement events

    -- M4: Node type discriminator
    kind TEXT DEFAULT 'MEMORY',       -- 'MEMORY' | 'SUMMARY'

    -- M5: Chunk provenance (hub link)
    chunk_id TEXT,                    -- Foreign key to chunks table (nullable for manual adds)

    -- Phase 2: Source abstraction - dual linking
    source_id TEXT                    -- Direct link to source (primary provenance)
);

-- memory_events: Event log for memory lifecycle tracking
CREATE TABLE memory_events (
    event_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    event_type TEXT NOT NULL,         -- ADD | REINFORCE_EXACT | REINFORCE_NEAR | DEPRECATE | SUPERSEDE | ENTITY_EXTRACTED | SUMMARY_CREATED
    occurred_at TEXT NOT NULL,        -- ISO 8601 timestamp (when event occurred)
    source TEXT NOT NULL,             -- manual | hook | import | batch | llm | summary_generation
    actor TEXT,                       -- User/agent identifier (optional)
    artifact_ref TEXT,                -- Session ID, filename, etc. (optional)
    payload_json TEXT NOT NULL,       -- JSON-serialized event details
    FOREIGN KEY(memory_id) REFERENCES memories(id)
);

-- =============================================================================
-- M4 Graph Layer: Entities and Edges
-- =============================================================================

-- entities: Canonical entity nodes (PERSON, ORG, SYSTEM, etc.)
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,        -- PERSON | ORG | SYSTEM | PROJECT | PLACE | SKILL | CAPABILITY | TOOL | FILE | CONCEPT
    canonical_name TEXT NOT NULL,     -- Canonical form of entity name
    norm_key TEXT NOT NULL,           -- Normalization key for deduplication (type:normalized_name)
    created_at TEXT NOT NULL,         -- ISO 8601 timestamp
    embedding TEXT,                   -- JSON-serialized embedding vector (for semantic matching)
    expired_at TEXT,                  -- When entity was merged/deprecated
    merged_into TEXT,                 -- ID of entity this was merged into

    -- M5: Chunk provenance (hub link)
    chunk_id TEXT,                    -- Foreign key to chunks table (nullable for cross-chunk entities)
    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
);

-- edges: Graph edges connecting memories and entities
CREATE TABLE edges (
    edge_id TEXT PRIMARY KEY,
    from_node TEXT NOT NULL,          -- Source node ID (mem_* or ent_*)
    to_node TEXT NOT NULL,            -- Target node ID (mem_* or ent_*)
    edge_type TEXT NOT NULL,          -- MENTIONS | RELATED | SUMMARIZES (deprecated) | CONTAINS | LINKED | SUMMARIZED_BY
    weight REAL NOT NULL,             -- Edge weight (1.0 default, or similarity score)

    -- M4: LLM extraction metadata
    confidence REAL,                  -- Extraction confidence (0.0-1.0, optional)
    evidence TEXT,                    -- Short explanation (max 200 chars, optional)

    -- M4: Bi-temporal fields (same as memories)
    t_valid TEXT,                     -- When relationship became true
    t_invalid TEXT,                   -- When relationship stopped being true
    t_created TEXT,                   -- When we learned about this relationship
    t_expired TEXT                    -- When edge was invalidated
);

-- =============================================================================
-- Indexes: Memories Table
-- =============================================================================

-- M2: Unique index on content_hash for deduplication
CREATE UNIQUE INDEX idx_content_hash
ON memories(content_hash);

-- M3: Partial index for expired memories
CREATE INDEX idx_memories_expired
ON memories(t_expired)
WHERE t_expired IS NOT NULL;

-- M4: Index for kind column (MEMORY vs SUMMARY queries)
CREATE INDEX idx_memories_kind
ON memories(kind);

-- =============================================================================
-- Indexes: Memory Events Table
-- =============================================================================

-- M3: Index for querying events by memory and time (most recent first)
CREATE INDEX idx_events_memory_time
ON memory_events(memory_id, occurred_at DESC);

-- M3: Index for filtering events by type
CREATE INDEX idx_events_type
ON memory_events(event_type);

-- =============================================================================
-- Indexes: Entities Table
-- =============================================================================

-- M4: Unique index on norm_key for entity deduplication
CREATE UNIQUE INDEX idx_entities_norm_key
ON entities(norm_key);

-- M4: Index for querying entities by type
CREATE INDEX idx_entities_type
ON entities(entity_type);

-- M4: Partial index for expired entities
CREATE INDEX idx_entities_expired
ON entities(expired_at)
WHERE expired_at IS NOT NULL;

-- =============================================================================
-- Indexes: Edges Table
-- =============================================================================

-- M4: Index for querying edges from a node (with edge type filter)
CREATE INDEX idx_edges_from_node
ON edges(from_node, edge_type);

-- M4: Index for querying edges to a node (with edge type filter)
CREATE INDEX idx_edges_to_node
ON edges(to_node, edge_type);

-- M4: Index for filtering edges by type
CREATE INDEX idx_edges_type
ON edges(edge_type);

-- M4: Partial index for high-confidence edges
CREATE INDEX idx_edges_confidence
ON edges(confidence)
WHERE confidence IS NOT NULL;

-- M4: Partial index for expired edges
CREATE INDEX idx_edges_expired
ON edges(t_expired)
WHERE t_expired IS NOT NULL;

-- M4: Unique constraint to prevent duplicate edges (belt and braces)
CREATE UNIQUE INDEX idx_edges_unique
ON edges(from_node, to_node, edge_type)
WHERE t_expired IS NULL;

-- =============================================================================
-- Indexes: Sources Table
-- =============================================================================

-- Phase 2: Index for querying sources by type
CREATE INDEX idx_sources_type
ON sources(source_type);

-- Phase 2: Index for querying sources by agent (for agentic sources)
CREATE INDEX idx_sources_agent
ON sources(agent)
WHERE agent IS NOT NULL;

-- Phase 2: Index for querying sources by path (for file sources)
CREATE INDEX idx_sources_path
ON sources(path)
WHERE path IS NOT NULL;

-- Phase 2: Index for querying sources by ingestion time
CREATE INDEX idx_sources_ingested_at
ON sources(ingested_at DESC);

-- Phase 2: Index for querying sources by session
CREATE INDEX idx_sources_session
ON sources(session_id)
WHERE session_id IS NOT NULL;

-- =============================================================================
-- Indexes: Files Table (DEPRECATED)
-- =============================================================================

-- M5: Index for querying files by path
CREATE INDEX idx_files_path
ON files(path);

-- M5: Index for querying files by ingestion time
CREATE INDEX idx_files_ingested_at
ON files(ingested_at DESC);

-- =============================================================================
-- Indexes: Chunks Table
-- =============================================================================

-- Phase 2: Index for querying chunks by source (ordered by sequence)
CREATE INDEX idx_chunks_source
ON chunks(source_id, sequence);

-- Phase 2: Index for querying chunks by position (for range lookups)
CREATE INDEX idx_chunks_position
ON chunks(source_id, start, length);

-- =============================================================================
-- Indexes: Phase 2 Source Provenance
-- =============================================================================

-- Phase 2: Index for querying memories by source (direct provenance)
CREATE INDEX idx_memories_source
ON memories(source_id)
WHERE source_id IS NOT NULL;

-- M5: Index for querying memories by chunk (positional metadata)
CREATE INDEX idx_memories_chunk
ON memories(chunk_id)
WHERE chunk_id IS NOT NULL;

-- M5: Index for querying entities by chunk
CREATE INDEX idx_entities_chunk
ON entities(chunk_id)
WHERE chunk_id IS NOT NULL;
