-- Vestig Schema v1.0
-- Milestone: M4 (Summary Nodes)
-- Last Updated: 2025-01-XX
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
    kind TEXT DEFAULT 'MEMORY'        -- 'MEMORY' | 'SUMMARY'
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
    embedding TEXT,                   -- JSON-serialized embedding vector (for semantic matching)
    created_at TEXT NOT NULL,         -- ISO 8601 timestamp
    expired_at TEXT,                  -- When entity was merged/deprecated
    merged_into TEXT                  -- ID of entity this was merged into
);

-- edges: Graph edges connecting memories and entities
CREATE TABLE edges (
    edge_id TEXT PRIMARY KEY,
    from_node TEXT NOT NULL,          -- Source node ID (mem_* or ent_*)
    to_node TEXT NOT NULL,            -- Target node ID (mem_* or ent_*)
    edge_type TEXT NOT NULL,          -- MENTIONS | RELATED | SUMMARIZES
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
