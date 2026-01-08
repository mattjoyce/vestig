// Vestig FalkorDB Schema v1.0
// Graph-native schema for memory system
//
// Node Types: Memory, Entity, Chunk, File, Event
// Edge Types: CONTAINS, LINKED, SUMMARIZED_BY, MENTIONS, RELATED, AFFECTS

// =============================================================================
// NODE CONSTRAINTS AND INDEXES
// =============================================================================

// Memory nodes (atomic facts and summaries)
// Properties: id, content, content_hash, content_embedding, kind, created_at,
//             metadata, t_valid, t_invalid, t_created, t_expired,
//             temporal_stability, last_seen_at, reinforce_count
// Note: chunk_id removed - use (Chunk)-[:CONTAINS]->(Memory) edges instead
CREATE CONSTRAINT unique_memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT unique_content_hash IF NOT EXISTS FOR (m:Memory) REQUIRE m.content_hash IS UNIQUE;
CREATE INDEX memory_kind IF NOT EXISTS FOR (m:Memory) ON (m.kind);
CREATE INDEX memory_created IF NOT EXISTS FOR (m:Memory) ON (m.created_at);
CREATE INDEX memory_expired IF NOT EXISTS FOR (m:Memory) ON (m.t_expired);

// Entity nodes (canonical named entities)
// Properties: id, entity_type, canonical_name, norm_key, created_at,
//             embedding, expired_at, merged_into
// Note: chunk_id removed - use (Chunk)-[:LINKED]->(Entity) edges instead
CREATE CONSTRAINT unique_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT unique_norm_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.norm_key IS UNIQUE;
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type);
CREATE INDEX entity_expired IF NOT EXISTS FOR (e:Entity) ON (e.expired_at);

// Chunk nodes (provenance hubs - location pointers within files)
// Properties: id, file_id, start, length, sequence, created_at
CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE INDEX chunk_file_seq IF NOT EXISTS FOR (c:Chunk) ON (c.file_id, c.sequence);

// File nodes (source documents)
// Properties: id, path, created_at, ingested_at, file_hash, metadata
CREATE CONSTRAINT unique_file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE;
CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path);
CREATE INDEX file_ingested IF NOT EXISTS FOR (f:File) ON (f.ingested_at);

// Event nodes (lifecycle audit trail)
// Properties: id, memory_id, event_type, occurred_at, source, actor,
//             artifact_ref, payload
CREATE CONSTRAINT unique_event_id IF NOT EXISTS FOR (evt:Event) REQUIRE evt.id IS UNIQUE;
CREATE INDEX event_memory_time IF NOT EXISTS FOR (evt:Event) ON (evt.memory_id, evt.occurred_at);
CREATE INDEX event_type IF NOT EXISTS FOR (evt:Event) ON (evt.event_type);

// =============================================================================
// VECTOR INDEXES (for semantic search)
// Note: FalkorDB vector index syntax - verify against your FalkorDB version
// =============================================================================

// Memory content embeddings (dimension depends on embedding model)
// CREATE VECTOR INDEX memory_embedding IF NOT EXISTS
//   FOR (m:Memory) ON (m.content_embedding)
//   OPTIONS {dimension: 768, similarity: 'cosine'};

// Entity embeddings
// CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
//   FOR (e:Entity) ON (e.embedding)
//   OPTIONS {dimension: 768, similarity: 'cosine'};

// =============================================================================
// EDGE TYPE DOCUMENTATION
// =============================================================================

// Hub-and-spoke relationships (Chunk as central hub):
// (Chunk)-[:CONTAINS]->(Memory)       - Chunk contains these memories
// (Chunk)-[:LINKED]->(Entity)         - Chunk mentions these entities (1st class)
// (Chunk)-[:SUMMARIZED_BY]->(Memory)  - Chunk is summarized by this memory (kind='SUMMARY')

// Graph layer relationships:
// (Memory)-[:MENTIONS]->(Entity)      - Memory mentions entity (2nd class, from extraction)
// (Memory)-[:RELATED]->(Memory)       - Memories are topically related

// Provenance relationships:
// (File)-[:HAS_CHUNK]->(Chunk)        - File contains this chunk
// (Event)-[:AFFECTS]->(Memory)        - Event affects this memory

// Edge properties:
// - weight: float (relevance/importance)
// - confidence: float (0.0-1.0, extraction confidence)
// - evidence: string (max 200 chars, supporting text)
// - t_valid, t_invalid, t_created, t_expired: temporal tracking
