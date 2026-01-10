"""FalkorDB database adapter implementing DatabaseInterface.

This adapter provides native graph database operations using FalkorDB,
a Redis-based graph database with Cypher query support.
"""

import json
from contextlib import contextmanager
from datetime import datetime, timezone

from falkordb import FalkorDB

from vestig.core.db_interface import DatabaseInterface, EventStorageInterface
from vestig.core.models import ChunkNode, EdgeNode, EntityNode, EventNode, FileNode, MemoryNode


class FalkorEventStorage(EventStorageInterface):
    """FalkorDB event storage using Event nodes with AFFECTS edges."""

    def __init__(self, graph):
        self._graph = graph

    def add_event(self, event: EventNode) -> str:
        """Insert event as node with AFFECTS edge to memory."""
        self._graph.query(
            """
            MATCH (m:Memory {id: $memory_id})
            CREATE (e:Event {
                id: $event_id,
                memory_id: $memory_id,
                event_type: $event_type,
                occurred_at: $occurred_at,
                source: $source,
                actor: $actor,
                artifact_ref: $artifact_ref,
                payload: $payload
            })-[:AFFECTS]->(m)
            """,
            {
                "memory_id": event.memory_id,
                "event_id": event.event_id,
                "event_type": event.event_type,
                "occurred_at": event.occurred_at,
                "source": event.source,
                "actor": event.actor,
                "artifact_ref": event.artifact_ref,
                "payload": json.dumps(event.payload),
            },
        )
        return event.event_id

    def get_events_for_memory(self, memory_id: str, limit: int = 100) -> list[EventNode]:
        """Retrieve events for a memory, newest first."""
        result = self._graph.ro_query(
            """
            MATCH (e:Event)-[:AFFECTS]->(m:Memory {id: $memory_id})
            RETURN e.id, e.memory_id, e.event_type, e.occurred_at,
                   e.source, e.actor, e.artifact_ref, e.payload
            ORDER BY e.occurred_at DESC
            LIMIT $limit
            """,
            {"memory_id": memory_id, "limit": limit},
        )
        return [
            EventNode(
                event_id=row[0],
                memory_id=row[1],
                event_type=row[2],
                occurred_at=row[3],
                source=row[4],
                actor=row[5],
                artifact_ref=row[6],
                payload=json.loads(row[7]) if row[7] else {},
            )
            for row in result.result_set
        ]

    def get_reinforcement_events(self, memory_id: str) -> list[EventNode]:
        """Get only REINFORCE_* events for TraceRank computation."""
        result = self._graph.ro_query(
            """
            MATCH (e:Event)-[:AFFECTS]->(m:Memory {id: $memory_id})
            WHERE e.event_type STARTS WITH 'REINFORCE'
            RETURN e.id, e.memory_id, e.event_type, e.occurred_at,
                   e.source, e.actor, e.artifact_ref, e.payload
            ORDER BY e.occurred_at DESC
            """,
            {"memory_id": memory_id},
        )
        return [
            EventNode(
                event_id=row[0],
                memory_id=row[1],
                event_type=row[2],
                occurred_at=row[3],
                source=row[4],
                actor=row[5],
                artifact_ref=row[6],
                payload=json.loads(row[7]) if row[7] else {},
            )
            for row in result.result_set
        ]


class FalkorDBDatabase(DatabaseInterface):
    """FalkorDB graph database adapter.

    Provides native graph operations using Cypher queries.
    """

    def __init__(self, host: str, port: int, graph_name: str):
        """Initialize FalkorDB connection.

        Args:
            host: FalkorDB server host
            port: FalkorDB server port
            graph_name: Name of the graph to use
        """
        self._client = FalkorDB(host=host, port=port)
        self._graph = self._client.select_graph(graph_name)
        self._graph_name = graph_name
        self._event_storage = FalkorEventStorage(self._graph)
        self._init_schema()

    def _init_schema(self):
        """Initialize schema constraints and indexes."""
        # Create unique constraints
        try:
            self._graph.create_node_unique_constraint("Memory", "id")
        except Exception:
            pass  # Constraint may already exist

        try:
            self._graph.create_node_unique_constraint("Memory", "content_hash")
        except Exception:
            pass

        try:
            self._graph.create_node_unique_constraint("Entity", "id")
        except Exception:
            pass

        try:
            self._graph.create_node_unique_constraint("Entity", "norm_key")
        except Exception:
            pass

        try:
            self._graph.create_node_unique_constraint("Chunk", "id")
        except Exception:
            pass

        try:
            self._graph.create_node_unique_constraint("File", "id")
        except Exception:
            pass

        try:
            self._graph.create_node_unique_constraint("Event", "id")
        except Exception:
            pass

        # Create range indexes for common queries
        try:
            self._graph.create_node_range_index("Memory", "kind")
        except Exception:
            pass

        try:
            self._graph.create_node_range_index("Memory", "t_expired")
        except Exception:
            pass

        try:
            self._graph.create_node_range_index("Entity", "entity_type")
        except Exception:
            pass

        try:
            self._graph.create_node_range_index("Entity", "expired_at")
        except Exception:
            pass

    # =========================================================================
    # Memory Operations
    # =========================================================================

    def store_memory(self, node: MemoryNode, kind: str = "MEMORY") -> str:
        """Store memory with deduplication via content_hash."""
        # Check for existing memory with same content_hash
        result = self._graph.ro_query(
            "MATCH (m:Memory {content_hash: $hash}) RETURN m.id",
            {"hash": node.content_hash},
        )
        if result.result_set:
            return result.result_set[0][0]

        # Create new memory node (no chunk_id - use edges instead)
        self._graph.query(
            """
            CREATE (m:Memory {
                id: $id,
                content: $content,
                content_embedding: $embedding,
                content_hash: $hash,
                created_at: $created_at,
                metadata: $metadata,
                kind: $kind,
                t_valid: $t_valid,
                t_invalid: $t_invalid,
                t_created: $t_created,
                t_expired: $t_expired,
                temporal_stability: $temporal_stability,
                last_seen_at: $last_seen_at,
                reinforce_count: $reinforce_count
            })
            """,
            {
                "id": node.id,
                "content": node.content,
                "embedding": json.dumps(node.content_embedding),
                "hash": node.content_hash,
                "created_at": node.created_at,
                "metadata": json.dumps(node.metadata),
                "kind": kind,
                "t_valid": node.t_valid,
                "t_invalid": node.t_invalid,
                "t_created": node.t_created,
                "t_expired": node.t_expired,
                "temporal_stability": node.temporal_stability,
                "last_seen_at": node.last_seen_at,
                "reinforce_count": node.reinforce_count,
            },
        )
        return node.id

    def get_memory(self, memory_id: str) -> MemoryNode | None:
        """Retrieve memory by ID."""
        result = self._graph.ro_query(
            """
            MATCH (m:Memory {id: $id})
            RETURN m.id, m.content, m.content_embedding, m.content_hash,
                   m.created_at, m.metadata, m.kind, m.t_valid, m.t_invalid,
                   m.t_created, m.t_expired, m.temporal_stability,
                   m.last_seen_at, m.reinforce_count
            """,
            {"id": memory_id},
        )
        if not result.result_set:
            return None

        row = result.result_set[0]
        return MemoryNode(
            id=row[0],
            content=row[1],
            content_embedding=json.loads(row[2]) if row[2] else [],
            content_hash=row[3],
            created_at=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
            kind=row[6] or "MEMORY",
            t_valid=row[7],
            t_invalid=row[8],
            t_created=row[9],
            t_expired=row[10],
            temporal_stability=row[11] or "unknown",
            last_seen_at=row[12],
            reinforce_count=row[13] or 0,
            chunk_id=None,  # Use edges instead
        )

    def get_all_memories(self) -> list[MemoryNode]:
        """Get all memories (including expired)."""
        result = self._graph.ro_query(
            """
            MATCH (m:Memory)
            RETURN m.id, m.content, m.content_embedding, m.content_hash,
                   m.created_at, m.metadata, m.kind, m.t_valid, m.t_invalid,
                   m.t_created, m.t_expired, m.temporal_stability,
                   m.last_seen_at, m.reinforce_count
            ORDER BY m.created_at DESC
            """
        )
        return [self._row_to_memory(row) for row in result.result_set]

    def get_active_memories(self) -> list[MemoryNode]:
        """Get all non-expired memories."""
        result = self._graph.ro_query(
            """
            MATCH (m:Memory)
            WHERE m.t_expired IS NULL
            RETURN m.id, m.content, m.content_embedding, m.content_hash,
                   m.created_at, m.metadata, m.kind, m.t_valid, m.t_invalid,
                   m.t_created, m.t_expired, m.temporal_stability,
                   m.last_seen_at, m.reinforce_count
            ORDER BY m.created_at DESC
            """
        )
        return [self._row_to_memory(row) for row in result.result_set]

    def _row_to_memory(self, row) -> MemoryNode:
        """Convert query result row to MemoryNode (without chunk_id - use edges)."""
        return MemoryNode(
            id=row[0],
            content=row[1],
            content_embedding=json.loads(row[2]) if row[2] else [],
            content_hash=row[3],
            created_at=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
            kind=row[6] or "MEMORY",
            t_valid=row[7],
            t_invalid=row[8],
            t_created=row[9],
            t_expired=row[10],
            temporal_stability=row[11] or "unknown",
            last_seen_at=row[12],
            reinforce_count=row[13] or 0,
            chunk_id=None,  # Use CONTAINS edges instead
        )

    def list_memories(self, include_expired: bool = False, limit: int | None = None) -> list[tuple]:
        """List memories for CLI display.

        Returns: List of (id, content, created_at, t_expired, metadata_json) tuples
        """
        expired_filter = "" if include_expired else "WHERE m.t_expired IS NULL"
        limit_clause = f"LIMIT {limit}" if limit else ""

        result = self._graph.ro_query(
            f"""
            MATCH (m:Memory)
            {expired_filter}
            RETURN m.id, m.content, m.created_at, m.t_expired, m.metadata
            ORDER BY m.created_at DESC
            {limit_clause}
            """
        )
        return [tuple(row) for row in result.result_set]

    def get_memories_for_entity_extraction(self, reprocess: bool = False) -> list[tuple[str, str]]:
        """Get memories needing entity extraction."""
        if reprocess:
            # Return all active memories
            result = self._graph.ro_query(
                """
                MATCH (m:Memory)
                WHERE m.t_expired IS NULL AND m.kind = 'MEMORY'
                RETURN m.id, m.content
                ORDER BY m.created_at ASC
                """
            )
        else:
            # Return memories without MENTIONS edges
            result = self._graph.ro_query(
                """
                MATCH (m:Memory)
                WHERE m.t_expired IS NULL AND m.kind = 'MEMORY'
                AND NOT (m)-[:MENTIONS]->(:Entity)
                RETURN m.id, m.content
                ORDER BY m.created_at ASC
                """
            )
        return [(row[0], row[1]) for row in result.result_set]

    def get_memories_by_chunk(
        self, chunk_id: str, include_expired: bool = False
    ) -> list[MemoryNode]:
        """Get memories by chunk_id property (M5 legacy).

        In FalkorDB, we don't store chunk_id as a property.
        Delegate to edge-based method instead.
        """
        return self.get_memories_in_chunk(chunk_id, include_expired)

    def get_memories_in_chunk(
        self, chunk_id: str, include_expired: bool = False
    ) -> list[MemoryNode]:
        """Get memories via CONTAINS edge (M6)."""
        expired_filter = "" if include_expired else "AND m.t_expired IS NULL"

        result = self._graph.ro_query(
            f"""
            MATCH (c:Chunk {{id: $chunk_id}})-[:CONTAINS]->(m:Memory)
            WHERE 1=1 {expired_filter}
            RETURN m.id, m.content, m.content_embedding, m.content_hash,
                   m.created_at, m.metadata, m.kind, m.t_valid, m.t_invalid,
                   m.t_created, m.t_expired, m.temporal_stability,
                   m.last_seen_at, m.reinforce_count
            ORDER BY m.created_at ASC
            """,
            {"chunk_id": chunk_id},
        )
        return [self._row_to_memory(row) for row in result.result_set]

    def get_summary_for_artifact(self, artifact_ref: str) -> MemoryNode | None:
        """Find SUMMARY by artifact reference."""
        result = self._graph.ro_query(
            """
            MATCH (e:Event)-[:AFFECTS]->(m:Memory {kind: 'SUMMARY'})
            WHERE e.artifact_ref = $artifact_ref
            RETURN m.id, m.content, m.content_embedding, m.content_hash,
                   m.created_at, m.metadata, m.kind, m.t_valid, m.t_invalid,
                   m.t_created, m.t_expired, m.temporal_stability,
                   m.last_seen_at, m.reinforce_count
            LIMIT 1
            """,
            {"artifact_ref": artifact_ref},
        )
        if not result.result_set:
            return None
        return self._row_to_memory(result.result_set[0])

    def get_summary_for_chunk(self, chunk_id: str) -> MemoryNode | None:
        """Find SUMMARY for chunk via SUMMARIZED_BY edge.

        In FalkorDB, we use edges instead of chunk_id property.
        Delegate to edge-based method.
        """
        return self.get_summary_for_chunk_via_edge(chunk_id)

    def update_node_embedding(self, node_id: str, embedding_json: str, node_type: str) -> None:
        """Update embedding for memory or entity."""
        if node_type == "memory":
            self._graph.query(
                "MATCH (m:Memory {id: $id}) SET m.content_embedding = $embedding",
                {"id": node_id, "embedding": embedding_json},
            )
        elif node_type == "entity":
            self._graph.query(
                "MATCH (e:Entity {id: $id}) SET e.embedding = $embedding",
                {"id": node_id, "embedding": embedding_json},
            )

    def increment_reinforce_count(self, memory_id: str) -> None:
        """Increment reinforce_count."""
        self._graph.query(
            """
            MATCH (m:Memory {id: $id})
            SET m.reinforce_count = COALESCE(m.reinforce_count, 0) + 1
            """,
            {"id": memory_id},
        )

    def update_last_seen(self, memory_id: str, timestamp: str) -> None:
        """Update last_seen_at timestamp."""
        self._graph.query(
            "MATCH (m:Memory {id: $id}) SET m.last_seen_at = $ts",
            {"id": memory_id, "ts": timestamp},
        )

    def deprecate_memory(self, memory_id: str, t_invalid: str | None = None) -> None:
        """Mark memory as expired."""
        now = datetime.now(timezone.utc).isoformat()
        self._graph.query(
            """
            MATCH (m:Memory {id: $id})
            SET m.t_expired = $now, m.t_invalid = $t_invalid
            """,
            {"id": memory_id, "now": now, "t_invalid": t_invalid},
        )

    # =========================================================================
    # Entity Operations
    # =========================================================================

    def store_entity(self, entity: EntityNode) -> str:
        """Store entity with deduplication via norm_key."""
        # Check for existing entity
        result = self._graph.ro_query(
            """
            MATCH (e:Entity {norm_key: $norm_key})
            WHERE e.expired_at IS NULL
            RETURN e.id
            """,
            {"norm_key": entity.norm_key},
        )
        if result.result_set:
            return result.result_set[0][0]

        # Create new entity (no chunk_id - use edges instead)
        self._graph.query(
            """
            CREATE (e:Entity {
                id: $id,
                entity_type: $entity_type,
                canonical_name: $canonical_name,
                norm_key: $norm_key,
                created_at: $created_at,
                embedding: $embedding,
                expired_at: $expired_at,
                merged_into: $merged_into
            })
            """,
            {
                "id": entity.id,
                "entity_type": entity.entity_type,
                "canonical_name": entity.canonical_name,
                "norm_key": entity.norm_key,
                "created_at": entity.created_at,
                "embedding": entity.embedding,
                "expired_at": entity.expired_at,
                "merged_into": entity.merged_into,
            },
        )
        return entity.id

    def get_entity(self, entity_id: str) -> EntityNode | None:
        """Retrieve entity by ID."""
        result = self._graph.ro_query(
            """
            MATCH (e:Entity {id: $id})
            RETURN e.id, e.entity_type, e.canonical_name, e.norm_key,
                   e.created_at, e.embedding, e.expired_at, e.merged_into
            """,
            {"id": entity_id},
        )
        if not result.result_set:
            return None

        row = result.result_set[0]
        return EntityNode(
            id=row[0],
            entity_type=row[1],
            canonical_name=row[2],
            norm_key=row[3],
            created_at=row[4],
            embedding=row[5],
            expired_at=row[6],
            merged_into=row[7],
            chunk_id=None,  # Use LINKED edges instead
        )

    def find_entity_by_norm_key(
        self, norm_key: str, include_expired: bool = False
    ) -> EntityNode | None:
        """Find entity by norm_key."""
        expired_filter = "" if include_expired else "AND e.expired_at IS NULL"

        result = self._graph.ro_query(
            f"""
            MATCH (e:Entity {{norm_key: $norm_key}})
            WHERE 1=1 {expired_filter}
            RETURN e.id, e.entity_type, e.canonical_name, e.norm_key,
                   e.created_at, e.embedding, e.expired_at, e.merged_into
            LIMIT 1
            """,
            {"norm_key": norm_key},
        )
        if not result.result_set:
            return None

        row = result.result_set[0]
        return EntityNode(
            id=row[0],
            entity_type=row[1],
            canonical_name=row[2],
            norm_key=row[3],
            created_at=row[4],
            embedding=row[5],
            expired_at=row[6],
            merged_into=row[7],
            chunk_id=None,  # Use LINKED edges instead
        )

    def get_all_entities(self, include_expired: bool = False) -> list[EntityNode]:
        """Get all entities."""
        expired_filter = "" if include_expired else "WHERE e.expired_at IS NULL"

        result = self._graph.ro_query(
            f"""
            MATCH (e:Entity)
            {expired_filter}
            RETURN e.id, e.entity_type, e.canonical_name, e.norm_key,
                   e.created_at, e.embedding, e.expired_at, e.merged_into
            ORDER BY e.created_at DESC
            """
        )
        return [self._row_to_entity(row) for row in result.result_set]

    def _row_to_entity(self, row) -> EntityNode:
        """Convert query result row to EntityNode (without chunk_id - use edges)."""
        return EntityNode(
            id=row[0],
            entity_type=row[1],
            canonical_name=row[2],
            norm_key=row[3],
            created_at=row[4],
            embedding=row[5],
            expired_at=row[6],
            merged_into=row[7],
            chunk_id=None,  # Use LINKED edges instead
        )

    def get_entities_by_type(
        self, entity_type: str, include_expired: bool = False
    ) -> list[EntityNode]:
        """Get entities by type."""
        expired_filter = "AND e.expired_at IS NULL" if not include_expired else ""

        result = self._graph.ro_query(
            f"""
            MATCH (e:Entity)
            WHERE e.entity_type = $type {expired_filter}
            RETURN e.id, e.entity_type, e.canonical_name, e.norm_key,
                   e.created_at, e.embedding, e.expired_at, e.merged_into
            ORDER BY e.created_at DESC
            """,
            {"type": entity_type},
        )
        return [self._row_to_entity(row) for row in result.result_set]

    def get_entities_in_chunk(
        self, chunk_id: str, include_expired: bool = False
    ) -> list[EntityNode]:
        """Get entities via LINKED edge (M6)."""
        expired_filter = "AND e.expired_at IS NULL" if not include_expired else ""

        result = self._graph.ro_query(
            f"""
            MATCH (c:Chunk {{id: $chunk_id}})-[:LINKED]->(e:Entity)
            WHERE 1=1 {expired_filter}
            RETURN e.id, e.entity_type, e.canonical_name, e.norm_key,
                   e.created_at, e.embedding, e.expired_at, e.merged_into
            ORDER BY e.created_at ASC
            """,
            {"chunk_id": chunk_id},
        )
        return [self._row_to_entity(row) for row in result.result_set]

    def list_entities_with_mention_counts(
        self, include_expired: bool = False, limit: int | None = None
    ) -> list[tuple]:
        """List entities with mention counts."""
        expired_filter = "WHERE e.expired_at IS NULL" if not include_expired else ""
        limit_clause = f"LIMIT {limit}" if limit else ""

        result = self._graph.ro_query(
            f"""
            MATCH (e:Entity)
            {expired_filter}
            OPTIONAL MATCH (m:Memory)-[r:MENTIONS]->(e)
            WHERE r.t_expired IS NULL
            WITH e, COUNT(r) AS mentions
            RETURN e.id, e.entity_type, e.canonical_name, e.created_at,
                   e.expired_at, e.merged_into, mentions
            ORDER BY mentions DESC, e.created_at DESC
            {limit_clause}
            """
        )
        return [tuple(row) for row in result.result_set]

    def expire_entity(self, entity_id: str, merged_into: str | None = None) -> None:
        """Mark entity as expired."""
        now = datetime.now(timezone.utc).isoformat()
        self._graph.query(
            """
            MATCH (e:Entity {id: $id})
            SET e.expired_at = $now, e.merged_into = $merged_into
            """,
            {"id": entity_id, "now": now, "merged_into": merged_into},
        )

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def store_edge(self, edge: EdgeNode) -> str:
        """Store edge with deduplication."""
        # Check for existing edge
        result = self._graph.ro_query(
            """
            MATCH (a {id: $from})-[r]->(b {id: $to})
            WHERE type(r) = $type AND r.t_expired IS NULL
            RETURN r.edge_id
            """,
            {"from": edge.from_node, "to": edge.to_node, "type": edge.edge_type},
        )
        if result.result_set:
            return result.result_set[0][0]

        # Create edge based on type
        # Note: Cypher requires different syntax for relationship creation
        self._graph.query(
            f"""
            MATCH (a {{id: $from}}), (b {{id: $to}})
            CREATE (a)-[r:{edge.edge_type} {{
                edge_id: $edge_id,
                weight: $weight,
                confidence: $confidence,
                evidence: $evidence,
                t_valid: $t_valid,
                t_invalid: $t_invalid,
                t_created: $t_created,
                t_expired: $t_expired
            }}]->(b)
            """,
            {
                "from": edge.from_node,
                "to": edge.to_node,
                "edge_id": edge.edge_id,
                "weight": edge.weight,
                "confidence": edge.confidence,
                "evidence": edge.evidence,
                "t_valid": edge.t_valid,
                "t_invalid": edge.t_invalid,
                "t_created": edge.t_created,
                "t_expired": edge.t_expired,
            },
        )
        return edge.edge_id

    def get_edge(self, edge_id: str) -> EdgeNode | None:
        """Retrieve edge by ID."""
        result = self._graph.ro_query(
            """
            MATCH (a)-[r {edge_id: $edge_id}]->(b)
            RETURN r.edge_id, a.id, b.id, type(r), r.weight, r.confidence,
                   r.evidence, r.t_valid, r.t_invalid, r.t_created, r.t_expired
            """,
            {"edge_id": edge_id},
        )
        if not result.result_set:
            return None

        row = result.result_set[0]
        return EdgeNode(
            edge_id=row[0],
            from_node=row[1],
            to_node=row[2],
            edge_type=row[3],
            weight=row[4] or 1.0,
            confidence=row[5],
            evidence=row[6],
            t_valid=row[7],
            t_invalid=row[8],
            t_created=row[9],
            t_expired=row[10],
        )

    def get_edges_from_memory(
        self,
        memory_id: str,
        edge_type: str | None = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list[EdgeNode]:
        """Get outgoing edges from memory."""
        type_filter = f"AND type(r) = '{edge_type}'" if edge_type else ""
        expired_filter = "" if include_expired else "AND r.t_expired IS NULL"
        conf_filter = f"AND (r.confidence IS NULL OR r.confidence >= {min_confidence})"

        result = self._graph.ro_query(
            f"""
            MATCH (m:Memory {{id: $memory_id}})-[r]->(n)
            WHERE 1=1 {type_filter} {expired_filter} {conf_filter}
            RETURN r.edge_id, m.id, n.id, type(r), r.weight, r.confidence,
                   r.evidence, r.t_valid, r.t_invalid, r.t_created, r.t_expired
            ORDER BY r.t_created DESC
            """,
            {"memory_id": memory_id},
        )
        return [self._row_to_edge(row) for row in result.result_set]

    def get_edges_to_entity(
        self,
        entity_id: str,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list[EdgeNode]:
        """Get incoming edges to entity."""
        expired_filter = "" if include_expired else "AND r.t_expired IS NULL"
        conf_filter = f"AND (r.confidence IS NULL OR r.confidence >= {min_confidence})"

        result = self._graph.ro_query(
            f"""
            MATCH (n)-[r]->(e:Entity {{id: $entity_id}})
            WHERE 1=1 {expired_filter} {conf_filter}
            RETURN r.edge_id, n.id, e.id, type(r), r.weight, r.confidence,
                   r.evidence, r.t_valid, r.t_invalid, r.t_created, r.t_expired
            ORDER BY r.t_created DESC
            """,
            {"entity_id": entity_id},
        )
        return [self._row_to_edge(row) for row in result.result_set]

    def get_edges_to_memory(
        self,
        memory_id: str,
        edge_type: str | None = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list[EdgeNode]:
        """Get incoming edges to memory."""
        type_filter = f"AND type(r) = '{edge_type}'" if edge_type else ""
        expired_filter = "" if include_expired else "AND r.t_expired IS NULL"
        conf_filter = f"AND (r.confidence IS NULL OR r.confidence >= {min_confidence})"

        result = self._graph.ro_query(
            f"""
            MATCH (n)-[r]->(m:Memory {{id: $memory_id}})
            WHERE 1=1 {type_filter} {expired_filter} {conf_filter}
            RETURN r.edge_id, n.id, m.id, type(r), r.weight, r.confidence,
                   r.evidence, r.t_valid, r.t_invalid, r.t_created, r.t_expired
            ORDER BY r.t_created DESC
            """,
            {"memory_id": memory_id},
        )
        return [self._row_to_edge(row) for row in result.result_set]

    def _row_to_edge(self, row) -> EdgeNode:
        """Convert query result row to EdgeNode."""
        return EdgeNode(
            edge_id=row[0],
            from_node=row[1],
            to_node=row[2],
            edge_type=row[3],
            weight=row[4] or 1.0,
            confidence=row[5],
            evidence=row[6],
            t_valid=row[7],
            t_invalid=row[8],
            t_created=row[9],
            t_expired=row[10],
        )

    def expire_edge(self, edge_id: str) -> None:
        """Mark edge as expired."""
        now = datetime.now(timezone.utc).isoformat()
        self._graph.query(
            """
            MATCH ()-[r {edge_id: $edge_id}]->()
            SET r.t_expired = $now
            """,
            {"edge_id": edge_id, "now": now},
        )

    def delete_edges_by_type(self, edge_type: str) -> int:
        """Hard delete all edges of type."""
        result = self._graph.query(
            f"""
            MATCH ()-[r:{edge_type}]->()
            WITH r, COUNT(r) as cnt
            DELETE r
            RETURN cnt
            """
        )
        return result.result_set[0][0] if result.result_set else 0

    def delete_edges_from_node(self, node_id: str, edge_type: str | None = None) -> int:
        """Delete outbound edges from node."""
        if edge_type:
            result = self._graph.query(
                f"""
                MATCH (n {{id: $node_id}})-[r:{edge_type}]->()
                WITH r, COUNT(r) as cnt
                DELETE r
                RETURN cnt
                """,
                {"node_id": node_id},
            )
        else:
            result = self._graph.query(
                """
                MATCH (n {id: $node_id})-[r]->()
                WITH r, COUNT(r) as cnt
                DELETE r
                RETURN cnt
                """,
                {"node_id": node_id},
            )
        return result.result_set[0][0] if result.result_set else 0

    def list_edges(
        self,
        edge_type: str | None = None,
        include_expired: bool = False,
        limit: int = 100,
    ) -> list[tuple]:
        """List edges for CLI display."""
        type_filter = f"AND type(r) = '{edge_type}'" if edge_type else ""
        expired_filter = "" if include_expired else "AND r.t_expired IS NULL"

        result = self._graph.ro_query(
            f"""
            MATCH (a)-[r]->(b)
            WHERE 1=1 {type_filter} {expired_filter}
            RETURN r.edge_id, a.id, b.id, type(r), r.weight, r.confidence,
                   r.t_created, r.t_expired
            ORDER BY r.t_created DESC
            LIMIT {limit}
            """
        )
        return [tuple(row) for row in result.result_set]

    # =========================================================================
    # File and Chunk Operations
    # =========================================================================

    def store_file(self, file_node: FileNode) -> str:
        """Store file metadata."""
        self._graph.query(
            """
            CREATE (f:File {
                id: $id,
                path: $path,
                created_at: $created_at,
                ingested_at: $ingested_at,
                file_hash: $file_hash,
                metadata: $metadata
            })
            """,
            {
                "id": file_node.file_id,
                "path": file_node.path,
                "created_at": file_node.created_at,
                "ingested_at": file_node.ingested_at,
                "file_hash": file_node.file_hash,
                "metadata": json.dumps(file_node.metadata),
            },
        )
        return file_node.file_id

    def get_file(self, file_id: str) -> FileNode | None:
        """Retrieve file by ID."""
        result = self._graph.ro_query(
            """
            MATCH (f:File {id: $id})
            RETURN f.id, f.path, f.created_at, f.ingested_at, f.file_hash, f.metadata
            """,
            {"id": file_id},
        )
        if not result.result_set:
            return None

        row = result.result_set[0]
        return FileNode(
            file_id=row[0],
            path=row[1],
            created_at=row[2],
            ingested_at=row[3],
            file_hash=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
        )

    def find_file_by_path(self, path: str) -> FileNode | None:
        """Find file by path."""
        result = self._graph.ro_query(
            """
            MATCH (f:File {path: $path})
            RETURN f.id, f.path, f.created_at, f.ingested_at, f.file_hash, f.metadata
            ORDER BY f.ingested_at DESC
            LIMIT 1
            """,
            {"path": path},
        )
        if not result.result_set:
            return None

        row = result.result_set[0]
        return FileNode(
            file_id=row[0],
            path=row[1],
            created_at=row[2],
            ingested_at=row[3],
            file_hash=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
        )

    def store_chunk(self, chunk_node: ChunkNode) -> str:
        """Store chunk pointer."""
        self._graph.query(
            """
            CREATE (c:Chunk {
                id: $id,
                file_id: $file_id,
                start: $start,
                length: $length,
                sequence: $sequence,
                created_at: $created_at
            })
            """,
            {
                "id": chunk_node.chunk_id,
                "file_id": chunk_node.file_id,
                "start": chunk_node.start,
                "length": chunk_node.length,
                "sequence": chunk_node.sequence,
                "created_at": chunk_node.created_at,
            },
        )
        return chunk_node.chunk_id

    def get_chunk(self, chunk_id: str) -> ChunkNode | None:
        """Retrieve chunk by ID."""
        result = self._graph.ro_query(
            """
            MATCH (c:Chunk {id: $id})
            RETURN c.id, c.file_id, c.start, c.length, c.sequence, c.created_at
            """,
            {"id": chunk_id},
        )
        if not result.result_set:
            return None

        row = result.result_set[0]
        return ChunkNode(
            chunk_id=row[0],
            file_id=row[1],
            start=row[2],
            length=row[3],
            sequence=row[4],
            created_at=row[5],
        )

    def get_chunks_by_file(self, file_id: str) -> list[ChunkNode]:
        """Get chunks for file, ordered by sequence."""
        result = self._graph.ro_query(
            """
            MATCH (c:Chunk {file_id: $file_id})
            RETURN c.id, c.file_id, c.start, c.length, c.sequence, c.created_at
            ORDER BY c.sequence ASC
            """,
            {"file_id": file_id},
        )
        return [
            ChunkNode(
                chunk_id=row[0],
                file_id=row[1],
                start=row[2],
                length=row[3],
                sequence=row[4],
                created_at=row[5],
            )
            for row in result.result_set
        ]

    def get_chunk_for_memory(self, memory_id: str) -> ChunkNode | None:
        """Get chunk via CONTAINS edge (M6)."""
        result = self._graph.ro_query(
            """
            MATCH (c:Chunk)-[:CONTAINS]->(m:Memory {id: $memory_id})
            RETURN c.id, c.file_id, c.start, c.length, c.sequence, c.created_at
            LIMIT 1
            """,
            {"memory_id": memory_id},
        )
        if not result.result_set:
            return None

        row = result.result_set[0]
        return ChunkNode(
            chunk_id=row[0],
            file_id=row[1],
            start=row[2],
            length=row[3],
            sequence=row[4],
            created_at=row[5],
        )

    def get_summary_for_chunk_via_edge(self, chunk_id: str) -> MemoryNode | None:
        """Get summary via SUMMARIZED_BY edge (M6)."""
        result = self._graph.ro_query(
            """
            MATCH (c:Chunk {id: $chunk_id})-[:SUMMARIZED_BY]->(m:Memory)
            RETURN m.id, m.content, m.content_embedding, m.content_hash,
                   m.created_at, m.metadata, m.kind, m.t_valid, m.t_invalid,
                   m.t_created, m.t_expired, m.temporal_stability,
                   m.last_seen_at, m.reinforce_count
            LIMIT 1
            """,
            {"chunk_id": chunk_id},
        )
        if not result.result_set:
            return None
        return self._row_to_memory(result.result_set[0])

    # =========================================================================
    # Count and Statistics Operations
    # =========================================================================

    def count_memories(self, kind: str | None = None) -> int:
        """Count memories."""
        if kind:
            result = self._graph.ro_query(
                "MATCH (m:Memory {kind: $kind}) RETURN COUNT(m)",
                {"kind": kind},
            )
        else:
            result = self._graph.ro_query("MATCH (m:Memory) RETURN COUNT(m)")
        return result.result_set[0][0] if result.result_set else 0

    def count_entities(self) -> int:
        """Count entities."""
        result = self._graph.ro_query("MATCH (e:Entity) RETURN COUNT(e)")
        return result.result_set[0][0] if result.result_set else 0

    def count_edges(self, edge_type: str | None = None) -> int:
        """Count edges."""
        if edge_type:
            result = self._graph.ro_query(f"MATCH ()-[r:{edge_type}]->() RETURN COUNT(r)")
        else:
            result = self._graph.ro_query("MATCH ()-[r]->() RETURN COUNT(r)")
        return result.result_set[0][0] if result.result_set else 0

    def count_memories_with_edge_type(self, edge_type: str) -> int:
        """Count distinct memories with outgoing edges of type."""
        result = self._graph.ro_query(
            f"""
            MATCH (m:Memory)-[r:{edge_type}]->()
            RETURN COUNT(DISTINCT m)
            """
        )
        return result.result_set[0][0] if result.result_set else 0

    # =========================================================================
    # Housekeeping Operations
    # =========================================================================

    def get_orphaned_memories(self) -> list[tuple[str, str, str]]:
        """Find memories without provenance (no CONTAINS edge) and no entity links.

        Returns:
            List of (memory_id, content, created_at) tuples for orphaned memories.
        """
        result = self._graph.ro_query(
            """
            MATCH (m:Memory)
            WHERE m.t_expired IS NULL AND m.kind = 'MEMORY'
              AND NOT (:Chunk)-[:CONTAINS]->(m)
              AND NOT (m)-[:MENTIONS]->(:Entity)
            RETURN m.id, m.content, m.created_at
            ORDER BY m.created_at
            """
        )
        return [(row[0], row[1], row[2]) for row in result.result_set]

    def get_memories_without_source(self) -> list[tuple[str, str, str]]:
        """Find memories without Source linkage (direct or via Chunk).

        For FalkorDB: memories without incoming CONTAINS edge from Chunk.
        This is effectively the same as orphaned for now, until Source abstraction.

        Returns:
            List of (memory_id, content, created_at) tuples.
        """
        result = self._graph.ro_query(
            """
            MATCH (m:Memory)
            WHERE m.t_expired IS NULL AND m.kind = 'MEMORY'
              AND NOT (:Chunk)-[:CONTAINS]->(m)
            RETURN m.id, m.content, m.created_at
            ORDER BY m.created_at
            """
        )
        return [(row[0], row[1], row[2]) for row in result.result_set]

    # =========================================================================
    # Bulk Delete Operations
    # =========================================================================

    def delete_all_entities(self) -> int:
        """Delete all entities."""
        result = self._graph.query(
            """
            MATCH (e:Entity)
            WITH e, COUNT(e) as cnt
            DETACH DELETE e
            RETURN cnt
            """
        )
        return result.result_set[0][0] if result.result_set else 0

    # =========================================================================
    # Transaction and Lifecycle Management
    # =========================================================================

    @contextmanager
    def transaction(self):
        """Context manager for transactions.

        Note: FalkorDB doesn't have traditional transaction support like SQL.
        Each query is atomic. This is provided for interface compatibility.
        """
        # FalkorDB queries are atomic, so just yield
        yield

    def commit(self) -> None:
        """Explicit commit - no-op for FalkorDB (queries auto-commit)."""
        pass

    def close(self) -> None:
        """Close database connection."""
        # FalkorDB uses connection pooling, no explicit close needed
        pass

    # =========================================================================
    # Event Storage
    # =========================================================================

    @property
    def event_storage(self) -> FalkorEventStorage:
        """Return event storage."""
        return self._event_storage
