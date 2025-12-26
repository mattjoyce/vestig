"""Storage layer using SQLite with M2 dedupe support"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from vestig.core.models import MemoryNode, EntityNode, EdgeNode


class MemoryStorage:
    """SQLite storage for memory nodes with M2 dedupe"""

    def __init__(self, db_path: str):
        """
        Initialize storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))

        # M3 FIX: Enable foreign key constraints (SQLite doesn't enforce by default)
        self.conn.execute("PRAGMA foreign_keys = ON")

        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist (additive migrations)"""
        # Create base table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )

        # M2: Add content_hash column if it doesn't exist (additive migration)
        cursor = self.conn.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]

        if "content_hash" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT")

        # M3 FIX: Always ensure unique index exists (even if column already present)
        # Create unique index on content_hash for dedupe
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_content_hash
            ON memories(content_hash)
            """
        )

        # M3: Add temporal columns (check first, then ALTER TABLE)
        cursor = self.conn.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]

        if "t_valid" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN t_valid TEXT")
        if "t_invalid" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN t_invalid TEXT")
        if "t_created" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN t_created TEXT")
        if "t_expired" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN t_expired TEXT")
        if "temporal_stability" not in columns:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN temporal_stability TEXT DEFAULT 'unknown'"
            )
        if "last_seen_at" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN last_seen_at TEXT")
        if "reinforce_count" not in columns:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN reinforce_count INTEGER DEFAULT 0"
            )

        # Backfill t_created from created_at for existing memories
        self.conn.execute(
            "UPDATE memories SET t_created = created_at WHERE t_created IS NULL"
        )

        # M3: Create memory_events table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                source TEXT NOT NULL,
                actor TEXT,
                artifact_ref TEXT,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
            """
        )

        # M3: Create indexes for temporal queries
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_memory_time
            ON memory_events(memory_id, occurred_at DESC)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON memory_events(event_type)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_expired
            ON memories(t_expired) WHERE t_expired IS NOT NULL
            """
        )

        # M4: Create entities table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                norm_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expired_at TEXT,
                merged_into TEXT
            )
            """
        )

        # M4: Create indexes for entity queries
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_norm_key
            ON entities(norm_key)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entities_type
            ON entities(entity_type)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entities_expired
            ON entities(expired_at) WHERE expired_at IS NOT NULL
            """
        )

        # M4: Create edges table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                from_node TEXT NOT NULL,
                to_node TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                confidence REAL,
                evidence TEXT,
                t_valid TEXT,
                t_invalid TEXT,
                t_created TEXT,
                t_expired TEXT
            )
            """
        )

        # M4: Create indexes for edge queries
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_from_node
            ON edges(from_node, edge_type)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_to_node
            ON edges(to_node, edge_type)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_type
            ON edges(edge_type)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_confidence
            ON edges(confidence) WHERE confidence IS NOT NULL
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_expired
            ON edges(t_expired) WHERE t_expired IS NOT NULL
            """
        )

        self.conn.commit()

    def store_memory(self, node: MemoryNode) -> str:
        """
        Persist a memory node (M2: handles exact duplicates).

        Args:
            node: MemoryNode to store

        Returns:
            Memory ID (existing ID if duplicate detected)
        """
        # Check for exact duplicate via content_hash
        cursor = self.conn.execute(
            "SELECT id FROM memories WHERE content_hash = ?",
            (node.content_hash,),
        )
        existing = cursor.fetchone()

        if existing:
            # Exact duplicate found - return existing ID (nice UX)
            return existing[0]

        # No duplicate - insert new memory
        self.conn.execute(
            """
            INSERT INTO memories (
                id, content, content_embedding, content_hash, created_at, metadata,
                t_valid, t_invalid, t_created, t_expired, temporal_stability,
                last_seen_at, reinforce_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.content,
                json.dumps(node.content_embedding),
                node.content_hash,
                node.created_at,
                json.dumps(node.metadata),
                node.t_valid,
                node.t_invalid,
                node.t_created,
                node.t_expired,
                node.temporal_stability,
                node.last_seen_at,
                node.reinforce_count,
            ),
        )
        # NOTE: Caller manages transaction commit
        return node.id

    def get_memory(self, memory_id: str) -> Optional[MemoryNode]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            MemoryNode if found, None otherwise
        """
        cursor = self.conn.execute(
            """
            SELECT id, content, content_embedding, content_hash, created_at, metadata,
                   t_valid, t_invalid, t_created, t_expired, temporal_stability,
                   last_seen_at, reinforce_count
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return MemoryNode(
            id=row[0],
            content=row[1],
            content_embedding=json.loads(row[2]),
            content_hash=row[3],
            created_at=row[4],
            metadata=json.loads(row[5]),
            t_valid=row[6],
            t_invalid=row[7],
            t_created=row[8],
            t_expired=row[9],
            temporal_stability=row[10] or "unknown",
            last_seen_at=row[11],
            reinforce_count=row[12] or 0,
        )

    def get_all_memories(self) -> List[MemoryNode]:
        """
        Load all memories (for brute-force search).

        Returns:
            List of all MemoryNode objects
        """
        cursor = self.conn.execute(
            """
            SELECT id, content, content_embedding, content_hash, created_at, metadata,
                   t_valid, t_invalid, t_created, t_expired, temporal_stability,
                   last_seen_at, reinforce_count
            FROM memories
            ORDER BY created_at DESC
            """
        )
        memories = []
        for row in cursor.fetchall():
            memories.append(
                MemoryNode(
                    id=row[0],
                    content=row[1],
                    content_embedding=json.loads(row[2]),
                    content_hash=row[3],
                    created_at=row[4],
                    metadata=json.loads(row[5]),
                    t_valid=row[6],
                    t_invalid=row[7],
                    t_created=row[8],
                    t_expired=row[9],
                    temporal_stability=row[10] or "unknown",
                    last_seen_at=row[11],
                    reinforce_count=row[12] or 0,
                )
            )
        return memories

    def close(self):
        """Close database connection"""
        self.conn.close()

    # M3: Temporal operations

    def increment_reinforce_count(self, memory_id: str) -> None:
        """Increment reinforce_count (convenience cache for TraceRank)"""
        self.conn.execute(
            "UPDATE memories SET reinforce_count = reinforce_count + 1 WHERE id = ?",
            (memory_id,),
        )
        # NOTE: Caller manages transaction commit

    def update_last_seen(self, memory_id: str, timestamp: str) -> None:
        """Update last_seen_at timestamp"""
        self.conn.execute(
            "UPDATE memories SET last_seen_at = ? WHERE id = ?",
            (timestamp, memory_id),
        )
        # NOTE: Caller manages transaction commit

    def deprecate_memory(
        self, memory_id: str, t_invalid: Optional[str] = None
    ) -> None:
        """Mark memory as deprecated/expired"""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            UPDATE memories
            SET t_expired = ?, t_invalid = COALESCE(?, t_invalid)
            WHERE id = ?
            """,
            (now, t_invalid, memory_id),
        )
        # NOTE: Caller manages transaction commit

    def get_active_memories(self) -> List[MemoryNode]:
        """Get all non-expired memories (for retrieval)"""
        cursor = self.conn.execute(
            """
            SELECT id, content, content_embedding, content_hash, created_at, metadata,
                   t_valid, t_invalid, t_created, t_expired, temporal_stability,
                   last_seen_at, reinforce_count
            FROM memories
            WHERE t_expired IS NULL
            ORDER BY created_at DESC
            """
        )
        memories = []
        for row in cursor.fetchall():
            memories.append(
                MemoryNode(
                    id=row[0],
                    content=row[1],
                    content_embedding=json.loads(row[2]),
                    content_hash=row[3],
                    created_at=row[4],
                    metadata=json.loads(row[5]),
                    t_valid=row[6],
                    t_invalid=row[7],
                    t_created=row[8],
                    t_expired=row[9],
                    temporal_stability=row[10] or "unknown",
                    last_seen_at=row[11],
                    reinforce_count=row[12] or 0,
                )
            )
        return memories

    # M4: Entity operations

    def store_entity(self, entity: EntityNode) -> str:
        """
        Persist an entity node (M4: Graph Layer).

        Deduplication via norm_key:
        - If entity with same norm_key exists and not expired, return existing ID
        - Otherwise insert new entity

        Args:
            entity: EntityNode to store

        Returns:
            Entity ID (existing ID if duplicate detected via norm_key)
        """
        # Check for existing entity via norm_key (not expired)
        cursor = self.conn.execute(
            "SELECT id FROM entities WHERE norm_key = ? AND expired_at IS NULL",
            (entity.norm_key,),
        )
        existing = cursor.fetchone()

        if existing:
            # Duplicate found via norm_key - return existing ID
            return existing[0]

        # No duplicate - insert new entity
        self.conn.execute(
            """
            INSERT INTO entities (id, entity_type, canonical_name, norm_key, created_at, expired_at, merged_into)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.id,
                entity.entity_type,
                entity.canonical_name,
                entity.norm_key,
                entity.created_at,
                entity.expired_at,
                entity.merged_into,
            ),
        )
        self.conn.commit()
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        """
        Retrieve entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            EntityNode if found, None otherwise
        """
        cursor = self.conn.execute(
            """
            SELECT id, entity_type, canonical_name, norm_key, created_at, expired_at, merged_into
            FROM entities
            WHERE id = ?
            """,
            (entity_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        return EntityNode(
            id=row[0],
            entity_type=row[1],
            canonical_name=row[2],
            norm_key=row[3],
            created_at=row[4],
            expired_at=row[5],
            merged_into=row[6],
        )

    def find_entity_by_norm_key(
        self, norm_key: str, include_expired: bool = False
    ) -> Optional[EntityNode]:
        """
        Find entity by normalization key (deduplication lookup).

        Args:
            norm_key: Normalization key (format: "TYPE:normalized_name")
            include_expired: Include expired entities in search

        Returns:
            EntityNode if found, None otherwise
        """
        if include_expired:
            query = "SELECT id, entity_type, canonical_name, norm_key, created_at, expired_at, merged_into FROM entities WHERE norm_key = ?"
        else:
            query = "SELECT id, entity_type, canonical_name, norm_key, created_at, expired_at, merged_into FROM entities WHERE norm_key = ? AND expired_at IS NULL"

        cursor = self.conn.execute(query, (norm_key,))
        row = cursor.fetchone()

        if not row:
            return None

        return EntityNode(
            id=row[0],
            entity_type=row[1],
            canonical_name=row[2],
            norm_key=row[3],
            created_at=row[4],
            expired_at=row[5],
            merged_into=row[6],
        )

    def get_all_entities(self, include_expired: bool = False) -> List[EntityNode]:
        """
        Get all entities across all types.

        Args:
            include_expired: Include expired entities

        Returns:
            List of EntityNode objects
        """
        if include_expired:
            query = """
                SELECT id, entity_type, canonical_name, norm_key, created_at, expired_at, merged_into
                FROM entities
                ORDER BY created_at DESC
            """
        else:
            query = """
                SELECT id, entity_type, canonical_name, norm_key, created_at, expired_at, merged_into
                FROM entities
                WHERE expired_at IS NULL
                ORDER BY created_at DESC
            """

        cursor = self.conn.execute(query)
        entities = []

        for row in cursor.fetchall():
            entities.append(
                EntityNode(
                    id=row[0],
                    entity_type=row[1],
                    canonical_name=row[2],
                    norm_key=row[3],
                    created_at=row[4],
                    expired_at=row[5],
                    merged_into=row[6],
                )
            )

        return entities

    def get_entities_by_type(
        self, entity_type: str, include_expired: bool = False
    ) -> List[EntityNode]:
        """
        Get all entities of a specific type.

        Args:
            entity_type: Entity type (PERSON, ORG, SYSTEM, etc.)
            include_expired: Include expired entities

        Returns:
            List of EntityNode objects
        """
        if include_expired:
            query = """
                SELECT id, entity_type, canonical_name, norm_key, created_at, expired_at, merged_into
                FROM entities
                WHERE entity_type = ?
                ORDER BY created_at DESC
            """
        else:
            query = """
                SELECT id, entity_type, canonical_name, norm_key, created_at, expired_at, merged_into
                FROM entities
                WHERE entity_type = ? AND expired_at IS NULL
                ORDER BY created_at DESC
            """

        cursor = self.conn.execute(query, (entity_type,))
        entities = []

        for row in cursor.fetchall():
            entities.append(
                EntityNode(
                    id=row[0],
                    entity_type=row[1],
                    canonical_name=row[2],
                    norm_key=row[3],
                    created_at=row[4],
                    expired_at=row[5],
                    merged_into=row[6],
                )
            )

        return entities

    def expire_entity(
        self, entity_id: str, merged_into: Optional[str] = None
    ) -> None:
        """
        Mark entity as expired (soft delete / merge).

        Args:
            entity_id: Entity ID to expire
            merged_into: Optional ID of entity this was merged into
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            "UPDATE entities SET expired_at = ?, merged_into = ? WHERE id = ?",
            (now, merged_into, entity_id),
        )
        self.conn.commit()

    # M4: Edge operations

    def store_edge(self, edge: EdgeNode) -> str:
        """
        Persist an edge (M4: Graph Layer).

        Args:
            edge: EdgeNode to store

        Returns:
            Edge ID

        Raises:
            ValueError: If edge_type is invalid (enforced in EdgeNode.create())
        """
        # Check for duplicate edge (same from/to/type, not expired)
        cursor = self.conn.execute(
            """
            SELECT edge_id FROM edges 
            WHERE from_node = ? AND to_node = ? AND edge_type = ? AND t_expired IS NULL
            """,
            (edge.from_node, edge.to_node, edge.edge_type),
        )
        existing = cursor.fetchone()

        if existing:
            # Duplicate edge found - return existing ID
            return existing[0]

        # Insert new edge
        self.conn.execute(
            """
            INSERT INTO edges (edge_id, from_node, to_node, edge_type, weight, 
                              confidence, evidence, t_valid, t_invalid, t_created, t_expired)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.edge_id,
                edge.from_node,
                edge.to_node,
                edge.edge_type,
                edge.weight,
                edge.confidence,
                edge.evidence,
                edge.t_valid,
                edge.t_invalid,
                edge.t_created,
                edge.t_expired,
            ),
        )
        self.conn.commit()
        return edge.edge_id

    def get_edges_from_memory(
        self,
        memory_id: str,
        edge_type: Optional[str] = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> List[EdgeNode]:
        """
        Get all outgoing edges from a memory node.

        Args:
            memory_id: Source memory ID
            edge_type: Optional edge type filter (MENTIONS, RELATED)
            include_expired: Include expired edges
            min_confidence: Minimum confidence threshold (0.0 = all)

        Returns:
            List of EdgeNode objects
        """
        # Build query based on filters
        if edge_type:
            if include_expired:
                query = """
                    SELECT edge_id, from_node, to_node, edge_type, weight,
                           confidence, evidence, t_valid, t_invalid, t_created, t_expired
                    FROM edges
                    WHERE from_node = ? AND edge_type = ? AND (confidence IS NULL OR confidence >= ?)
                    ORDER BY t_created DESC
                """
                params = (memory_id, edge_type, min_confidence)
            else:
                query = """
                    SELECT edge_id, from_node, to_node, edge_type, weight,
                           confidence, evidence, t_valid, t_invalid, t_created, t_expired
                    FROM edges
                    WHERE from_node = ? AND edge_type = ? AND t_expired IS NULL AND (confidence IS NULL OR confidence >= ?)
                    ORDER BY t_created DESC
                """
                params = (memory_id, edge_type, min_confidence)
        else:
            if include_expired:
                query = """
                    SELECT edge_id, from_node, to_node, edge_type, weight,
                           confidence, evidence, t_valid, t_invalid, t_created, t_expired
                    FROM edges
                    WHERE from_node = ? AND (confidence IS NULL OR confidence >= ?)
                    ORDER BY t_created DESC
                """
                params = (memory_id, min_confidence)
            else:
                query = """
                    SELECT edge_id, from_node, to_node, edge_type, weight,
                       confidence, evidence, t_valid, t_invalid, t_created, t_expired
                FROM edges
                WHERE from_node = ? AND t_expired IS NULL AND (confidence IS NULL OR confidence >= ?)
                ORDER BY t_created DESC
                """
                params = (memory_id, min_confidence)

        cursor = self.conn.execute(query, params)
        edges = []

        for row in cursor.fetchall():
            edges.append(
                EdgeNode(
                    edge_id=row[0],
                    from_node=row[1],
                    to_node=row[2],
                    edge_type=row[3],
                    weight=row[4],
                    confidence=row[5],
                    evidence=row[6],
                    t_valid=row[7],
                    t_invalid=row[8],
                    t_created=row[9],
                    t_expired=row[10],
                )
            )

        return edges

    def get_edges_to_entity(
        self,
        entity_id: str,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> List[EdgeNode]:
        """
        Get all incoming edges to an entity node.

        Args:
            entity_id: Target entity ID
            include_expired: Include expired edges
            min_confidence: Minimum confidence threshold

        Returns:
            List of EdgeNode objects
        """
        if include_expired:
            query = """
                SELECT edge_id, from_node, to_node, edge_type, weight, 
                       confidence, evidence, t_valid, t_invalid, t_created, t_expired
                FROM edges
                WHERE to_node = ? AND (confidence IS NULL OR confidence >= ?)
                ORDER BY t_created DESC
            """
        else:
            query = """
                SELECT edge_id, from_node, to_node, edge_type, weight, 
                       confidence, evidence, t_valid, t_invalid, t_created, t_expired
                FROM edges
                WHERE to_node = ? AND t_expired IS NULL AND (confidence IS NULL OR confidence >= ?)
                ORDER BY t_created DESC
            """

        cursor = self.conn.execute(query, (entity_id, min_confidence))
        edges = []

        for row in cursor.fetchall():
            edges.append(
                EdgeNode(
                    edge_id=row[0],
                    from_node=row[1],
                    to_node=row[2],
                    edge_type=row[3],
                    weight=row[4],
                    confidence=row[5],
                    evidence=row[6],
                    t_valid=row[7],
                    t_invalid=row[8],
                    t_created=row[9],
                    t_expired=row[10],
                )
            )

        return edges

    def get_edge(self, edge_id: str) -> Optional[EdgeNode]:
        """
        Retrieve edge by ID.

        Args:
            edge_id: Edge ID

        Returns:
            EdgeNode if found, None otherwise
        """
        cursor = self.conn.execute(
            """
            SELECT edge_id, from_node, to_node, edge_type, weight, 
                   confidence, evidence, t_valid, t_invalid, t_created, t_expired
            FROM edges
            WHERE edge_id = ?
            """,
            (edge_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        return EdgeNode(
            edge_id=row[0],
            from_node=row[1],
            to_node=row[2],
            edge_type=row[3],
            weight=row[4],
            confidence=row[5],
            evidence=row[6],
            t_valid=row[7],
            t_invalid=row[8],
            t_created=row[9],
            t_expired=row[10],
        )

    def expire_edge(self, edge_id: str) -> None:
        """
        Mark edge as expired (soft delete / invalidation).

        Args:
            edge_id: Edge ID to expire
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            "UPDATE edges SET t_expired = ? WHERE edge_id = ?",
            (now, edge_id),
        )
        self.conn.commit()
