"""Storage layer using SQLite with M2 dedupe support"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from vestig.core.models import EdgeNode, EntityNode, MemoryNode


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
        self._validate_schema()

    def _is_fresh_database(self) -> bool:
        """Check if database is empty (no tables exist)

        Returns:
            True if this is a fresh database with no memories table
        """
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        return cursor.fetchone() is None

    def _apply_schema_file(self) -> None:
        """Apply schema.sql for fresh database creation

        Raises:
            FileNotFoundError: If schema.sql is not found
        """
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"schema.sql not found at {schema_path}")

        schema_sql = schema_path.read_text()
        self.conn.executescript(schema_sql)
        self.conn.commit()
        # Note: Using logger here would require import at top
        # logger.info("Applied schema.sql for fresh database creation")

    def _init_schema(self):
        """Initialize or migrate database schema

        Fresh databases: Apply schema.sql as single source of truth
        Existing databases: Use migration logic for backward compatibility
        """
        if self._is_fresh_database():
            # Fresh DB: Apply schema.sql as single source of truth
            self._apply_schema_file()
        else:
            # Existing DB: Keep migration logic for backward compatibility
            self._migrate_existing_database()

    def _migrate_existing_database(self):
        """Legacy migration path for pre-M0 databases (additive migrations)"""
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
            self.conn.execute("ALTER TABLE memories ADD COLUMN reinforce_count INTEGER DEFAULT 0")

        # M4: Add kind column for MEMORY vs SUMMARY nodes
        if "kind" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN kind TEXT DEFAULT 'MEMORY'")

        # Backfill t_created from created_at for existing memories
        self.conn.execute("UPDATE memories SET t_created = created_at WHERE t_created IS NULL")

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
        # M4: Create index for kind column (for summary queries)
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_kind
            ON memories(kind)
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
        # M4: UNIQUE constraint to prevent duplicate edges (belt and braces)
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
            ON edges(from_node, to_node, edge_type)
            WHERE t_expired IS NULL
            """
        )

        # M4: Add embedding column to entities (if missing)
        cursor = self.conn.execute("PRAGMA table_info(entities)")
        entity_columns = [row[1] for row in cursor.fetchall()]
        if "embedding" not in entity_columns:
            self.conn.execute("ALTER TABLE entities ADD COLUMN embedding TEXT")

        # M5: Create files table (hub layer)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                file_hash TEXT,
                metadata TEXT
            )
            """
        )

        # M5: Create chunks table (hub layer)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                start INTEGER NOT NULL,
                length INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(file_id)
            )
            """
        )

        # M5: Add chunk_id column to memories (if missing)
        cursor = self.conn.execute("PRAGMA table_info(memories)")
        memory_columns = [row[1] for row in cursor.fetchall()]
        if "chunk_id" not in memory_columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN chunk_id TEXT")

        # M5: Add chunk_id column to entities (if missing)
        cursor = self.conn.execute("PRAGMA table_info(entities)")
        entity_columns = [row[1] for row in cursor.fetchall()]
        if "chunk_id" not in entity_columns:
            self.conn.execute("ALTER TABLE entities ADD COLUMN chunk_id TEXT")

        # M5: Create indexes for files table
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_files_path
            ON files(path)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_files_ingested_at
            ON files(ingested_at DESC)
            """
        )

        # M5: Create indexes for chunks table
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_file
            ON chunks(file_id, sequence)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_position
            ON chunks(file_id, start, length)
            """
        )

        # M5: Create indexes for chunk provenance
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_chunk
            ON memories(chunk_id) WHERE chunk_id IS NOT NULL
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entities_chunk
            ON entities(chunk_id) WHERE chunk_id IS NOT NULL
            """
        )

        self.conn.commit()

    def _validate_schema(self) -> None:
        """Validate database schema matches expectations (fail loudly on mismatch)

        Raises:
            RuntimeError: If required tables or columns are missing
        """
        # Check critical columns exist in memories table
        cursor = self.conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}

        required_columns = {
            "id",
            "content",
            "content_embedding",
            "content_hash",
            "created_at",
            "metadata",
            "t_valid",
            "t_invalid",
            "t_created",
            "t_expired",
            "temporal_stability",
            "last_seen_at",
            "reinforce_count",
            "kind",
        }

        missing = required_columns - columns
        if missing:
            raise RuntimeError(
                f"Schema validation failed: Missing columns in memories table: {missing}. "
                f"Database schema is incompatible with this version of Vestig. "
                f"Expected schema from M4 (Summary Nodes)."
            )

        # Check critical tables exist
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        required_tables = {"memories", "memory_events", "entities", "edges"}
        missing_tables = required_tables - tables
        if missing_tables:
            raise RuntimeError(f"Schema validation failed: Missing tables: {missing_tables}")

        # Note: Using logger here would require import at top
        # logger.debug("Schema validation passed")

    def store_memory(self, node: MemoryNode, kind: str = "MEMORY") -> str:
        """
        Persist a memory node (M2: handles exact duplicates, M4: supports kind).

        Args:
            node: MemoryNode to store
            kind: Memory kind - "MEMORY" (default) or "SUMMARY"

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
                last_seen_at, reinforce_count, kind, chunk_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                kind,
                node.chunk_id,
            ),
        )
        # NOTE: Caller manages transaction commit
        return node.id

    def get_memory(self, memory_id: str) -> MemoryNode | None:
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
                   last_seen_at, reinforce_count, chunk_id, kind
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
            chunk_id=row[13],
            kind=row[14],
        )

    def get_all_memories(self) -> list[MemoryNode]:
        """
        Load all memories (for brute-force search).

        Returns:
            List of all MemoryNode objects
        """
        cursor = self.conn.execute(
            """
            SELECT id, content, content_embedding, content_hash, created_at, metadata,
                   t_valid, t_invalid, t_created, t_expired, temporal_stability,
                   last_seen_at, reinforce_count, chunk_id, kind
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
                    chunk_id=row[13],
                    kind=row[14],
                )
            )
        return memories

    # M5: File operations

    def store_file(self, file_node: "FileNode") -> str:
        """
        Store a file node in the database.

        Args:
            file_node: FileNode to store

        Returns:
            file_id of the stored file
        """
        from vestig.core.models import FileNode

        self.conn.execute(
            """
            INSERT INTO files (file_id, path, created_at, ingested_at, file_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_node.file_id,
                file_node.path,
                file_node.created_at,
                file_node.ingested_at,
                file_node.file_hash,
                json.dumps(file_node.metadata),
            ),
        )
        self.conn.commit()
        return file_node.file_id

    def get_file(self, file_id: str) -> "FileNode | None":
        """
        Get a file by ID.

        Args:
            file_id: File ID to retrieve

        Returns:
            FileNode if found, None otherwise
        """
        from vestig.core.models import FileNode

        cursor = self.conn.execute(
            """
            SELECT file_id, path, created_at, ingested_at, file_hash, metadata
            FROM files
            WHERE file_id = ?
            """,
            (file_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return FileNode(
            file_id=row[0],
            path=row[1],
            created_at=row[2],
            ingested_at=row[3],
            file_hash=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
        )

    def find_file_by_path(self, path: str) -> "FileNode | None":
        """
        Find a file by path.

        Args:
            path: File path to search for

        Returns:
            FileNode if found, None otherwise
        """
        from vestig.core.models import FileNode

        cursor = self.conn.execute(
            """
            SELECT file_id, path, created_at, ingested_at, file_hash, metadata
            FROM files
            WHERE path = ?
            ORDER BY ingested_at DESC
            LIMIT 1
            """,
            (path,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return FileNode(
            file_id=row[0],
            path=row[1],
            created_at=row[2],
            ingested_at=row[3],
            file_hash=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
        )

    # M5: Chunk operations

    def store_chunk(self, chunk_node: "ChunkNode") -> str:
        """
        Store a chunk node in the database.

        Args:
            chunk_node: ChunkNode to store

        Returns:
            chunk_id of the stored chunk
        """
        from vestig.core.models import ChunkNode

        self.conn.execute(
            """
            INSERT INTO chunks (chunk_id, file_id, start, length, sequence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_node.chunk_id,
                chunk_node.file_id,
                chunk_node.start,
                chunk_node.length,
                chunk_node.sequence,
                chunk_node.created_at,
            ),
        )
        self.conn.commit()
        return chunk_node.chunk_id

    def get_chunk(self, chunk_id: str) -> "ChunkNode | None":
        """
        Get a chunk by ID.

        Args:
            chunk_id: Chunk ID to retrieve

        Returns:
            ChunkNode if found, None otherwise
        """
        from vestig.core.models import ChunkNode

        cursor = self.conn.execute(
            """
            SELECT chunk_id, file_id, start, length, sequence, created_at
            FROM chunks
            WHERE chunk_id = ?
            """,
            (chunk_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return ChunkNode(
            chunk_id=row[0],
            file_id=row[1],
            start=row[2],
            length=row[3],
            sequence=row[4],
            created_at=row[5],
        )

    def get_chunks_by_file(self, file_id: str) -> list["ChunkNode"]:
        """
        Get all chunks for a file, ordered by sequence.

        Args:
            file_id: File ID to get chunks for

        Returns:
            List of ChunkNode objects ordered by sequence
        """
        from vestig.core.models import ChunkNode

        cursor = self.conn.execute(
            """
            SELECT chunk_id, file_id, start, length, sequence, created_at
            FROM chunks
            WHERE file_id = ?
            ORDER BY sequence ASC
            """,
            (file_id,),
        )

        chunks = []
        for row in cursor.fetchall():
            chunks.append(
                ChunkNode(
                    chunk_id=row[0],
                    file_id=row[1],
                    start=row[2],
                    length=row[3],
                    sequence=row[4],
                    created_at=row[5],
                )
            )
        return chunks

    def get_memories_by_chunk(self, chunk_id: str, include_expired: bool = False) -> list[MemoryNode]:
        """
        Get all memories linked to a specific chunk (M5).

        Args:
            chunk_id: Chunk ID
            include_expired: Include expired memories

        Returns:
            List of MemoryNode objects in this chunk
        """
        expired_filter = "" if include_expired else "AND t_expired IS NULL"

        cursor = self.conn.execute(
            f"""
            SELECT id, content, content_embedding, content_hash, created_at, metadata,
                   t_valid, t_invalid, t_created, t_expired, temporal_stability,
                   last_seen_at, reinforce_count, chunk_id, kind
            FROM memories
            WHERE chunk_id = ? {expired_filter}
            ORDER BY created_at ASC
            """,
            (chunk_id,),
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
                    chunk_id=row[13],
                    kind=row[14],
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

    def deprecate_memory(self, memory_id: str, t_invalid: str | None = None) -> None:
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

    def get_active_memories(self) -> list[MemoryNode]:
        """Get all non-expired memories (for retrieval)"""
        cursor = self.conn.execute(
            """
            SELECT id, content, content_embedding, content_hash, created_at, metadata,
                   t_valid, t_invalid, t_created, t_expired, temporal_stability,
                   last_seen_at, reinforce_count, chunk_id, kind
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
                    chunk_id=row[13],
                    kind=row[14],
                )
            )
        return memories

    def get_summary_for_artifact(self, artifact_ref: str) -> MemoryNode | None:
        """
        Get summary node for a specific artifact (M4).

        Args:
            artifact_ref: Artifact reference (filename/session ID)

        Returns:
            MemoryNode if summary exists, None otherwise
        """
        cursor = self.conn.execute(
            """
            SELECT id, content, content_embedding, content_hash, created_at, metadata,
                   t_valid, t_invalid, t_created, t_expired, temporal_stability,
                   last_seen_at, reinforce_count, chunk_id, kind
            FROM memories
            WHERE kind = 'SUMMARY' AND t_expired IS NULL
            ORDER BY created_at DESC
            """
        )

        for row in cursor.fetchall():
            metadata = json.loads(row[5])
            if metadata.get("artifact_ref") == artifact_ref:
                return MemoryNode(
                    id=row[0],
                    content=row[1],
                    content_embedding=json.loads(row[2]),
                    content_hash=row[3],
                    created_at=row[4],
                    metadata=metadata,
                    t_valid=row[6],
                    t_invalid=row[7],
                    t_created=row[8],
                    t_expired=row[9],
                    temporal_stability=row[10] or "unknown",
                    last_seen_at=row[11],
                    reinforce_count=row[12] or 0,
                    chunk_id=row[13],
                    kind=row[14],
                )

        return None

    def get_summary_for_chunk(self, chunk_id: str) -> MemoryNode | None:
        """
        Get summary node for a specific chunk (M5).

        Args:
            chunk_id: Chunk ID

        Returns:
            MemoryNode if summary exists, None otherwise
        """
        cursor = self.conn.execute(
            """
            SELECT id, content, content_embedding, content_hash, created_at, metadata,
                   t_valid, t_invalid, t_created, t_expired, temporal_stability,
                   last_seen_at, reinforce_count, chunk_id, kind
            FROM memories
            WHERE kind = 'SUMMARY' AND chunk_id = ? AND t_expired IS NULL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (chunk_id,),
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
            chunk_id=row[13],
            kind=row[14],
        )

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
            INSERT INTO entities (id, entity_type, canonical_name, norm_key, created_at, embedding, expired_at, merged_into, chunk_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.id,
                entity.entity_type,
                entity.canonical_name,
                entity.norm_key,
                entity.created_at,
                entity.embedding,
                entity.expired_at,
                entity.merged_into,
                entity.chunk_id,
            ),
        )
        self.conn.commit()
        return entity.id

    def get_entity(self, entity_id: str) -> EntityNode | None:
        """
        Retrieve entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            EntityNode if found, None otherwise
        """
        cursor = self.conn.execute(
            """
            SELECT id, entity_type, canonical_name, norm_key, created_at, embedding, expired_at, merged_into, chunk_id
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
            embedding=row[5],
            expired_at=row[6],
            merged_into=row[7],
            chunk_id=row[8],
        )

    def find_entity_by_norm_key(
        self, norm_key: str, include_expired: bool = False
    ) -> EntityNode | None:
        """
        Find entity by normalization key (deduplication lookup).

        Args:
            norm_key: Normalization key (format: "TYPE:normalized_name")
            include_expired: Include expired entities in search

        Returns:
            EntityNode if found, None otherwise
        """
        if include_expired:
            query = "SELECT id, entity_type, canonical_name, norm_key, created_at, embedding, expired_at, merged_into, chunk_id FROM entities WHERE norm_key = ?"
        else:
            query = "SELECT id, entity_type, canonical_name, norm_key, created_at, embedding, expired_at, merged_into, chunk_id FROM entities WHERE norm_key = ? AND expired_at IS NULL"

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
            embedding=row[5],
            expired_at=row[6],
            merged_into=row[7],
            chunk_id=row[8],
        )

    def get_all_entities(self, include_expired: bool = False) -> list[EntityNode]:
        """
        Get all entities across all types.

        Args:
            include_expired: Include expired entities

        Returns:
            List of EntityNode objects
        """
        if include_expired:
            query = """
                SELECT id, entity_type, canonical_name, norm_key, created_at, embedding, expired_at, merged_into, chunk_id
                FROM entities
                ORDER BY created_at DESC
            """
        else:
            query = """
                SELECT id, entity_type, canonical_name, norm_key, created_at, embedding, expired_at, merged_into, chunk_id
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
    ) -> list[EntityNode]:
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
                SELECT id, entity_type, canonical_name, norm_key, created_at, embedding, expired_at, merged_into, chunk_id
                FROM entities
                WHERE entity_type = ?
                ORDER BY created_at DESC
            """
        else:
            query = """
                SELECT id, entity_type, canonical_name, norm_key, created_at, embedding, expired_at, merged_into, chunk_id
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

    def expire_entity(self, entity_id: str, merged_into: str | None = None) -> None:
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

    def migrate_fk_to_edges(self) -> dict[str, int]:
        """
        M5→M6 Migration: Convert chunk_id FKs to graph edges.

        Creates:
        - Chunk-(CONTAINS)→Memory edges from memory.chunk_id
        - Chunk-(LINKED)→Entity edges from entity.chunk_id
        - Chunk-(SUMMARIZED_BY)→Summary edges from summary.chunk_id
        - Removes old Summary-(SUMMARIZES)→Memory edges

        Returns:
            Dict with counts: {contains: N, linked: N, summarized_by: N, removed_summarizes: N}
        """
        from vestig.core.models import EdgeNode

        stats = {
            "contains": 0,
            "linked": 0,
            "summarized_by": 0,
            "removed_summarizes": 0
        }

        # 1. Create CONTAINS edges from memory.chunk_id
        cursor = self.conn.execute("""
            SELECT chunk_id, id FROM memories
            WHERE chunk_id IS NOT NULL AND kind = 'MEMORY'
        """)
        for chunk_id, memory_id in cursor.fetchall():
            edge = EdgeNode.create(
                from_node=chunk_id,
                to_node=memory_id,
                edge_type="CONTAINS",
                weight=1.0
            )
            # Check if edge already exists
            existing = self.conn.execute(
                "SELECT edge_id FROM edges WHERE from_node=? AND to_node=? AND edge_type=? AND t_expired IS NULL",
                (edge.from_node, edge.to_node, edge.edge_type)
            ).fetchone()
            if not existing:
                self.store_edge(edge)
                stats["contains"] += 1

        # 2. Create LINKED edges from entity.chunk_id
        cursor = self.conn.execute("""
            SELECT chunk_id, id FROM entities
            WHERE chunk_id IS NOT NULL AND expired_at IS NULL
        """)
        for chunk_id, entity_id in cursor.fetchall():
            edge = EdgeNode.create(
                from_node=chunk_id,
                to_node=entity_id,
                edge_type="LINKED",
                weight=1.0
            )
            existing = self.conn.execute(
                "SELECT edge_id FROM edges WHERE from_node=? AND to_node=? AND edge_type=? AND t_expired IS NULL",
                (edge.from_node, edge.to_node, edge.edge_type)
            ).fetchone()
            if not existing:
                self.store_edge(edge)
                stats["linked"] += 1

        # 3. Create SUMMARIZED_BY edges from summary.chunk_id
        cursor = self.conn.execute("""
            SELECT chunk_id, id FROM memories
            WHERE chunk_id IS NOT NULL AND kind = 'SUMMARY'
        """)
        for chunk_id, summary_id in cursor.fetchall():
            edge = EdgeNode.create(
                from_node=chunk_id,
                to_node=summary_id,
                edge_type="SUMMARIZED_BY",
                weight=1.0
            )
            existing = self.conn.execute(
                "SELECT edge_id FROM edges WHERE from_node=? AND to_node=? AND edge_type=? AND t_expired IS NULL",
                (edge.from_node, edge.to_node, edge.edge_type)
            ).fetchone()
            if not existing:
                self.store_edge(edge)
                stats["summarized_by"] += 1

        # 4. Remove old SUMMARIZES edges (Summary→Memory)
        cursor = self.conn.execute("""
            UPDATE edges SET t_expired = ?
            WHERE edge_type = 'SUMMARIZES' AND t_expired IS NULL
        """, (datetime.now(timezone.utc).isoformat(),))
        stats["removed_summarizes"] = cursor.rowcount

        self.conn.commit()
        return stats

    def get_edges_from_memory(
        self,
        memory_id: str,
        edge_type: str | None = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list[EdgeNode]:
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
    ) -> list[EdgeNode]:
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

    def get_edges_to_memory(
        self,
        memory_id: str,
        edge_type: str | None = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list[EdgeNode]:
        """
        Get all incoming edges to a memory node.

        Args:
            memory_id: Target memory ID
            edge_type: Optional edge type filter (RELATED)
            include_expired: Include expired edges
            min_confidence: Minimum confidence threshold

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
                    WHERE to_node = ? AND edge_type = ? AND (confidence IS NULL OR confidence >= ?)
                    ORDER BY t_created DESC
                """
                params = (memory_id, edge_type, min_confidence)
            else:
                query = """
                    SELECT edge_id, from_node, to_node, edge_type, weight,
                           confidence, evidence, t_valid, t_invalid, t_created, t_expired
                    FROM edges
                    WHERE to_node = ? AND edge_type = ? AND t_expired IS NULL AND (confidence IS NULL OR confidence >= ?)
                    ORDER BY t_created DESC
                """
                params = (memory_id, edge_type, min_confidence)
        else:
            if include_expired:
                query = """
                    SELECT edge_id, from_node, to_node, edge_type, weight,
                           confidence, evidence, t_valid, t_invalid, t_created, t_expired
                    FROM edges
                    WHERE to_node = ? AND (confidence IS NULL OR confidence >= ?)
                    ORDER BY t_created DESC
                """
                params = (memory_id, min_confidence)
            else:
                query = """
                    SELECT edge_id, from_node, to_node, edge_type, weight,
                           confidence, evidence, t_valid, t_invalid, t_created, t_expired
                    FROM edges
                    WHERE to_node = ? AND t_expired IS NULL AND (confidence IS NULL OR confidence >= ?)
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

    def get_chunk_for_memory(self, memory_id: str) -> ChunkNode | None:
        """Get chunk containing this memory via CONTAINS edge (M6)."""
        cursor = self.conn.execute("""
            SELECT c.chunk_id, c.file_id, c.start, c.length, c.sequence, c.created_at
            FROM chunks c
            JOIN edges e ON e.from_node = c.chunk_id
            WHERE e.to_node = ? AND e.edge_type = 'CONTAINS' AND e.t_expired IS NULL
        """, (memory_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return ChunkNode(
            chunk_id=row[0],
            file_id=row[1],
            start=row[2],
            length=row[3],
            sequence=row[4],
            created_at=row[5]
        )

    def get_memories_in_chunk(self, chunk_id: str, include_expired: bool = False) -> list[MemoryNode]:
        """Get all memories in chunk via CONTAINS edges (M6, replaces get_memories_by_chunk)."""
        expired_filter = "" if include_expired else "AND m.t_expired IS NULL"

        cursor = self.conn.execute(f"""
            SELECT m.id, m.content, m.content_embedding, m.content_hash, m.created_at, m.metadata,
                   m.t_valid, m.t_invalid, m.t_created, m.t_expired, m.temporal_stability,
                   m.last_seen_at, m.reinforce_count, m.chunk_id, m.kind
            FROM memories m
            JOIN edges e ON e.to_node = m.id
            WHERE e.from_node = ? AND e.edge_type = 'CONTAINS' AND e.t_expired IS NULL {expired_filter}
            ORDER BY m.created_at ASC
        """, (chunk_id,))

        memories = []
        for row in cursor.fetchall():
            memories.append(MemoryNode(
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
                chunk_id=row[13],
                kind=row[14]
            ))
        return memories

    def get_summary_for_chunk_via_edge(self, chunk_id: str) -> MemoryNode | None:
        """Get chunk summary via SUMMARIZED_BY edge (M6)."""
        cursor = self.conn.execute("""
            SELECT m.id, m.content, m.content_embedding, m.content_hash, m.created_at, m.metadata,
                   m.t_valid, m.t_invalid, m.t_created, m.t_expired, m.temporal_stability,
                   m.last_seen_at, m.reinforce_count, m.chunk_id, m.kind
            FROM memories m
            JOIN edges e ON e.to_node = m.id
            WHERE e.from_node = ? AND e.edge_type = 'SUMMARIZED_BY' AND e.t_expired IS NULL
            LIMIT 1
        """, (chunk_id,))

        row = cursor.fetchone()
        if not row:
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
            chunk_id=row[13],
            kind=row[14]
        )

    def get_entities_in_chunk(self, chunk_id: str, include_expired: bool = False) -> list[EntityNode]:
        """Get entities linked to chunk via LINKED edges (M6)."""
        expired_filter = "" if include_expired else "AND ent.expired_at IS NULL"

        cursor = self.conn.execute(f"""
            SELECT ent.id, ent.entity_type, ent.canonical_name, ent.norm_key,
                   ent.created_at, ent.embedding, ent.expired_at, ent.merged_into, ent.chunk_id
            FROM entities ent
            JOIN edges e ON e.to_node = ent.id
            WHERE e.from_node = ? AND e.edge_type = 'LINKED' AND e.t_expired IS NULL {expired_filter}
            ORDER BY ent.created_at ASC
        """, (chunk_id,))

        entities = []
        for row in cursor.fetchall():
            entities.append(EntityNode(
                id=row[0],
                entity_type=row[1],
                canonical_name=row[2],
                norm_key=row[3],
                created_at=row[4],
                embedding=row[5],
                expired_at=row[6],
                merged_into=row[7],
                chunk_id=row[8]
            ))
        return entities

    def get_edge(self, edge_id: str) -> EdgeNode | None:
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
