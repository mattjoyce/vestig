"""Storage layer using SQLite with M2 dedupe support"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from vestig.core.models import MemoryNode


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
