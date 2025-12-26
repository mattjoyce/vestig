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

            # Create unique index on content_hash for dedupe
            self.conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_content_hash
                ON memories(content_hash)
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
            INSERT INTO memories (id, content, content_embedding, content_hash, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.content,
                json.dumps(node.content_embedding),
                node.content_hash,
                node.created_at,
                json.dumps(node.metadata),
            ),
        )
        self.conn.commit()
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
            SELECT id, content, content_embedding, content_hash, created_at, metadata
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
        )

    def get_all_memories(self) -> List[MemoryNode]:
        """
        Load all memories (for brute-force search).

        Returns:
            List of all MemoryNode objects
        """
        cursor = self.conn.execute(
            """
            SELECT id, content, content_embedding, content_hash, created_at, metadata
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
                )
            )
        return memories

    def close(self):
        """Close database connection"""
        self.conn.close()
