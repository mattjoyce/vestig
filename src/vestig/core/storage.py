"""Storage layer using SQLite for M1 (minimal graph support)"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from vestig.core.models import MemoryNode


class MemoryStorage:
    """Simple SQLite storage for memory nodes (M1: no graph edges yet)"""

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
        """Create tables if they don't exist"""
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
        self.conn.commit()

    def store_memory(self, node: MemoryNode) -> str:
        """
        Persist a memory node.

        Args:
            node: MemoryNode to store

        Returns:
            Memory ID
        """
        self.conn.execute(
            """
            INSERT INTO memories (id, content, content_embedding, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.content,
                json.dumps(node.content_embedding),
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
            SELECT id, content, content_embedding, created_at, metadata
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
            created_at=row[3],
            metadata=json.loads(row[4]),
        )

    def get_all_memories(self) -> List[MemoryNode]:
        """
        Load all memories (for brute-force search in M1).

        Returns:
            List of all MemoryNode objects
        """
        cursor = self.conn.execute(
            """
            SELECT id, content, content_embedding, created_at, metadata
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
                    created_at=row[3],
                    metadata=json.loads(row[4]),
                )
            )
        return memories

    def close(self):
        """Close database connection"""
        self.conn.close()
