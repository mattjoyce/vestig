"""Event storage for M3 lifecycle tracking"""

import json
import sqlite3
from typing import List

from vestig.core.models import EventNode


class MemoryEventStorage:
    """Event CRUD operations (shares DB connection with MemoryStorage)"""

    def __init__(self, conn: sqlite3.Connection):
        """
        Use same DB connection as MemoryStorage for transaction consistency.

        Args:
            conn: SQLite connection from MemoryStorage
        """
        self.conn = conn

    def add_event(self, event: EventNode) -> str:
        """
        Insert event (append-only, never update).

        Args:
            event: EventNode to insert

        Returns:
            Event ID
        """
        self.conn.execute(
            """
            INSERT INTO memory_events
            (event_id, memory_id, event_type, occurred_at, source, actor, artifact_ref, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.memory_id,
                event.event_type,
                event.occurred_at,
                event.source,
                event.actor,
                event.artifact_ref,
                json.dumps(event.payload),
            ),
        )
        self.conn.commit()
        return event.event_id

    def get_events_for_memory(self, memory_id: str, limit: int = 100) -> List[EventNode]:
        """
        Retrieve events for a memory, newest first.

        Args:
            memory_id: Memory ID to retrieve events for
            limit: Maximum number of events to return

        Returns:
            List of EventNode objects
        """
        cursor = self.conn.execute(
            """
            SELECT event_id, memory_id, event_type, occurred_at, source, actor, artifact_ref, payload_json
            FROM memory_events
            WHERE memory_id = ?
            ORDER BY occurred_at DESC
            LIMIT ?
            """,
            (memory_id, limit),
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
                payload=json.loads(row[7]),
            )
            for row in cursor.fetchall()
        ]

    def get_reinforcement_events(self, memory_id: str) -> List[EventNode]:
        """
        Get only REINFORCE_* events for TraceRank computation.

        Args:
            memory_id: Memory ID to retrieve reinforcement events for

        Returns:
            List of EventNode objects with event_type like 'REINFORCE_%'
        """
        cursor = self.conn.execute(
            """
            SELECT event_id, memory_id, event_type, occurred_at, source, actor, artifact_ref, payload_json
            FROM memory_events
            WHERE memory_id = ? AND event_type LIKE 'REINFORCE_%'
            ORDER BY occurred_at DESC
            """,
            (memory_id,),
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
                payload=json.loads(row[7]),
            )
            for row in cursor.fetchall()
        ]
