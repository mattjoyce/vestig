"""SQLite database adapter wrapping existing MemoryStorage.

This is a thin wrapper around the existing MemoryStorage class,
implementing the DatabaseInterface without any refactoring of the
underlying storage code. This adapter will be deprecated once
FalkorDB is fully functional.
"""

from contextlib import contextmanager

from vestig.core.db_interface import DatabaseInterface, EventStorageInterface
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.models import ChunkNode, EdgeNode, EntityNode, FileNode, MemoryNode
from vestig.core.storage import MemoryStorage


class SQLiteEventStorage(EventStorageInterface):
    """SQLite event storage adapter wrapping MemoryEventStorage."""

    def __init__(self, event_storage: MemoryEventStorage):
        self._storage = event_storage

    def add_event(self, event) -> str:
        return self._storage.add_event(event)

    def get_events_for_memory(self, memory_id: str, limit: int = 100) -> list:
        return self._storage.get_events_for_memory(memory_id, limit)

    def get_reinforcement_events(self, memory_id: str) -> list:
        return self._storage.get_reinforcement_events(memory_id)


class SQLiteDatabase(DatabaseInterface):
    """SQLite adapter wrapping existing MemoryStorage (legacy, will be deleted).

    This class implements DatabaseInterface by delegating all operations
    to the existing MemoryStorage and MemoryEventStorage classes.
    No refactoring of the underlying code - just a thin wrapper.
    """

    def __init__(self, db_path: str):
        """Initialize SQLite database adapter.

        Args:
            db_path: Path to SQLite database file
        """
        self._storage = MemoryStorage(db_path)
        self._event_storage = SQLiteEventStorage(
            MemoryEventStorage(self._storage.conn)
        )

    # =========================================================================
    # Memory Operations
    # =========================================================================

    def store_memory(self, node: MemoryNode, kind: str = "MEMORY") -> str:
        return self._storage.store_memory(node, kind)

    def get_memory(self, memory_id: str) -> MemoryNode | None:
        return self._storage.get_memory(memory_id)

    def get_all_memories(self) -> list[MemoryNode]:
        return self._storage.get_all_memories()

    def get_active_memories(self) -> list[MemoryNode]:
        return self._storage.get_active_memories()

    def list_memories(
        self, include_expired: bool = False, limit: int | None = None
    ) -> list[tuple]:
        return self._storage.list_memories(include_expired, limit)

    def get_memories_for_entity_extraction(
        self, reprocess: bool = False
    ) -> list[tuple[str, str]]:
        return self._storage.get_memories_for_entity_extraction(reprocess)

    def get_memories_by_chunk(
        self, chunk_id: str, include_expired: bool = False
    ) -> list[MemoryNode]:
        return self._storage.get_memories_by_chunk(chunk_id, include_expired)

    def get_memories_in_chunk(
        self, chunk_id: str, include_expired: bool = False
    ) -> list[MemoryNode]:
        return self._storage.get_memories_in_chunk(chunk_id, include_expired)

    def get_summary_for_artifact(self, artifact_ref: str) -> MemoryNode | None:
        return self._storage.get_summary_for_artifact(artifact_ref)

    def get_summary_for_chunk(self, chunk_id: str) -> MemoryNode | None:
        return self._storage.get_summary_for_chunk(chunk_id)

    def update_node_embedding(
        self, node_id: str, embedding_json: str, node_type: str
    ) -> None:
        return self._storage.update_node_embedding(node_id, embedding_json, node_type)

    def increment_reinforce_count(self, memory_id: str) -> None:
        return self._storage.increment_reinforce_count(memory_id)

    def update_last_seen(self, memory_id: str, timestamp: str) -> None:
        return self._storage.update_last_seen(memory_id, timestamp)

    def deprecate_memory(self, memory_id: str, t_invalid: str | None = None) -> None:
        return self._storage.deprecate_memory(memory_id, t_invalid)

    # =========================================================================
    # Entity Operations
    # =========================================================================

    def store_entity(self, entity: EntityNode) -> str:
        return self._storage.store_entity(entity)

    def get_entity(self, entity_id: str) -> EntityNode | None:
        return self._storage.get_entity(entity_id)

    def find_entity_by_norm_key(
        self, norm_key: str, include_expired: bool = False
    ) -> EntityNode | None:
        return self._storage.find_entity_by_norm_key(norm_key, include_expired)

    def get_all_entities(self, include_expired: bool = False) -> list[EntityNode]:
        return self._storage.get_all_entities(include_expired)

    def get_entities_by_type(
        self, entity_type: str, include_expired: bool = False
    ) -> list[EntityNode]:
        return self._storage.get_entities_by_type(entity_type, include_expired)

    def get_entities_in_chunk(
        self, chunk_id: str, include_expired: bool = False
    ) -> list[EntityNode]:
        return self._storage.get_entities_in_chunk(chunk_id, include_expired)

    def list_entities_with_mention_counts(
        self, include_expired: bool = False, limit: int | None = None
    ) -> list[tuple]:
        return self._storage.list_entities_with_mention_counts(include_expired, limit)

    def expire_entity(self, entity_id: str, merged_into: str | None = None) -> None:
        return self._storage.expire_entity(entity_id, merged_into)

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def store_edge(self, edge: EdgeNode) -> str:
        return self._storage.store_edge(edge)

    def get_edge(self, edge_id: str) -> EdgeNode | None:
        return self._storage.get_edge(edge_id)

    def get_edges_from_memory(
        self,
        memory_id: str,
        edge_type: str | None = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list[EdgeNode]:
        return self._storage.get_edges_from_memory(
            memory_id, edge_type, include_expired, min_confidence
        )

    def get_edges_to_entity(
        self,
        entity_id: str,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list[EdgeNode]:
        return self._storage.get_edges_to_entity(
            entity_id, include_expired, min_confidence
        )

    def get_edges_to_memory(
        self,
        memory_id: str,
        edge_type: str | None = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list[EdgeNode]:
        return self._storage.get_edges_to_memory(
            memory_id, edge_type, include_expired, min_confidence
        )

    def expire_edge(self, edge_id: str) -> None:
        return self._storage.expire_edge(edge_id)

    def delete_edges_by_type(self, edge_type: str) -> int:
        return self._storage.delete_edges_by_type(edge_type)

    def delete_edges_from_node(
        self, node_id: str, edge_type: str | None = None
    ) -> int:
        return self._storage.delete_edges_from_node(node_id, edge_type)

    def list_edges(
        self,
        edge_type: str | None = None,
        include_expired: bool = False,
        limit: int = 100,
    ) -> list[tuple]:
        """List edges for CLI display."""
        query = (
            "SELECT edge_id, from_node, to_node, edge_type, weight, confidence, "
            "t_created, t_expired FROM edges "
        )
        params = []

        if edge_type is not None:
            query += "WHERE edge_type = ? "
            params.append(edge_type)
            if not include_expired:
                query += "AND t_expired IS NULL "
        else:
            if not include_expired:
                query += "WHERE t_expired IS NULL "

        query += "ORDER BY t_created DESC LIMIT ?"
        params.append(limit)

        cursor = self._storage.conn.execute(query, params)
        return cursor.fetchall()

    # =========================================================================
    # File and Chunk Operations
    # =========================================================================

    def store_file(self, file_node: FileNode) -> str:
        return self._storage.store_file(file_node)

    def get_file(self, file_id: str) -> FileNode | None:
        return self._storage.get_file(file_id)

    def find_file_by_path(self, path: str) -> FileNode | None:
        return self._storage.find_file_by_path(path)

    def store_chunk(self, chunk_node: ChunkNode) -> str:
        return self._storage.store_chunk(chunk_node)

    def get_chunk(self, chunk_id: str) -> ChunkNode | None:
        return self._storage.get_chunk(chunk_id)

    def get_chunks_by_file(self, file_id: str) -> list[ChunkNode]:
        return self._storage.get_chunks_by_file(file_id)

    def get_chunk_for_memory(self, memory_id: str) -> ChunkNode | None:
        return self._storage.get_chunk_for_memory(memory_id)

    def get_summary_for_chunk_via_edge(self, chunk_id: str) -> MemoryNode | None:
        return self._storage.get_summary_for_chunk_via_edge(chunk_id)

    # =========================================================================
    # Count and Statistics Operations
    # =========================================================================

    def count_memories(self, kind: str | None = None) -> int:
        return self._storage.count_memories(kind)

    def count_entities(self) -> int:
        return self._storage.count_entities()

    def count_edges(self, edge_type: str | None = None) -> int:
        return self._storage.count_edges(edge_type)

    def count_memories_with_edge_type(self, edge_type: str) -> int:
        return self._storage.count_memories_with_edge_type(edge_type)

    # =========================================================================
    # Housekeeping Operations
    # =========================================================================

    def get_orphaned_memories(self) -> list[tuple[str, str, str]]:
        return self._storage.get_orphaned_memories()

    def get_memories_without_source(self) -> list[tuple[str, str, str]]:
        return self._storage.get_memories_without_source()

    # =========================================================================
    # Bulk Delete Operations
    # =========================================================================

    def delete_all_entities(self) -> int:
        return self._storage.delete_all_entities()

    # =========================================================================
    # Transaction and Lifecycle Management
    # =========================================================================

    @contextmanager
    def transaction(self):
        """Use SQLite connection's context manager for transactions."""
        with self._storage.conn:
            yield

    def commit(self) -> None:
        """Explicit commit for operations outside transaction context."""
        self._storage.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self._storage.close()

    # =========================================================================
    # Event Storage
    # =========================================================================

    @property
    def event_storage(self) -> SQLiteEventStorage:
        """Return event storage sharing this DB's transaction context."""
        return self._event_storage

    # =========================================================================
    # Legacy Access (for gradual migration)
    # =========================================================================

    @property
    def conn(self):
        """Direct connection access (for legacy code during migration).

        DEPRECATED: This property is for backward compatibility during migration.
        Use transaction() context manager instead.
        """
        return self._storage.conn
