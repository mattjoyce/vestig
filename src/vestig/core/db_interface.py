"""Database abstraction layer for Vestig storage backends.

This module provides an abstract interface that mirrors the existing MemoryStorage API,
enabling pluggable backends (SQLite, FalkorDB) via configuration.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vestig.core.models import ChunkNode, EdgeNode, EntityNode, EventNode, FileNode, MemoryNode


class EventStorageInterface(ABC):
    """Abstract interface for event storage operations."""

    @abstractmethod
    def add_event(self, event: "EventNode") -> str:
        """Insert event (append-only, never update)."""
        ...

    @abstractmethod
    def get_events_for_memory(self, memory_id: str, limit: int = 100) -> list:
        """Retrieve events for a memory, newest first."""
        ...

    @abstractmethod
    def get_reinforcement_events(self, memory_id: str) -> list:
        """Get only REINFORCE_* events for TraceRank computation."""
        ...


class DatabaseInterface(ABC):
    """Abstract interface for Vestig storage backends.

    This interface mirrors the existing MemoryStorage API to enable
    a thin wrapper pattern. All methods delegate to the underlying
    storage implementation.
    """

    # =========================================================================
    # Memory Operations
    # =========================================================================

    @abstractmethod
    def store_memory(self, node: "MemoryNode", kind: str = "MEMORY") -> str:
        """Store a memory node with deduplication.

        Args:
            node: MemoryNode to store
            kind: "MEMORY" or "SUMMARY"

        Returns:
            Memory ID (existing ID if duplicate detected)
        """
        ...

    @abstractmethod
    def get_memory(self, memory_id: str) -> "MemoryNode | None":
        """Retrieve memory by ID."""
        ...

    @abstractmethod
    def get_all_memories(self) -> list["MemoryNode"]:
        """Get all memories (including expired)."""
        ...

    @abstractmethod
    def get_active_memories(self) -> list["MemoryNode"]:
        """Get all non-expired memories."""
        ...

    @abstractmethod
    def list_memories(
        self, include_expired: bool = False, limit: int | None = None
    ) -> list[tuple]:
        """List memories for CLI display.

        Returns:
            List of (id, content_preview, created_at, t_expired, kind) tuples
        """
        ...

    @abstractmethod
    def get_memories_for_entity_extraction(
        self, reprocess: bool = False
    ) -> list[tuple[str, str]]:
        """Get memories needing entity extraction.

        Args:
            reprocess: If True, return all memories; if False, only unprocessed

        Returns:
            List of (memory_id, content) tuples
        """
        ...

    @abstractmethod
    def get_memories_by_chunk(
        self, chunk_id: str, include_expired: bool = False
    ) -> list["MemoryNode"]:
        """Get memories linked to chunk via chunk_id FK (M5 legacy)."""
        ...

    @abstractmethod
    def get_memories_in_chunk(
        self, chunk_id: str, include_expired: bool = False
    ) -> list["MemoryNode"]:
        """Get memories in chunk via CONTAINS edges (M6)."""
        ...

    @abstractmethod
    def get_summary_for_artifact(self, artifact_ref: str) -> "MemoryNode | None":
        """Find SUMMARY node by artifact reference."""
        ...

    @abstractmethod
    def get_summary_for_chunk(self, chunk_id: str) -> "MemoryNode | None":
        """Find SUMMARY node for specific chunk via kind='SUMMARY'."""
        ...

    @abstractmethod
    def update_node_embedding(
        self, node_id: str, embedding_json: str, node_type: str
    ) -> None:
        """Update embedding for a node (memory or entity)."""
        ...

    @abstractmethod
    def increment_reinforce_count(self, memory_id: str) -> None:
        """Increment cached reinforce_count for TraceRank."""
        ...

    @abstractmethod
    def update_last_seen(self, memory_id: str, timestamp: str) -> None:
        """Update last_seen_at timestamp."""
        ...

    @abstractmethod
    def deprecate_memory(self, memory_id: str, t_invalid: str | None = None) -> None:
        """Mark memory as expired (soft delete)."""
        ...

    # =========================================================================
    # Entity Operations
    # =========================================================================

    @abstractmethod
    def store_entity(self, entity: "EntityNode") -> str:
        """Store entity with deduplication via norm_key.

        Returns:
            Entity ID (existing ID if duplicate detected)
        """
        ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> "EntityNode | None":
        """Retrieve entity by ID."""
        ...

    @abstractmethod
    def find_entity_by_norm_key(
        self, norm_key: str, include_expired: bool = False
    ) -> "EntityNode | None":
        """Find entity by normalization key (fast dedup lookup)."""
        ...

    @abstractmethod
    def get_all_entities(self, include_expired: bool = False) -> list["EntityNode"]:
        """Get all entities."""
        ...

    @abstractmethod
    def get_entities_by_type(
        self, entity_type: str, include_expired: bool = False
    ) -> list["EntityNode"]:
        """Get entities of a specific type (PERSON, ORG, etc.)."""
        ...

    @abstractmethod
    def get_entities_in_chunk(
        self, chunk_id: str, include_expired: bool = False
    ) -> list["EntityNode"]:
        """Get entities linked to chunk via LINKED edges (M6)."""
        ...

    @abstractmethod
    def list_entities_with_mention_counts(
        self, include_expired: bool = False, limit: int | None = None
    ) -> list[tuple]:
        """List entities with mention counts for CLI display.

        Returns:
            List of (id, entity_type, canonical_name, created_at,
                    expired_at, merged_into, mention_count) tuples
        """
        ...

    @abstractmethod
    def expire_entity(self, entity_id: str, merged_into: str | None = None) -> None:
        """Mark entity as expired (soft delete / merge)."""
        ...

    # =========================================================================
    # Edge Operations
    # =========================================================================

    @abstractmethod
    def store_edge(self, edge: "EdgeNode") -> str:
        """Store edge with deduplication.

        Returns:
            Edge ID (existing ID if duplicate detected)
        """
        ...

    @abstractmethod
    def get_edge(self, edge_id: str) -> "EdgeNode | None":
        """Retrieve edge by ID."""
        ...

    @abstractmethod
    def get_edges_from_memory(
        self,
        memory_id: str,
        edge_type: str | None = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list["EdgeNode"]:
        """Get outgoing edges from a memory node."""
        ...

    @abstractmethod
    def get_edges_to_entity(
        self,
        entity_id: str,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list["EdgeNode"]:
        """Get incoming edges to an entity node."""
        ...

    @abstractmethod
    def get_edges_to_memory(
        self,
        memory_id: str,
        edge_type: str | None = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
    ) -> list["EdgeNode"]:
        """Get incoming edges to a memory node."""
        ...

    @abstractmethod
    def expire_edge(self, edge_id: str) -> None:
        """Mark edge as expired (soft delete)."""
        ...

    @abstractmethod
    def delete_edges_by_type(self, edge_type: str) -> int:
        """Hard delete all edges of a type. Returns count deleted."""
        ...

    @abstractmethod
    def delete_edges_from_node(
        self, node_id: str, edge_type: str | None = None
    ) -> int:
        """Delete outbound edges from a node. Returns count deleted."""
        ...

    @abstractmethod
    def list_edges(
        self,
        edge_type: str | None = None,
        include_expired: bool = False,
        limit: int = 100,
    ) -> list[tuple]:
        """List edges for CLI display.

        Args:
            edge_type: Filter by edge type, or None for all types
            include_expired: Include expired edges
            limit: Maximum number of edges to return

        Returns:
            List of (edge_id, from_node, to_node, edge_type, weight,
                    confidence, t_created, t_expired) tuples
        """
        ...

    # =========================================================================
    # File and Chunk Operations (M5 Hub Layer)
    # =========================================================================

    @abstractmethod
    def store_file(self, file_node: "FileNode") -> str:
        """Store file metadata."""
        ...

    @abstractmethod
    def get_file(self, file_id: str) -> "FileNode | None":
        """Retrieve file by ID."""
        ...

    @abstractmethod
    def find_file_by_path(self, path: str) -> "FileNode | None":
        """Find file by path."""
        ...

    @abstractmethod
    def store_chunk(self, chunk_node: "ChunkNode") -> str:
        """Store chunk pointer."""
        ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> "ChunkNode | None":
        """Retrieve chunk by ID."""
        ...

    @abstractmethod
    def get_chunks_by_file(self, file_id: str) -> list["ChunkNode"]:
        """Get all chunks for a file, ordered by sequence."""
        ...

    @abstractmethod
    def get_chunk_for_memory(self, memory_id: str) -> "ChunkNode | None":
        """Get chunk containing memory via CONTAINS edge (M6)."""
        ...

    @abstractmethod
    def get_summary_for_chunk_via_edge(self, chunk_id: str) -> "MemoryNode | None":
        """Get summary for chunk via SUMMARIZED_BY edge (M6)."""
        ...

    # =========================================================================
    # Count and Statistics Operations
    # =========================================================================

    @abstractmethod
    def count_memories(self, kind: str | None = None) -> int:
        """Count memories, optionally filtered by kind."""
        ...

    @abstractmethod
    def count_entities(self) -> int:
        """Count all entities."""
        ...

    @abstractmethod
    def count_edges(self, edge_type: str | None = None) -> int:
        """Count edges, optionally filtered by type."""
        ...

    @abstractmethod
    def count_memories_with_edge_type(self, edge_type: str) -> int:
        """Count distinct memories with outgoing edges of type."""
        ...

    # =========================================================================
    # Housekeeping Operations
    # =========================================================================

    @abstractmethod
    def get_orphaned_memories(self) -> list[tuple[str, str, str]]:
        """Find memories without provenance (no CONTAINS edge) and no entity links.

        Returns:
            List of (memory_id, content, created_at) tuples for orphaned memories.
        """
        ...

    @abstractmethod
    def get_memories_without_source(self) -> list[tuple[str, str, str]]:
        """Find memories without Source linkage (direct or via Chunk).

        Returns:
            List of (memory_id, content, created_at) tuples.
        """
        ...

    # =========================================================================
    # Bulk Delete Operations
    # =========================================================================

    @abstractmethod
    def delete_all_entities(self) -> int:
        """Delete all entities (used by purge command). Returns count deleted."""
        ...

    # =========================================================================
    # Transaction and Lifecycle Management
    # =========================================================================

    @abstractmethod
    @contextmanager
    def transaction(self):
        """Context manager for atomic transactions.

        Usage:
            with storage.transaction():
                storage.store_memory(...)
                storage.store_edge(...)
                # Commits on exit, rolls back on exception
        """
        ...

    @abstractmethod
    def commit(self) -> None:
        """Explicit commit (for cases outside transaction context)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        ...

    # =========================================================================
    # Event Storage (Composed)
    # =========================================================================

    @property
    @abstractmethod
    def event_storage(self) -> EventStorageInterface:
        """Return event storage sharing this DB's transaction context."""
        ...
