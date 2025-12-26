"""Data models for Vestig"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


@dataclass
class MemoryNode:
    """Memory node with M2 dedupe support and M3 temporal fields"""

    id: str
    content: str
    content_embedding: List[float]
    content_hash: str  # SHA256 of normalized content (M2: dedupe)
    created_at: str  # ISO 8601 timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    # M3: Bi-temporal fields
    t_valid: Optional[str] = None  # When fact became true (event time)
    t_invalid: Optional[str] = None  # When fact stopped being true (event time)
    t_created: Optional[str] = None  # When we learned it (transaction time)
    t_expired: Optional[str] = None  # When deprecated/superseded
    temporal_stability: str = "unknown"  # "static" | "dynamic" | "unknown"

    # M3: Reinforcement tracking (cached from events)
    last_seen_at: Optional[str] = None  # Most recent reinforcement
    reinforce_count: int = 0  # Total reinforcement events

    @classmethod
    def create(
        cls,
        memory_id: str,
        content: str,
        embedding: List[float],
        source: str = "manual",
        tags: List[str] = None,
        content_hash: Optional[str] = None,  # M3 FIX: Allow passing pre-computed hash
    ) -> "MemoryNode":
        """
        Create a new memory node with M3 temporal initialization.

        Args:
            memory_id: Unique identifier (e.g., mem_uuid)
            content: Memory content text (normalized)
            embedding: Content embedding vector
            source: Source of the memory (manual, hook, batch)
            tags: Optional tags for filtering
            content_hash: Pre-computed content hash (optional, computed if not provided)

        Returns:
            MemoryNode instance
        """
        # Compute content hash if not provided (backward compatibility)
        if content_hash is None:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Build metadata
        metadata = {"source": source}
        if tags:
            metadata["tags"] = tags

        # M3: Initialize temporal fields
        now = datetime.now(timezone.utc).isoformat()

        return cls(
            id=memory_id,
            content=content,
            content_embedding=embedding,
            content_hash=content_hash,
            created_at=now,
            metadata=metadata,
            # M3: Bi-temporal initialization
            t_valid=now,  # Assume valid from creation
            t_invalid=None,  # Not yet invalidated
            t_created=now,  # Transaction time = creation time
            t_expired=None,  # Not deprecated
            temporal_stability="unknown",  # Default stability
            last_seen_at=None,  # No reinforcement yet
            reinforce_count=0,  # No reinforcement yet
        )


@dataclass
class EventNode:
    """Memory lifecycle event (M3)"""

    event_id: str  # evt_<uuid>
    memory_id: str  # FK to memories table
    event_type: str  # ADD | REINFORCE_EXACT | REINFORCE_NEAR | DEPRECATE | SUPERSEDE
    occurred_at: str  # UTC timestamp (ISO 8601)
    source: str  # manual | hook | import | batch
    actor: Optional[str] = None  # User/agent identifier
    artifact_ref: Optional[str] = None  # Session ID, filename, etc.
    payload: Dict[str, Any] = field(default_factory=dict)  # Event details

    @classmethod
    def create(
        cls,
        memory_id: str,
        event_type: str,
        source: str = "manual",
        actor: Optional[str] = None,
        artifact_ref: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "EventNode":
        """Create new event with generated ID and timestamp"""
        return cls(
            event_id=f"evt_{uuid.uuid4()}",
            memory_id=memory_id,
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            actor=actor,
            artifact_ref=artifact_ref,
            payload=payload or {},
        )
