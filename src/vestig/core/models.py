"""Data models for Vestig"""

import hashlib
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryNode:
    """Memory node with M2 dedupe support and M3 temporal fields"""

    id: str
    content: str
    content_embedding: list[float]
    content_hash: str  # SHA256 of normalized content (M2: dedupe)
    created_at: str  # ISO 8601 timestamp
    metadata: dict[str, Any] = field(default_factory=dict)

    # M3: Bi-temporal fields
    t_valid: str | None = None  # When fact became true (event time)
    t_invalid: str | None = None  # When fact stopped being true (event time)
    t_created: str | None = None  # When we learned it (transaction time)
    t_expired: str | None = None  # When deprecated/superseded
    temporal_stability: str = "unknown"  # "static" | "dynamic" | "ephemeral" | "unknown"

    # M3: Reinforcement tracking (cached from events)
    last_seen_at: str | None = None  # Most recent reinforcement
    reinforce_count: int = 0  # Total reinforcement events

    @classmethod
    def create(
        cls,
        memory_id: str,
        content: str,
        embedding: list[float],
        source: str = "manual",
        tags: list[str] = None,
        content_hash: str | None = None,  # M3 FIX: Allow passing pre-computed hash
        t_valid_hint: str | None = None,  # Temporal hint: when fact became true
        temporal_stability_hint: str | None = None,  # Temporal hint: static/dynamic/ephemeral/unknown
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
            t_valid_hint: Optional temporal hint for when fact became true
            temporal_stability_hint: Optional stability classification (static/dynamic/ephemeral/unknown)

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

        # Use temporal hints if provided, otherwise default to now
        t_valid = t_valid_hint if t_valid_hint else now
        temporal_stability = temporal_stability_hint if temporal_stability_hint else "unknown"

        return cls(
            id=memory_id,
            content=content,
            content_embedding=embedding,
            content_hash=content_hash,
            created_at=now,
            metadata=metadata,
            # M3: Bi-temporal initialization with hints
            t_valid=t_valid,  # Use hint or fallback to now
            t_invalid=None,  # Not yet invalidated
            t_created=now,  # Transaction time is always now (when we learned it)
            t_expired=None,  # Not deprecated
            temporal_stability=temporal_stability,  # Use hint or fallback to "unknown"
            last_seen_at=None,  # No reinforcement yet
            reinforce_count=0,  # No reinforcement yet
        )


@dataclass
class EntityNode:
    """Entity node (M4: Graph Layer)

    Represents a canonical entity extracted from memories.
    Entities are deduplicated via norm_key.
    """

    id: str  # ent_<uuid>
    entity_type: str  # PERSON | ORG | SYSTEM | PROJECT | PLACE | SKILL | CAPABILITY | TOOL | FILE | CONCEPT (from config)
    canonical_name: str  # Canonical form of entity name
    norm_key: str  # Normalization key for deduplication (type:normalized_name)
    embedding: str | None = None  # JSON-serialized embedding vector (for semantic matching)
    created_at: str  # ISO 8601 timestamp
    expired_at: str | None = None  # When entity was merged/deprecated
    merged_into: str | None = None  # ID of entity this was merged into

    @classmethod
    def create(
        cls,
        entity_type: str,
        canonical_name: str,
        entity_id: str | None = None,
    ) -> "EntityNode":
        """
        Create a new entity node with computed norm_key.

        Args:
            entity_type: Type of entity (must be in config.allowed_types)
            canonical_name: Canonical form of entity name
            entity_id: Optional entity ID (generated if not provided)

        Returns:
            EntityNode instance
        """
        if entity_id is None:
            entity_id = f"ent_{uuid.uuid4()}"

        # Compute norm_key deterministically
        norm_key = compute_norm_key(canonical_name, entity_type)

        now = datetime.now(timezone.utc).isoformat()

        return cls(
            id=entity_id,
            entity_type=entity_type,
            canonical_name=canonical_name,
            norm_key=norm_key,
            created_at=now,
            expired_at=None,
            merged_into=None,
        )


@dataclass
class EdgeNode:
    """Edge node (M4: Graph Layer)

    Represents a relationship between nodes with bi-temporal tracking.
    Edge types: MENTIONS (Memory→Entity), RELATED (Memory→Memory)
    """

    edge_id: str  # edge_<uuid>
    from_node: str  # Source node ID (mem_* or ent_*)
    to_node: str  # Target node ID (mem_* or ent_*)
    edge_type: str  # MENTIONS | RELATED (enforced)
    weight: float  # Edge weight (1.0 default, or similarity score)

    # M4: LLM extraction metadata
    confidence: float | None = None  # Extraction confidence (0.0-1.0)
    evidence: str | None = None  # Short explanation (max 200 chars)

    # M4: Bi-temporal fields (same as entities)
    t_valid: str | None = None  # When relationship became true
    t_invalid: str | None = None  # When relationship stopped being true
    t_created: str | None = None  # When we learned about this relationship
    t_expired: str | None = None  # When edge was invalidated

    @classmethod
    def create(
        cls,
        from_node: str,
        to_node: str,
        edge_type: str,
        weight: float = 1.0,
        confidence: float | None = None,
        evidence: str | None = None,
        edge_id: str | None = None,
    ) -> "EdgeNode":
        """
        Create a new edge node with bi-temporal initialization.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            edge_type: MENTIONS or RELATED (validated)
            weight: Edge weight (default 1.0)
            confidence: Optional extraction confidence (0.0-1.0)
            evidence: Optional short explanation
            edge_id: Optional edge ID (generated if not provided)

        Returns:
            EdgeNode instance

        Raises:
            ValueError: If edge_type is invalid
        """
        # Enforce edge type constraints
        allowed_edge_types = {"MENTIONS", "RELATED", "SUMMARIZES"}
        if edge_type not in allowed_edge_types:
            raise ValueError(f"Invalid edge_type: {edge_type}. Allowed: {allowed_edge_types}")

        if edge_id is None:
            edge_id = f"edge_{uuid.uuid4()}"

        # Truncate evidence if too long
        if evidence and len(evidence) > 200:
            evidence = evidence[:197] + "..."

        now = datetime.now(timezone.utc).isoformat()

        return cls(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node,
            edge_type=edge_type,
            weight=weight,
            confidence=confidence,
            evidence=evidence,
            # Bi-temporal initialization
            t_valid=now,
            t_invalid=None,
            t_created=now,
            t_expired=None,
        )


@dataclass
class EventNode:
    """Memory lifecycle event (M3)"""

    event_id: str  # evt_<uuid>
    memory_id: str  # FK to memories table
    event_type: (
        str  # ADD | REINFORCE_EXACT | REINFORCE_NEAR | DEPRECATE | SUPERSEDE | ENTITY_EXTRACTED | SUMMARY_CREATED
    )
    occurred_at: str  # UTC timestamp (ISO 8601)
    source: str  # manual | hook | import | batch | llm | summary_generation
    actor: str | None = None  # User/agent identifier
    artifact_ref: str | None = None  # Session ID, filename, etc.
    payload: dict[str, Any] = field(default_factory=dict)  # Event details

    @classmethod
    def create(
        cls,
        memory_id: str,
        event_type: str,
        source: str = "manual",
        actor: str | None = None,
        artifact_ref: str | None = None,
        payload: dict[str, Any] | None = None,
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


# M4: Utility Functions


def compute_norm_key(text: str, entity_type: str) -> str:
    """
    Compute normalization key for entity deduplication.

    Deterministic canonicalization:
    - Lowercase
    - Collapse whitespace
    - Strip leading/trailing punctuation
    - Prefix with entity_type for scoped deduplication

    Args:
        text: Entity name/text to normalize
        entity_type: Entity type (for scoping)

    Returns:
        Normalized key in format "TYPE:normalized_text"

    Examples:
        >>> compute_norm_key("Alice Smith", "PERSON")
        'PERSON:alice smith'
        >>> compute_norm_key("PostgreSQL", "SYSTEM")
        'SYSTEM:postgresql'
        >>> compute_norm_key("  Dr. Alice  ", "PERSON")
        'PERSON:dr alice'
    """
    # Lowercase and collapse whitespace
    normalized = " ".join(text.lower().strip().split())

    # Strip leading/trailing punctuation
    normalized = normalized.strip(string.punctuation)

    # Prefix with entity type for scoped deduplication
    return f"{entity_type}:{normalized}"
