"""Event and outcome types for M2→M3 bridge"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable


@dataclass
class CommitOutcome:
    """
    Structured outcome of a commit attempt (M2→M3 bridge).

    This captures the decision made during commit_memory() in a way that
    can be consumed by M3 event logging and TraceRank mechanics.

    Outcome states:
    - INSERTED_NEW: A new Memory Node was created
    - EXACT_DUPE: Hash match to existing memory (same ID returned)
    - NEAR_DUPE: Semantic match above threshold to existing memory
    - REJECTED_HYGIENE: Blocked by hygiene rule(s)
    """

    outcome: str  # INSERTED_NEW | EXACT_DUPE | NEAR_DUPE | REJECTED_HYGIENE
    memory_id: str  # The canonical ID (existing or new)
    content_hash: str  # SHA256 of normalized content
    occurred_at: str  # UTC timestamp (ISO 8601)
    source: str = "manual"  # manual | hook | import | batch

    # Dupe/near-dupe fields
    matched_memory_id: Optional[str] = None  # ID of matched memory (if dupe)
    query_score: Optional[float] = None  # Similarity score (for near-dupe)

    # Hygiene rejection fields
    hygiene_reasons: List[str] = field(default_factory=list)  # Rejection reasons

    # Metadata
    thresholds: Dict[str, Any] = field(default_factory=dict)  # Config thresholds used
    tags: Optional[List[str]] = None
    artifact_ref: Optional[str] = None  # Session ID, filename, URL, etc.

    @classmethod
    def inserted_new(
        cls,
        memory_id: str,
        content_hash: str,
        source: str = "manual",
        tags: Optional[List[str]] = None,
        artifact_ref: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> "CommitOutcome":
        """Create outcome for a new memory insertion."""
        return cls(
            outcome="INSERTED_NEW",
            memory_id=memory_id,
            content_hash=content_hash,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            tags=tags,
            artifact_ref=artifact_ref,
            thresholds=thresholds or {},
        )

    @classmethod
    def exact_dupe(
        cls,
        memory_id: str,
        content_hash: str,
        source: str = "manual",
        tags: Optional[List[str]] = None,
        artifact_ref: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> "CommitOutcome":
        """Create outcome for an exact duplicate (hash match)."""
        return cls(
            outcome="EXACT_DUPE",
            memory_id=memory_id,
            content_hash=content_hash,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            matched_memory_id=memory_id,
            tags=tags,
            artifact_ref=artifact_ref,
            thresholds=thresholds or {},
        )

    @classmethod
    def near_dupe(
        cls,
        memory_id: str,
        matched_memory_id: str,
        query_score: float,
        content_hash: str,
        source: str = "manual",
        tags: Optional[List[str]] = None,
        artifact_ref: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> "CommitOutcome":
        """Create outcome for a near-duplicate (semantic match)."""
        return cls(
            outcome="NEAR_DUPE",
            memory_id=memory_id,
            content_hash=content_hash,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            matched_memory_id=matched_memory_id,
            query_score=query_score,
            tags=tags,
            artifact_ref=artifact_ref,
            thresholds=thresholds or {},
        )

    @classmethod
    def rejected_hygiene(
        cls,
        content_hash: str,
        hygiene_reasons: List[str],
        source: str = "manual",
        tags: Optional[List[str]] = None,
        artifact_ref: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> "CommitOutcome":
        """Create outcome for a hygiene rejection."""
        return cls(
            outcome="REJECTED_HYGIENE",
            memory_id="",  # No ID assigned
            content_hash=content_hash,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            hygiene_reasons=hygiene_reasons,
            tags=tags,
            artifact_ref=artifact_ref,
            thresholds=thresholds or {},
        )


# Hook type for M3 event logging
OnCommitHook = Callable[[CommitOutcome], None]
