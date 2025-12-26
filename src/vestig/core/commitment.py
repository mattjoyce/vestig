"""Commitment pipeline with M2 quality firewall"""

import hashlib
import re
import uuid
from typing import Dict, Any, Optional

from vestig.core.embeddings import EmbeddingEngine
from vestig.core.events import CommitOutcome, OnCommitHook
from vestig.core.models import MemoryNode
from vestig.core.storage import MemoryStorage

# Default hygiene settings (used if not in config)
DEFAULT_HYGIENE = {
    "min_chars": 12,
    "max_chars": 4000,
    "normalize_whitespace": True,
    "reject_exact_duplicates": True,
    "near_duplicate": {"enabled": True, "threshold": 0.92},
}


def normalize_content(content: str, normalize_whitespace: bool = True) -> str:
    """
    Normalize content for storage.

    Args:
        content: Raw content string
        normalize_whitespace: Whether to normalize whitespace

    Returns:
        Normalized content string
    """
    # Strip leading/trailing whitespace
    normalized = content.strip()

    if normalize_whitespace:
        # Collapse multiple spaces/newlines
        normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def validate_content(content: str, hygiene_config: Dict[str, Any]) -> None:
    """
    Validate content against hygiene rules.

    Args:
        content: Normalized content string
        hygiene_config: Hygiene configuration dict

    Raises:
        ValueError: If content fails validation
    """
    # Check minimum length
    min_chars = hygiene_config.get("min_chars", DEFAULT_HYGIENE["min_chars"])
    if len(content) < min_chars:
        raise ValueError(
            f"Content too short: {len(content)} chars (minimum: {min_chars})"
        )

    # Check maximum length
    max_chars = hygiene_config.get("max_chars", DEFAULT_HYGIENE["max_chars"])
    if len(content) > max_chars:
        raise ValueError(
            f"Content too long: {len(content)} chars (maximum: {max_chars}). "
            f"Consider splitting or summarizing."
        )

    # Reject obviously useless content (simple rules)
    useless_patterns = [
        r"^(ok|okay|thanks|thx|lol|haha|yeah|yep|nope|k)$",
        r"^\.+$",  # Just dots
        r"^\s*$",  # Just whitespace (shouldn't happen after normalize)
    ]

    content_lower = content.lower()
    for pattern in useless_patterns:
        if re.match(pattern, content_lower):
            raise ValueError(
                f"Content rejected: appears to be non-substantive ('{content[:50]}')"
            )


def commit_memory(
    content: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    source: str = "manual",
    hygiene_config: Dict[str, Any] = None,
    tags: list[str] = None,
    artifact_ref: Optional[str] = None,
    on_commit: Optional[OnCommitHook] = None,
) -> CommitOutcome:
    """
    Commit a memory to storage with M2 quality firewall.

    Args:
        content: Memory content text
        storage: Storage instance
        embedding_engine: Embedding engine instance
        source: Source of the memory (manual, hook, batch)
        hygiene_config: Hygiene configuration (optional, uses defaults if None)
        tags: Optional tags for filtering
        artifact_ref: Optional reference to source artifact (session_id, filename, etc.)
        on_commit: Optional hook called with CommitOutcome (M2→M3 bridge)

    Returns:
        CommitOutcome with decision details

    Raises:
        ValueError: If content fails hygiene validation
    """
    from vestig.core.retrieval import cosine_similarity

    # Merge with defaults
    hygiene = {**DEFAULT_HYGIENE, **(hygiene_config or {})}

    # Track thresholds for outcome
    thresholds = {
        "min_chars": hygiene.get("min_chars", DEFAULT_HYGIENE["min_chars"]),
        "max_chars": hygiene.get("max_chars", DEFAULT_HYGIENE["max_chars"]),
    }

    # Basic empty check
    if not content or not content.strip():
        raise ValueError("Memory content cannot be empty")

    # Normalize content
    normalized = normalize_content(
        content, normalize_whitespace=hygiene.get("normalize_whitespace", True)
    )

    # Calculate content hash early (needed for all outcomes)
    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # Validate content against hygiene rules
    try:
        validate_content(normalized, hygiene)
    except ValueError as e:
        # Hygiene rejection - build outcome and optionally invoke hook
        outcome = CommitOutcome.rejected_hygiene(
            content_hash=content_hash,
            hygiene_reasons=[str(e)],
            source=source,
            tags=tags,
            artifact_ref=artifact_ref,
            thresholds=thresholds,
        )
        if on_commit:
            on_commit(outcome)
        raise  # Re-raise for backward compatibility

    # Generate embedding
    embedding = embedding_engine.embed_text(normalized)

    # M2: Check for near-duplicates (semantic similarity)
    near_dup_config = hygiene.get("near_duplicate", {})
    near_dup_threshold = near_dup_config.get("threshold", 0.92)
    thresholds["near_duplicate_threshold"] = near_dup_threshold

    duplicate_metadata = None
    matched_id = None
    matched_score = None

    if near_dup_config.get("enabled", True):
        all_memories = storage.get_all_memories()

        if all_memories:
            # Find most similar existing memory
            max_score = 0.0
            most_similar_id = None

            for existing in all_memories:
                score = cosine_similarity(embedding, existing.content_embedding)
                if score > max_score:
                    max_score = score
                    most_similar_id = existing.id

            # Check if near-duplicate
            if max_score >= near_dup_threshold:
                # Mark as near-duplicate in metadata
                matched_id = most_similar_id
                matched_score = max_score
                duplicate_metadata = {
                    "duplicate_of": most_similar_id,
                    "duplicate_score": round(max_score, 4),
                }

    # Generate ID
    memory_id = f"mem_{uuid.uuid4()}"

    # Create memory node
    node = MemoryNode.create(
        memory_id=memory_id,
        content=normalized,
        embedding=embedding,
        source=source,
        tags=tags,
    )

    # Add near-duplicate metadata if detected
    if duplicate_metadata:
        node.metadata.update(duplicate_metadata)

    # Store (exact dedupe handled in storage layer)
    stored_id = storage.store_memory(node)

    # Build outcome based on what happened
    if stored_id != memory_id:
        # Exact duplicate detected by storage layer (hash match)
        outcome = CommitOutcome.exact_dupe(
            memory_id=stored_id,
            content_hash=content_hash,
            source=source,
            tags=tags,
            artifact_ref=artifact_ref,
            thresholds=thresholds,
        )
    elif matched_id is not None:
        # Near-duplicate detected
        outcome = CommitOutcome.near_dupe(
            memory_id=stored_id,
            matched_memory_id=matched_id,
            query_score=matched_score,
            content_hash=content_hash,
            source=source,
            tags=tags,
            artifact_ref=artifact_ref,
            thresholds=thresholds,
        )
    else:
        # New memory inserted
        outcome = CommitOutcome.inserted_new(
            memory_id=stored_id,
            content_hash=content_hash,
            source=source,
            tags=tags,
            artifact_ref=artifact_ref,
            thresholds=thresholds,
        )

    # Invoke hook if provided (M2→M3 bridge)
    if on_commit:
        on_commit(outcome)

    return outcome
