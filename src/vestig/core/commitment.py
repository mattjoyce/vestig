"""Commitment pipeline with M2 quality firewall"""

import hashlib
import re
import uuid
from typing import TYPE_CHECKING, Any

from vestig.core.embeddings import EmbeddingEngine
from vestig.core.events import CommitOutcome, OnCommitHook
from vestig.core.models import MemoryNode
from vestig.core.storage import MemoryStorage

if TYPE_CHECKING:
    from vestig.core.event_storage import MemoryEventStorage

# Default hygiene settings (used if not in config)
DEFAULT_HYGIENE = {
    "min_chars": 12,
    "max_chars": 4000,
    "normalize_whitespace": True,
    "reject_exact_duplicates": True,
    "near_duplicate": {
        "enabled": True,
        "threshold": 0.92,
        "skip_manual_source": True,  # M3 FIX #6: Skip expensive scan for CLI adds
    },
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


def validate_content(content: str, hygiene_config: dict[str, Any]) -> None:
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
        raise ValueError(f"Content too short: {len(content)} chars (minimum: {min_chars})")

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
            raise ValueError(f"Content rejected: appears to be non-substantive ('{content[:50]}')")


def commit_memory(
    content: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    source: str = "manual",
    hygiene_config: dict[str, Any] = None,
    tags: list[str] = None,
    artifact_ref: str | None = None,
    on_commit: OnCommitHook | None = None,
    event_storage: MemoryEventStorage | None = None,  # M3: Event logging
    m4_config: dict[str, Any] | None = None,  # M4: Graph config
    pre_extracted_entities: list[tuple[str, str, float, str]]
    | None = None,  # M4: Pre-extracted entities
    temporal_hints: Any | None = None,  # ExtractedMemory with temporal fields
    chunk_id: str | None = None,  # M5: Chunk ID (hub link for provenance)
) -> CommitOutcome:
    """
    Commit a memory to storage with M2 quality firewall, M3 event logging, M4 entity extraction, and temporal hints.

    Args:
        content: Memory content text
        storage: Storage instance
        embedding_engine: Embedding engine instance
        source: Source of the memory (manual, hook, batch)
        hygiene_config: Hygiene configuration (optional, uses defaults if None)
        tags: Optional tags for filtering
        artifact_ref: Optional reference to source artifact (session_id, filename, etc.)
        on_commit: Optional hook called with CommitOutcome (M2→M3 bridge)
        event_storage: Optional event storage for M3 event logging
        m4_config: Optional M4 config for entity extraction + graph construction
        pre_extracted_entities: Optional pre-extracted entities (name, type, confidence, evidence)
                                Skips LLM extraction if provided
        temporal_hints: Optional ExtractedMemory with temporal fields (t_valid_hint, temporal_stability_hint)
        chunk_id: Optional chunk ID (M5 hub link for provenance, NULL for manual adds)

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

    # M4: One-shot entity extraction for manual adds (if not pre-extracted)
    # Treat manual content as a "chunk" and extract entities using the same prompt
    extraction_model = None
    extraction_min_confidence = None
    prompt_hash = None
    if pre_extracted_entities is None and m4_config is not None:
        extraction_config = m4_config.get("entity_extraction", {})
        if extraction_config.get("enabled", True):
            from vestig.core.ingestion import extract_memories_from_chunk

            # Get extraction model from M4 config
            extraction_model = extraction_config.get("llm", {}).get("model")
            extraction_min_confidence = extraction_config.get("llm", {}).get("min_confidence", 0.75)

            if extraction_model:
                try:
                    if event_storage:
                        from vestig.core.entity_extraction import compute_prompt_hash, load_prompts

                        prompts = load_prompts()
                        template = prompts.get("extract_memories_from_session")
                        if template:
                            # Handle both string (legacy) and dict (M4+) template formats
                            if isinstance(template, dict):
                                # For dict format, concatenate system + user for hashing
                                template_str = (
                                    f"{template.get('system', '')}\n{template.get('user', '')}"
                                )
                            else:
                                template_str = template
                            prompt_hash = compute_prompt_hash(template_str)

                    # Extract entities one-shot (treating manual input as a chunk)
                    extracted = extract_memories_from_chunk(
                        normalized,
                        model=extraction_model,
                        min_confidence=extraction_min_confidence,
                    )

                    # Use entities from first extracted memory if available
                    if extracted and len(extracted) > 0:
                        pre_extracted_entities = extracted[0].entities
                    else:
                        pre_extracted_entities = []

                except Exception:
                    # Fall back to empty entities if extraction fails
                    pre_extracted_entities = []

    # Calculate content hash early (needed for all outcomes)
    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # Extract temporal hints if provided
    t_valid_hint = None
    temporal_stability_hint = None
    if temporal_hints is not None:
        # Check if it has temporal fields (duck typing)
        if hasattr(temporal_hints, "t_valid_hint"):
            t_valid_hint = temporal_hints.t_valid_hint
        if hasattr(temporal_hints, "temporal_stability_hint"):
            temporal_stability_hint = temporal_hints.temporal_stability_hint

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

    matched_id = None
    matched_score = None

    # M3 FIX #6: Performance optimization - skip near-dupe for manual adds
    # Manual CLI adds are intentional, so we trust the user and avoid expensive scan
    skip_manual = near_dup_config.get("skip_manual_source", True)
    should_check_near_dup = near_dup_config.get("enabled", True)

    if should_check_near_dup and (not skip_manual or source != "manual"):
        # TODO(M4): Optimize with time-window (last 30 days) or rolling window (last 1000)
        # For now: full scan for hook/batch/import sources
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
                # M3 FIX: Don't create new memory - reinforce canonical instead
                matched_id = most_similar_id
                matched_score = max_score

    # M3 FIX: Atomic transaction for store + event + cache updates
    # All DB writes happen in one transaction to prevent partial state
    with storage.conn:
        # M3 FIX (Option A): Near-dupe reinforces canonical, doesn't create new memory
        if matched_id is not None:
            # Near-duplicate detected - reinforce canonical memory, don't insert new
            outcome = CommitOutcome.near_dupe(
                memory_id=matched_id,  # Return canonical ID, not new ID
                matched_memory_id=matched_id,
                query_score=matched_score,
                content_hash=content_hash,
                source=source,
                tags=tags,
                artifact_ref=artifact_ref,
                thresholds=thresholds,
            )
        else:
            # Not a near-duplicate - proceed with normal flow
            # Generate ID
            memory_id = f"mem_{uuid.uuid4()}"

            # Create memory node
            # M3 FIX: Pass pre-computed content_hash to avoid duplication
            node = MemoryNode.create(
                memory_id=memory_id,
                content=normalized,
                embedding=embedding,
                source=source,
                tags=tags,
                content_hash=content_hash,
                t_valid_hint=t_valid_hint,  # Temporal extraction
                temporal_stability_hint=temporal_stability_hint,  # Temporal classification
            )

            # M5: Set chunk_id for hub-and-spoke provenance
            if chunk_id:
                node.chunk_id = chunk_id

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

        # M4: Entity extraction + MENTIONS edge creation
        # (Only for new memories, not duplicates)
        # (Must happen inside transaction before commit)
        if m4_config and outcome.outcome == "INSERTED_NEW":
            _extract_and_link_entities(
                content=normalized,
                memory_id=outcome.memory_id,
                storage=storage,
                m4_config=m4_config,
                artifact_ref=artifact_ref,
                pre_extracted_entities=pre_extracted_entities,
                event_storage=event_storage,
                extraction_model=extraction_model,
                prompt_hash=prompt_hash,
                min_confidence=extraction_min_confidence,
                embedding_engine=embedding_engine,
            )

            # M4: RELATED edge creation (Memory → Memory)
            # Create semantic similarity edges to related memories
            _create_related_edges(
                memory_id=outcome.memory_id,
                embedding=embedding,
                storage=storage,
                m4_config=m4_config,
            )

        # M3: Log event if event_storage provided
        # (Must happen inside transaction before commit)
        if event_storage:
            _log_commit_event(outcome, storage, event_storage)

        # Transaction commits here automatically (context manager)

    # Invoke hook if provided (M2→M3 bridge)
    # (Happens after successful commit)
    if on_commit:
        on_commit(outcome)

    return outcome


def _extract_and_link_entities(
    content: str,
    memory_id: str,
    storage: MemoryStorage,
    m4_config: dict[str, Any],
    artifact_ref: str | None = None,
    pre_extracted_entities: list[tuple[str, str, float, str]] | None = None,
    event_storage: MemoryEventStorage | None = None,
    extraction_model: str | None = None,
    prompt_hash: str | None = None,
    min_confidence: float | None = None,
    embedding_engine: EmbeddingEngine | None = None,
) -> None:
    """
    Store entities and create MENTIONS edges (M4).

    Called from commit_memory() inside transaction.
    Entities are always pre-extracted using one-shot extraction in commit_memory().

    Args:
        content: Memory content (not used, kept for backward compatibility)
        memory_id: Memory ID (for MENTIONS edges)
        storage: Storage instance
        m4_config: M4 configuration dict
        artifact_ref: Optional artifact reference (not used, kept for backward compatibility)
        pre_extracted_entities: Pre-extracted entities (name, type, confidence, evidence)
    """
    from vestig.core.entity_extraction import store_entities
    from vestig.core.models import EdgeNode

    # Check if entity extraction enabled
    extraction_config = m4_config.get("entity_extraction", {})
    if not extraction_config.get("enabled", True):
        return

    # Store pre-extracted entities (with deduplication)
    # Entities are always extracted one-shot in commit_memory()
    if pre_extracted_entities is None:
        pre_extracted_entities = []

    entities = store_entities(
        entities=pre_extracted_entities,
        memory_id=memory_id,
        storage=storage,
        config=m4_config,
        embedding_engine=embedding_engine,
    )

    # Check if MENTIONS edge creation enabled
    mentions_config = m4_config.get("edge_creation", {}).get("mentions", {})
    if not mentions_config.get("enabled", True):
        return

    if event_storage and extraction_model:
        from vestig.core.models import EventNode

        event = EventNode.create(
            memory_id=memory_id,
            event_type="ENTITY_EXTRACTED",
            source="system",
            artifact_ref=artifact_ref,
            payload={
                "model_name": extraction_model,
                "prompt_hash": prompt_hash,
                "entity_count": len(entities),
                "min_confidence": min_confidence,
            },
        )
        event_storage.add_event(event)

    # Create MENTIONS edges for each entity (confidence-gated)
    for entity_id, entity_type, confidence, evidence in entities:
        edge = EdgeNode.create(
            from_node=memory_id,
            to_node=entity_id,
            edge_type="MENTIONS",
            weight=1.0,
            confidence=confidence,
            evidence=evidence,
        )
        storage.store_edge(edge)


def _create_related_edges(
    memory_id: str,
    embedding: list[float],
    storage: MemoryStorage,
    m4_config: dict[str, Any],
) -> None:
    """
    Create RELATED edges to semantically similar memories (M4).

    Called from commit_memory() inside transaction.

    Args:
        memory_id: New memory ID (source of RELATED edges)
        embedding: Embedding vector of new memory
        storage: Storage instance
        m4_config: M4 configuration dict
    """
    from vestig.core.models import EdgeNode
    from vestig.core.retrieval import cosine_similarity

    # Check if RELATED edge creation enabled
    related_config = m4_config.get("edge_creation", {}).get("related", {})
    if not related_config.get("enabled", True):
        return

    # Get config parameters
    similarity_threshold = related_config.get("similarity_threshold", 0.6)
    max_edges_per_memory = related_config.get("max_edges_per_memory", 10)

    # Get all existing memories (exclude current)
    all_memories = storage.get_all_memories()

    # Compute similarity scores
    candidates = []
    for existing in all_memories:
        if existing.id == memory_id:
            continue  # Skip self

        score = cosine_similarity(embedding, existing.content_embedding)

        if score >= similarity_threshold:
            candidates.append((existing.id, score))

    # Sort by score descending and take top-K
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:max_edges_per_memory]

    # Create RELATED edges
    for target_id, score in top_candidates:
        edge = EdgeNode.create(
            from_node=memory_id,
            to_node=target_id,
            edge_type="RELATED",
            weight=score,  # Weight = similarity score
            confidence=score,  # Confidence = similarity score
            evidence=f"semantic_similarity={score:.3f}",
        )
        storage.store_edge(edge)


def _log_commit_event(
    outcome: CommitOutcome,
    storage: MemoryStorage,
    event_storage: MemoryEventStorage,
) -> None:
    """
    Convert CommitOutcome to EventNode and persist (M3).

    Args:
        outcome: The commit outcome to log
        storage: Storage instance for updating convenience fields
        event_storage: Event storage instance
    """
    from vestig.core.models import EventNode

    if outcome.outcome == "REJECTED_HYGIENE":
        return  # Don't log hygiene rejections

    # Map outcome to event type
    event_type_map = {
        "INSERTED_NEW": "ADD",
        "EXACT_DUPE": "REINFORCE_EXACT",
        "NEAR_DUPE": "REINFORCE_NEAR",
    }
    event_type = event_type_map[outcome.outcome]

    # Create event
    event = EventNode.create(
        memory_id=outcome.memory_id,
        event_type=event_type,
        source=outcome.source,
        artifact_ref=outcome.artifact_ref,
        payload={
            "content_hash": outcome.content_hash,
            "tags": outcome.tags,
            "artifact_ref": outcome.artifact_ref,
            "matched_memory_id": outcome.matched_memory_id,
            "query_score": outcome.query_score,
        },
    )
    event_storage.add_event(event)

    # Update convenience fields for reinforcement
    if event_type.startswith("REINFORCE"):
        storage.increment_reinforce_count(outcome.memory_id)
        storage.update_last_seen(outcome.memory_id, outcome.occurred_at)
