"""Retrieval: M1 semantic similarity + M3 TraceRank + M5 hybrid entity retrieval"""

import json
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from vestig.core.embeddings import EmbeddingEngine
from vestig.core.models import MemoryNode
from vestig.core.storage import MemoryStorage

if TYPE_CHECKING:
    from vestig.core.event_storage import MemoryEventStorage
    from vestig.core.tracerank import TraceRankConfig


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0-1, higher is more similar)
    """
    a_arr = np.array(a)
    b_arr = np.array(b)

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    # Guard against zero vectors (avoid NaN)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def match_query_entities_to_db(
    query_entities: list[tuple[str, str, float, str]],
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    similarity_threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """
    Match query entities to database entities using semantic similarity.

    Args:
        query_entities: Entities extracted from query (name, type, conf, evidence)
        storage: Storage for entity lookup
        embedding_engine: For embedding query entity names
        similarity_threshold: Minimum similarity to consider a match

    Returns:
        List of (db_entity_id, query_entity_name, similarity_score) tuples
    """
    from vestig.core.models import compute_norm_key

    matched_entities = []

    for q_ent_name, q_ent_type, q_conf, q_evidence in query_entities:
        # First try exact match via norm_key (fast path)
        norm_key = compute_norm_key(q_ent_name, q_ent_type)
        exact_match = storage.find_entity_by_norm_key(norm_key, include_expired=False)

        if exact_match:
            # Perfect match
            matched_entities.append((exact_match.id, q_ent_name, 1.0))
            continue

        # No exact match - try semantic similarity
        q_ent_embedding = embedding_engine.embed_text(q_ent_name.lower())

        # Get all entities of same type
        all_entities = storage.get_all_entities(include_expired=False)
        type_entities = [e for e in all_entities if e.entity_type == q_ent_type]

        # Find best match above threshold
        best_match = None
        best_similarity = 0.0

        for db_entity in type_entities:
            if db_entity.embedding:
                db_embedding = json.loads(db_entity.embedding)
                similarity = cosine_similarity(q_ent_embedding, db_embedding)

                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_match = db_entity.id
                    best_similarity = similarity

        if best_match:
            matched_entities.append((best_match, q_ent_name, best_similarity))

    return matched_entities


def retrieve_memories_by_entities(
    matched_entities: list[tuple[str, float]],
    storage: MemoryStorage,
    include_expired: bool = False,
) -> dict[str, float]:
    """
    Retrieve memories that mention matched entities via MENTIONS edges.

    Args:
        matched_entities: DB entities that matched query (entity_id, match_score)
        storage: Storage for edge lookup
        include_expired: Include expired memories

    Returns:
        Dict mapping memory_id to entity-based score
    """
    memory_scores = {}

    for entity_id, entity_similarity in matched_entities:
        # Get all MENTIONS edges pointing to this entity
        # Schema: edges.from_node = memory_id, edges.to_node = entity_id
        all_edges = storage.get_edges_to_entity(entity_id, include_expired=include_expired)
        # Filter for MENTIONS edges only
        edges = [e for e in all_edges if e.edge_type == "MENTIONS"]

        for edge in edges:
            memory_id = edge.from_node

            # Score = entity_similarity × edge_confidence
            score = entity_similarity * (edge.confidence or 1.0)

            # If memory mentions multiple matched entities, keep max score
            memory_scores[memory_id] = max(memory_scores.get(memory_id, 0.0), score)

    return memory_scores


def search_memories(
    query: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    limit: int = 5,
    event_storage: MemoryEventStorage | None = None,  # M3
    tracerank_config: TraceRankConfig | None = None,  # M3
    include_expired: bool = False,  # M3
    show_timing: bool = False,  # Performance instrumentation
    entity_config: dict | None = None,  # M5: Entity-based retrieval config
    model: str | None = None,  # M5: Model for entity extraction
) -> list[tuple[MemoryNode, float]]:
    """
    Search memories with hybrid semantic + entity-based retrieval (M5) and TraceRank (M3).

    Args:
        query: Search query text
        storage: Storage instance
        embedding_engine: Embedding engine instance
        limit: Number of top results to return
        event_storage: Optional event storage for TraceRank (M3)
        tracerank_config: Optional TraceRank configuration (M3)
        include_expired: Include deprecated/expired memories (M3)
        show_timing: Display performance timing breakdown
        entity_config: Optional config for entity-based retrieval (M5)
            Expected keys: enabled (bool), entity_weight (float), similarity_threshold (float)
        model: Optional LLM model name for entity extraction (M5)

    Returns:
        List of (MemoryNode, final_score) tuples, sorted by score descending
        final_score = (semantic_score + entity_score) * tracerank_multiplier
    """
    t_start = time.perf_counter()
    timings = {}

    # Generate query embedding
    t0 = time.perf_counter()
    query_embedding = embedding_engine.embed_text(query)
    timings["1_embedding_generation"] = time.perf_counter() - t0

    # Load memories (active only or all) - M3
    t0 = time.perf_counter()
    if include_expired or event_storage is None:
        all_memories = storage.get_all_memories()
    else:
        all_memories = storage.get_active_memories()
    timings["2_load_memories"] = time.perf_counter() - t0

    if not all_memories:
        if show_timing:
            print(f"\n[TIMING] Total: {(time.perf_counter() - t_start) * 1000:.0f}ms (no memories)")
        return []

    # Compute semantic scores
    t0 = time.perf_counter()
    scored_memories = []
    for memory in all_memories:
        semantic_score = cosine_similarity(query_embedding, memory.content_embedding)
        scored_memories.append((memory, semantic_score))
    timings["3_semantic_scoring"] = time.perf_counter() - t0

    # M5: Entity-based retrieval (optional)
    entity_scores = {}
    if entity_config and entity_config.get("enabled", False) and model:
        from vestig.core.entity_extraction import extract_entities_from_text

        t0 = time.perf_counter()

        # Extract entities from query
        try:
            query_entities = extract_entities_from_text(query, model)
            timings["4_entity_extraction"] = time.perf_counter() - t0

            if query_entities:
                # Match to DB entities
                t0 = time.perf_counter()
                similarity_threshold = entity_config.get("similarity_threshold", 0.7)
                matched_entities = match_query_entities_to_db(
                    query_entities, storage, embedding_engine, similarity_threshold
                )
                timings["5_entity_matching"] = time.perf_counter() - t0

                if matched_entities:
                    # Get memories via MENTIONS edges
                    t0 = time.perf_counter()
                    entity_scores = retrieve_memories_by_entities(
                        [(ent_id, similarity) for ent_id, _, similarity in matched_entities],
                        storage,
                        include_expired,
                    )
                    timings["6_entity_retrieval"] = time.perf_counter() - t0

                    # Combine semantic + entity scores
                    t0 = time.perf_counter()
                    entity_weight = entity_config.get("entity_weight", 0.5)

                    for i, (memory, semantic_score) in enumerate(scored_memories):
                        entity_score = entity_scores.get(memory.id, 0.0)

                        # Weighted combination
                        combined_score = (
                            1 - entity_weight
                        ) * semantic_score + entity_weight * entity_score

                        scored_memories[i] = (memory, combined_score)

                    timings["7_score_combination"] = time.perf_counter() - t0
        except Exception as e:
            # Entity path failed - fall back to semantic only
            if show_timing:
                print(f"[WARNING] Entity retrieval failed: {e}")
            timings["4_entity_extraction"] = time.perf_counter() - t0

    # M3: Apply Enhanced TraceRank if enabled
    if event_storage and tracerank_config and tracerank_config.enabled:
        from vestig.core.tracerank import compute_enhanced_multiplier

        # Compute Enhanced TraceRank for all memories
        t0 = time.perf_counter()
        tracerank_timings = {"events": 0, "edges": 0, "compute": 0}

        for i, (memory, semantic_score) in enumerate(scored_memories):
            # Get reinforcement events
            t1 = time.perf_counter()
            events = event_storage.get_reinforcement_events(memory.id)
            tracerank_timings["events"] += time.perf_counter() - t1

            # Get inbound edge count (graph connectivity)
            t1 = time.perf_counter()
            inbound_edges = storage.get_edges_to_memory(memory.id, include_expired=False)
            edge_count = len(inbound_edges)
            tracerank_timings["edges"] += time.perf_counter() - t1

            # Compute comprehensive multiplier
            t1 = time.perf_counter()
            multiplier = compute_enhanced_multiplier(
                memory_id=memory.id,
                temporal_stability=memory.temporal_stability,
                t_valid=memory.t_valid or memory.created_at,  # Fallback to created_at
                inbound_edge_count=edge_count,
                reinforcement_events=events,
                config=tracerank_config,
            )
            tracerank_timings["compute"] += time.perf_counter() - t1

            # Multiply semantic score by enhanced multiplier
            scored_memories[i] = (memory, semantic_score * multiplier)

        timings["4_tracerank_total"] = time.perf_counter() - t0
        timings["4a_tracerank_events"] = tracerank_timings["events"]
        timings["4b_tracerank_edges"] = tracerank_timings["edges"]
        timings["4c_tracerank_compute"] = tracerank_timings["compute"]

    # Sort by final score descending and return top-K
    t0 = time.perf_counter()
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    result = scored_memories[:limit]
    timings["5_sort_and_slice"] = time.perf_counter() - t0

    timings["TOTAL"] = time.perf_counter() - t_start

    if show_timing:
        print("\n" + "=" * 60)
        print("PERFORMANCE BREAKDOWN")
        print("=" * 60)

        # Show entity path status if configured
        if entity_config and entity_config.get("enabled", False):
            entity_weight = entity_config.get("entity_weight", 0.5)
            print(f"Entity path: ENABLED (weight={entity_weight:.2f})")
            if entity_scores:
                print(f"  Memories with entity matches: {len(entity_scores)}")
        else:
            print("Entity path: DISABLED")
        print()

        for key, duration in timings.items():
            ms = duration * 1000
            pct = (duration / timings["TOTAL"] * 100) if timings["TOTAL"] > 0 else 0
            indent = "  " if key.startswith(("4a", "4b", "4c")) else ""
            print(f"{indent}{key:30s} {ms:8.0f}ms  ({pct:5.1f}%)")
        print("=" * 60 + "\n")

    return result


def format_search_results(results: list[tuple[MemoryNode, float]]) -> str:
    """
    Format search results for display.

    Args:
        results: List of (MemoryNode, similarity_score) tuples

    Returns:
        Formatted string for terminal output
    """
    if not results:
        return "No memories found."

    lines = []
    for memory, score in results:
        # Truncate content to 100 chars
        content_preview = memory.content[:100]
        if len(memory.content) > 100:
            content_preview += "..."

        # Parse and format timestamp
        created_dt = datetime.fromisoformat(memory.created_at.replace("Z", "+00:00"))
        created_str = created_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        lines.append(
            f"ID: {memory.id}\n"
            f"Score: {score:.4f}\n"
            f"Created: {created_str}\n"
            f"Content: {content_preview}\n"
        )

    return "\n".join(lines)


def _format_age(timestamp_str: str) -> str:
    """
    Format timestamp as human-readable age.

    Args:
        timestamp_str: ISO 8601 timestamp string

    Returns:
        Human-readable age (e.g., "3d", "2h", "45m", "just now")
    """
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        now = datetime.now(timezone.utc)

        # Ensure both are timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        delta = now - timestamp

        # Format as compact age
        total_seconds = delta.total_seconds()

        if total_seconds < 60:
            return "just now"
        elif total_seconds < 3600:  # < 1 hour
            minutes = int(total_seconds / 60)
            return f"{minutes}m"
        elif total_seconds < 86400:  # < 1 day
            hours = int(total_seconds / 3600)
            return f"{hours}h"
        elif total_seconds < 604800:  # < 1 week
            days = int(total_seconds / 86400)
            return f"{days}d"
        elif total_seconds < 2592000:  # < 30 days
            weeks = int(total_seconds / 604800)
            return f"{weeks}w"
        else:
            months = int(total_seconds / 2592000)
            return f"{months}mo"
    except Exception:
        return "unknown"


def format_recall_results(results: list[tuple[MemoryNode, float]]) -> str:
    """
    Format recall results for agent context.

    Optimized for AI consumption with minimal metadata:
    - score (confidence)
    - age (temporal context/freshness)
    - stability (trust/reliability)

    Args:
        results: List of (MemoryNode, similarity_score) tuples

    Returns:
        Formatted string suitable for LLM context

    Format:
        (score=0.82, age=3d, stability=static)
        <content>
    """
    if not results:
        return "No memories found."

    blocks = []
    for memory, score in results:
        # Compute age from created_at
        age = _format_age(memory.created_at)

        # Get stability (default to unknown if not present)
        stability = getattr(memory, "temporal_stability", "unknown")

        # Minimal header: score, age, stability (no ID - not useful for AI)
        header = f"(score={score:.4f}, age={age}, stability={stability})"

        blocks.append(f"{header}\n{memory.content}")

    return "\n\n---\n\n".join(blocks)


def format_recall_results_with_explanation(
    results: list[tuple[MemoryNode, float]],
    event_storage: "MemoryEventStorage",
    storage: "MemoryStorage",
    tracerank_config: "TraceRankConfig",
) -> str:
    """
    Format recall results with explanations for why each memory was retrieved.

    Args:
        results: List of (MemoryNode, final_score) tuples
        event_storage: Event storage for TraceRank analysis
        storage: Memory storage for graph queries
        tracerank_config: TraceRank configuration

    Returns:
        Formatted string with explanations

    Format:
        [META] (score=0.82, age=3d, stability=static)
        Semantic match. TraceRank: 1.42x (3x reinforced, 2 conn). Static.
        [MEMORY]
        <content>
    """
    if not results:
        return "No memories found."

    blocks = []
    for memory, final_score in results:
        # Compute age from created_at
        age = _format_age(memory.created_at)

        # Get stability (default to unknown if not present)
        stability = getattr(memory, "temporal_stability", "unknown")

        # Header (same as standard format, no ID)
        header = f"(score={final_score:.4f}, age={age}, stability={stability})"

        # Generate explanation
        explanation_parts = []

        # TraceRank analysis
        try:
            from vestig.core.tracerank import compute_enhanced_multiplier

            # Get reinforcement events
            events = event_storage.get_reinforcement_events(memory.id)
            reinforcement_count = len(events)

            # Get graph connectivity (inbound edges)
            inbound_edges = storage.get_edges_to_memory(memory.id)
            edge_count = len(inbound_edges)

            # Compute TraceRank multiplier
            tracerank_mult = compute_enhanced_multiplier(
                memory_id=memory.id,
                temporal_stability=stability,
                t_valid=getattr(memory, "t_valid", None) or memory.created_at,
                inbound_edge_count=edge_count,
                reinforcement_events=events,
                config=tracerank_config,
            )

            # Build explanation (token-efficient)
            explanation_parts.append("Semantic match.")

            # Show TraceRank boost if significant
            if tracerank_mult > 1.0:
                tracerank_details = []
                if reinforcement_count > 0:
                    tracerank_details.append(f"{reinforcement_count}x reinforced")
                if edge_count > 0:
                    tracerank_details.append(f"{edge_count} conn")

                if tracerank_details:
                    details_str = ", ".join(tracerank_details)
                    explanation_parts.append(f"TraceRank: {tracerank_mult:.2f}x ({details_str}).")
                else:
                    explanation_parts.append(f"TraceRank: {tracerank_mult:.2f}x.")

            # Temporal stability note (compact)
            if stability == "dynamic":
                explanation_parts.append("Dynamic (may decay).")
            elif stability == "static":
                explanation_parts.append("Static.")

        except Exception as e:
            explanation_parts.append(f"Semantic match. (Analysis error: {e})")

        explanation = " ".join(explanation_parts)

        # Combine header, explanation, and content with clear labels
        blocks.append(f"[META] {header}\n{explanation}\n[MEMORY]\n{memory.content}")

    return "\n\n---\n\n".join(blocks)
