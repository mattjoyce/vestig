"""Retrieval logic for M1 (brute-force cosine similarity) with M3 TraceRank"""

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


def search_memories(
    query: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    limit: int = 5,
    event_storage: MemoryEventStorage | None = None,  # M3
    tracerank_config: TraceRankConfig | None = None,  # M3
    include_expired: bool = False,  # M3
    show_timing: bool = False,  # Performance instrumentation
) -> list[tuple[MemoryNode, float]]:
    """
    Search memories by semantic similarity (brute-force) with M3 TraceRank.

    Args:
        query: Search query text
        storage: Storage instance
        embedding_engine: Embedding engine instance
        limit: Number of top results to return
        event_storage: Optional event storage for TraceRank (M3)
        tracerank_config: Optional TraceRank configuration (M3)
        include_expired: Include deprecated/expired memories (M3)
        show_timing: Display performance timing breakdown

    Returns:
        List of (MemoryNode, final_score) tuples, sorted by score descending
        final_score = semantic_score * tracerank_multiplier
    """
    t_start = time.perf_counter()
    timings = {}

    # Generate query embedding
    t0 = time.perf_counter()
    query_embedding = embedding_engine.embed_text(query)
    timings['1_embedding_generation'] = time.perf_counter() - t0

    # Load memories (active only or all) - M3
    t0 = time.perf_counter()
    if include_expired or event_storage is None:
        all_memories = storage.get_all_memories()
    else:
        all_memories = storage.get_active_memories()
    timings['2_load_memories'] = time.perf_counter() - t0

    if not all_memories:
        if show_timing:
            print(f"\n[TIMING] Total: {(time.perf_counter() - t_start)*1000:.0f}ms (no memories)")
        return []

    # Compute semantic scores
    t0 = time.perf_counter()
    scored_memories = []
    for memory in all_memories:
        semantic_score = cosine_similarity(query_embedding, memory.content_embedding)
        scored_memories.append((memory, semantic_score))
    timings['3_semantic_scoring'] = time.perf_counter() - t0

    # M3: Apply Enhanced TraceRank if enabled
    if event_storage and tracerank_config and tracerank_config.enabled:
        from vestig.core.tracerank import compute_enhanced_multiplier

        # Compute Enhanced TraceRank for all memories
        t0 = time.perf_counter()
        tracerank_timings = {'events': 0, 'edges': 0, 'compute': 0}

        for i, (memory, semantic_score) in enumerate(scored_memories):
            # Get reinforcement events
            t1 = time.perf_counter()
            events = event_storage.get_reinforcement_events(memory.id)
            tracerank_timings['events'] += time.perf_counter() - t1

            # Get inbound edge count (graph connectivity)
            t1 = time.perf_counter()
            inbound_edges = storage.get_edges_to_memory(memory.id, include_expired=False)
            edge_count = len(inbound_edges)
            tracerank_timings['edges'] += time.perf_counter() - t1

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
            tracerank_timings['compute'] += time.perf_counter() - t1

            # Multiply semantic score by enhanced multiplier
            scored_memories[i] = (memory, semantic_score * multiplier)

        timings['4_tracerank_total'] = time.perf_counter() - t0
        timings['4a_tracerank_events'] = tracerank_timings['events']
        timings['4b_tracerank_edges'] = tracerank_timings['edges']
        timings['4c_tracerank_compute'] = tracerank_timings['compute']

    # Sort by final score descending and return top-K
    t0 = time.perf_counter()
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    result = scored_memories[:limit]
    timings['5_sort_and_slice'] = time.perf_counter() - t0

    timings['TOTAL'] = time.perf_counter() - t_start

    if show_timing:
        print("\n" + "="*60)
        print("PERFORMANCE BREAKDOWN")
        print("="*60)
        for key, duration in timings.items():
            ms = duration * 1000
            pct = (duration / timings['TOTAL'] * 100) if timings['TOTAL'] > 0 else 0
            indent = "  " if key.startswith(('4a', '4b', '4c')) else ""
            print(f"{indent}{key:30s} {ms:8.0f}ms  ({pct:5.1f}%)")
        print("="*60 + "\n")

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
