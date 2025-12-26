"""Retrieval logic for M1 (brute-force cosine similarity) with M3 TraceRank"""

from datetime import datetime
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

    Returns:
        List of (MemoryNode, final_score) tuples, sorted by score descending
        final_score = semantic_score * tracerank_multiplier
    """
    # Generate query embedding
    query_embedding = embedding_engine.embed_text(query)

    # Load memories (active only or all) - M3
    if include_expired or event_storage is None:
        all_memories = storage.get_all_memories()
    else:
        all_memories = storage.get_active_memories()

    if not all_memories:
        return []

    # Compute semantic scores
    scored_memories = []
    for memory in all_memories:
        semantic_score = cosine_similarity(query_embedding, memory.content_embedding)
        scored_memories.append((memory, semantic_score))

    # M3: Apply TraceRank if enabled
    if event_storage and tracerank_config and tracerank_config.enabled:
        from vestig.core.tracerank import compute_tracerank_multiplier

        # Compute TraceRank for all memories
        for i, (memory, semantic_score) in enumerate(scored_memories):
            events = event_storage.get_reinforcement_events(memory.id)
            tracerank = compute_tracerank_multiplier(events, tracerank_config)
            # Multiply semantic score by TraceRank
            scored_memories[i] = (memory, semantic_score * tracerank)

    # Sort by final score descending and return top-K
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    return scored_memories[:limit]


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


def format_recall_results(results: list[tuple[MemoryNode, float]]) -> str:
    """
    Format recall results for agent context (M2: stable contract, M3: temporal hints).

    Args:
        results: List of (MemoryNode, similarity_score) tuples

    Returns:
        Formatted string suitable for LLM context

    Format:
        [mem_...] (source=manual, created=2025-12-26T08:12:00Z, score=0.8123, reinforced=3x, last_seen=..., status=EXPIRED)
        <content>
    """
    if not results:
        return "No memories found."

    blocks = []
    for memory, score in results:
        # Extract source from metadata
        source = memory.metadata.get("source", "unknown")

        # Format: [id] (source=..., created=..., score=...)
        header = f"[{memory.id}] (source={source}, created={memory.created_at}, score={score:.4f}"

        # M3: Add reinforcement + validity hints
        if hasattr(memory, "reinforce_count") and memory.reinforce_count > 0:
            header += f", reinforced={memory.reinforce_count}x"
        if hasattr(memory, "last_seen_at") and memory.last_seen_at:
            header += f", last_seen={memory.last_seen_at}"
        if hasattr(memory, "t_expired") and memory.t_expired:
            header += ", status=EXPIRED"

        header += ")"

        blocks.append(f"{header}\n{memory.content}")

    return "\n\n---\n\n".join(blocks)
