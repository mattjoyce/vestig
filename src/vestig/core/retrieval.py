"""Retrieval logic for M1 (brute-force cosine similarity)"""

from datetime import datetime
from typing import List, Tuple

import numpy as np

from vestig.core.embeddings import EmbeddingEngine
from vestig.core.models import MemoryNode
from vestig.core.storage import MemoryStorage


def cosine_similarity(a: List[float], b: List[float]) -> float:
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
) -> List[Tuple[MemoryNode, float]]:
    """
    Search memories by semantic similarity (brute-force).

    Args:
        query: Search query text
        storage: Storage instance
        embedding_engine: Embedding engine instance
        limit: Number of top results to return

    Returns:
        List of (MemoryNode, similarity_score) tuples, sorted by score descending
    """
    # Generate query embedding
    query_embedding = embedding_engine.embed_text(query)

    # Load all memories (brute-force for M1)
    all_memories = storage.get_all_memories()

    if not all_memories:
        return []

    # Compute similarity scores
    scored_memories = []
    for memory in all_memories:
        score = cosine_similarity(query_embedding, memory.content_embedding)
        scored_memories.append((memory, score))

    # Sort by score descending and return top-K
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    return scored_memories[:limit]


def format_search_results(results: List[Tuple[MemoryNode, float]]) -> str:
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


def format_recall_results(results: List[Tuple[MemoryNode, float]]) -> str:
    """
    Format recall results for agent context (clean text blocks).

    Args:
        results: List of (MemoryNode, similarity_score) tuples

    Returns:
        Formatted string suitable for LLM context
    """
    if not results:
        return "No memories found."

    blocks = []
    for memory, _ in results:
        # Boring UTC timestamp (no humanization)
        created_dt = datetime.fromisoformat(memory.created_at.replace("Z", "+00:00"))
        timestamp_str = created_dt.strftime("%Y-%m-%d %H:%M UTC")

        blocks.append(f"{memory.content}\n\nCreated: {timestamp_str}")

    return "\n\n---\n\n".join(blocks)
