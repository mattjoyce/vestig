"""Graph traversal and expansion (M4: Graph Layer)"""

from typing import List, Dict, Any, Set
from vestig.core.models import MemoryNode
from vestig.core.storage import MemoryStorage


def expand_via_entities(
    memory_ids: List[str],
    storage: MemoryStorage,
    limit: int = 5,
    include_expired: bool = False,
    min_confidence: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Expand via shared entities (1-hop through MENTIONS edges).

    Given a list of memory IDs, find memories that share entities with them.

    Args:
        memory_ids: List of source memory IDs
        storage: Storage instance
        limit: Maximum number of expanded memories to return
        include_expired: Include expired memories/edges
        min_confidence: Minimum confidence for MENTIONS edges

    Returns:
        List of dicts with:
            - memory: MemoryNode object
            - retrieval_reason: "graph_expansion_entity"
            - shared_entities: List of shared entity IDs
            - expansion_score: Number of shared entities (for ranking)
    """
    # Track candidate memories and their shared entities
    candidates: Dict[str, Set[str]] = {}  # memory_id -> set of shared entity IDs

    # For each source memory, find connected entities
    for source_id in memory_ids:
        # Get MENTIONS edges from source memory
        mentions_edges = storage.get_edges_from_memory(
            source_id,
            edge_type="MENTIONS",
            include_expired=include_expired,
            min_confidence=min_confidence,
        )

        # For each entity, find other memories that mention it
        for edge in mentions_edges:
            entity_id = edge.to_node

            # Find all memories that mention this entity (incoming edges to entity)
            incoming_edges = storage.get_edges_to_entity(
                entity_id,
                include_expired=include_expired,
                min_confidence=min_confidence,
            )

            # Add candidates (exclude source memories)
            for incoming_edge in incoming_edges:
                candidate_id = incoming_edge.from_node

                # Skip if same as source or already in source list
                if candidate_id == source_id or candidate_id in memory_ids:
                    continue

                # Track shared entity
                if candidate_id not in candidates:
                    candidates[candidate_id] = set()
                candidates[candidate_id].add(entity_id)

    # Convert to list and score by number of shared entities
    results = []
    for memory_id, shared_entities in candidates.items():
        memory = storage.get_memory(memory_id)

        if memory is None:
            continue  # Memory deleted or not found

        # Skip expired memories (unless include_expired=True)
        if not include_expired and memory.t_expired is not None:
            continue

        results.append(
            {
                "memory": memory,
                "retrieval_reason": "graph_expansion_entity",
                "shared_entities": list(shared_entities),
                "expansion_score": len(shared_entities),
            }
        )

    # Sort by expansion score descending (more shared entities = higher score)
    results.sort(key=lambda x: x["expansion_score"], reverse=True)

    # Return top-K
    return results[:limit]


def expand_via_related(
    memory_ids: List[str],
    storage: MemoryStorage,
    limit: int = 5,
    include_expired: bool = False,
    min_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Expand via RELATED edges (1-hop semantic similarity).

    Given a list of memory IDs, find memories connected via RELATED edges.

    Args:
        memory_ids: List of source memory IDs
        storage: Storage instance
        limit: Maximum number of expanded memories to return
        include_expired: Include expired memories/edges
        min_confidence: Minimum confidence for RELATED edges

    Returns:
        List of dicts with:
            - memory: MemoryNode object
            - retrieval_reason: "graph_expansion_related"
            - similarity_score: Edge weight (semantic similarity)
            - source_memory_id: Which source memory connected to this
    """
    # Track candidate memories and their best similarity score
    candidates: Dict[str, tuple] = {}  # memory_id -> (score, source_id)

    # For each source memory, find RELATED edges
    for source_id in memory_ids:
        # Get RELATED edges from source memory
        related_edges = storage.get_edges_from_memory(
            source_id,
            edge_type="RELATED",
            include_expired=include_expired,
            min_confidence=min_confidence,
        )

        # Add candidates (exclude source memories)
        for edge in related_edges:
            candidate_id = edge.to_node

            # Skip if same as source or already in source list
            if candidate_id == source_id or candidate_id in memory_ids:
                continue

            # Track best (highest) similarity score for this candidate
            score = edge.weight
            if candidate_id not in candidates or score > candidates[candidate_id][0]:
                candidates[candidate_id] = (score, source_id)

    # Convert to list
    results = []
    for memory_id, (score, source_id) in candidates.items():
        memory = storage.get_memory(memory_id)

        if memory is None:
            continue  # Memory deleted or not found

        # Skip expired memories (unless include_expired=True)
        if not include_expired and memory.t_expired is not None:
            continue

        results.append(
            {
                "memory": memory,
                "retrieval_reason": "graph_expansion_related",
                "similarity_score": score,
                "source_memory_id": source_id,
            }
        )

    # Sort by similarity score descending (higher similarity = better)
    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Return top-K
    return results[:limit]


def expand_with_graph(
    memory_ids: List[str],
    storage: MemoryStorage,
    entity_limit: int = 3,
    related_limit: int = 3,
    include_expired: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform both entity and related expansions in one call.

    Convenience function that combines entity and related expansion.

    Args:
        memory_ids: List of source memory IDs
        storage: Storage instance
        entity_limit: Max memories to return from entity expansion
        related_limit: Max memories to return from related expansion
        include_expired: Include expired memories/edges

    Returns:
        Dict with two keys:
            - "via_entities": Results from expand_via_entities
            - "via_related": Results from expand_via_related
    """
    entity_expansion = expand_via_entities(
        memory_ids=memory_ids,
        storage=storage,
        limit=entity_limit,
        include_expired=include_expired,
    )

    related_expansion = expand_via_related(
        memory_ids=memory_ids,
        storage=storage,
        limit=related_limit,
        include_expired=include_expired,
    )

    return {
        "via_entities": entity_expansion,
        "via_related": related_expansion,
    }
