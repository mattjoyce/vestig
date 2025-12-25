"""Simple commitment pipeline for M1 (no filtering, no LLM calls)"""

import uuid

from vestig.core.embeddings import EmbeddingEngine
from vestig.core.models import MemoryNode
from vestig.core.storage import MemoryStorage


def commit_memory(
    content: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    source: str = "manual",
) -> str:
    """
    Commit a memory to storage (M1: minimal pipeline, no filtering).

    Args:
        content: Memory content text
        storage: Storage instance
        embedding_engine: Embedding engine instance
        source: Source of the memory (manual, hook, batch)

    Returns:
        Memory ID

    Raises:
        ValueError: If content is empty
    """
    if not content or not content.strip():
        raise ValueError("Memory content cannot be empty")

    # Generate ID
    memory_id = f"mem_{uuid.uuid4()}"

    # Generate embedding
    embedding = embedding_engine.embed_text(content)

    # Create memory node
    node = MemoryNode.create(
        memory_id=memory_id,
        content=content.strip(),
        embedding=embedding,
        source=source,
    )

    # Store
    storage.store_memory(node)

    return memory_id
