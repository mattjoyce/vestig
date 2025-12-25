"""Data models for Vestig M1"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any


@dataclass
class MemoryNode:
    """Memory node for M1 (minimal schema)"""

    id: str
    content: str
    content_embedding: List[float]
    created_at: str  # ISO 8601 timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        memory_id: str,
        content: str,
        embedding: List[float],
        source: str = "manual",
    ) -> "MemoryNode":
        """
        Create a new memory node.

        Args:
            memory_id: Unique identifier (e.g., mem_uuid)
            content: Memory content text
            embedding: Content embedding vector
            source: Source of the memory (manual, hook, batch)

        Returns:
            MemoryNode instance
        """
        return cls(
            id=memory_id,
            content=content,
            content_embedding=embedding,
            created_at=datetime.utcnow().isoformat() + "Z",
            metadata={"source": source},
        )
