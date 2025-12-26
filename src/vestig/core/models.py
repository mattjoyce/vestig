"""Data models for Vestig"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any


@dataclass
class MemoryNode:
    """Memory node with M2 dedupe support"""

    id: str
    content: str
    content_embedding: List[float]
    content_hash: str  # SHA256 of normalized content (M2: dedupe)
    created_at: str  # ISO 8601 timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        memory_id: str,
        content: str,
        embedding: List[float],
        source: str = "manual",
        tags: List[str] = None,
    ) -> "MemoryNode":
        """
        Create a new memory node.

        Args:
            memory_id: Unique identifier (e.g., mem_uuid)
            content: Memory content text (normalized)
            embedding: Content embedding vector
            source: Source of the memory (manual, hook, batch)
            tags: Optional tags for filtering

        Returns:
            MemoryNode instance
        """
        # Compute content hash for dedupe
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Build metadata
        metadata = {"source": source}
        if tags:
            metadata["tags"] = tags

        return cls(
            id=memory_id,
            content=content,
            content_embedding=embedding,
            content_hash=content_hash,
            created_at=datetime.utcnow().isoformat() + "Z",
            metadata=metadata,
        )
