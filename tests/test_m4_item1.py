#!/usr/bin/env python3
"""Test M4 Work Item #1: Entity Schema & Storage

Tests entity creation, storage, retrieval, deduplication, and expiration.
Runs against FalkorDB backend.

Extended for Issue #9 Phase 1.4: Model validation tests.
"""

from __future__ import annotations

import os
import sys
import uuid

# Add src to path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.db_interface import DatabaseInterface
from vestig.core.models import EdgeNode, EntityNode, MemoryNode, compute_norm_key


class TestNormKey:
    """Test normalization key computation (no database needed)."""

    def test_basic_normalization(self):
        assert compute_norm_key("Alice Smith", "PERSON") == "PERSON:alice smith"
        assert compute_norm_key("PostgreSQL", "SYSTEM") == "SYSTEM:postgresql"

    def test_whitespace_normalization(self):
        assert compute_norm_key("  Dr. Alice  ", "PERSON") == "PERSON:dr. alice"

    def test_punctuation_preserved(self):
        assert compute_norm_key("Acme, Inc.", "ORG") == "ORG:acme, inc"


class TestEntityNode:
    """Test EntityNode model creation (no database needed)."""

    def test_entity_creation(self):
        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Alice Smith",
        )
        assert entity.id.startswith("ent_")
        assert entity.entity_type == "PERSON"
        assert entity.canonical_name == "Alice Smith"
        assert entity.norm_key == "PERSON:alice smith"
        assert entity.expired_at is None
        assert entity.merged_into is None


class TestEntityStorage:
    """Test entity storage operations against database backends."""

    def test_store_and_retrieve(self, storage: DatabaseInterface):
        """Store an entity and retrieve it by ID."""
        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Alice Smith",
        )
        entity_id = storage.store_entity(entity)
        assert entity_id == entity.id

        retrieved = storage.get_entity(entity_id)
        assert retrieved is not None
        assert retrieved.id == entity.id
        assert retrieved.canonical_name == entity.canonical_name
        assert retrieved.norm_key == entity.norm_key

    def test_find_by_norm_key(self, storage: DatabaseInterface):
        """Find entity by normalized key."""
        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Bob Jones",
        )
        storage.store_entity(entity)

        found = storage.find_entity_by_norm_key("PERSON:bob jones")
        assert found is not None
        assert found.id == entity.id

    def test_deduplication_via_norm_key(self, storage: DatabaseInterface):
        """Entities with same norm_key should deduplicate."""
        entity1 = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Alice Smith",
        )
        entity2 = EntityNode.create(
            entity_type="PERSON",
            canonical_name="alice smith",  # Different capitalization
        )

        # Same norm_key
        assert entity2.norm_key == entity1.norm_key

        id1 = storage.store_entity(entity1)
        id2 = storage.store_entity(entity2)

        # Should return same ID (deduped)
        assert id2 == id1

    def test_different_type_same_name(self, storage: DatabaseInterface):
        """Same name with different type creates separate entities."""
        entity1 = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Alice Smith",
        )
        entity2 = EntityNode.create(
            entity_type="ORG",
            canonical_name="Alice Smith",
        )

        assert entity1.norm_key != entity2.norm_key

        id1 = storage.store_entity(entity1)
        id2 = storage.store_entity(entity2)

        assert id1 != id2

    def test_get_entities_by_type(self, storage: DatabaseInterface):
        """Retrieve entities filtered by type."""
        storage.store_entity(EntityNode.create(entity_type="SYSTEM", canonical_name="PostgreSQL"))
        storage.store_entity(EntityNode.create(entity_type="SYSTEM", canonical_name="Redis"))
        storage.store_entity(EntityNode.create(entity_type="PERSON", canonical_name="Alice"))

        systems = storage.get_entities_by_type("SYSTEM")
        assert len(systems) == 2
        system_names = [e.canonical_name for e in systems]
        assert "PostgreSQL" in system_names
        assert "Redis" in system_names

    def test_expire_entity(self, storage: DatabaseInterface):
        """Expire an entity (soft delete)."""
        entity = EntityNode.create(entity_type="SYSTEM", canonical_name="OldSystem")
        entity_id = storage.store_entity(entity)

        storage.expire_entity(entity_id)

        expired = storage.get_entity(entity_id)
        assert expired is not None
        assert expired.expired_at is not None

    def test_expired_excluded_from_lookup(self, storage: DatabaseInterface):
        """Expired entities excluded from norm_key lookup by default."""
        entity = EntityNode.create(entity_type="SYSTEM", canonical_name="OldSystem")
        entity_id = storage.store_entity(entity)
        storage.expire_entity(entity_id)

        # Excluded by default
        found = storage.find_entity_by_norm_key(entity.norm_key, include_expired=False)
        assert found is None

        # Included when requested
        found = storage.find_entity_by_norm_key(entity.norm_key, include_expired=True)
        assert found is not None

    def test_merge_entity(self, storage: DatabaseInterface):
        """Merge one entity into another."""
        entity1 = EntityNode.create(entity_type="PERSON", canonical_name="Bob Jones")
        entity2 = EntityNode.create(entity_type="PERSON", canonical_name="Robert Jones")
        id1 = storage.store_entity(entity1)
        id2 = storage.store_entity(entity2)

        # Merge entity2 into entity1
        storage.expire_entity(id2, merged_into=id1)

        merged = storage.get_entity(id2)
        assert merged.merged_into == id1
        assert merged.expired_at is not None


# Issue #9 Phase 1.4: Model Validation Tests


class TestMemoryNodeValidation:
    """Test MemoryNode model creation and field validation (no database needed)."""

    def test_memory_creation_basic(self):
        """Test basic MemoryNode creation with required fields."""
        embedding = [0.1] * 768  # Sample embedding
        memory = MemoryNode.create(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            content="Test memory content",
            embedding=embedding,
            source="test",
        )
        assert memory.id.startswith("mem_")
        assert memory.content == "Test memory content"
        assert memory.kind == "MEMORY"  # default

    def test_memory_embedding_is_list_of_floats(self):
        """Test that memory embedding is stored as list[float]."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        memory = MemoryNode.create(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            content="Test",
            embedding=embedding,
            source="test",
        )
        assert isinstance(memory.content_embedding, list)
        assert all(isinstance(x, float) for x in memory.content_embedding)
        assert memory.content_embedding == embedding

    def test_memory_content_hash_computed(self):
        """Test that content_hash is computed when not provided."""
        memory = MemoryNode.create(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            content="Unique content for hashing",
            embedding=[0.1] * 10,
            source="test",
        )
        assert memory.content_hash is not None
        assert len(memory.content_hash) == 64  # SHA256 hex length

    def test_memory_content_hash_deterministic(self):
        """Test that same content produces same hash."""
        content = "Deterministic content test"
        mem1 = MemoryNode.create(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            content=content,
            embedding=[0.1] * 10,
            source="test",
        )
        mem2 = MemoryNode.create(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            content=content,
            embedding=[0.2] * 10,  # Different embedding
            source="test",
        )
        assert mem1.content_hash == mem2.content_hash

    def test_memory_timestamps_initialized(self):
        """Test that temporal fields are initialized."""
        memory = MemoryNode.create(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            content="Test",
            embedding=[0.1] * 10,
            source="test",
        )
        assert memory.created_at is not None
        assert memory.t_valid is not None
        assert memory.t_created is not None
        assert memory.t_expired is None  # Not expired
        assert memory.t_invalid is None  # Not invalidated

    def test_memory_temporal_hints(self):
        """Test that temporal hints are respected."""
        memory = MemoryNode.create(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            content="Historical fact",
            embedding=[0.1] * 10,
            source="test",
            t_valid_hint="2020-01-01T00:00:00Z",
            temporal_stability_hint="static",
        )
        assert memory.t_valid == "2020-01-01T00:00:00Z"
        assert memory.temporal_stability == "static"

    def test_memory_metadata_includes_source(self):
        """Test that metadata includes source field."""
        memory = MemoryNode.create(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            content="Test",
            embedding=[0.1] * 10,
            source="manual",
        )
        assert "source" in memory.metadata
        assert memory.metadata["source"] == "manual"

    def test_memory_metadata_includes_tags(self):
        """Test that tags are included in metadata."""
        memory = MemoryNode.create(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            content="Tagged memory",
            embedding=[0.1] * 10,
            source="test",
            tags=["tag1", "tag2"],
        )
        assert "tags" in memory.metadata
        assert memory.metadata["tags"] == ["tag1", "tag2"]


class TestEntityNodeEmbedding:
    """Test EntityNode embedding format validation (no database needed)."""

    def test_entity_embedding_default_none(self):
        """Test that entity embedding defaults to None."""
        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Test Person",
        )
        assert entity.embedding is None

    def test_entity_embedding_assignable(self):
        """Test that embedding can be assigned after creation."""
        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Test Person",
        )
        # Assign embedding (list[float])
        entity.embedding = [0.1, 0.2, 0.3]
        assert entity.embedding == [0.1, 0.2, 0.3]

    def test_entity_id_format(self):
        """Test that entity ID has correct format."""
        entity = EntityNode.create(
            entity_type="ORG",
            canonical_name="Acme Corp",
        )
        assert entity.id.startswith("ent_")
        # UUID part should be valid
        uuid_part = entity.id[4:]  # Remove "ent_" prefix
        assert len(uuid_part) == 36  # Standard UUID length with dashes

    def test_entity_custom_id(self):
        """Test that custom entity ID is respected."""
        custom_id = "ent_custom_12345"
        entity = EntityNode.create(
            entity_type="SYSTEM",
            canonical_name="Custom System",
            entity_id=custom_id,
        )
        assert entity.id == custom_id


class TestEdgeNodeValidation:
    """Test EdgeNode model validation (no database needed)."""

    def test_edge_creation_basic(self):
        """Test basic edge creation with required fields."""
        edge = EdgeNode.create(
            from_node="mem_123",
            to_node="ent_456",
            edge_type="MENTIONS",
        )
        assert edge.edge_id.startswith("edge_")
        assert edge.from_node == "mem_123"
        assert edge.to_node == "ent_456"
        assert edge.edge_type == "MENTIONS"
        assert edge.weight == 1.0  # default

    def test_edge_invalid_type_raises(self):
        """Test that invalid edge type raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Invalid edge_type"):
            EdgeNode.create(
                from_node="mem_123",
                to_node="ent_456",
                edge_type="INVALID_TYPE",
            )

    def test_edge_all_valid_types(self):
        """Test that all documented edge types are valid."""
        valid_types = [
            "MENTIONS",
            "RELATED",
            "SUMMARIZES",
            "CONTAINS",
            "LINKED",
            "SUMMARIZED_BY",
            "PRODUCED",
            "HAS_CHUNK",
            "AFFECTS",
        ]
        for edge_type in valid_types:
            edge = EdgeNode.create(
                from_node="node_a",
                to_node="node_b",
                edge_type=edge_type,
            )
            assert edge.edge_type == edge_type

    def test_edge_confidence_range(self):
        """Test that confidence can be set to valid values."""
        edge = EdgeNode.create(
            from_node="mem_123",
            to_node="ent_456",
            edge_type="MENTIONS",
            confidence=0.85,
        )
        assert edge.confidence == 0.85

    def test_edge_evidence_truncation(self):
        """Test that long evidence is truncated."""
        long_evidence = "A" * 300  # Longer than 200 char limit
        edge = EdgeNode.create(
            from_node="mem_123",
            to_node="ent_456",
            edge_type="MENTIONS",
            evidence=long_evidence,
        )
        assert len(edge.evidence) == 200
        assert edge.evidence.endswith("...")

    def test_edge_timestamps_initialized(self):
        """Test that edge temporal fields are initialized."""
        edge = EdgeNode.create(
            from_node="mem_123",
            to_node="ent_456",
            edge_type="MENTIONS",
        )
        assert edge.t_valid is not None
        assert edge.t_created is not None
        assert edge.t_expired is None
        assert edge.t_invalid is None
