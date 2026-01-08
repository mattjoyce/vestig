#!/usr/bin/env python3
"""Test M4 Work Item #1: Entity Schema & Storage

Tests entity creation, storage, retrieval, deduplication, and expiration.
Runs against both SQLite and FalkorDB backends when VESTIG_TEST_FALKORDB=1.
"""

import os
import sys

# Add src to path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.db_interface import DatabaseInterface
from vestig.core.models import EntityNode, compute_norm_key


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


# Legacy standalone execution support
def run_standalone():
    """Run tests standalone without pytest (for backwards compatibility)."""
    import tempfile

    from vestig.core.storage import MemoryStorage

    print("=== M4 Work Item #1: Entity Schema & Storage ===\n")
    print("Running standalone with SQLite backend...\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        storage = MemoryStorage(db_path)

        # Run a subset of tests
        print("Test: Normalization key computation")
        assert compute_norm_key("Alice Smith", "PERSON") == "PERSON:alice smith"
        print("  PASSED\n")

        print("Test: Entity creation and storage")
        entity = EntityNode.create(entity_type="PERSON", canonical_name="Alice Smith")
        entity_id = storage.store_entity(entity)
        retrieved = storage.get_entity(entity_id)
        assert retrieved.canonical_name == "Alice Smith"
        print("  PASSED\n")

        print("Test: Deduplication")
        entity2 = EntityNode.create(entity_type="PERSON", canonical_name="alice smith")
        id2 = storage.store_entity(entity2)
        assert id2 == entity_id
        print("  PASSED\n")

        storage.close()

        print("=" * 50)
        print("All standalone tests passed!")
        print("=" * 50)
        print("\nFor full test coverage, run: pytest tests/test_m4_item1.py -v")

    finally:
        os.remove(db_path)


if __name__ == "__main__":
    run_standalone()
