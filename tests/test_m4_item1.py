#!/usr/bin/env python3
"""Test M4 Work Item #1: Entity Schema & Storage"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.models import EntityNode, compute_norm_key
from vestig.core.storage import MemoryStorage


def test_entity_schema_and_storage():
    """Test Entity node creation and storage operations"""
    print("=== M4 Work Item #1: Entity Schema & Storage ===\n")

    # Use test database
    test_db = "./data/test_m4_item1.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    storage = MemoryStorage(test_db)

    # Test 1: compute_norm_key function
    print("Test 1: Normalization key computation")
    assert compute_norm_key("Alice Smith", "PERSON") == "PERSON:alice smith"
    assert compute_norm_key("PostgreSQL", "SYSTEM") == "SYSTEM:postgresql"
    assert compute_norm_key("  Dr. Alice  ", "PERSON") == "PERSON:dr. alice"  # Period preserved in middle
    assert compute_norm_key("Acme, Inc.", "ORG") == "ORG:acme, inc"  # Comma preserved in middle
    print("✓ Normalization keys computed correctly\n")

    # Test 2: EntityNode creation
    print("Test 2: EntityNode creation")
    entity1 = EntityNode.create(
        entity_type="PERSON",
        canonical_name="Alice Smith",
    )
    assert entity1.id.startswith("ent_")
    assert entity1.entity_type == "PERSON"
    assert entity1.canonical_name == "Alice Smith"
    assert entity1.norm_key == "PERSON:alice smith"
    assert entity1.expired_at is None
    assert entity1.merged_into is None
    print(f"✓ EntityNode created: {entity1.id} ({entity1.canonical_name})\n")

    # Test 3: Store entity
    print("Test 3: Store entity")
    entity_id = storage.store_entity(entity1)
    assert entity_id == entity1.id
    print(f"✓ Entity stored with ID: {entity_id}\n")

    # Test 4: Retrieve entity by ID
    print("Test 4: Retrieve entity by ID")
    retrieved = storage.get_entity(entity_id)
    assert retrieved is not None
    assert retrieved.id == entity1.id
    assert retrieved.canonical_name == entity1.canonical_name
    assert retrieved.norm_key == entity1.norm_key
    print(f"✓ Entity retrieved: {retrieved.canonical_name}\n")

    # Test 5: Find entity by norm_key
    print("Test 5: Find entity by norm_key")
    found = storage.find_entity_by_norm_key("PERSON:alice smith")
    assert found is not None
    assert found.id == entity1.id
    print(f"✓ Entity found by norm_key: {found.canonical_name}\n")

    # Test 6: Deduplication via norm_key
    print("Test 6: Deduplication via norm_key")
    entity2 = EntityNode.create(
        entity_type="PERSON",
        canonical_name="alice smith",  # Different capitalization
    )
    # Should get same norm_key
    assert entity2.norm_key == entity1.norm_key

    # Store should return existing ID
    entity2_id = storage.store_entity(entity2)
    assert entity2_id == entity1.id  # Same ID (deduped!)
    print(f"✓ Deduplication works: '{entity2.canonical_name}' → existing ID {entity1.id}\n")

    # Test 7: Store different entity type (same name)
    print("Test 7: Different entity type (same name, different norm_key)")
    entity3 = EntityNode.create(
        entity_type="ORG",
        canonical_name="Alice Smith",  # Same name, different type
    )
    assert entity3.norm_key == "ORG:alice smith"  # Different norm_key
    entity3_id = storage.store_entity(entity3)
    assert entity3_id != entity1.id  # Different ID (different type)
    print(f"✓ Different type creates separate entity: {entity3_id}\n")

    # Test 8: Get entities by type
    print("Test 8: Get entities by type")
    # Add more entities
    entity4 = EntityNode.create(entity_type="SYSTEM", canonical_name="PostgreSQL")
    entity5 = EntityNode.create(entity_type="SYSTEM", canonical_name="Redis")
    storage.store_entity(entity4)
    storage.store_entity(entity5)

    systems = storage.get_entities_by_type("SYSTEM")
    assert len(systems) == 2
    system_names = [e.canonical_name for e in systems]
    assert "PostgreSQL" in system_names
    assert "Redis" in system_names
    print(f"✓ Retrieved {len(systems)} SYSTEM entities\n")

    # Test 9: Expire entity (soft delete)
    print("Test 9: Expire entity")
    storage.expire_entity(entity4.id)
    expired = storage.get_entity(entity4.id)
    assert expired is not None
    assert expired.expired_at is not None
    print(f"✓ Entity {entity4.id} expired at {expired.expired_at}\n")

    # Test 10: Exclude expired from find_by_norm_key
    print("Test 10: Exclude expired from norm_key lookup")
    found_expired = storage.find_entity_by_norm_key(
        entity4.norm_key, include_expired=False
    )
    assert found_expired is None  # Excluded by default

    found_with_expired = storage.find_entity_by_norm_key(
        entity4.norm_key, include_expired=True
    )
    assert found_with_expired is not None
    print("✓ Expired entities excluded by default\n")

    # Test 11: Soft merge (expire with merged_into)
    print("Test 11: Soft merge (expire with merged_into)")
    entity6 = EntityNode.create(entity_type="PERSON", canonical_name="Bob Jones")
    entity7 = EntityNode.create(entity_type="PERSON", canonical_name="Robert Jones")
    id6 = storage.store_entity(entity6)
    id7 = storage.store_entity(entity7)

    # Merge entity7 into entity6
    storage.expire_entity(id7, merged_into=id6)
    merged = storage.get_entity(id7)
    assert merged.merged_into == id6
    print(f"✓ Entity {id7} merged into {id6}\n")

    # Cleanup
    storage.close()
    os.remove(test_db)

    print("=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
    print("\nWork Item #1 (Entity Schema & Storage) complete!")


if __name__ == "__main__":
    test_entity_schema_and_storage()
