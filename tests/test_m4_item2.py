#!/usr/bin/env python3
"""Test M4 Work Item #2: Edge Schema & Storage (Bi-temporal)"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vestig.core.models import EdgeNode, EntityNode, MemoryNode
from vestig.core.storage import MemoryStorage


def test_edge_schema_and_storage():
    """Test Edge node creation and storage operations"""
    print("=== M4 Work Item #2: Edge Schema & Storage ===\n")

    # Use test database
    test_db = "./data/test_m4_item2.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    storage = MemoryStorage(test_db)

    # Create test nodes (memory + entities)
    memory1 = MemoryNode.create(
        memory_id="mem_test1",
        content="Alice fixed PostgreSQL bug",
        embedding=[0.1] * 1024,
    )
    entity1 = EntityNode.create(entity_type="PERSON", canonical_name="Alice")
    entity2 = EntityNode.create(entity_type="SYSTEM", canonical_name="PostgreSQL")

    storage.store_memory(memory1)
    storage.store_entity(entity1)
    storage.store_entity(entity2)

    # Test 1: EdgeNode creation with type enforcement
    print("Test 1: EdgeNode creation with type enforcement")
    edge1 = EdgeNode.create(
        from_node=memory1.id,
        to_node=entity1.id,
        edge_type="MENTIONS",
        weight=1.0,
        confidence=0.92,
        evidence="Alice is mentioned as the person who fixed the bug",
    )
    assert edge1.edge_id.startswith("edge_")
    assert edge1.edge_type == "MENTIONS"
    assert edge1.confidence == 0.92
    assert edge1.t_valid is not None
    assert edge1.t_expired is None
    print(f"✓ EdgeNode created: {edge1.edge_id} ({edge1.edge_type})\n")

    # Test 2: Invalid edge type rejection
    print("Test 2: Invalid edge type rejection")
    try:
        invalid_edge = EdgeNode.create(
            from_node=memory1.id,
            to_node=entity1.id,
            edge_type="INVALID_TYPE",  # Should fail
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid edge_type" in str(e)
        print(f"✓ Invalid edge type rejected: {e}\n")

    # Test 3: Store edge
    print("Test 3: Store edge")
    edge_id = storage.store_edge(edge1)
    assert edge_id == edge1.edge_id
    print(f"✓ Edge stored: {edge_id}\n")

    # Test 4: Retrieve edge by ID
    print("Test 4: Retrieve edge by ID")
    retrieved = storage.get_edge(edge_id)
    assert retrieved is not None
    assert retrieved.edge_id == edge1.edge_id
    assert retrieved.from_node == memory1.id
    assert retrieved.to_node == entity1.id
    assert retrieved.confidence == 0.92
    print(f"✓ Edge retrieved: {retrieved.edge_type} (confidence={retrieved.confidence})\n")

    # Test 5: Get edges from memory
    print("Test 5: Get outgoing edges from memory")
    # Add another edge
    edge2 = EdgeNode.create(
        from_node=memory1.id,
        to_node=entity2.id,
        edge_type="MENTIONS",
        weight=1.0,
        confidence=0.95,
        evidence="PostgreSQL is the system that had the bug",
    )
    storage.store_edge(edge2)

    outgoing = storage.get_edges_from_memory(memory1.id)
    assert len(outgoing) == 2
    print(f"✓ Retrieved {len(outgoing)} outgoing edges\n")

    # Test 6: Get edges to entity
    print("Test 6: Get incoming edges to entity")
    incoming = storage.get_edges_to_entity(entity1.id)
    assert len(incoming) == 1
    assert incoming[0].from_node == memory1.id
    print(f"✓ Retrieved {len(incoming)} incoming edges to entity\n")

    # Test 7: Confidence filtering
    print("Test 7: Confidence filtering")
    # Add low-confidence edge
    edge3 = EdgeNode.create(
        from_node=memory1.id,
        to_node=entity1.id,
        edge_type="RELATED",  # Different type to avoid duplicate detection
        weight=0.5,
        confidence=0.60,  # Below 0.75 threshold
    )
    storage.store_edge(edge3)

    # Get all edges
    all_edges = storage.get_edges_from_memory(memory1.id, min_confidence=0.0)
    assert len(all_edges) == 3

    # Filter by confidence >= 0.75
    high_conf = storage.get_edges_from_memory(memory1.id, min_confidence=0.75)
    assert len(high_conf) == 2  # Only edge1 and edge2
    print(f"✓ Confidence filtering works: {len(all_edges)} total, {len(high_conf)} above 0.75\n")

    # Test 8: Evidence truncation
    print("Test 8: Evidence truncation (max 200 chars)")
    long_evidence = "x" * 250
    edge4 = EdgeNode.create(
        from_node=memory1.id,
        to_node=entity2.id,
        edge_type="RELATED",
        evidence=long_evidence,
    )
    assert len(edge4.evidence) == 200  # Truncated to 200
    assert edge4.evidence.endswith("...")
    print(f"✓ Evidence truncated from 250 to {len(edge4.evidence)} chars\n")

    # Test 9: Duplicate edge detection
    print("Test 9: Duplicate edge detection")
    edge5 = EdgeNode.create(
        from_node=memory1.id,
        to_node=entity1.id,
        edge_type="MENTIONS",  # Same as edge1
        confidence=0.80,  # Different confidence
    )
    edge5_id = storage.store_edge(edge5)
    assert edge5_id == edge1.edge_id  # Returns existing edge ID
    print(f"✓ Duplicate edge detected: returned existing ID {edge1.edge_id}\n")

    # Test 10: Expire edge
    print("Test 10: Expire edge (soft delete)")
    storage.expire_edge(edge1.edge_id)
    expired = storage.get_edge(edge1.edge_id)
    assert expired.t_expired is not None
    print(f"✓ Edge expired at {expired.t_expired}\n")

    # Test 11: Exclude expired from queries
    print("Test 11: Exclude expired edges from queries")
    active_edges = storage.get_edges_from_memory(memory1.id, include_expired=False)
    expired_count = len(storage.get_edges_from_memory(memory1.id, include_expired=True))
    assert expired_count > len(active_edges)
    print(f"✓ Expired edges excluded: {len(active_edges)} active, {expired_count} total\n")

    # Test 12: Bi-temporal fields initialized
    print("Test 12: Bi-temporal fields initialized correctly")
    edge6 = EdgeNode.create(
        from_node=memory1.id,
        to_node=entity2.id,
        edge_type="RELATED",
    )
    assert edge6.t_valid is not None
    assert edge6.t_created is not None
    assert edge6.t_invalid is None
    assert edge6.t_expired is None
    print(f"✓ Bi-temporal fields: t_valid={edge6.t_valid[:19]}, t_created={edge6.t_created[:19]}\n")

    # Cleanup
    storage.close()
    os.remove(test_db)

    print("=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
    print("\nWork Item #2 (Edge Schema & Storage) complete!")


if __name__ == "__main__":
    test_edge_schema_and_storage()
