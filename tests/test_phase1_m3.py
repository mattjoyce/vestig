#!/usr/bin/env python3
"""Phase 1 validation: Test M3 schema foundation"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vestig.core.models import MemoryNode, EventNode
from vestig.core.storage import MemoryStorage
from vestig.core.event_storage import MemoryEventStorage


def test_phase1():
    """Test Phase 1: Schema Foundation"""
    print("=== Phase 1 Validation: M3 Schema Foundation ===\n")

    # Use temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"Using temp database: {db_path}\n")

    # Test 1: MemoryNode creation with M3 fields
    print("Test 1: MemoryNode creation with M3 temporal fields")
    try:
        node = MemoryNode.create(
            memory_id="mem_test_123",
            content="Testing M3 temporal fields",
            embedding=[0.1, 0.2, 0.3],
            source="manual",
            tags=["test", "m3"],
        )

        # Verify M3 fields are initialized
        assert node.t_created is not None, "t_created should be set"
        assert node.t_valid is not None, "t_valid should be set"
        assert node.t_created == node.t_valid == node.created_at, "Temporal fields should match"
        assert node.temporal_stability == "unknown", "Default stability should be 'unknown'"
        assert node.reinforce_count == 0, "Default reinforce_count should be 0"
        assert node.last_seen_at is None, "Default last_seen_at should be None"
        assert node.t_invalid is None, "Default t_invalid should be None"
        assert node.t_expired is None, "Default t_expired should be None"

        print("  ✓ MemoryNode created with all M3 fields initialized")
        print(f"    - t_created: {node.t_created}")
        print(f"    - temporal_stability: {node.temporal_stability}")
        print(f"    - reinforce_count: {node.reinforce_count}")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False

    print()

    # Test 2: Storage initialization (migrations)
    print("Test 2: Storage initialization (M3 migrations)")
    try:
        storage = MemoryStorage(db_path)

        # Check that memory_events table was created
        cursor = storage.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_events'"
        )
        result = cursor.fetchone()
        assert result is not None, "memory_events table should exist"

        print("  ✓ Storage initialized successfully")
        print("  ✓ memory_events table created")

        # Check that M3 columns were added to memories table
        cursor = storage.conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}

        m3_columns = {
            "t_valid", "t_invalid", "t_created", "t_expired",
            "temporal_stability", "last_seen_at", "reinforce_count"
        }

        missing = m3_columns - columns
        assert not missing, f"Missing M3 columns: {missing}"

        print(f"  ✓ All M3 columns present: {', '.join(sorted(m3_columns))}")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Test 3: Store and retrieve memory with M3 fields
    print("Test 3: Store and retrieve memory with M3 fields")
    try:
        stored_id = storage.store_memory(node)
        assert stored_id == node.id, "Stored ID should match node ID"
        print(f"  ✓ Memory stored: {stored_id}")

        # Retrieve memory
        retrieved = storage.get_memory(stored_id)
        assert retrieved is not None, "Memory should be retrieved"
        assert retrieved.id == node.id, "IDs should match"
        assert retrieved.t_created == node.t_created, "t_created should match"
        assert retrieved.temporal_stability == node.temporal_stability, "temporal_stability should match"
        assert retrieved.reinforce_count == node.reinforce_count, "reinforce_count should match"

        print("  ✓ Memory retrieved with all M3 fields intact")
        print(f"    - t_created: {retrieved.t_created}")
        print(f"    - temporal_stability: {retrieved.temporal_stability}")
        print(f"    - reinforce_count: {retrieved.reinforce_count}")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Test 4: M3 storage methods
    print("Test 4: M3 storage methods (increment_reinforce_count, etc.)")
    try:
        # Test increment_reinforce_count
        storage.increment_reinforce_count(stored_id)
        retrieved = storage.get_memory(stored_id)
        assert retrieved.reinforce_count == 1, "reinforce_count should be incremented"
        print("  ✓ increment_reinforce_count() works")

        # Test update_last_seen
        test_timestamp = "2025-12-26T10:00:00Z"
        storage.update_last_seen(stored_id, test_timestamp)
        retrieved = storage.get_memory(stored_id)
        assert retrieved.last_seen_at == test_timestamp, "last_seen_at should be updated"
        print("  ✓ update_last_seen() works")

        # Test get_active_memories (should include our memory)
        active = storage.get_active_memories()
        assert len(active) == 1, "Should have 1 active memory"
        assert active[0].id == stored_id, "Active memory should match"
        print("  ✓ get_active_memories() works")

        # Test deprecate_memory
        storage.deprecate_memory(stored_id, t_invalid="2025-12-26T12:00:00Z")
        retrieved = storage.get_memory(stored_id)
        assert retrieved.t_expired is not None, "t_expired should be set"
        assert retrieved.t_invalid == "2025-12-26T12:00:00Z", "t_invalid should be set"
        print("  ✓ deprecate_memory() works")

        # Test get_active_memories (should NOT include deprecated memory)
        active = storage.get_active_memories()
        assert len(active) == 0, "Should have 0 active memories after deprecation"
        print("  ✓ get_active_memories() filters deprecated memories")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Test 5: EventNode creation
    print("Test 5: EventNode creation")
    try:
        event = EventNode.create(
            memory_id=stored_id,
            event_type="ADD",
            source="manual",
            payload={"test": "data"}
        )

        assert event.event_id.startswith("evt_"), "Event ID should start with 'evt_'"
        assert event.memory_id == stored_id, "memory_id should match"
        assert event.event_type == "ADD", "event_type should be 'ADD'"
        assert event.occurred_at is not None, "occurred_at should be set"

        print("  ✓ EventNode created successfully")
        print(f"    - event_id: {event.event_id}")
        print(f"    - event_type: {event.event_type}")
        print(f"    - occurred_at: {event.occurred_at}")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Test 6: Event storage
    print("Test 6: Event storage (MemoryEventStorage)")
    try:
        event_storage = MemoryEventStorage(storage.conn)

        # Add event
        event_id = event_storage.add_event(event)
        assert event_id == event.event_id, "Event ID should match"
        print("  ✓ Event stored successfully")

        # Retrieve events
        events = event_storage.get_events_for_memory(stored_id)
        assert len(events) == 1, "Should have 1 event"
        assert events[0].event_id == event_id, "Event ID should match"
        assert events[0].event_type == "ADD", "Event type should be 'ADD'"
        print("  ✓ Events retrieved successfully")

        # Add a REINFORCE event
        reinforce_event = EventNode.create(
            memory_id=stored_id,
            event_type="REINFORCE_EXACT",
            source="manual"
        )
        event_storage.add_event(reinforce_event)

        # Get reinforcement events
        reinforce_events = event_storage.get_reinforcement_events(stored_id)
        assert len(reinforce_events) == 1, "Should have 1 REINFORCE event"
        assert reinforce_events[0].event_type == "REINFORCE_EXACT", "Should be REINFORCE_EXACT"
        print("  ✓ get_reinforcement_events() filters correctly")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Cleanup
    storage.close()
    Path(db_path).unlink()

    print("=== All Phase 1 Tests Passed! ===\n")
    print("✓ M3 data models working")
    print("✓ M3 schema migrations working")
    print("✓ M3 storage methods working")
    print("✓ M3 event storage working")
    print("\nPhase 1 (Schema Foundation) is ready for Phase 2!")

    return True


if __name__ == "__main__":
    success = test_phase1()
    sys.exit(0 if success else 1)
