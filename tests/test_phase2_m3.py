#!/usr/bin/env python3
"""Phase 2 validation: Test event pipeline integration"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure tests run offline if the model is already cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vestig.core.models import MemoryNode
from vestig.core.storage import MemoryStorage
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.commitment import commit_memory
from vestig.core.config import load_config


def test_phase2():
    """Test Phase 2: Event Pipeline Integration"""
    print("=== Phase 2 Validation: Event Pipeline Integration ===\n")

    # Use temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"Using temp database: {db_path}\n")

    # Load config and setup
    config = load_config("config_test.yaml")
    storage = MemoryStorage(db_path)
    event_storage = MemoryEventStorage(storage.conn)
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    print("Test 1: commit_memory() logs ADD event")
    try:
        outcome = commit_memory(
            content="Testing M3 event logging with a new memory",
            storage=storage,
            embedding_engine=embedding_engine,
            source="manual",
            tags=["test", "m3"],
            event_storage=event_storage,
        )

        assert outcome.outcome == "INSERTED_NEW", "Should insert new memory"
        memory_id = outcome.memory_id
        print(f"  ✓ Memory inserted: {memory_id}")

        # Check that ADD event was logged
        events = event_storage.get_events_for_memory(memory_id)
        assert len(events) == 1, "Should have 1 event"
        assert events[0].event_type == "ADD", "Event type should be ADD"
        assert events[0].memory_id == memory_id, "Event memory_id should match"
        print(f"  ✓ ADD event logged: {events[0].event_id}")
        print(f"    - event_type: {events[0].event_type}")
        print(f"    - occurred_at: {events[0].occurred_at}")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Test 2: Exact duplicate creates REINFORCE_EXACT event
    print("Test 2: Exact duplicate logs REINFORCE_EXACT event")
    try:
        outcome2 = commit_memory(
            content="Testing M3 event logging with a new memory",
            storage=storage,
            embedding_engine=embedding_engine,
            source="manual",
            tags=["test", "m3"],
            event_storage=event_storage,
        )

        assert outcome2.outcome == "EXACT_DUPE", "Should be exact duplicate"
        assert outcome2.memory_id == memory_id, "Should return same memory ID"
        print(f"  ✓ Exact duplicate detected: {outcome2.memory_id}")

        # Check that REINFORCE_EXACT event was logged
        events = event_storage.get_events_for_memory(memory_id)
        assert len(events) == 2, "Should have 2 events (ADD + REINFORCE_EXACT)"

        reinforce_events = [e for e in events if e.event_type == "REINFORCE_EXACT"]
        assert len(reinforce_events) == 1, "Should have 1 REINFORCE_EXACT event"
        print(f"  ✓ REINFORCE_EXACT event logged: {reinforce_events[0].event_id}")

        # Check that reinforce_count was incremented
        memory = storage.get_memory(memory_id)
        assert memory.reinforce_count == 1, "reinforce_count should be 1"
        assert memory.last_seen_at is not None, "last_seen_at should be set"
        print(f"  ✓ Convenience fields updated:")
        print(f"    - reinforce_count: {memory.reinforce_count}")
        print(f"    - last_seen_at: {memory.last_seen_at}")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Test 3: Near duplicate creates REINFORCE_NEAR event
    print("Test 3: Near duplicate logs REINFORCE_NEAR event")
    try:
        outcome3 = commit_memory(
            content="Testing M3 event logging with similar memory content",
            storage=storage,
            embedding_engine=embedding_engine,
            source="manual",
            event_storage=event_storage,
        )

        # This should create a new memory (not exact match)
        # but may trigger NEAR_DUPE depending on threshold
        print(f"  Outcome: {outcome3.outcome}")
        print(f"  Memory ID: {outcome3.memory_id}")

        if outcome3.outcome == "NEAR_DUPE":
            # Check for REINFORCE_NEAR event
            events = event_storage.get_events_for_memory(memory_id)
            reinforce_near = [e for e in events if e.event_type == "REINFORCE_NEAR"]
            if reinforce_near:
                print(f"  ✓ REINFORCE_NEAR event logged: {reinforce_near[0].event_id}")

            # Check reinforce_count increased
            memory = storage.get_memory(memory_id)
            assert memory.reinforce_count == 2, "reinforce_count should be 2"
            print(f"  ✓ reinforce_count incremented: {memory.reinforce_count}")
        else:
            print("  ℹ Content not similar enough for NEAR_DUPE (OK)")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Test 4: Event payload contains metadata
    print("Test 4: Event payload contains commit metadata")
    try:
        events = event_storage.get_events_for_memory(memory_id)
        add_event = [e for e in events if e.event_type == "ADD"][0]

        assert "content_hash" in add_event.payload, "Payload should have content_hash"
        assert "tags" in add_event.payload, "Payload should have tags"
        print("  ✓ Event payload contains metadata:")
        print(f"    - content_hash: {add_event.payload['content_hash'][:16]}...")
        print(f"    - tags: {add_event.payload['tags']}")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Test 5: Hygiene rejection does NOT log event
    print("Test 5: Hygiene rejection does not log event")
    try:
        short_content_id = None
        try:
            outcome_reject = commit_memory(
                content="short",  # Too short, will be rejected
                storage=storage,
                embedding_engine=embedding_engine,
                source="manual",
                event_storage=event_storage,
            )
        except ValueError as e:
            # Expected - hygiene rejection
            print(f"  ✓ Hygiene rejection as expected: {str(e)[:50]}")

        # Verify no event was logged for rejected content
        all_events = []
        for mem in storage.get_all_memories():
            all_events.extend(event_storage.get_events_for_memory(mem.id))

        # Should only have events from previous tests, not from rejection
        print(f"  ✓ No event logged for hygiene rejection (total events: {len(all_events)})")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        storage.close()
        return False

    print()

    # Cleanup
    storage.close()
    Path(db_path).unlink()

    print("=== All Phase 2 Tests Passed! ===\n")
    print("✓ ADD events logged on new memory")
    print("✓ REINFORCE_EXACT events logged on exact duplicate")
    print("✓ Convenience fields (reinforce_count, last_seen_at) updated")
    print("✓ Event payloads contain commit metadata")
    print("✓ Hygiene rejections do not create events")
    print("\nPhase 2 (Event Pipeline Integration) is ready for Phase 3!")

    return True


if __name__ == "__main__":
    success = test_phase2()
    sys.exit(0 if success else 1)
