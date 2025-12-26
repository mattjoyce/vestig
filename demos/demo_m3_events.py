#!/usr/bin/env python3
"""Demo M3 event logging in action"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vestig.core.config import load_config
from vestig.core.storage import MemoryStorage
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.commitment import commit_memory


def demo_m3_events():
    """Demonstrate M3 event logging"""
    print("=== M3 Event Logging Demo ===\n")

    # Setup
    config = load_config("config.yaml")
    storage = MemoryStorage("data/memory_m3_demo.db")
    event_storage = MemoryEventStorage(storage.conn)
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    print("Step 1: Adding a new memory\n")
    outcome1 = commit_memory(
        content="Python async/await enables concurrent programming without threads",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        tags=["python", "async"],
        event_storage=event_storage,
    )

    print(f"Result: {outcome1.outcome}")
    print(f"Memory ID: {outcome1.memory_id}")
    print(f"Occurred at: {outcome1.occurred_at}\n")

    # Show the event
    events = event_storage.get_events_for_memory(outcome1.memory_id)
    print(f"Events logged: {len(events)}")
    for event in events:
        print(f"  [{event.event_id}] {event.event_type}")
        print(f"    Occurred: {event.occurred_at}")
        print(f"    Source: {event.source}")
        print()

    print("-" * 60)
    print("\nStep 2: Adding exact duplicate (should reinforce)\n")

    outcome2 = commit_memory(
        content="Python async/await enables concurrent programming without threads",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        event_storage=event_storage,
    )

    print(f"Result: {outcome2.outcome}")
    print(f"Memory ID: {outcome2.memory_id} (same as before)")
    print()

    # Show updated events
    events = event_storage.get_events_for_memory(outcome2.memory_id)
    print(f"Events logged: {len(events)}")
    for event in events:
        print(f"  [{event.event_id}] {event.event_type}")
        print(f"    Occurred: {event.occurred_at}")
        print()

    # Show reinforcement tracking
    memory = storage.get_memory(outcome2.memory_id)
    print(f"Reinforcement tracking:")
    print(f"  reinforce_count: {memory.reinforce_count}")
    print(f"  last_seen_at: {memory.last_seen_at}")
    print()

    print("-" * 60)
    print("\nStep 3: Adding similar content (may trigger near-duplicate)\n")

    outcome3 = commit_memory(
        content="Python's async/await syntax allows for concurrent execution without using threads",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        event_storage=event_storage,
    )

    print(f"Result: {outcome3.outcome}")
    print(f"Memory ID: {outcome3.memory_id}")
    print()

    if outcome3.outcome == "NEAR_DUPE":
        print("Near-duplicate detected!")
        events = event_storage.get_events_for_memory(outcome3.memory_id)
        print(f"Total events for original memory: {len(events)}")

        memory = storage.get_memory(outcome3.memory_id)
        print(f"Updated reinforcement:")
        print(f"  reinforce_count: {memory.reinforce_count}")
        print(f"  last_seen_at: {memory.last_seen_at}")
    else:
        print("New memory created (not similar enough for near-duplicate)")
        events = event_storage.get_events_for_memory(outcome3.memory_id)
        print(f"Events for new memory: {len(events)}")
        for event in events:
            print(f"  [{event.event_id}] {event.event_type}")

    print()
    print("-" * 60)
    print("\n=== Summary ===")
    print(f"\nTotal memories: {len(storage.get_all_memories())}")
    print("\nAll events across all memories:")

    for mem in storage.get_all_memories():
        events = event_storage.get_events_for_memory(mem.id)
        if events:
            print(f"\n  Memory: {mem.id}")
            print(f"  Content: {mem.content[:60]}...")
            print(f"  Events: {len(events)}")
            for event in events:
                print(f"    - {event.event_type} at {event.occurred_at}")

    storage.close()
    print("\n✓ Demo complete! Database saved to: data/memory_m3_demo.db")
    print("  (You can inspect it or delete it)")


if __name__ == "__main__":
    demo_m3_events()
