#!/usr/bin/env python3
"""
Demo: CommitOutcome hook for M3 event logging bridge

This demonstrates how the M2 CommitOutcome hook can be used to capture
reinforcement events for M3's TraceRank system.
"""

from vestig.core.commitment import commit_memory
from vestig.core.config import load_config
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.events import CommitOutcome
from vestig.core.storage import MemoryStorage


def event_logger(outcome: CommitOutcome) -> None:
    """
    Example M3 event logger hook.

    In M3, this would write to a memory_events table for TraceRank.
    For now, just print structured data.
    """
    print(f"\n📊 Event captured:")
    print(f"  Outcome: {outcome.outcome}")
    print(f"  Memory ID: {outcome.memory_id}")
    print(f"  Occurred: {outcome.occurred_at}")
    print(f"  Source: {outcome.source}")
    print(f"  Content hash: {outcome.content_hash[:16]}...")

    if outcome.outcome == "EXACT_DUPE":
        print(f"  ✓ Exact duplicate of: {outcome.matched_memory_id}")

    elif outcome.outcome == "NEAR_DUPE":
        print(f"  ✓ Near-duplicate of: {outcome.matched_memory_id}")
        print(f"  ✓ Similarity score: {outcome.query_score:.4f}")
        print(f"  ✓ Threshold used: {outcome.thresholds.get('near_duplicate_threshold', 'N/A')}")

    elif outcome.outcome == "REJECTED_HYGIENE":
        print(f"  ✗ Rejected: {', '.join(outcome.hygiene_reasons)}")

    print()


def main():
    print("=== CommitOutcome Hook Demo (M2→M3 Bridge) ===\n")

    # Load config and build runtime
    config = load_config("config.yaml")
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
        normalize=config["embedding"]["normalize"],
    )
    storage = MemoryStorage(config["storage"]["db_path"])

    try:
        print("Scenario 1: Insert new memory")
        print("-" * 50)
        outcome1 = commit_memory(
            content="Learned how to use Python dataclasses for structured data",
            storage=storage,
            embedding_engine=embedding_engine,
            source="manual",
            tags=["python", "learning"],
            on_commit=event_logger,  # <-- M3 bridge hook
        )

        print("\nScenario 2: Add exact duplicate")
        print("-" * 50)
        outcome2 = commit_memory(
            content="Learned how to use Python dataclasses for structured data",  # Same
            storage=storage,
            embedding_engine=embedding_engine,
            source="hook",
            on_commit=event_logger,
        )

        print("\nScenario 3: Add near-duplicate")
        print("-" * 50)
        outcome3 = commit_memory(
            content="Learned to use dataclasses in Python for structured information",  # Similar
            storage=storage,
            embedding_engine=embedding_engine,
            source="import",
            on_commit=event_logger,
        )

        print("\nScenario 4: Hygiene rejection (too short)")
        print("-" * 50)
        try:
            outcome4 = commit_memory(
                content="ok",
                storage=storage,
                embedding_engine=embedding_engine,
                source="manual",
                on_commit=event_logger,
            )
        except ValueError as e:
            print(f"✓ Correctly rejected: {e}")

        print("\n" + "=" * 50)
        print("\n📝 Summary:")
        print("  The on_commit hook receives structured CommitOutcome data")
        print("  that M3 can use to:")
        print("    - Log reinforcement events (exact/near dupes)")
        print("    - Track temporal patterns (occurred_at, source)")
        print("    - Compute TraceRank (frequency, spacing, recency)")
        print("    - Implement provenance (artifact_ref, tags)")
        print("\n  All without polluting M2's retrieval contract!\n")

    finally:
        storage.close()


if __name__ == "__main__":
    main()
