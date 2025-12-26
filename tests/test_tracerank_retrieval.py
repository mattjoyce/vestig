#!/usr/bin/env python3
"""Test TraceRank integration with retrieval system"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vestig.core.storage import MemoryStorage
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.retrieval import search_memories
from vestig.core.commitment import commit_memory
from vestig.core.tracerank import TraceRankConfig
from vestig.core.config import load_config


def test_tracerank_retrieval():
    """Test that TraceRank affects retrieval ranking"""
    print("=== TraceRank Retrieval Integration Test ===\n")

    # Use temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"Using temp database: {db_path}\n")

    # Load config and setup
    config = load_config("config.yaml")
    storage = MemoryStorage(db_path)
    event_storage = MemoryEventStorage(storage.conn)
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    # TraceRank config
    tracerank_config = TraceRankConfig(
        enabled=True,
        tau_days=21.0,
        cooldown_hours=24.0,
        burst_discount=0.2,
        k=0.35,
    )

    print("Step 1: Add two similar memories about Python")

    # Memory A: Unreinforced
    outcome_a = commit_memory(
        content="Python is a high-level programming language with dynamic typing",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        tags=["python", "intro"],
        event_storage=event_storage,
    )
    memory_a_id = outcome_a.memory_id
    print(f"  Memory A (unreinforced): {memory_a_id}")

    # Memory B: Reinforced (add same content twice)
    outcome_b1 = commit_memory(
        content="Python supports multiple programming paradigms including OOP",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        tags=["python", "paradigms"],
        event_storage=event_storage,
    )
    memory_b_id = outcome_b1.memory_id
    print(f"  Memory B (will be reinforced): {memory_b_id}")

    # Reinforce Memory B
    outcome_b2 = commit_memory(
        content="Python supports multiple programming paradigms including OOP",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        event_storage=event_storage,
    )
    assert outcome_b2.memory_id == memory_b_id, "Should return same ID for duplicate"
    print(f"    ✓ Memory B reinforced (REINFORCE_EXACT event created)")

    # Verify reinforcement
    memory_b = storage.get_memory(memory_b_id)
    print(f"    ✓ Memory B reinforce_count: {memory_b.reinforce_count}")
    print()

    print("Step 2: Search WITHOUT TraceRank (baseline)")
    results_no_tr = search_memories(
        query="Python programming language features",
        storage=storage,
        embedding_engine=embedding_engine,
        limit=5,
        event_storage=None,  # Disable TraceRank
        tracerank_config=None,
    )

    print("  Results (semantic similarity only):")
    for i, (memory, score) in enumerate(results_no_tr, 1):
        is_b = "B (reinforced)" if memory.id == memory_b_id else "A (unreinforced)"
        print(f"    {i}. Memory {is_b}: score={score:.4f}")
    print()

    print("Step 3: Search WITH TraceRank enabled")
    results_with_tr = search_memories(
        query="Python programming language features",
        storage=storage,
        embedding_engine=embedding_engine,
        limit=5,
        event_storage=event_storage,
        tracerank_config=tracerank_config,
    )

    print("  Results (semantic × TraceRank):")
    for i, (memory, score) in enumerate(results_with_tr, 1):
        is_b = "B (reinforced)" if memory.id == memory_b_id else "A (unreinforced)"
        print(f"    {i}. Memory {is_b}: score={score:.4f}")
    print()

    # Get event count for Memory B
    events_b = event_storage.get_reinforcement_events(memory_b_id)
    print(f"  Memory B has {len(events_b)} reinforcement event(s)")
    print()

    print("Step 4: Verify TraceRank boosted Memory B")

    # Find Memory B in both result sets
    score_no_tr_b = next(score for mem, score in results_no_tr if mem.id == memory_b_id)
    score_with_tr_b = next(score for mem, score in results_with_tr if mem.id == memory_b_id)

    boost = score_with_tr_b / score_no_tr_b
    print(f"  Memory B score WITHOUT TraceRank: {score_no_tr_b:.4f}")
    print(f"  Memory B score WITH TraceRank:    {score_with_tr_b:.4f}")
    print(f"  Boost factor: {boost:.4f}x")
    print()

    if boost > 1.0:
        print("  ✓ TraceRank boosted reinforced memory!")
    else:
        print("  ✗ TraceRank did not boost (unexpected)")

    print()

    # Cleanup
    storage.close()
    Path(db_path).unlink()

    print("=== TraceRank Retrieval Integration Test Complete ===")


if __name__ == "__main__":
    test_tracerank_retrieval()
