#!/usr/bin/env python3
"""Performance benchmark for vestig memory ingestion"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vestig.core.commitment import commit_memory
from vestig.core.config import load_config
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.storage import MemoryStorage


def benchmark_memory_commit(
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    m4_config: dict,
    test_memories: list[str],
    scenario_name: str,
    hygiene_config: dict | None = None,
) -> dict:
    """
    Benchmark memory commit performance.

    Args:
        storage: Storage instance
        embedding_engine: Embedding engine
        m4_config: M4 config (empty dict to disable)
        test_memories: List of test memory strings
        scenario_name: Name for this benchmark scenario

    Returns:
        Dict with timing results
    """
    times = []
    entity_counts = []

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")

    for i, content in enumerate(test_memories, 1):
        start = time.time()

        try:
            outcome = commit_memory(
                content=content,
                storage=storage,
                embedding_engine=embedding_engine,
                source="benchmark",
                hygiene_config=hygiene_config,
                m4_config=m4_config,
            )
            elapsed = time.time() - start
            times.append(elapsed)

            # Count entities if M4 enabled
            entity_count = 0
            if m4_config:
                edges = storage.get_edges_from_memory(
                    outcome.memory_id, edge_type="MENTIONS", include_expired=False
                )
                entity_count = len(edges)
            entity_counts.append(entity_count)

            print(f"  Memory {i}: {elapsed:.3f}s (entities: {entity_count})")

        except Exception as e:
            print(f"  Memory {i}: ERROR - {e}")
            times.append(0)
            entity_counts.append(0)

    avg_time = sum(times) / len(times) if times else 0
    total_time = sum(times)
    avg_entities = sum(entity_counts) / len(entity_counts) if entity_counts else 0

    print(f"\nResults:")
    print(f"  Average time:     {avg_time:.3f}s")
    print(f"  Total time:       {total_time:.3f}s")
    print(f"  Avg entities:     {avg_entities:.1f}")
    print(f"  Min time:         {min(times):.3f}s")
    print(f"  Max time:         {max(times):.3f}s")

    return {
        "scenario": scenario_name,
        "avg_time": avg_time,
        "total_time": total_time,
        "avg_entities": avg_entities,
        "times": times,
    }


def main():
    """Run performance benchmarks"""
    print("VESTIG PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Clean up any existing benchmark DB
    db_path = "benchmark_test.db"
    if os.path.exists(db_path):
        os.unlink(db_path)
        print(f"Cleaned up existing {db_path}")

    # Load test memories from file
    test_memories_file = Path("benchmark_test_memories.txt")
    if not test_memories_file.exists():
        print(f"Error: {test_memories_file} not found", file=sys.stderr)
        sys.exit(1)

    test_memories = [line.strip() for line in test_memories_file.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(test_memories)} test memories from {test_memories_file}")

    # Load config
    config = load_config("config.yaml")

    # Initialize embedding engine once (reuse across scenarios)
    print("Loading embedding model...")
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    # Disable near-duplicate to ensure all memories are inserted
    hygiene_no_neardup = {
        "near_duplicate": {"enabled": False},
        "reject_exact_duplicates": False,  # Allow exact dupes for benchmark
    }

    results = []

    # Scenario 1: No M4 (baseline)
    print("\n" + "=" * 70)
    print("SCENARIO 1: No M4 (Embedding + Storage only)")
    storage = MemoryStorage(db_path)
    result = benchmark_memory_commit(
        storage, embedding_engine, {}, test_memories, "No M4 (baseline)", hygiene_no_neardup
    )
    results.append(result)
    storage.close()

    # Clean DB between scenarios
    if os.path.exists(db_path):
        os.unlink(db_path)

    # Scenario 2: M4 with Sonnet 4.5
    print("\n" + "=" * 70)
    print("SCENARIO 2: M4 with claude-sonnet-4.5")
    storage = MemoryStorage(db_path)
    m4_config_sonnet = config.get("m4", {}).copy()
    if m4_config_sonnet:
        m4_config_sonnet["entity_extraction"]["llm"]["model"] = "claude-sonnet-4.5"
    result = benchmark_memory_commit(
        storage,
        embedding_engine,
        m4_config_sonnet,
        test_memories,
        "M4 (claude-sonnet-4.5)",
        hygiene_no_neardup,
    )
    results.append(result)
    storage.close()

    # Clean DB between scenarios
    if os.path.exists(db_path):
        os.unlink(db_path)

    # Scenario 3: M4 with Haiku 4.5
    print("\n" + "=" * 70)
    print("SCENARIO 3: M4 with claude-haiku-4.5")
    storage = MemoryStorage(db_path)
    m4_config_haiku = config.get("m4", {}).copy()
    if m4_config_haiku:
        m4_config_haiku["entity_extraction"]["llm"]["model"] = "claude-haiku-4.5"
    result = benchmark_memory_commit(
        storage,
        embedding_engine,
        m4_config_haiku,
        test_memories,
        "M4 (claude-haiku-4.5)",
        hygiene_no_neardup,
    )
    results.append(result)
    storage.close()

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(
        f"{'Scenario':<30} {'Avg Time':>12} {'Total Time':>12} {'Avg Entities':>15}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['scenario']:<30} {r['avg_time']:>11.3f}s {r['total_time']:>11.3f}s {r['avg_entities']:>15.1f}"
        )

    # Speedup analysis
    if len(results) >= 3:
        baseline = results[0]["avg_time"]
        sonnet = results[1]["avg_time"]
        haiku = results[2]["avg_time"]

        print(f"\nSpeedup vs baseline:")
        print(f"  Sonnet overhead: {sonnet/baseline:.1f}x slower")
        print(f"  Haiku overhead:  {haiku/baseline:.1f}x slower")
        print(f"  Haiku vs Sonnet: {sonnet/haiku:.1f}x faster")

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
        print(f"\nCleaned up {db_path}")


if __name__ == "__main__":
    main()
