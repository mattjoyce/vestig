#!/usr/bin/env python3
"""Test native vector search implementation (Issue #7)"""

import sys
import uuid
from pathlib import Path

# Embedding provider is llm CLI (Ollama), not HuggingFace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vestig.core.commitment import commit_memory
from vestig.core.config import load_config
from vestig.core.db_interface import DatabaseInterface
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.models import MemoryNode


def test_native_vector_search_memories(storage: DatabaseInterface):
    """Test that native vector search works end-to-end for memories."""
    print("\n=== Test: Native Vector Search for Memories ===\n")

    # Setup
    config = load_config("config_test.yaml")
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    # Add test memories
    print("Step 1: Adding test memories...")
    memories = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Rust provides memory safety",
    ]

    memory_ids = []
    for content in memories:
        outcome = commit_memory(
            content=content,
            storage=storage,
            embedding_engine=embedding_engine,
            source="test",
        )
        memory_ids.append(outcome.memory_id)
        print(f"  Added: {content[:50]}...")

    print("\nStep 2: Testing native vector search...")
    query = "programming languages"
    query_embedding = embedding_engine.embed_text(query)

    # Test the native vector search method directly
    results = storage.search_memories_by_vector(
        query_vector=query_embedding,
        limit=5,
        kind_filter=None,
        include_expired=False,
    )

    print(f"  Query: '{query}'")
    print(f"  Found {len(results)} results:\n")

    assert len(results) > 0, "Native vector search returned no results"

    for i, (memory, score) in enumerate(results, 1):
        print(f"  {i}. Score: {score:.4f} | {memory.content}")

    # Verify scores are valid
    for memory, score in results:
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0,1]"

    # Verify results are sorted by score
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    print("\n✓ Native vector search working correctly!")


def test_native_vector_search_entities(storage: DatabaseInterface):
    """Test that native vector search works for entities."""
    print("\n=== Test: Native Vector Search for Entities ===\n")

    # Setup
    config = load_config("config_test.yaml")
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    from vestig.core.models import EntityNode

    # Add test entities
    print("Step 1: Adding test entities...")
    entities = [
        ("Python Software Foundation", "ORG"),
        ("Mozilla Foundation", "ORG"),
        ("Rust Foundation", "ORG"),
    ]

    for name, entity_type in entities:
        entity = EntityNode.create(
            entity_type=entity_type,
            canonical_name=name,
        )
        # Generate embedding
        entity.embedding = embedding_engine.embed_text(name.lower())
        storage.store_entity(entity)
        print(f"  Added: {name} ({entity_type})")

    print("\nStep 2: Testing entity vector search...")
    query = "software foundation"
    query_embedding = embedding_engine.embed_text(query)

    # Test entity vector search
    results = storage.search_entities_by_vector(
        query_vector=query_embedding,
        entity_type="ORG",
        limit=5,
        include_expired=False,
    )

    print(f"  Query: '{query}'")
    print(f"  Found {len(results)} results:\n")

    assert len(results) > 0, "Entity vector search returned no results"

    for i, (entity, score) in enumerate(results, 1):
        print(f"  {i}. Score: {score:.4f} | {entity.canonical_name} ({entity.entity_type})")

    # Verify scores are valid
    for entity, score in results:
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0,1]"

    print("\n✓ Entity vector search working correctly!")


def test_vector_search_with_filters(storage: DatabaseInterface):
    """Test vector search with kind filters."""
    print("\n=== Test: Vector Search with Filters ===\n")

    config = load_config("config_test.yaml")
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    # Add memories with different kinds
    print("Step 1: Adding MEMORY and SUMMARY...")

    # Regular memory
    commit_memory(
        content="This is a regular memory about testing",
        storage=storage,
        embedding_engine=embedding_engine,
        source="test",
    )

    # Create a summary manually (SUMMARY kind)

    summary_content = "This is a summary about testing frameworks"
    summary_embedding = embedding_engine.embed_text(summary_content)
    summary = MemoryNode.create(
        memory_id=f"mem_{uuid.uuid4().hex[:16]}",
        content=summary_content,
        embedding=summary_embedding,
        source="test",
    )
    storage.store_memory(summary, kind="SUMMARY")

    print("\nStep 2: Test filtering by kind...")
    query_embedding = embedding_engine.embed_text("testing")

    # Search for SUMMARY only
    summary_results = storage.search_memories_by_vector(
        query_vector=query_embedding,
        limit=10,
        kind_filter="SUMMARY",
        include_expired=False,
    )

    print(f"  SUMMARY results: {len(summary_results)}")
    for memory, score in summary_results:
        assert memory.kind == "SUMMARY", f"Expected SUMMARY, got {memory.kind}"
        print(f"    ✓ {memory.content[:50]}... (kind={memory.kind})")

    # Search for MEMORY only
    memory_results = storage.search_memories_by_vector(
        query_vector=query_embedding,
        limit=10,
        kind_filter="MEMORY",
        include_expired=False,
    )

    print(f"  MEMORY results: {len(memory_results)}")
    for memory, score in memory_results:
        assert memory.kind == "MEMORY", f"Expected MEMORY, got {memory.kind}"
        print(f"    ✓ {memory.content[:50]}... (kind={memory.kind})")

    print("\n✓ Kind filtering working correctly!")
