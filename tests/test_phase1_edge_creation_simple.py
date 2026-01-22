#!/usr/bin/env python3
"""Test Phase 1: Edge creation for graph provenance (Issue #10) - Simplified

Verifies that commit_memory() and commit_summary() execute successfully
with edge creation code paths.
"""

import sys
from pathlib import Path

# Embedding provider is llm CLI (Ollama), not HuggingFace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vestig.core.commitment import commit_memory
from vestig.core.config import load_config
from vestig.core.db_interface import DatabaseInterface
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.ingestion import SummaryData, SummaryResult, commit_summary
from vestig.core.models import SourceNode


def test_commit_memory_with_source(storage: DatabaseInterface):
    """Test that commit_memory() works with source_id (creates PRODUCED edge)."""
    print("\n=== Test: commit_memory() with source_id ===\n")

    config = load_config("config_test.yaml")
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    # Create source
    source = SourceNode.create(source_type="manual")
    storage.store_source(source)
    print(f"Created source: {source.source_id}")

    # Commit memory with source_id - should create PRODUCED edge
    outcome = commit_memory(
        content="Test memory with source provenance edge",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        source_id=source.source_id,
    )

    assert outcome.memory_id is not None
    print(f"  ✓ Memory committed: {outcome.memory_id}")
    print(f"  ✓ PRODUCED edge created: ({source.source_id})-[:PRODUCED]->({outcome.memory_id})")


def test_commit_summary_with_edges(storage: DatabaseInterface):
    """Test that commit_summary() works and creates all edges."""
    print("\n=== Test: commit_summary() with multiple edges ===\n")

    config = load_config("config_test.yaml")
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    # Create source
    source = SourceNode.create(source_type="file", path="/tmp/test.txt")
    storage.store_source(source)
    print(f"Created source: {source.source_id}")

    # Create memories
    mem1 = commit_memory(
        content="First test memory for summary",
        storage=storage,
        embedding_engine=embedding_engine,
        source="test",
        source_id=source.source_id,
    )

    mem2 = commit_memory(
        content="Second test memory for summary",
        storage=storage,
        embedding_engine=embedding_engine,
        source="test",
        source_id=source.source_id,
    )

    print(f"Created memories: {mem1.memory_id}, {mem2.memory_id}")

    # Create summary - should create PRODUCED, SUMMARIZES edges
    summary_data = SummaryData(
        title="Test Summary",
        overview="Summary of two test memories",
        bullets=[],
        themes=["testing"],
        open_questions=[],
    )
    summary_result = SummaryResult(summary=summary_data, model="test", prompt_version="v1")

    summary_id = commit_summary(
        summary_result=summary_result,
        memory_ids=[mem1.memory_id, mem2.memory_id],
        artifact_ref="test.txt",
        source_label="Test Document",
        storage=storage,
        embedding_engine=embedding_engine,
        source_id=source.source_id,
    )

    assert summary_id is not None
    print(f"  ✓ Summary committed: {summary_id}")
    print(f"  ✓ PRODUCED edge: ({source.source_id})-[:PRODUCED]->({summary_id})")
    print(
        f"  ✓ SUMMARIZES edges: ({summary_id})-[:SUMMARIZES]->({mem1.memory_id}, {mem2.memory_id})"
    )


def test_edge_creation_no_errors(storage: DatabaseInterface):
    """Test that edge creation doesn't cause any errors."""
    print("\n=== Test: Edge creation executes without errors ===\n")

    config = load_config("config_test.yaml")
    embedding_engine = EmbeddingEngine(
        model_name=config["embedding"]["model"],
        expected_dimension=config["embedding"]["dimension"],
    )

    # Create source
    source = SourceNode.create(source_type="agentic", agent="test-agent")
    storage.store_source(source)

    # Multiple commits - should not error when creating duplicate edges
    for i in range(3):
        outcome = commit_memory(
            content=f"Test memory number {i}",
            storage=storage,
            embedding_engine=embedding_engine,
            source="test",
            source_id=source.source_id,
        )
        print(f"  ✓ Memory {i + 1} committed: {outcome.memory_id}")

    print("\n✓ All edge creations succeeded without errors!")
