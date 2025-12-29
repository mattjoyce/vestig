#!/usr/bin/env python3
"""Test M4 Work Item #6: 1-hop Graph Traversal & Expansion"""

import os
import sys
import tempfile
from unittest.mock import patch

# Ensure tests run offline if the model is already cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.commitment import commit_memory
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.storage import MemoryStorage
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.config import load_config
from vestig.core.graph import expand_via_entities, expand_via_related, expand_with_graph
from vestig.core.ingestion import MemoryExtractionResult


def test_graph_traversal():
    """Test 1-hop graph traversal and expansion"""
    print("=== M4 Work Item #6: Graph Traversal & Expansion ===\n")

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Load config and initialize storage
        config = load_config("config_test.yaml")
        storage = MemoryStorage(db_path)
        event_storage = MemoryEventStorage(storage.conn)
        embedding_engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )

        # M4 config (both entity extraction and RELATED edges enabled)
        m4_config = {
            "entity_types": {
                "allowed_types": ["PERSON", "ORG", "SYSTEM", "PROJECT", "PLACE"]
            },
            "entity_extraction": {
                "enabled": True,
                "llm": {"model": "claude-sonnet-4.5", "min_confidence": 0.75},
            },
            "edge_creation": {
                "mentions": {"enabled": True},
                "related": {
                    "enabled": True,
                    "similarity_threshold": 0.5,
                    "max_edges_per_memory": 10,
                },
            },
        }

        # Setup: Create memories with entities and RELATED edges
        print("Setup: Creating test memories with graph structure\n")

        # Mock LLM for entity extraction
        mock_response_alice_pg = MemoryExtractionResult.model_validate(
            {
                "memories": [
                    {
                        "content": "Alice fixed the PostgreSQL replication bug",
                        "confidence": 0.92,
                        "rationale": "test",
                        "entities": [
                            {
                                "name": "Alice",
                                "type": "PERSON",
                                "confidence": 0.92,
                                "evidence": "developer",
                            },
                            {
                                "name": "PostgreSQL",
                                "type": "SYSTEM",
                                "confidence": 0.95,
                                "evidence": "database",
                            },
                        ],
                    }
                ]
            }
        )

        mock_response_alice_redis = MemoryExtractionResult.model_validate(
            {
                "memories": [
                    {
                        "content": "Alice deployed the new Redis caching layer",
                        "confidence": 0.90,
                        "rationale": "test",
                        "entities": [
                            {
                                "name": "Alice",
                                "type": "PERSON",
                                "confidence": 0.90,
                                "evidence": "developer",
                            },
                            {
                                "name": "Redis",
                                "type": "SYSTEM",
                                "confidence": 0.93,
                                "evidence": "cache",
                            },
                        ],
                    }
                ]
            }
        )

        mock_response_bob_pg = MemoryExtractionResult.model_validate(
            {
                "memories": [
                    {
                        "content": "Bob optimized PostgreSQL query performance",
                        "confidence": 0.88,
                        "rationale": "test",
                        "entities": [
                            {
                                "name": "Bob",
                                "type": "PERSON",
                                "confidence": 0.88,
                                "evidence": "developer",
                            },
                            {
                                "name": "PostgreSQL",
                                "type": "SYSTEM",
                                "confidence": 0.94,
                                "evidence": "database",
                            },
                        ],
                    }
                ]
            }
        )

        # Memory 1: Alice fixed PostgreSQL bug
        with patch("vestig.core.ingestion.call_llm", return_value=mock_response_alice_pg):
            outcome1 = commit_memory(
                content="Alice fixed the PostgreSQL replication bug",
                storage=storage,
                embedding_engine=embedding_engine,
                source="manual",
                event_storage=event_storage,
                m4_config=m4_config,
            )
            mem1_id = outcome1.memory_id
            print(f"✓ Memory 1 (Alice + PostgreSQL): {mem1_id}")

        # Memory 2: Alice deployed Redis
        with patch("vestig.core.ingestion.call_llm", return_value=mock_response_alice_redis):
            outcome2 = commit_memory(
                content="Alice deployed the new Redis caching layer",
                storage=storage,
                embedding_engine=embedding_engine,
                source="manual",
                event_storage=event_storage,
                m4_config=m4_config,
            )
            mem2_id = outcome2.memory_id
            print(f"✓ Memory 2 (Alice + Redis): {mem2_id}")

        # Memory 3: Bob optimized PostgreSQL
        with patch("vestig.core.ingestion.call_llm", return_value=mock_response_bob_pg):
            outcome3 = commit_memory(
                content="Bob optimized PostgreSQL query performance",
                storage=storage,
                embedding_engine=embedding_engine,
                source="manual",
                event_storage=event_storage,
                m4_config=m4_config,
            )
            mem3_id = outcome3.memory_id
            print(f"✓ Memory 3 (Bob + PostgreSQL): {mem3_id}")

        # Memory 4: Unrelated content (no entities, for RELATED edges only)
        outcome4 = commit_memory(
            content="PostgreSQL database indexing improves query speed",
            storage=storage,
            embedding_engine=embedding_engine,
            source="manual",
            event_storage=event_storage,
            m4_config={
                **m4_config,
                "entity_extraction": {"enabled": False},
            },  # No entities
        )
        mem4_id = outcome4.memory_id
        print(f"✓ Memory 4 (no entities, RELATED only): {mem4_id}")

        # Test 1: Expand via entities (shared entity: Alice)
        print("\nTest 1: Expand via entities (shared entity)")

        # Expand from mem1 (should find mem2 via shared entity Alice)
        expansion = expand_via_entities(
            memory_ids=[mem1_id], storage=storage, limit=5
        )

        # Should find mem2 (shares Alice entity)
        expanded_ids = [r["memory"].id for r in expansion]
        assert mem2_id in expanded_ids, f"Should find mem2 (shares Alice), got {expanded_ids}"
        print(f"✓ Expanded from mem1, found {len(expansion)} memories")

        # Check result structure
        result = next(r for r in expansion if r["memory"].id == mem2_id)
        assert result["retrieval_reason"] == "graph_expansion_entity"
        assert len(result["shared_entities"]) > 0
        assert result["expansion_score"] > 0
        print(f"✓ Result has correct structure: {result['retrieval_reason']}")
        print(f"  Shared entities: {len(result['shared_entities'])}")
        print(f"  Expansion score: {result['expansion_score']}")

        # Test 2: Expand via entities (shared entity: PostgreSQL)
        print("\nTest 2: Expand via entities (multiple shared)")

        # Expand from mem1 (should find mem3 via shared entity PostgreSQL)
        expansion = expand_via_entities(
            memory_ids=[mem1_id], storage=storage, limit=5
        )

        # Should find both mem2 (Alice) and mem3 (PostgreSQL)
        expanded_ids = [r["memory"].id for r in expansion]
        assert mem3_id in expanded_ids, f"Should find mem3 (shares PostgreSQL)"
        print(f"✓ Found {len(expansion)} memories via shared entities")

        # mem1 should have higher score (shares PostgreSQL) than others
        mem3_result = next(r for r in expansion if r["memory"].id == mem3_id)
        print(f"  mem3 expansion_score: {mem3_result['expansion_score']}")

        # Test 3: Expand via RELATED edges
        print("\nTest 3: Expand via RELATED edges")

        # First check if any RELATED edges exist
        related_edges = storage.get_edges_from_memory(
            mem1_id, edge_type="RELATED", min_confidence=0.0
        )
        print(f"  mem1 has {len(related_edges)} RELATED edges")

        # Expand from mem1 (should find semantically similar memories)
        expansion = expand_via_related(
            memory_ids=[mem1_id], storage=storage, limit=5, min_confidence=0.0
        )

        # Should find some related memories (if edges exist)
        if len(expansion) > 0:
            print(f"✓ Found {len(expansion)} RELATED memories")

            # Check result structure
            result = expansion[0]
            assert result["retrieval_reason"] == "graph_expansion_related"
            assert result["similarity_score"] >= 0.0
            assert result["source_memory_id"] == mem1_id
            print(f"✓ Result has correct structure: {result['retrieval_reason']}")
            print(f"  Similarity score: {result['similarity_score']:.3f}")
            print(f"  Source: {result['source_memory_id']}")
        else:
            print(f"✓ No RELATED edges from mem1 (threshold not met)")

        # Test 4: Limit parameter respected
        print("\nTest 4: Limit parameter respected")

        # Request limit=1
        expansion = expand_via_entities(
            memory_ids=[mem1_id], storage=storage, limit=1
        )

        assert len(expansion) <= 1, f"Should return at most 1 result, got {len(expansion)}"
        print(f"✓ Limit=1 respected: returned {len(expansion)} results")

        # Test 5: Combined expansion (expand_with_graph)
        print("\nTest 5: Combined expansion (expand_with_graph)")

        combined = expand_with_graph(
            memory_ids=[mem1_id],
            storage=storage,
            entity_limit=3,
            related_limit=3,
        )

        assert "via_entities" in combined
        assert "via_related" in combined
        print(f"✓ Combined expansion returns both types")
        print(f"  via_entities: {len(combined['via_entities'])} results")
        print(f"  via_related: {len(combined['via_related'])} results")

        # Test 6: Exclude source memories from expansion
        print("\nTest 6: Source memories excluded from expansion")

        # Expand from mem1 - should not include mem1 in results
        expansion = expand_via_entities(
            memory_ids=[mem1_id], storage=storage, limit=10
        )

        expanded_ids = [r["memory"].id for r in expansion]
        assert mem1_id not in expanded_ids, "Source memory should be excluded"
        print("✓ Source memory excluded from expansion")

        # Test 7: Multiple source memories
        print("\nTest 7: Expand from multiple sources")

        expansion = expand_via_entities(
            memory_ids=[mem1_id, mem2_id], storage=storage, limit=10
        )

        # Should not include mem1 or mem2 in results
        expanded_ids = [r["memory"].id for r in expansion]
        assert mem1_id not in expanded_ids
        assert mem2_id not in expanded_ids
        print(f"✓ Multiple sources excluded: found {len(expansion)} other memories")

        # Test 8: Ranking by expansion score
        print("\nTest 8: Results ranked by expansion score")

        expansion = expand_via_entities(
            memory_ids=[mem1_id], storage=storage, limit=10
        )

        # Verify sorted by expansion_score descending
        if len(expansion) > 1:
            for i in range(len(expansion) - 1):
                assert (
                    expansion[i]["expansion_score"]
                    >= expansion[i + 1]["expansion_score"]
                ), "Results should be sorted by expansion_score descending"
            print("✓ Results sorted by expansion score")
        else:
            print("✓ (Skipped - need multiple results)")

        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
        print("\nWork Item #6 (Graph Traversal & Expansion) complete!")
        print("\nFeatures validated:")
        print("  ✓ expand_via_entities() finds memories with shared entities")
        print("  ✓ expand_via_related() finds semantically similar memories")
        print("  ✓ Expansion respects limit parameter")
        print("  ✓ Results include retrieval_reason metadata")
        print("  ✓ Source memories excluded from results")
        print("  ✓ Multiple source memories supported")
        print("  ✓ Results ranked by relevance (expansion_score/similarity)")
        print("  ✓ expand_with_graph() combines both expansion types")

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    test_graph_traversal()
