#!/usr/bin/env python3
"""Test M4 Work Item #4: MENTIONS Edge Creation Integration"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

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
from vestig.core.ingestion import MemoryExtractionResult


def test_mentions_edge_creation():
    """Test end-to-end MENTIONS edge creation during memory commit"""
    print("=== M4 Work Item #4: MENTIONS Edge Creation ===\n")

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

        # M4 config (with entity extraction enabled)
        m4_config = {
            "entity_types": {
                "allowed_types": ["PERSON", "ORG", "SYSTEM", "PROJECT", "PLACE"]
            },
            "entity_extraction": {
                "enabled": True,
                "mode": "llm",
                "llm": {
                    "model": "claude-sonnet-4.5",
                    "min_confidence": 0.75,
                },
                "heuristics": {
                    "strip_titles": True,
                    "normalize_org_suffixes": True,
                    "reject_garbage": True,
                },
            },
            "edge_creation": {
                "mentions": {
                    "enabled": True,
                    "confidence_gated": True,
                },
            },
        }

        # Test 1: Mock LLM extraction and verify entity + edge creation
        print("Test 1: Entity extraction creates entities and MENTIONS edges")

        # Mock the extract_entities_llm function to return test entities
        mock_llm_response = MemoryExtractionResult.model_validate(
            {
                "memories": [
                    {
                        "content": "Alice Smith fixed the PostgreSQL replication bug",
                        "confidence": 0.95,
                        "rationale": "test",
                        "entities": [
                            {
                                "name": "Alice Smith",
                                "type": "PERSON",
                                "confidence": 0.92,
                                "evidence": "mentioned as the developer",
                            },
                            {
                                "name": "PostgreSQL",
                                "type": "SYSTEM",
                                "confidence": 0.95,
                                "evidence": "the database system",
                            },
                        ],
                    }
                ]
            }
        )

        with patch("vestig.core.ingestion.call_llm", return_value=mock_llm_response):
            # Commit memory with M4 config
            outcome = commit_memory(
                content="Alice Smith fixed the PostgreSQL replication bug",
                storage=storage,
                embedding_engine=embedding_engine,
                source="manual",
                event_storage=event_storage,
                m4_config=m4_config,
            )

            assert outcome.outcome == "INSERTED_NEW"
            memory_id = outcome.memory_id
            print(f"✓ Memory committed: {memory_id}")

        # Verify entities were created
        entities = storage.get_all_entities()
        assert len(entities) == 2
        print(f"✓ Created {len(entities)} entities")

        # Find entities by norm_key
        alice_norm = "PERSON:alice smith"
        postgres_norm = "SYSTEM:postgresql"

        alice = storage.find_entity_by_norm_key(alice_norm)
        postgres = storage.find_entity_by_norm_key(postgres_norm)

        assert alice is not None
        assert alice.canonical_name == "Alice Smith"
        assert alice.entity_type == "PERSON"
        print(f"✓ Entity created: {alice.canonical_name} ({alice.id})")

        assert postgres is not None
        assert postgres.canonical_name == "PostgreSQL"
        assert postgres.entity_type == "SYSTEM"
        print(f"✓ Entity created: {postgres.canonical_name} ({postgres.id})")

        # Verify MENTIONS edges were created
        edges = storage.get_edges_from_memory(memory_id, edge_type="MENTIONS")
        assert len(edges) == 2
        print(f"✓ Created {len(edges)} MENTIONS edges")

        # Check edge details
        edge_to_ids = [e.to_node for e in edges]
        assert alice.id in edge_to_ids
        assert postgres.id in edge_to_ids
        print("✓ Edges point to correct entities")

        # Check confidence and evidence stored
        for edge in edges:
            assert edge.confidence is not None
            assert edge.confidence >= 0.75  # Min threshold
            assert edge.evidence is not None
            assert len(edge.evidence) > 0
        print("✓ Edges have confidence and evidence")

        # Test 2: Low confidence entities filtered out
        print("\nTest 2: Low confidence entities filtered by threshold")

        mock_llm_response_low = MemoryExtractionResult.model_validate(
            {
                "memories": [
                    {
                        "content": "Bob mentioned something about the bug",
                        "confidence": 0.90,
                        "rationale": "test",
                        "entities": [
                            {
                                "name": "Bob",
                                "type": "PERSON",
                                "confidence": 0.60,  # Below 0.75 threshold
                                "evidence": "mentioned briefly",
                            }
                        ],
                    }
                ]
            }
        )

        with patch("vestig.core.ingestion.call_llm", return_value=mock_llm_response_low):
            outcome2 = commit_memory(
                content="Bob mentioned something about the bug",
                storage=storage,
                embedding_engine=embedding_engine,
                source="manual",
                event_storage=event_storage,
                m4_config=m4_config,
            )

            memory_id2 = outcome2.memory_id

        # Should have no new entities (filtered by confidence)
        bob_norm = "PERSON:bob"
        bob = storage.find_entity_by_norm_key(bob_norm)
        assert bob is None
        print("✓ Low confidence entity not created")

        # Should have no MENTIONS edges
        edges2 = storage.get_edges_from_memory(memory_id2, edge_type="MENTIONS")
        assert len(edges2) == 0
        print("✓ No MENTIONS edges for low confidence entities")

        # Test 3: Entity deduplication reuses existing entity
        print("\nTest 3: Entity deduplication reuses existing entity")

        mock_llm_response_dupe = MemoryExtractionResult.model_validate(
            {
                "memories": [
                    {
                        "content": "Alice Smith also fixed the backup script",
                        "confidence": 0.90,
                        "rationale": "test",
                        "entities": [
                            {
                                "name": "Alice Smith",  # Same person as before
                                "type": "PERSON",
                                "confidence": 0.88,
                                "evidence": "the developer again",
                            }
                        ],
                    }
                ]
            }
        )

        with patch("vestig.core.ingestion.call_llm", return_value=mock_llm_response_dupe):
            outcome3 = commit_memory(
                content="Alice Smith also fixed the backup script",
                storage=storage,
                embedding_engine=embedding_engine,
                source="manual",
                event_storage=event_storage,
                m4_config=m4_config,
            )

            memory_id3 = outcome3.memory_id

        # Should still have only 2 entities (no duplicate)
        all_entities = storage.get_all_entities()
        assert len(all_entities) == 2
        print("✓ Entity deduplicated (reused existing)")

        # Should have 1 MENTIONS edge pointing to existing Alice entity
        edges3 = storage.get_edges_from_memory(memory_id3, edge_type="MENTIONS")
        assert len(edges3) == 1
        assert edges3[0].to_node == alice.id
        print(f"✓ MENTIONS edge reuses existing entity ({alice.id})")

        # Test 4: Extraction disabled in config
        print("\nTest 4: Entity extraction respects enabled flag")

        m4_config_disabled = {
            **m4_config,
            "entity_extraction": {
                **m4_config["entity_extraction"],
                "enabled": False,
            },
        }

        outcome4 = commit_memory(
            content="Carol fixed the authentication system",
            storage=storage,
            embedding_engine=embedding_engine,
            source="manual",
            event_storage=event_storage,
            m4_config=m4_config_disabled,
        )

        memory_id4 = outcome4.memory_id

        # Should have no new entities
        all_entities_after = storage.get_all_entities()
        assert len(all_entities_after) == 2  # Same as before
        print("✓ No extraction when disabled")

        # Should have no MENTIONS edges
        edges4 = storage.get_edges_from_memory(memory_id4, edge_type="MENTIONS")
        assert len(edges4) == 0
        print("✓ No MENTIONS edges when extraction disabled")

        # Test 5: ENTITY_EXTRACTED event logged
        print("\nTest 5: ENTITY_EXTRACTED event logged")

        events = event_storage.get_events_for_memory(memory_id)
        entity_events = [e for e in events if e.event_type == "ENTITY_EXTRACTED"]
        assert len(entity_events) == 1
        print(f"✓ ENTITY_EXTRACTED event logged for {memory_id}")

        # Check event payload
        event = entity_events[0]
        assert event.payload.get("model_name") == "claude-sonnet-4.5"
        assert event.payload.get("prompt_hash") is not None
        assert event.payload.get("entity_count") == 2
        assert event.payload.get("min_confidence") == 0.75
        print("✓ Event payload contains model, prompt_hash, counts")

        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
        print("\nWork Item #4 (MENTIONS Edge Creation) complete!")
        print("\nFeatures validated:")
        print("  ✓ Entity extraction integrated into commit pipeline")
        print("  ✓ MENTIONS edges created with confidence + evidence")
        print("  ✓ Confidence gating filters low-quality extractions")
        print("  ✓ Entity deduplication via norm_key")
        print("  ✓ ENTITY_EXTRACTED event logging")
        print("  ✓ Config flags respected (enabled/disabled)")

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    test_mentions_edge_creation()
