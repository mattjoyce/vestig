#!/usr/bin/env python3
"""Test ingestion with mocked LLM"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

# Ensure tests run offline if the model is already cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vestig.core.ingestion import ingest_document, MemoryExtractionResult
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.storage import MemoryStorage
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.config import load_config


def test_ingestion():
    """Test document ingestion with mocked LLM"""
    print("=== Testing Document Ingestion ===\n")

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Load config and setup
        config = load_config("config_test.yaml")
        storage = MemoryStorage(db_path)
        event_storage = MemoryEventStorage(storage.conn)
        embedding_engine = EmbeddingEngine(
            model_name=config["embedding"]["model"],
            expected_dimension=config["embedding"]["dimension"],
        )

        # Mock LLM response - extract memories from conversation
        mock_response = MemoryExtractionResult.model_validate(
            {
                "memories": [
                    {
                        "content": "PostgreSQL queries on user_events table were experiencing performance issues, with some queries taking over 5 seconds according to slow query log",
                        "confidence": 0.9,
                        "rationale": "Clear technical problem statement with metrics",
                        "entities": [],
                    },
                    {
                        "content": "The slow PostgreSQL queries were caused by missing index on event_timestamp column, leading to full table scans",
                        "confidence": 0.95,
                        "rationale": "Root cause identified with specific solution",
                        "entities": [],
                    },
                    {
                        "content": "Bob recommended creating B-tree index on (user_id, event_timestamp) for PostgreSQL optimization",
                        "confidence": 0.9,
                        "rationale": "Specific technical decision with person attribution",
                        "entities": [],
                    },
                    {
                        "content": "PgBouncer should be configured with pool size of 20 connections to reduce connection overhead",
                        "confidence": 0.85,
                        "rationale": "Configuration recommendation with specific value",
                        "entities": [],
                    },
                    {
                        "content": "Redis caching strategy should include event queries with 5-minute TTL to prevent stale reads",
                        "confidence": 0.85,
                        "rationale": "Caching strategy decision with time parameter",
                        "entities": [],
                    },
                ]
            }
        )

        # Test with mocked LLM
        with patch("vestig.core.ingestion.call_llm", return_value=mock_response):
            result = ingest_document(
                document_path=str(Path(__file__).resolve().parents[1] / "test_session_sample.txt"),
                storage=storage,
                embedding_engine=embedding_engine,
                event_storage=event_storage,
                m4_config=config.get("m4", {}),
                chunk_size=20000,
                extraction_model="claude-sonnet-4.5",
                min_confidence=0.6,
            )

        # Verify results
        print("\n" + "=" * 70)
        print("INGESTION RESULTS")
        print("=" * 70)
        print(f"Document: {result.document_path}")
        print(f"Chunks processed: {result.chunks_processed}")
        print(f"Memories extracted: {result.memories_extracted}")
        print(f"Memories committed: {result.memories_committed}")
        print(f"Duplicates skipped: {result.memories_deduplicated}")
        print(f"Entities created: {result.entities_created}")

        if result.errors:
            print(f"\nErrors: {len(result.errors)}")
            for error in result.errors:
                print(f"  - {error}")

        print("=" * 70)

        # Verify memories in storage
        all_memories = storage.get_all_memories()
        print(f"\nTotal memories in database: {len(all_memories)}")

        # Show sample memories
        print("\nSample committed memories:")
        for i, memory in enumerate(all_memories[:3], 1):
            print(f"\n{i}. {memory.content[:100]}...")

        # Show entities
        entities = storage.get_all_entities()
        if entities:
            print(f"\n\nEntities extracted:")
            for entity in entities:
                print(f"  - {entity.canonical_name} ({entity.entity_type})")

        # Verify entity extraction events
        for memory in all_memories[:1]:
            events = event_storage.get_events_for_memory(memory.id)
            entity_events = [e for e in events if e.event_type == "ENTITY_EXTRACTED"]
            if entity_events:
                print(f"\nENTITY_EXTRACTED event for {memory.id[:16]}...")
                print(f"  Model: {entity_events[0].payload.get('model_name')}")
                print(f"  Entities found: {entity_events[0].payload.get('entity_count')}")

        print("\n" + "=" * 70)
        print("✅ Ingestion test complete!")
        print("=" * 70)

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    test_ingestion()
