#!/usr/bin/env python3
"""Test M4 Work Item #3: Entity Extraction & Deduplication (current API)"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.db_interface import DatabaseInterface
from vestig.core.entity_extraction import load_prompts, store_entities, substitute_tokens


def test_entity_extraction(storage: DatabaseInterface):
    """Test entity extraction helpers and storage integration"""
    print("=== M4 Work Item #3: Entity Extraction ===\n")

    # Test 1: Load prompts from default location
    print("Test 1: Load prompts from default location")
    prompts = load_prompts()
    assert "extract_memories_from_session" in prompts
    # Prompts are now dicts with system/user/description keys (M4+ format)
    prompt = prompts["extract_memories_from_session"]
    assert isinstance(prompt, dict), "Prompt should be a dict with system/user keys"
    assert "user" in prompt, "Prompt should have 'user' key"
    assert "{{content}}" in prompt["user"], "User prompt should contain {{content}} token"
    print("✓ Prompts loaded successfully\n")

    # Test 2: Token substitution
    print("Test 2: Token substitution")
    template = "Hello {{name}}, your score is {{score}}"
    result = substitute_tokens(template, name="Alice", score=95)
    assert result == "Hello Alice, your score is 95"
    # Use the 'user' field from the prompt dict
    user_prompt = substitute_tokens(
        prompts["extract_memories_from_session"]["user"],
        content="Alice fixed PostgreSQL bug",
    )
    assert "Alice fixed PostgreSQL bug" in user_prompt
    assert "{{" not in user_prompt
    print("✓ Token substitution works\n")

    # Test 3: store_entities respects confidence threshold + dedup
    print("Test 3: store_entities confidence gating + dedup")

    from vestig.core.config import load_config
    from vestig.core.embeddings import EmbeddingEngine

    try:
        # Load full config for embedding settings
        full_config = load_config("config_test.yaml")
        embedding_engine = EmbeddingEngine(
            model_name=full_config["embedding"]["model"],
            expected_dimension=full_config["embedding"]["dimension"],
        )

        # Override min_confidence for this test
        config = full_config.copy()
        config["m4"] = config.get("m4", {})
        config["m4"]["entity_extraction"] = {"llm": {"min_confidence": 0.75}}

        entities = [
            ("Alice Smith", "PERSON", 0.92, "developer"),
            ("PostgreSQL", "SYSTEM", 0.95, "database"),
            ("Bob", "PERSON", 0.60, "low confidence"),
        ]
        stored = store_entities(
            entities,
            memory_id="mem_test",
            storage=storage,
            config=config,
            embedding_engine=embedding_engine,
        )
        assert len(stored) == 2, f"Expected 2 entities (above threshold), got {len(stored)}"

        # Dedup by norm_key (same entity, different casing)
        entities_dupe = [("alice smith", "PERSON", 0.88, "duplicate")]
        stored_dupe = store_entities(
            entities_dupe,
            memory_id="mem_test",
            storage=storage,
            config=config,
            embedding_engine=embedding_engine,
        )
        # Should return 1 (the existing entity is returned, not created again)
        assert len(stored_dupe) == 1, f"Expected 1 entity returned, got {len(stored_dupe)}"

        all_entities = storage.get_all_entities()
        assert len(all_entities) == 2, f"Expected 2 unique entities, got {len(all_entities)}"
        print("✓ Confidence gating + dedup working\n")

    finally:
        # Cleanup handled by fixture
        pass

    print("=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
    print("\nWork Item #3 (Entity Extraction) complete!")
