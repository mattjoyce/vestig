#!/usr/bin/env python3
"""Test M4 Work Item #3: Entity Extraction & Deduplication (current API)"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.entity_extraction import load_prompts, substitute_tokens, store_entities
from vestig.core.storage import MemoryStorage


def test_entity_extraction():
    """Test entity extraction helpers and storage integration"""
    print("=== M4 Work Item #3: Entity Extraction ===\n")

    # Test 1: Load prompts from default location
    print("Test 1: Load prompts from default location")
    prompts = load_prompts()
    assert "extract_memories_from_session" in prompts
    assert "{{content}}" in prompts["extract_memories_from_session"]
    print("✓ Prompts loaded successfully\n")

    # Test 2: Token substitution
    print("Test 2: Token substitution")
    template = "Hello {{name}}, your score is {{score}}"
    result = substitute_tokens(template, name="Alice", score=95)
    assert result == "Hello Alice, your score is 95"
    prompt = substitute_tokens(
        prompts["extract_memories_from_session"],
        content="Alice fixed PostgreSQL bug",
    )
    assert "Alice fixed PostgreSQL bug" in prompt
    assert "{{" not in prompt
    print("✓ Token substitution works\n")

    # Test 3: store_entities respects confidence threshold + dedup
    print("Test 3: store_entities confidence gating + dedup")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        storage = MemoryStorage(db_path)
        config = {
            "entity_extraction": {
                "llm": {"min_confidence": 0.75},
            }
        }

        entities = [
            ("Alice Smith", "PERSON", 0.92, "developer"),
            ("PostgreSQL", "SYSTEM", 0.95, "database"),
            ("Bob", "PERSON", 0.60, "low confidence"),
        ]
        stored = store_entities(entities, memory_id="mem_test", storage=storage, config=config)
        assert len(stored) == 2

        # Dedup by norm_key (same entity, different casing)
        entities_dupe = [("alice smith", "PERSON", 0.88, "duplicate")]
        stored_dupe = store_entities(
            entities_dupe, memory_id="mem_test", storage=storage, config=config
        )
        assert len(stored_dupe) == 1

        all_entities = storage.get_all_entities()
        assert len(all_entities) == 2
        print("✓ Confidence gating + dedup working\n")

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

    print("=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
    print("\nWork Item #3 (Entity Extraction) complete!")


if __name__ == "__main__":
    test_entity_extraction()
