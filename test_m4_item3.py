#!/usr/bin/env python3
"""Test M4 Work Item #3: Entity Extraction & Deduplication"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vestig.core.entity_extraction import (
    load_prompts,
    substitute_tokens,
    validate_extraction_result,
    apply_heuristic_cleanup,
    ExtractedEntity,
    compute_prompt_hash,
)


def test_entity_extraction():
    """Test entity extraction functions"""
    print("=== M4 Work Item #3: Entity Extraction ===\n")

    # Test 1: Load prompts from prompts.yaml
    print("Test 1: Load prompts from prompts.yaml")
    prompts = load_prompts("prompts.yaml")
    assert "extract_entities" in prompts
    assert "{{allowed_types}}" in prompts["extract_entities"]
    assert "{{content}}" in prompts["extract_entities"]
    print("✓ Prompts loaded successfully\n")

    # Test 2: Token substitution
    print("Test 2: Token substitution")
    template = "Hello {{name}}, your score is {{score}}"
    result = substitute_tokens(template, name="Alice", score=95)
    assert result == "Hello Alice, your score is 95"

    # Substitute in actual prompt
    prompt = substitute_tokens(
        prompts["extract_entities"],
        allowed_types="PERSON, SYSTEM",
        content="Alice fixed PostgreSQL bug",
    )
    assert "PERSON, SYSTEM" in prompt
    assert "Alice fixed PostgreSQL bug" in prompt
    assert "{{" not in prompt  # All tokens substituted
    print("✓ Token substitution works\n")

    # Test 3: Validate extraction result (valid)
    print("Test 3: Validate extraction result (valid)")
    valid_result = {
        "entities": [
            {
                "name": "Alice Smith",
                "type": "PERSON",
                "confidence": 0.92,
                "evidence": "mentioned as the person who fixed the bug",
            },
            {
                "name": "PostgreSQL",
                "type": "SYSTEM",
                "confidence": 0.95,
                "evidence": "the database system that had the bug",
            },
        ]
    }
    allowed_types = ["PERSON", "ORG", "SYSTEM", "PROJECT", "PLACE"]
    entities = validate_extraction_result(valid_result, allowed_types)
    assert len(entities) == 2
    assert entities[0].name == "Alice Smith"
    assert entities[0].entity_type == "PERSON"
    assert entities[0].confidence == 0.92
    assert entities[1].name == "PostgreSQL"
    print(f"✓ Validated {len(entities)} entities\n")

    # Test 4: Validation rejects invalid type
    print("Test 4: Validation rejects invalid entity type")
    invalid_type = {
        "entities": [{"name": "Test", "type": "INVALID", "confidence": 0.9, "evidence": "test"}]
    }
    try:
        validate_extraction_result(invalid_type, allowed_types)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid type" in str(e).lower()
        print(f"✓ Invalid type rejected: {e}\n")

    # Test 5: Validation rejects invalid confidence
    print("Test 5: Validation rejects invalid confidence")
    invalid_conf = {
        "entities": [{"name": "Test", "type": "PERSON", "confidence": 1.5, "evidence": "test"}]
    }
    try:
        validate_extraction_result(invalid_conf, allowed_types)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "confidence" in str(e).lower()
        print(f"✓ Invalid confidence rejected: {e}\n")

    # Test 6: Validation requires all fields
    print("Test 6: Validation requires all fields")
    missing_evidence = {
        "entities": [{"name": "Test", "type": "PERSON", "confidence": 0.9}]
    }
    try:
        validate_extraction_result(missing_evidence, allowed_types)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "evidence" in str(e).lower()
        print(f"✓ Missing field rejected: {e}\n")

    # Test 7: Evidence truncation
    print("Test 7: Evidence truncation (max 200 chars)")
    long_evidence = {
        "entities": [
            {
                "name": "Test",
                "type": "PERSON",
                "confidence": 0.9,
                "evidence": "x" * 250,
            }
        ]
    }
    entities = validate_extraction_result(long_evidence, allowed_types)
    assert len(entities[0].evidence) == 200
    assert entities[0].evidence.endswith("...")
    print(f"✓ Evidence truncated to {len(entities[0].evidence)} chars\n")

    # Test 8: Heuristic cleanup - strip titles
    print("Test 8: Heuristic cleanup - strip titles")
    entities_with_titles = [
        ExtractedEntity("Dr. Alice Smith", "PERSON", 0.9, "test"),
        ExtractedEntity("Mr. Bob Jones", "PERSON", 0.9, "test"),
        ExtractedEntity("Prof. Carol White", "PERSON", 0.9, "test"),
    ]
    cleaned = apply_heuristic_cleanup(entities_with_titles)
    assert cleaned[0].name == "Alice Smith"
    assert cleaned[1].name == "Bob Jones"
    assert cleaned[2].name == "Carol White"
    print("✓ Titles stripped correctly\n")

    # Test 9: Heuristic cleanup - normalize org suffixes
    print("Test 9: Heuristic cleanup - normalize org suffixes")
    entities_with_suffixes = [
        ExtractedEntity("Acme Ltd.", "ORG", 0.9, "test"),
        ExtractedEntity("XYZ Inc", "ORG", 0.9, "test"),
        ExtractedEntity("ABC Corporation", "ORG", 0.9, "test"),
    ]
    cleaned = apply_heuristic_cleanup(entities_with_suffixes)
    assert cleaned[0].name == "Acme"
    assert cleaned[1].name == "XYZ"
    assert cleaned[2].name == "ABC"
    print("✓ Org suffixes normalized\n")

    # Test 10: Heuristic cleanup - reject garbage
    print("Test 10: Heuristic cleanup - reject garbage")
    entities_with_garbage = [
        ExtractedEntity("Alice", "PERSON", 0.9, "test"),  # Valid
        ExtractedEntity("...", "PERSON", 0.9, "test"),  # All punctuation
        ExtractedEntity("a", "PERSON", 0.9, "test"),  # Too short
        ExtractedEntity("123", "SYSTEM", 0.9, "test"),  # Numeric only
        ExtractedEntity("PostgreSQL", "SYSTEM", 0.9, "test"),  # Valid
    ]
    cleaned = apply_heuristic_cleanup(entities_with_garbage)
    assert len(cleaned) == 2
    assert cleaned[0].name == "Alice"
    assert cleaned[1].name == "PostgreSQL"
    print(f"✓ Garbage rejected: {len(entities_with_garbage)} → {len(cleaned)} entities\n")

    # Test 11: Prompt hash computation
    print("Test 11: Compute prompt hash for reproducibility")
    template = prompts["extract_entities"]
    hash1 = compute_prompt_hash(template)
    hash2 = compute_prompt_hash(template)
    assert hash1 == hash2  # Deterministic
    assert len(hash1) == 16  # First 16 chars
    print(f"✓ Prompt hash: {hash1}\n")

    # Test 12: Empty entities list
    print("Test 12: Handle empty entities list")
    empty_result = {"entities": []}
    entities = validate_extraction_result(empty_result, allowed_types)
    assert len(entities) == 0
    print("✓ Empty entities list handled\n")

    print("=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
    print("\nWork Item #3 (Entity Extraction - Core Logic) complete!")
    print("\nNote: LLM integration is a placeholder (call_llm not implemented).")
    print("Production will use Anthropic SDK or llm CLI tool.")


if __name__ == "__main__":
    test_entity_extraction()
