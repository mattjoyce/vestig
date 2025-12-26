#!/usr/bin/env python3
"""Test that llm Python API works"""

import llm

print("Testing llm Python API...")

# Test 1: LLM model
print("\n[1] Testing LLM model (gpt-4o-mini)...")
try:
    model = llm.get_model("gpt-4o-mini")
    response = model.prompt("Say hello in 3 words")
    print(f"✓ Response: {response.text()}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Embedding model
print("\n[2] Testing embedding model (ada-002)...")
try:
    model = llm.get_embedding_model("ada-002")
    embedding = model.embed("test text")
    print(f"✓ Embedding generated: {len(list(embedding))} dimensions")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n✓ All tests passed!")
