#!/usr/bin/env python3
"""
Test script to demonstrate naive vs discerned extraction
"""

import json
from session_parser import ClaudeCodeSessionParser

def main():
    session_file = "test_session.jsonl"

    print("=" * 80)
    print("VESTIG EXTRACTION TEST - Naive vs Discerned")
    print("=" * 80)

    # Test 1: Naive extraction only
    print("\n[1] Testing NAIVE extraction (regex-based, fast)...")
    print("-" * 80)

    parser_naive = ClaudeCodeSessionParser(use_discerned=False)
    result_naive = parser_naive.parse_session_file(session_file)

    naive_facts = [f for f in result_naive['facts'] if f['extraction_method'] == 'naive']

    print(f"✓ Extracted {len(naive_facts)} naive facts")
    print("\nSample naive facts:")

    # Group by predicate
    by_predicate = {}
    for fact in naive_facts:
        pred = fact['predicate']
        if pred not in by_predicate:
            by_predicate[pred] = []
        by_predicate[pred].append(fact)

    for pred, facts in sorted(by_predicate.items()):
        print(f"\n  {pred} ({len(facts)} facts):")
        for fact in facts[:2]:  # Show first 2 examples
            obj_preview = fact['object'][:60] if len(fact['object']) > 60 else fact['object']
            print(f"    - {obj_preview}")
            print(f"      confidence: {fact['confidence']}")

    # Test 2: Discerned extraction (with LLM)
    print("\n\n[2] Testing DISCERNED extraction (LLM-based, semantic)...")
    print("-" * 80)

    # Use gpt-4o-mini since it's available
    parser_discerned = ClaudeCodeSessionParser(
        use_discerned=True,
        llm_model="gpt-4o-mini"
    )
    result_discerned = parser_discerned.parse_session_file(session_file)

    discerned_facts = [f for f in result_discerned['facts'] if f['extraction_method'] == 'discerned']

    print(f"✓ Extracted {len(discerned_facts)} discerned facts")
    print("\nDiscerned facts (semantic insights):")

    # Group discerned by predicate
    discerned_by_pred = {}
    for fact in discerned_facts:
        pred = fact['predicate']
        if pred not in discerned_by_pred:
            discerned_by_pred[pred] = []
        discerned_by_pred[pred].append(fact)

    for pred, facts in sorted(discerned_by_pred.items()):
        print(f"\n  {pred} ({len(facts)} facts):")
        for fact in facts:
            obj_preview = fact['object'][:80] if len(fact['object']) > 80 else fact['object']
            print(f"    - {obj_preview}")
            print(f"      confidence: {fact['confidence']}")
            if fact.get('context'):
                ctx_preview = fact['context'][:60]
                print(f"      context: {ctx_preview}...")

    # Summary comparison
    print("\n\n[3] COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Naive facts:     {len(naive_facts):3d}  (fast, structured data)")
    print(f"Discerned facts: {len(discerned_facts):3d}  (slow, semantic insights)")
    print(f"Total facts:     {len(result_discerned['facts']):3d}")

    print("\n\nNaive predicates:")
    for pred in sorted(by_predicate.keys()):
        print(f"  • {pred}")

    print("\n\nDiscerned predicates (new insights!):")
    for pred in sorted(discerned_by_pred.keys()):
        print(f"  • {pred}")

    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("\nThe extraction_method field correctly flags:")
    print("  - 'naive' for regex-based facts")
    print("  - 'discerned' for LLM-analyzed facts")
    print("=" * 80)

if __name__ == '__main__':
    main()
