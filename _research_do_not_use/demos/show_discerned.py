#!/usr/bin/env python3
"""Show discerned extraction results in a readable format"""

import json
from session_parser import ClaudeCodeSessionParser

def main():
    print("Running discerned extraction on test session...")
    print("=" * 80)

    parser = ClaudeCodeSessionParser(use_discerned=True, llm_model="gpt-4o-mini")
    result = parser.parse_session_file("test_session.jsonl")

    # Filter discerned facts
    discerned_facts = [f for f in result['facts'] if f['extraction_method'] == 'discerned']

    print(f"\n✓ Extracted {len(discerned_facts)} discerned facts\n")
    print("=" * 80)

    for i, fact in enumerate(discerned_facts, 1):
        print(f"\n[{i}] {fact['predicate'].upper()}")
        print("-" * 80)
        print(f"Subject:    {fact['subject']}")
        print(f"Object:     {fact['object']}")
        print(f"Confidence: {fact['confidence']}")
        if fact.get('context'):
            print(f"Context:    {fact['context']}")
        print(f"Timestamp:  {fact['timestamp']}")

    print("\n" + "=" * 80)
    print("\nDISCERNED INSIGHTS SUMMARY:")
    print("=" * 80)

    # Group by predicate for summary
    by_pred = {}
    for fact in discerned_facts:
        pred = fact['predicate']
        if pred not in by_pred:
            by_pred[pred] = []
        by_pred[pred].append(fact)

    for pred, facts in sorted(by_pred.items()):
        print(f"\n{pred}:")
        for fact in facts:
            print(f"  → {fact['object']}")

if __name__ == '__main__':
    main()
