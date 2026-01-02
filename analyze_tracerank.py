#!/usr/bin/env python3
"""
Analyze TraceRank behavior from test results with --explain output
"""

import json
import re
from collections import defaultdict
from pathlib import Path
import sys


def parse_explain_output(output: str) -> list:
    """
    Parse --explain output to extract TraceRank data
    Returns list of {
        'score': float,
        'tracerank_boost': float,
        'connections': int,
        'reinforced': bool,
        'age': str,
        'stability': str,
        'content': str,
        'rank': int  # Position in results (0 = first)
    }
    """
    results = []

    # Split by memory blocks
    blocks = output.split('\n---\n')

    for rank, block in enumerate(blocks):
        if '[META]' not in block:
            continue

        # Extract metadata line
        meta_match = re.search(r'\[META\] \(score=([\d.]+), age=([^,]+), stability=(\w+)\)', block)
        if not meta_match:
            continue

        score = float(meta_match.group(1))
        age = meta_match.group(2)
        stability = meta_match.group(3)

        # Extract TraceRank boost
        tracerank_match = re.search(r'TraceRank: ([\d.]+)x', block)
        if not tracerank_match:
            continue
        tracerank_boost = float(tracerank_match.group(1))

        # Extract connections
        conn_match = re.search(r'(\d+) conn', block)
        connections = int(conn_match.group(1)) if conn_match else 0

        # Check if reinforced
        reinforced = 'reinforced' in block

        # Extract content
        content_match = re.search(r'\[MEMORY\]\n(.+?)(?:\n\n|$)', block, re.DOTALL)
        content = content_match.group(1).strip() if content_match else ""

        results.append({
            'score': score,
            'tracerank_boost': tracerank_boost,
            'connections': connections,
            'reinforced': reinforced,
            'age': age,
            'stability': stability,
            'content': content,
            'rank': rank
        })

    return results


def analyze_results(results_file: str):
    """Analyze test results for TraceRank patterns"""

    with open(results_file) as f:
        data = json.load(f)

    print(f"{'='*80}")
    print(f"TRACERANK ANALYSIS")
    print(f"{'='*80}\n")

    # Overall stats
    total_queries = len(data)
    print(f"Total queries: {total_queries}\n")

    # Analyze each query
    tracerank_helps = 0
    tracerank_hurts = 0

    for result in data:
        qa_id = result['qa_id']
        question = result['question']
        expected = result['expected_answer']
        output = result['vestig_output']
        relevant = result.get('evaluation', {}).get('appears_relevant', False)

        # Parse explain output
        memories = parse_explain_output(output)

        if not memories:
            continue

        # Analyze top result
        top = memories[0]

        # Check if TraceRank boosted the wrong answer to the top
        # Compare semantic score vs final rank
        sorted_by_semantic = sorted(memories, key=lambda m: m['score'], reverse=True)
        top_semantic = sorted_by_semantic[0]

        # If top result by semantic score is NOT the top overall result,
        # TraceRank changed the ranking
        tracerank_changed_order = (top['content'] != top_semantic['content'])

        if tracerank_changed_order:
            print(f"\n{'='*80}")
            print(f"Q{qa_id}: {question[:60]}...")
            print(f"Expected: {expected[:60]}...")
            print(f"Relevant: {'✓' if relevant else '✗'}\n")

            print(f"TraceRank REORDERED results:")
            print(f"\n  Top by semantic score (would be #1 without TraceRank):")
            print(f"    Score: {top_semantic['score']:.4f}, TraceRank: {top_semantic['tracerank_boost']:.2f}x")
            print(f"    Conn: {top_semantic['connections']}, Reinforced: {top_semantic['reinforced']}")
            print(f"    Content: {top_semantic['content'][:100]}...")

            print(f"\n  Actually shown first (boosted by TraceRank):")
            print(f"    Score: {top['score']:.4f}, TraceRank: {top['tracerank_boost']:.2f}x")
            print(f"    Conn: {top['connections']}, Reinforced: {top['reinforced']}")
            print(f"    Content: {top['content'][:100]}...")

            # Assess if this helped or hurt
            # Simple heuristic: check if top result contains key terms from expected answer
            expected_words = set(expected.lower().split()) - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            top_content_lower = top['content'].lower()
            semantic_content_lower = top_semantic['content'].lower()

            top_matches = sum(1 for word in expected_words if word in top_content_lower)
            semantic_matches = sum(1 for word in expected_words if word in semantic_content_lower)

            if top_matches > semantic_matches:
                print(f"    → TraceRank HELPED (boosted more relevant result)")
                tracerank_helps += 1
            elif top_matches < semantic_matches:
                print(f"    → TraceRank HURT (boosted less relevant result)")
                tracerank_hurts += 1
            else:
                print(f"    → TraceRank NEUTRAL (equal relevance)")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")
    print(f"Queries where TraceRank changed ranking:")
    print(f"  Helped:  {tracerank_helps}")
    print(f"  Hurt:    {tracerank_hurts}")
    print(f"  Neutral: {total_queries - tracerank_helps - tracerank_hurts}")

    # Statistics on TraceRank boosts
    print(f"\n{'='*80}")
    print(f"TRACERANK BOOST STATISTICS")
    print(f"{'='*80}\n")

    all_boosts = []
    all_connections = []
    reinforced_count = 0

    for result in data:
        memories = parse_explain_output(result['vestig_output'])
        for m in memories:
            all_boosts.append(m['tracerank_boost'])
            all_connections.append(m['connections'])
            if m['reinforced']:
                reinforced_count += 1

    if all_boosts:
        print(f"TraceRank boost range: {min(all_boosts):.2f}x - {max(all_boosts):.2f}x")
        print(f"Average boost: {sum(all_boosts)/len(all_boosts):.2f}x")
        print(f"Connections range: {min(all_connections)} - {max(all_connections)}")
        print(f"Average connections: {sum(all_connections)/len(all_connections):.1f}")
        print(f"Reinforced memories: {reinforced_count}/{len(all_boosts)} ({reinforced_count/len(all_boosts):.1%})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_tracerank.py <results_file.json>")
        sys.exit(1)

    analyze_results(sys.argv[1])
