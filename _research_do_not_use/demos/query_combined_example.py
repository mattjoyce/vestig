#!/usr/bin/env python3
"""
Example: Query naive + discerned facts together to gain insights
"""

from agent_memory import AgentMemory
from session_parser import ClaudeCodeSessionParser

def main():
    print("=" * 80)
    print("COMBINED QUERYING: Naive + Discerned Facts")
    print("=" * 80)

    # Parse and import test session with BOTH extraction methods
    print("\n[1] Importing test session with both naive and discerned extraction...")
    parser = ClaudeCodeSessionParser(use_discerned=True, llm_model="gpt-4o-mini")
    session_data = parser.parse_session_file("test_session.jsonl")

    memory = AgentMemory("test_combined.db")
    memory.import_session(session_data)

    # Query 1: What hosts did we connect to? (naive)
    print("\n[2] NAIVE QUERY: What hosts did we SSH to?")
    print("-" * 80)
    hosts = memory.query_facts(predicate='connects_to_host')
    for fact in hosts:
        print(f"  → {fact['object']} (confidence: {fact['confidence']})")

    # Query 2: What was the session intent? (discerned)
    print("\n[3] DISCERNED QUERY: What was the overall intent?")
    print("-" * 80)
    intents = memory.query_facts(predicate='session_intent')
    for fact in intents:
        print(f"  → {fact['object']}")

    # Query 3: What workflow pattern? (discerned)
    print("\n[4] DISCERNED QUERY: What workflow pattern was detected?")
    print("-" * 80)
    workflows = memory.query_facts(predicate='workflow_pattern')
    for fact in workflows:
        print(f"  → {fact['object']}")

    # Query 4: COMBINED - Find sessions where:
    #   - User connected to a specific host (naive)
    #   - AND got approving feedback (discerned)
    print("\n[5] COMBINED QUERY: Sessions with SSH + User Approval")
    print("-" * 80)
    cursor = memory.conn.cursor()
    cursor.execute("""
        SELECT DISTINCT
            n.object as host,
            d.object as feedback_type,
            n.source_session
        FROM facts n
        JOIN facts d ON n.source_session = d.source_session
        WHERE n.extraction_method = 'naive'
          AND n.predicate = 'connects_to_host'
          AND d.extraction_method = 'discerned'
          AND d.predicate = 'user_feedback_type'
          AND d.object = 'approving'
    """)

    results = cursor.fetchall()
    for host, feedback, session in results:
        print(f"  ✓ Host: {host}")
        print(f"    Feedback: {feedback}")
        print(f"    Session: {session}")

    # Query 5: COMBINED - What commands worked well?
    # (Commands executed + direct assistant tone + approval)
    print("\n[6] COMBINED QUERY: Effective Commands (executed + direct tone + approval)")
    print("-" * 80)
    cursor.execute("""
        SELECT DISTINCT
            cmd.object as command,
            tone.object as tone,
            fb.object as feedback
        FROM facts cmd
        LEFT JOIN facts tone ON cmd.source_session = tone.source_session
        LEFT JOIN facts fb ON cmd.source_session = fb.source_session
        WHERE cmd.predicate = 'executed_command'
          AND cmd.extraction_method = 'naive'
          AND tone.predicate = 'assistant_tone'
          AND tone.extraction_method = 'discerned'
          AND tone.object = 'direct'
          AND fb.predicate = 'user_feedback_type'
          AND fb.extraction_method = 'discerned'
          AND fb.object = 'approving'
        LIMIT 3
    """)

    results = cursor.fetchall()
    for command, tone, feedback in results:
        print(f"  ✓ {command[:70]}...")
        print(f"    Tone: {tone}, Feedback: {feedback}")

    # Summary statistics
    print("\n[7] STATISTICS: Breakdown by extraction method")
    print("=" * 80)
    cursor.execute("""
        SELECT extraction_method, COUNT(*) as count
        FROM facts
        GROUP BY extraction_method
    """)

    for method, count in cursor.fetchall():
        print(f"  {method:12s}: {count:3d} facts")

    print("\n[8] KEY INSIGHT")
    print("=" * 80)
    print("By combining naive + discerned facts, we can answer questions like:")
    print("  • 'What SSH commands got user approval?' (naive + discerned)")
    print("  • 'What workflows are effective for host X?' (naive + discerned)")
    print("  • 'When user corrects me, what was I doing?' (naive + discerned)")
    print("\nThis is the POWER of dual-extraction! 🚀")
    print("=" * 80)

    memory.close()

if __name__ == '__main__':
    main()
