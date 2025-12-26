#!/usr/bin/env python3
"""
Import all Claude Code sessions from ~/.claude/projects
Creates a comprehensive agent memory database
"""

import glob
import os
from pathlib import Path
from agent_memory import AgentMemory
from session_parser import ClaudeCodeSessionParser

def main():
    # Configuration
    sessions_dir = os.path.expanduser("~/.claude/projects")
    db_path = "vestig_memory.db"
    use_discerned = True  # Enable LLM-based extraction
    llm_model = "gpt-4o-mini"

    print("=" * 80)
    print("IMPORTING ALL CLAUDE CODE SESSIONS")
    print("=" * 80)
    print(f"Sessions directory: {sessions_dir}")
    print(f"Output database:    {db_path}")
    print(f"Discerned mode:     {use_discerned}")
    print(f"LLM model:          {llm_model}")
    print("=" * 80)

    # Find all session files
    pattern = os.path.join(sessions_dir, "**/*.jsonl")
    session_files = glob.glob(pattern, recursive=True)

    print(f"\nFound {len(session_files)} session files")

    if len(session_files) == 0:
        print("No session files found!")
        return

    # Initialize memory and parser
    memory = AgentMemory(db_path)
    parser = ClaudeCodeSessionParser(
        use_discerned=use_discerned,
        llm_model=llm_model
    )

    # Process each session
    imported_count = 0
    skipped_count = 0
    error_count = 0

    for i, session_file in enumerate(session_files, 1):
        print(f"\n[{i}/{len(session_files)}] Processing: {Path(session_file).name}")

        try:
            # Parse session
            session_data = parser.parse_session_file(session_file)

            # Import with deduplication
            result = memory.import_session(
                session_data,
                file_path=session_file,
                skip_if_exists=True
            )

            if result:
                imported_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            error_count += 1
            continue

    # Summary
    print("\n" + "=" * 80)
    print("IMPORT COMPLETE")
    print("=" * 80)
    print(f"Total files:     {len(session_files)}")
    print(f"Imported:        {imported_count}")
    print(f"Skipped (dups):  {skipped_count}")
    print(f"Errors:          {error_count}")

    # Database stats
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)
    stats = memory.stats()
    print(f"Total sessions:  {stats['total_sessions']}")
    print(f"Total facts:     {stats['total_facts']}")
    print(f"Unique entities: {stats['unique_entities']}")
    print(f"Unique predicates: {stats['unique_predicates']}")

    print("\nTop predicates:")
    for pred, count in list(stats['top_predicates'].items())[:10]:
        print(f"  {pred:30s}: {count:5d}")

    # Count by extraction method
    cursor = memory.conn.cursor()
    cursor.execute("""
        SELECT extraction_method, COUNT(*)
        FROM facts
        GROUP BY extraction_method
    """)
    print("\nFacts by extraction method:")
    for method, count in cursor.fetchall():
        print(f"  {method:12s}: {count:5d}")

    print("\n" + "=" * 80)
    print(f"✓ Memory database created: {db_path}")
    print("=" * 80)

    memory.close()

    # Prompt for embeddings
    print("\n\nNext steps:")
    print("1. Generate embeddings (optional, requires OpenAI API):")
    print(f"   python -c \"from agent_memory import AgentMemory; m = AgentMemory('{db_path}'); m.embed_all_facts(model='ada-002')\"")
    print("\n2. Query the memory:")
    print(f"   python -c \"from agent_memory import AgentMemory; m = AgentMemory('{db_path}'); print(m.stats())\"")
    print("\n3. RAG retrieval:")
    print(f"   python -c \"from agent_memory import AgentMemory; m = AgentMemory('{db_path}'); print(m.find_similar_facts('how to...', top_k=5))\"")

if __name__ == '__main__':
    main()
