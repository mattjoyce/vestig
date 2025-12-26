#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Demo
Shows how to use Vestig memory as a RAG system
"""

from agent_memory import AgentMemory
from session_parser import ClaudeCodeSessionParser

def main():
    print("=" * 80)
    print("VESTIG RAG SYSTEM DEMO")
    print("=" * 80)

    # Step 1: Import session with deduplication
    print("\n[1] Importing session with deduplication...")
    print("-" * 80)

    memory = AgentMemory("rag_demo.db")
    parser = ClaudeCodeSessionParser(use_discerned=True, llm_model="gpt-4o-mini")

    session_file = "test_session.jsonl"
    session_data = parser.parse_session_file(session_file)

    # First import
    result = memory.import_session(session_data, file_path=session_file)
    if result:
        print("✓ Session imported")
    else:
        print("✓ Session already exists (deduplication working!)")

    # Try importing again (should be skipped)
    print("\nAttempting to import same session again...")
    result = memory.import_session(session_data, file_path=session_file)
    if not result:
        print("✓ Duplicate detected and skipped!")

    # Step 2: Generate embeddings
    print("\n\n[2] Generating embeddings for facts...")
    print("-" * 80)
    print("Note: This requires OpenAI API key configured with 'llm'")
    print("Using model: ada-002 (OpenAI text-embedding-ada-002)")

    try:
        memory.embed_all_facts(model="ada-002", batch_size=5)
    except Exception as e:
        print(f"Embedding generation skipped: {e}")
        print("To enable embeddings, configure OpenAI: llm keys set openai")

    # Step 3: RAG Retrieval - Find relevant memories
    print("\n\n[3] RAG RETRIEVAL: Finding relevant memories...")
    print("=" * 80)

    queries = [
        "How do I verify a backup archive?",
        "SSH connection to server",
        "User was happy with the result"
    ]

    for query in queries:
        print(f"\n📝 QUERY: \"{query}\"")
        print("-" * 80)

        try:
            results = memory.find_similar_facts(
                query_text=query,
                top_k=3,
                model="ada-002",
                min_confidence=0.5
            )

            if not results:
                print("  No embeddings found (run step 2 first)")
                continue

            for i, fact in enumerate(results, 1):
                print(f"\n  [{i}] Similarity: {fact['similarity']:.3f} | "
                      f"Method: {fact['extraction_method']} | "
                      f"Confidence: {fact['confidence']}")
                print(f"      {fact['predicate']}: {fact['object'][:80]}")
                if fact.get('context'):
                    print(f"      Context: {fact['context'][:80]}...")

        except Exception as e:
            print(f"  Search failed: {e}")
            print("  Make sure embeddings are generated (step 2)")

    # Step 4: RAG-Augmented Prompt
    print("\n\n[4] RAG-AUGMENTED PROMPT GENERATION")
    print("=" * 80)

    user_question = "How should I verify my unraid backups?"
    print(f"\nUser asks: \"{user_question}\"")

    try:
        # Retrieve relevant memories
        relevant_facts = memory.find_similar_facts(
            query_text=user_question,
            top_k=5,
            model="ada-002"
        )

        if relevant_facts:
            # Build augmented prompt
            print("\nRetrieved Memories:")
            memories_text = []
            for fact in relevant_facts:
                memories_text.append(
                    f"- {fact['predicate']}: {fact['object']} "
                    f"(similarity: {fact['similarity']:.2f}, {fact['extraction_method']})"
                )

            print("\n".join(memories_text))

            # Full RAG prompt
            print("\n" + "=" * 80)
            print("AUGMENTED PROMPT FOR LLM:")
            print("=" * 80)
            rag_prompt = f"""You are an AI assistant with access to memory from past sessions.

RELEVANT MEMORIES:
{chr(10).join(memories_text)}

USER QUESTION:
{user_question}

Based on the memories above, provide a helpful answer that references what you learned from past sessions."""

            print(rag_prompt)

        else:
            print("No relevant memories found (embeddings not generated)")

    except Exception as e:
        print(f"RAG prompt generation failed: {e}")

    # Summary
    print("\n\n[5] SYSTEM CAPABILITIES")
    print("=" * 80)
    stats = memory.stats()
    print(f"Total facts:     {stats['total_facts']}")
    print(f"Total sessions:  {stats['total_sessions']}")

    cursor = memory.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE entity_type = 'fact'")
    embedding_count = cursor.fetchone()[0]
    print(f"Embeddings:      {embedding_count}")

    print("\n✓ RAG System Features:")
    print("  • Deduplication: Prevents re-processing same sessions")
    print("  • Semantic Search: Find relevant facts by meaning, not keywords")
    print("  • Hybrid Memory: Combines naive (structured) + discerned (semantic) facts")
    print("  • RAG Ready: Use retrieved memories to augment LLM prompts")

    print("\n" + "=" * 80)
    print("Demo complete! See rag_demo.db for the memory database.")
    print("=" * 80)

    memory.close()


if __name__ == '__main__':
    main()
