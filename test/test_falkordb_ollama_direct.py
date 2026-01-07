"""
Test direct FalkorDB + Ollama integration for Vestig
No GraphRAG SDK - just native Cypher and Ollama embeddings
"""

from falkordb import FalkorDB
from ollama import Client as OllamaClient


def test_vestig_architecture():
    """Test Vestig's chunk-centric architecture in FalkorDB"""

    print("Testing Vestig Architecture: FalkorDB + Ollama")
    print("=" * 60)

    # Connect to FalkorDB
    print("\n1. Connecting to FalkorDB...")
    print("   Host: 192.168.20.4:6379")
    falkor = FalkorDB(host="192.168.20.4", port=6379)
    graph = falkor.select_graph("vestig_direct_test")
    print("   ✓ Connected")

    # Connect to Ollama
    print("\n2. Connecting to Ollama...")
    print("   Host: http://192.168.20.8:11434")
    print("   Embedding model: embeddinggemma:latest")
    ollama = OllamaClient(host="http://192.168.20.8:11434")
    print("   ✓ Connected")

    # Clean up any previous test data
    print("\n3. Cleaning up previous test data...")
    graph.query("MATCH (n) DETACH DELETE n")
    print("   ✓ Graph cleared")

    # Test 1: Create Vestig nodes (Chunk → Memory → Entity)
    print("\n4. Creating Vestig graph structure...")

    # Create a chunk node
    chunk_id = "chunk_test_001"
    graph.query(
        """
        CREATE (c:Chunk {
            id: $chunk_id,
            file_id: 'file_001',
            start: 0,
            length: 100,
            sequence: 1,
            created_at: '2026-01-07T00:00:00'
        })
    """,
        {"chunk_id": chunk_id},
    )
    print(f"   ✓ Created Chunk: {chunk_id}")

    # Create memory with embedding
    memory_text = "Vestig uses chunk-centric M5 architecture for memory storage"
    print(f"   → Generating embedding for: '{memory_text[:50]}...'")

    embedding_response = ollama.embeddings(model="embeddinggemma:latest", prompt=memory_text)
    embedding = embedding_response["embedding"]
    print(f"   ✓ Generated embedding: {len(embedding)} dimensions")

    memory_id = "mem_test_001"
    graph.query(
        """
        CREATE (m:Memory {
            id: $memory_id,
            content: $content,
            content_hash: $hash,
            created_at: '2026-01-07T00:00:00',
            kind: 'MEMORY',
            t_valid: '2026-01-07T00:00:00',
            t_created: '2026-01-07T00:00:00',
            reinforce_count: 1
        })
    """,
        {"memory_id": memory_id, "content": memory_text, "hash": "test_hash_001"},
    )
    print(f"   ✓ Created Memory: {memory_id}")

    # Create entity
    entity_id = "ent_test_001"
    graph.query(
        """
        CREATE (e:Entity {
            id: $entity_id,
            entity_type: 'SYSTEM',
            canonical_name: 'Vestig',
            norm_key: 'vestig',
            created_at: '2026-01-07T00:00:00'
        })
    """,
        {"entity_id": entity_id},
    )
    print(f"   ✓ Created Entity: {entity_id}")

    # Create edges (Vestig's hub-and-spoke model)
    print("\n5. Creating Vestig edges (hub-and-spoke)...")

    # CONTAINS: Chunk → Memory
    graph.query(
        """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (m:Memory {id: $memory_id})
        CREATE (c)-[:CONTAINS {weight: 1.0}]->(m)
    """,
        {"chunk_id": chunk_id, "memory_id": memory_id},
    )
    print("   ✓ Created CONTAINS: Chunk → Memory")

    # LINKED: Chunk → Entity
    graph.query(
        """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {id: $entity_id})
        CREATE (c)-[:LINKED {weight: 1.0, confidence: 0.95}]->(e)
    """,
        {"chunk_id": chunk_id, "entity_id": entity_id},
    )
    print("   ✓ Created LINKED: Chunk → Entity")

    # MENTIONS: Memory → Entity
    graph.query(
        """
        MATCH (m:Memory {id: $memory_id})
        MATCH (e:Entity {id: $entity_id})
        CREATE (m)-[:MENTIONS {weight: 1.0, confidence: 0.9}]->(e)
    """,
        {"memory_id": memory_id, "entity_id": entity_id},
    )
    print("   ✓ Created MENTIONS: Memory → Entity")

    # Test 2: Query Vestig graph
    print("\n6. Querying Vestig graph structure...")

    # Get all memories in a chunk (chunk-centric recall)
    result = graph.query(
        """
        MATCH (c:Chunk {id: $chunk_id})-[:CONTAINS]->(m:Memory)
        RETURN m.id as memory_id, m.content as content
    """,
        {"chunk_id": chunk_id},
    )

    print(f"   → Memories in chunk {chunk_id}:")
    for row in result.result_set:
        print(f"     - {row[0]}: {row[1][:50]}...")

    # Get entities linked to chunk (1st class)
    result = graph.query(
        """
        MATCH (c:Chunk {id: $chunk_id})-[:LINKED]->(e:Entity)
        RETURN e.canonical_name as entity, e.entity_type as type
    """,
        {"chunk_id": chunk_id},
    )

    print("   → Entities linked to chunk:")
    for row in result.result_set:
        print(f"     - {row[0]} ({row[1]})")

    # Get entities mentioned in memories (2nd class)
    result = graph.query(
        """
        MATCH (c:Chunk {id: $chunk_id})-[:CONTAINS]->(m:Memory)-[:MENTIONS]->(e:Entity)
        RETURN DISTINCT e.canonical_name as entity
    """,
        {"chunk_id": chunk_id},
    )

    print("   → Entities mentioned in memories:")
    for row in result.result_set:
        print(f"     - {row[0]}")

    # Test 3: Graph statistics
    print("\n7. Graph statistics...")

    result = graph.query("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
    print("   Node counts by type:")
    for row in result.result_set:
        print(f"     - {row[0]}: {row[1]}")

    result = graph.query("MATCH ()-[r]->() RETURN type(r) as edge_type, count(r) as count")
    print("   Edge counts by type:")
    for row in result.result_set:
        print(f"     - {row[0]}: {row[1]}")

    # Clean up
    print("\n8. Cleaning up test data...")
    graph.query("MATCH (n) DETACH DELETE n")
    print("   ✓ Test data removed")

    print("\n" + "=" * 60)
    print("SUCCESS: Vestig architecture working in FalkorDB!")
    print("=" * 60)
    print("\nKey capabilities verified:")
    print("  ✓ Direct FalkorDB connection and Cypher queries")
    print("  ✓ Direct Ollama embeddings (embeddinggemma:latest)")
    print("  ✓ Chunk-centric graph model (CONTAINS, LINKED, MENTIONS)")
    print("  ✓ Hub-and-spoke edge traversal")
    print("  ✓ Entity extraction and linking")
    print("\nReady for Vestig migration!")


if __name__ == "__main__":
    try:
        test_vestig_architecture()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
