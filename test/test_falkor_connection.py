"""Test FalkorDB connection and basic operations"""

from falkordb import FalkorDB


def test_connection():
    """Test basic FalkorDB connectivity"""
    print("Connecting to FalkorDB at 192.168.20.4:6379...")

    try:
        client = FalkorDB(host="192.168.20.4", port=6379)
        graph = client.select_graph("vestig_test")
        print("✓ Connected to FalkorDB")

        # Test node creation
        print("\nTesting node creation...")
        result = graph.query(
            "CREATE (n:Test {name: 'connection_test', timestamp: timestamp()}) RETURN n"
        )
        print(f"✓ Node created: {result.result_set}")

        # Test edge creation
        print("\nTesting edge creation...")
        result = graph.query("""
            CREATE (a:TestNode {id: 'test_a'})
              -[:TEST_EDGE {weight: 1.0}]->
              (b:TestNode {id: 'test_b'})
            RETURN a, b
        """)
        print(f"✓ Edge created: {result.result_set}")

        # Test Cypher query
        print("\nTesting Cypher query...")
        result = graph.query("MATCH (n:Test) RETURN n LIMIT 5")
        print(f"✓ Query successful, found {len(result.result_set)} nodes")

        # Test vector index support (if available)
        print("\nTesting vector search capabilities...")
        try:
            # Try to get vector index info
            result = graph.query("CALL db.idx.vector.list()")
            print(f"✓ Vector search available: {result.result_set}")
        except Exception as e:
            print(f"⚠ Vector search not available or requires setup: {e}")

        # Test graph statistics
        print("\nGetting graph statistics...")
        result = graph.query("MATCH (n) RETURN count(n) as node_count")
        node_count = result.result_set[0][0] if result.result_set else 0
        print(f"✓ Total nodes in graph: {node_count}")

        result = graph.query("MATCH ()-[r]->() RETURN count(r) as edge_count")
        edge_count = result.result_set[0][0] if result.result_set else 0
        print(f"✓ Total edges in graph: {edge_count}")

        # Cleanup
        print("\nCleaning up test data...")
        graph.query("MATCH (n:Test) DELETE n")
        graph.query("MATCH (n:TestNode) DELETE n")
        print("✓ Cleanup complete")

        print("\n" + "=" * 60)
        print("SUCCESS: FalkorDB connection test passed!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)
