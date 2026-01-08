#!/usr/bin/env python3
"""Export SQLite database to FalkorDB.

This tool migrates all data from a SQLite vestig database to FalkorDB.
Data is exported in dependency order to maintain referential integrity:
1. Files
2. Chunks
3. Memories
4. Entities
5. Events
6. Edges

Usage:
    python -m vestig.tools.export_to_falkor \
        --sqlite ./vestig.db \
        --host 192.168.20.4 \
        --port 6379 \
        --graph vestig

    # Or with dry-run to see what would be migrated:
    python -m vestig.tools.export_to_falkor \
        --sqlite ./vestig.db \
        --dry-run
"""

import argparse
import sqlite3
import sys


def get_sqlite_counts(conn: sqlite3.Connection) -> dict:
    """Get counts of all entities in SQLite database."""
    counts = {}

    for table in ["files", "chunks", "memories", "entities", "memory_events", "edges"]:
        try:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            counts[table] = 0

    return counts


def export_files(conn: sqlite3.Connection, falkor_db, batch_size: int = 100) -> int:
    """Export files table."""
    cursor = conn.execute(
        """
        SELECT file_id, path, created_at, ingested_at, file_hash, metadata
        FROM files
        """
    )

    count = 0
    for row in cursor:
        file_id, path, created_at, ingested_at, file_hash, metadata = row

        falkor_db._graph.query(
            """
            CREATE (f:File {
                id: $id,
                path: $path,
                created_at: $created_at,
                ingested_at: $ingested_at,
                file_hash: $file_hash,
                metadata: $metadata
            })
            """,
            {
                "id": file_id,
                "path": path,
                "created_at": created_at,
                "ingested_at": ingested_at,
                "file_hash": file_hash,
                "metadata": metadata,
            },
        )
        count += 1

        if count % batch_size == 0:
            print(f"  Exported {count} files...")

    return count


def export_chunks(conn: sqlite3.Connection, falkor_db, batch_size: int = 100) -> int:
    """Export chunks table."""
    cursor = conn.execute(
        """
        SELECT chunk_id, file_id, start, length, sequence, created_at
        FROM chunks
        """
    )

    count = 0
    for row in cursor:
        chunk_id, file_id, start, length, sequence, created_at = row

        falkor_db._graph.query(
            """
            CREATE (c:Chunk {
                id: $id,
                file_id: $file_id,
                start: $start,
                length: $length,
                sequence: $sequence,
                created_at: $created_at
            })
            """,
            {
                "id": chunk_id,
                "file_id": file_id,
                "start": start,
                "length": length,
                "sequence": sequence,
                "created_at": created_at,
            },
        )
        count += 1

        if count % batch_size == 0:
            print(f"  Exported {count} chunks...")

    return count


def export_memories(conn: sqlite3.Connection, falkor_db, batch_size: int = 100) -> int:
    """Export memories table (without chunk_id - use edges instead)."""
    cursor = conn.execute(
        """
        SELECT id, content, content_embedding, content_hash, created_at, metadata,
               t_valid, t_invalid, t_created, t_expired, temporal_stability,
               last_seen_at, reinforce_count, kind
        FROM memories
        """
    )

    count = 0
    for row in cursor:
        (
            id,
            content,
            content_embedding,
            content_hash,
            created_at,
            metadata,
            t_valid,
            t_invalid,
            t_created,
            t_expired,
            temporal_stability,
            last_seen_at,
            reinforce_count,
            kind,
        ) = row

        falkor_db._graph.query(
            """
            CREATE (m:Memory {
                id: $id,
                content: $content,
                content_embedding: $embedding,
                content_hash: $hash,
                created_at: $created_at,
                metadata: $metadata,
                kind: $kind,
                t_valid: $t_valid,
                t_invalid: $t_invalid,
                t_created: $t_created,
                t_expired: $t_expired,
                temporal_stability: $temporal_stability,
                last_seen_at: $last_seen_at,
                reinforce_count: $reinforce_count
            })
            """,
            {
                "id": id,
                "content": content,
                "embedding": content_embedding,  # Already JSON string
                "hash": content_hash,
                "created_at": created_at,
                "metadata": metadata,  # Already JSON string
                "kind": kind or "MEMORY",
                "t_valid": t_valid,
                "t_invalid": t_invalid,
                "t_created": t_created,
                "t_expired": t_expired,
                "temporal_stability": temporal_stability or "unknown",
                "last_seen_at": last_seen_at,
                "reinforce_count": reinforce_count or 0,
            },
        )
        count += 1

        if count % batch_size == 0:
            print(f"  Exported {count} memories...")

    return count


def export_entities(conn: sqlite3.Connection, falkor_db, batch_size: int = 100) -> int:
    """Export entities table (without chunk_id - use edges instead)."""
    cursor = conn.execute(
        """
        SELECT id, entity_type, canonical_name, norm_key, created_at,
               embedding, expired_at, merged_into
        FROM entities
        """
    )

    count = 0
    for row in cursor:
        (
            id,
            entity_type,
            canonical_name,
            norm_key,
            created_at,
            embedding,
            expired_at,
            merged_into,
        ) = row

        falkor_db._graph.query(
            """
            CREATE (e:Entity {
                id: $id,
                entity_type: $entity_type,
                canonical_name: $canonical_name,
                norm_key: $norm_key,
                created_at: $created_at,
                embedding: $embedding,
                expired_at: $expired_at,
                merged_into: $merged_into
            })
            """,
            {
                "id": id,
                "entity_type": entity_type,
                "canonical_name": canonical_name,
                "norm_key": norm_key,
                "created_at": created_at,
                "embedding": embedding,
                "expired_at": expired_at,
                "merged_into": merged_into,
            },
        )
        count += 1

        if count % batch_size == 0:
            print(f"  Exported {count} entities...")

    return count


def export_events(conn: sqlite3.Connection, falkor_db, batch_size: int = 100) -> int:
    """Export memory_events table."""
    cursor = conn.execute(
        """
        SELECT event_id, memory_id, event_type, occurred_at, source,
               actor, artifact_ref, payload_json
        FROM memory_events
        """
    )

    count = 0
    for row in cursor:
        (
            event_id,
            memory_id,
            event_type,
            occurred_at,
            source,
            actor,
            artifact_ref,
            payload_json,
        ) = row

        # Create event node and AFFECTS edge to memory
        falkor_db._graph.query(
            """
            MATCH (m:Memory {id: $memory_id})
            CREATE (e:Event {
                id: $event_id,
                memory_id: $memory_id,
                event_type: $event_type,
                occurred_at: $occurred_at,
                source: $source,
                actor: $actor,
                artifact_ref: $artifact_ref,
                payload: $payload
            })-[:AFFECTS]->(m)
            """,
            {
                "memory_id": memory_id,
                "event_id": event_id,
                "event_type": event_type,
                "occurred_at": occurred_at,
                "source": source,
                "actor": actor,
                "artifact_ref": artifact_ref,
                "payload": payload_json,
            },
        )
        count += 1

        if count % batch_size == 0:
            print(f"  Exported {count} events...")

    return count


def export_edges(conn: sqlite3.Connection, falkor_db, batch_size: int = 100) -> int:
    """Export edges table."""
    cursor = conn.execute(
        """
        SELECT edge_id, from_node, to_node, edge_type, weight,
               confidence, evidence, t_valid, t_invalid, t_created, t_expired
        FROM edges
        """
    )

    count = 0
    errors = 0
    for row in cursor:
        (
            edge_id,
            from_node,
            to_node,
            edge_type,
            weight,
            confidence,
            evidence,
            t_valid,
            t_invalid,
            t_created,
            t_expired,
        ) = row

        try:
            # Create edge with dynamic type
            falkor_db._graph.query(
                f"""
                MATCH (a {{id: $from}}), (b {{id: $to}})
                CREATE (a)-[r:{edge_type} {{
                    edge_id: $edge_id,
                    weight: $weight,
                    confidence: $confidence,
                    evidence: $evidence,
                    t_valid: $t_valid,
                    t_invalid: $t_invalid,
                    t_created: $t_created,
                    t_expired: $t_expired
                }}]->(b)
                """,
                {
                    "from": from_node,
                    "to": to_node,
                    "edge_id": edge_id,
                    "weight": weight,
                    "confidence": confidence,
                    "evidence": evidence,
                    "t_valid": t_valid,
                    "t_invalid": t_invalid,
                    "t_created": t_created,
                    "t_expired": t_expired,
                },
            )
            count += 1
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Warning: Failed to create edge {edge_id}: {e}")

        if count % batch_size == 0:
            print(f"  Exported {count} edges...")

    if errors > 0:
        print(f"  Warning: {errors} edges failed to export (missing nodes?)")

    return count


def verify_migration(conn: sqlite3.Connection, falkor_db) -> bool:
    """Verify data integrity after migration."""
    sqlite_counts = get_sqlite_counts(conn)

    print("\nVerification:")
    print("-" * 50)

    all_match = True

    # Count memories
    falkor_memories = falkor_db.count_memories()
    match = sqlite_counts["memories"] == falkor_memories
    status = "OK" if match else "MISMATCH"
    print(f"  Memories: SQLite={sqlite_counts['memories']}, FalkorDB={falkor_memories} [{status}]")
    all_match = all_match and match

    # Count entities
    falkor_entities = falkor_db.count_entities()
    match = sqlite_counts["entities"] == falkor_entities
    status = "OK" if match else "MISMATCH"
    print(f"  Entities: SQLite={sqlite_counts['entities']}, FalkorDB={falkor_entities} [{status}]")
    all_match = all_match and match

    # Count edges (excluding AFFECTS edges which are created for events)
    # FalkorDB has additional AFFECTS edges from Event nodes
    falkor_edges = falkor_db.count_edges()
    falkor_affects = falkor_db._graph.ro_query(
        "MATCH ()-[r:AFFECTS]->() RETURN COUNT(r)"
    )
    affects_count = falkor_affects.result_set[0][0] if falkor_affects.result_set else 0
    falkor_edges_without_affects = falkor_edges - affects_count

    match = sqlite_counts["edges"] == falkor_edges_without_affects
    status = "OK" if match else "MISMATCH"
    print(f"  Edges: SQLite={sqlite_counts['edges']}, FalkorDB={falkor_edges_without_affects} "
          f"(+{affects_count} AFFECTS) [{status}]")
    all_match = all_match and match

    # Count events
    falkor_events = falkor_db._graph.ro_query("MATCH (e:Event) RETURN COUNT(e)")
    falkor_event_count = falkor_events.result_set[0][0] if falkor_events.result_set else 0
    match = sqlite_counts["memory_events"] == falkor_event_count
    status = "OK" if match else "MISMATCH"
    print(f"  Events: SQLite={sqlite_counts['memory_events']}, FalkorDB={falkor_event_count} [{status}]")
    all_match = all_match and match

    return all_match


def drop_graph(falkor_host: str, falkor_port: int, graph_name: str) -> bool:
    """Drop (delete) an existing FalkorDB graph.

    Args:
        falkor_host: FalkorDB server host
        falkor_port: FalkorDB server port
        graph_name: Name of graph to drop

    Returns:
        True if successful, False otherwise
    """
    try:
        from falkordb import FalkorDB

        client = FalkorDB(host=falkor_host, port=falkor_port)
        graph = client.select_graph(graph_name)

        # Delete the graph
        graph.delete()
        print(f"Dropped graph '{graph_name}' successfully")
        return True
    except Exception as e:
        print(f"Warning: Failed to drop graph '{graph_name}': {e}")
        return False


def migrate_database(
    sqlite_path: str,
    falkor_host: str,
    falkor_port: int,
    graph_name: str,
    batch_size: int = 100,
    dry_run: bool = False,
    drop_graph_first: bool = False,
) -> bool:
    """Full migration from SQLite to FalkorDB.

    Args:
        sqlite_path: Path to SQLite database
        falkor_host: FalkorDB server host
        falkor_port: FalkorDB server port
        graph_name: Name of graph to create
        batch_size: Batch size for progress reporting
        dry_run: If True, only show what would be migrated
        drop_graph_first: If True, drop existing graph before migration

    Returns:
        True if migration successful, False otherwise
    """
    print("=" * 60)
    print("VESTIG SQLite to FalkorDB Migration")
    print("=" * 60)
    print(f"Source: {sqlite_path}")
    print(f"Target: {falkor_host}:{falkor_port}/{graph_name}")
    print(f"Batch size: {batch_size}")
    print(f"Dry run: {dry_run}")
    print(f"Drop graph first: {drop_graph_first}")
    print()

    # Connect to SQLite
    print("Connecting to SQLite...")
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    # Get counts
    counts = get_sqlite_counts(conn)
    print("\nSource database contents:")
    print(f"  Files: {counts['files']}")
    print(f"  Chunks: {counts['chunks']}")
    print(f"  Memories: {counts['memories']}")
    print(f"  Entities: {counts['entities']}")
    print(f"  Events: {counts['memory_events']}")
    print(f"  Edges: {counts['edges']}")

    if dry_run:
        print("\n[DRY RUN] Would migrate the above data to FalkorDB.")
        if drop_graph_first:
            print(f"[DRY RUN] Would drop graph '{graph_name}' before migration.")
        conn.close()
        return True

    # Drop existing graph if requested
    if drop_graph_first:
        print(f"\nDropping existing graph '{graph_name}'...")
        drop_graph(falkor_host, falkor_port, graph_name)

    # Connect to FalkorDB
    print("\nConnecting to FalkorDB...")
    from vestig.core.db_falkordb import FalkorDBDatabase

    try:
        falkor_db = FalkorDBDatabase(
            host=falkor_host,
            port=falkor_port,
            graph_name=graph_name,
        )
    except Exception as e:
        print(f"ERROR: Failed to connect to FalkorDB: {e}")
        conn.close()
        return False

    # Export in order
    print("\nExporting data...")

    print("\n1. Exporting files...")
    file_count = export_files(conn, falkor_db, batch_size)
    print(f"   Exported {file_count} files")

    print("\n2. Exporting chunks...")
    chunk_count = export_chunks(conn, falkor_db, batch_size)
    print(f"   Exported {chunk_count} chunks")

    print("\n3. Exporting memories...")
    memory_count = export_memories(conn, falkor_db, batch_size)
    print(f"   Exported {memory_count} memories")

    print("\n4. Exporting entities...")
    entity_count = export_entities(conn, falkor_db, batch_size)
    print(f"   Exported {entity_count} entities")

    print("\n5. Exporting events...")
    event_count = export_events(conn, falkor_db, batch_size)
    print(f"   Exported {event_count} events")

    print("\n6. Exporting edges...")
    edge_count = export_edges(conn, falkor_db, batch_size)
    print(f"   Exported {edge_count} edges")

    # Verify
    all_ok = verify_migration(conn, falkor_db)

    # Cleanup
    conn.close()

    print("\n" + "=" * 60)
    if all_ok:
        print("MIGRATION COMPLETE - All counts match!")
    else:
        print("MIGRATION COMPLETE - Some counts mismatch (check warnings)")
    print("=" * 60)

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Export SQLite database to FalkorDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sqlite",
        required=True,
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="FalkorDB host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6379,
        help="FalkorDB port (default: 6379)",
    )
    parser.add_argument(
        "--graph",
        default="vestig",
        help="FalkorDB graph name (default: vestig)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for progress reporting (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating",
    )
    parser.add_argument(
        "--drop-graph",
        action="store_true",
        help="Drop (delete) the target graph before migration (creates clean graph)",
    )

    args = parser.parse_args()

    success = migrate_database(
        sqlite_path=args.sqlite,
        falkor_host=args.host,
        falkor_port=args.port,
        graph_name=args.graph,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        drop_graph_first=args.drop_graph,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
