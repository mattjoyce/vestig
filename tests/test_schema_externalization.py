#!/usr/bin/env python3
"""Tests for M0 schema externalization"""

import os
import sys
import sqlite3
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vestig.core.storage import MemoryStorage


def test_fresh_db_uses_schema_sql():
    """Test that fresh databases use schema.sql"""
    print("Test 1: Fresh DB uses schema.sql")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "fresh.db"

        # Create fresh database
        storage = MemoryStorage(str(db_path))

        # Verify all tables exist
        cursor = storage.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert tables == {"edges", "entities", "memories", "memory_events"}
        print(f"✅ All tables created: {sorted(tables)}")

        # Verify all critical indexes exist
        cursor = storage.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        indexes = {row[0] for row in cursor.fetchall()}

        # Check for key indexes (not exhaustive, but representative)
        assert "idx_content_hash" in indexes
        assert "idx_memories_kind" in indexes
        assert "idx_entities_norm_key" in indexes
        assert "idx_edges_unique" in indexes
        print(f"✅ Key indexes created: {len(indexes)} total indexes")

        # Verify memories table has all M4 columns
        cursor = storage.conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        expected_columns = {
            "id", "content", "content_embedding", "created_at", "metadata",
            "content_hash", "t_valid", "t_invalid", "t_created", "t_expired",
            "temporal_stability", "last_seen_at", "reinforce_count", "kind"
        }
        assert columns == expected_columns
        print(f"✅ Memories table has all {len(columns)} M4 columns")

        storage.close()
    print()


def test_existing_db_uses_migration():
    """Test that existing databases use migration logic"""
    print("Test 2: Existing DB uses migration")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "existing.db"

        # Create database with old schema (pre-M4: no 'kind' column)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL,
                content_hash TEXT,
                t_valid TEXT,
                t_invalid TEXT,
                t_created TEXT,
                t_expired TEXT,
                temporal_stability TEXT DEFAULT 'unknown',
                last_seen_at TEXT,
                reinforce_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE memory_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                source TEXT NOT NULL,
                actor TEXT,
                artifact_ref TEXT,
                payload_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                norm_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expired_at TEXT,
                merged_into TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE edges (
                edge_id TEXT PRIMARY KEY,
                from_node TEXT NOT NULL,
                to_node TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                confidence REAL,
                evidence TEXT,
                t_valid TEXT,
                t_invalid TEXT,
                t_created TEXT,
                t_expired TEXT
            )
        """)
        conn.commit()
        conn.close()
        print("✅ Created pre-M4 database (no 'kind' column)")

        # Open with MemoryStorage (should trigger migration)
        storage = MemoryStorage(str(db_path))

        # Verify kind column was added
        cursor = storage.conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "kind" in columns
        print("✅ Migration added 'kind' column")

        # Verify no errors occurred
        assert storage.conn is not None
        print("✅ Database opened successfully after migration")

        storage.close()
    print()


def test_schema_validation_passes_after_migration():
    """Test that schema validation passes after successful migration"""
    print("Test 3: Schema validation passes after migration")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "old.db"

        # Create a minimal old database (just memories table)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        print("✅ Created minimal old database")

        # Open with MemoryStorage - migration should add missing columns/tables
        # and validation should pass
        try:
            storage = MemoryStorage(str(db_path))
            print("✅ Migration and validation succeeded")

            # Verify schema is now complete
            cursor = storage.conn.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "kind" in columns
            assert "content_hash" in columns
            print(f"✅ Schema validated with {len(columns)} columns")

            storage.close()
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)
    print()


def test_schema_sql_is_valid_sqlite():
    """Test that schema.sql is valid SQLite syntax"""
    print("Test 4: schema.sql is valid SQLite")
    print("-" * 40)

    # Read schema.sql
    schema_path = Path(__file__).parent.parent / "src" / "vestig" / "core" / "schema.sql"
    assert schema_path.exists(), "schema.sql not found"
    print(f"✅ Found schema.sql at {schema_path}")

    schema_sql = schema_path.read_text()

    # Apply to in-memory DB
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql)
    print("✅ schema.sql executed without syntax errors")

    # Verify all tables created
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cursor.fetchall()}
    assert tables == {"edges", "entities", "memories", "memory_events"}
    print(f"✅ All tables created: {sorted(tables)}")

    # Verify indexes created
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
    )
    indexes = {row[0] for row in cursor.fetchall()}

    # Check for key indexes
    assert "idx_content_hash" in indexes
    assert "idx_memories_kind" in indexes
    assert "idx_entities_norm_key" in indexes
    assert "idx_edges_unique" in indexes
    print(f"✅ Key indexes created: {len(indexes)} total indexes")

    conn.close()
    print()


def test_fresh_and_migrated_schemas_match():
    """Test that fresh schema.sql and migrated schema produce same structure"""
    print("Test 5: Fresh and migrated schemas match")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        fresh_db = Path(tmpdir) / "fresh.db"
        migrated_db = Path(tmpdir) / "migrated.db"

        # Create fresh database from schema.sql
        storage_fresh = MemoryStorage(str(fresh_db))
        print("✅ Created fresh database from schema.sql")

        # Create minimal old database and migrate it
        conn = sqlite3.connect(str(migrated_db))
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE memory_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                source TEXT NOT NULL,
                actor TEXT,
                artifact_ref TEXT,
                payload_json TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        storage_migrated = MemoryStorage(str(migrated_db))
        print("✅ Created and migrated old database")

        # Compare column sets in memories table
        cursor_fresh = storage_fresh.conn.execute("PRAGMA table_info(memories)")
        columns_fresh = {row[1] for row in cursor_fresh.fetchall()}

        cursor_migrated = storage_migrated.conn.execute("PRAGMA table_info(memories)")
        columns_migrated = {row[1] for row in cursor_migrated.fetchall()}

        assert columns_fresh == columns_migrated, "Column mismatch between fresh and migrated"
        print(f"✅ Column sets match: {len(columns_fresh)} columns")

        # Compare tables
        cursor_fresh = storage_fresh.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables_fresh = {row[0] for row in cursor_fresh.fetchall()}

        cursor_migrated = storage_migrated.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables_migrated = {row[0] for row in cursor_migrated.fetchall()}

        assert tables_fresh == tables_migrated, "Table mismatch between fresh and migrated"
        print(f"✅ Table sets match: {sorted(tables_fresh)}")

        storage_fresh.close()
        storage_migrated.close()
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("M0: Schema Externalization Tests")
    print("=" * 60)
    print()

    test_fresh_db_uses_schema_sql()
    test_existing_db_uses_migration()
    test_schema_validation_passes_after_migration()
    test_schema_sql_is_valid_sqlite()
    test_fresh_and_migrated_schemas_match()

    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
