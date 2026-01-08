"""Pytest configuration and fixtures for vestig tests.

Provides parametrized fixtures to run tests against both SQLite and FalkorDB backends.
"""

import os

# Add src to path for imports
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.db_interface import DatabaseInterface
from vestig.core.db_sqlite import SQLiteDatabase


def get_falkordb_config():
    """Get FalkorDB connection config from environment or defaults."""
    return {
        "host": os.environ.get("VESTIG_FALKORDB_HOST", "192.168.20.4"),
        "port": int(os.environ.get("VESTIG_FALKORDB_PORT", "6379")),
        "graph_name": os.environ.get(
            "VESTIG_FALKORDB_GRAPH", f"vestig_test_{uuid.uuid4().hex[:8]}"
        ),
    }


def falkordb_available() -> bool:
    """Check if FalkorDB is available for testing."""
    try:
        from falkordb import FalkorDB

        config = get_falkordb_config()
        client = FalkorDB(host=config["host"], port=config["port"])
        client.list_graphs()  # Quick connectivity check
        return True
    except Exception:
        return False


# Determine which backends to test
def get_backends():
    """Get list of backends to test based on availability."""
    backends = ["sqlite"]

    # Check if FalkorDB testing is enabled
    if os.environ.get("VESTIG_TEST_FALKORDB", "").lower() in ("1", "true", "yes"):
        if falkordb_available():
            backends.append("falkordb")
        else:
            print("Warning: VESTIG_TEST_FALKORDB enabled but FalkorDB not available")

    return backends


@pytest.fixture(params=get_backends())
def storage(request, tmp_path) -> DatabaseInterface:
    """
    Parametrized fixture providing a storage backend.

    Runs each test once per available backend (SQLite, and optionally FalkorDB).

    Usage:
        def test_something(storage):
            storage.store_entity(...)

    Environment variables:
        VESTIG_TEST_FALKORDB=1  - Enable FalkorDB testing
        VESTIG_FALKORDB_HOST    - FalkorDB host (default: 192.168.20.4)
        VESTIG_FALKORDB_PORT    - FalkorDB port (default: 6379)
        VESTIG_FALKORDB_GRAPH   - Graph name (default: random per test)
    """
    backend = request.param

    if backend == "sqlite":
        db_path = str(tmp_path / "test.db")
        db = SQLiteDatabase(db_path)
        yield db
        db.close()
        # tmp_path cleanup is automatic

    elif backend == "falkordb":
        from vestig.core.db_falkordb import FalkorDBDatabase

        config = get_falkordb_config()
        # Use unique graph name per test to avoid conflicts
        graph_name = f"vestig_test_{uuid.uuid4().hex[:8]}"

        db = FalkorDBDatabase(
            host=config["host"],
            port=config["port"],
            graph_name=graph_name,
        )
        yield db

        # Cleanup: drop the test graph
        try:
            db._graph.delete()
        except Exception:
            pass  # Best effort cleanup
        db.close()


@pytest.fixture
def sqlite_storage(tmp_path) -> DatabaseInterface:
    """Fixture for tests that specifically need SQLite only."""
    db_path = str(tmp_path / "test.db")
    db = SQLiteDatabase(db_path)
    yield db
    db.close()


@pytest.fixture
def falkordb_storage() -> DatabaseInterface:
    """Fixture for tests that specifically need FalkorDB only."""
    if not falkordb_available():
        pytest.skip("FalkorDB not available")

    from vestig.core.db_falkordb import FalkorDBDatabase

    config = get_falkordb_config()
    graph_name = f"vestig_test_{uuid.uuid4().hex[:8]}"

    db = FalkorDBDatabase(
        host=config["host"],
        port=config["port"],
        graph_name=graph_name,
    )
    yield db

    try:
        db._graph.delete()
    except Exception:
        pass
    db.close()


@pytest.fixture
def backend_name(request) -> str:
    """Returns the current backend name for logging/debugging."""
    # Works with the parametrized storage fixture
    if hasattr(request, "param"):
        return request.param
    return "unknown"
