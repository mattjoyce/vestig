"""Pytest configuration and fixtures for vestig tests.

Provides fixtures for FalkorDB backend testing.
"""

import os

# Add src to path for imports
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vestig.core.db_interface import DatabaseInterface


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
    """Get list of backends to test based on availability.

    Returns list of available backends, or pytest.skip if none available
    and VESTIG_REQUIRE_BACKEND is set.
    """
    if falkordb_available():
        return ["falkordb"]

    # No backend available - check if we should fail
    if os.environ.get("VESTIG_REQUIRE_BACKEND", "").lower() in ("1", "true", "yes"):
        pytest.fail("No storage backend available and VESTIG_REQUIRE_BACKEND=1")

    return []


def pytest_collection_modifyitems(config, items):
    """Warn if no storage-dependent tests will run."""
    if not falkordb_available():
        # Count tests that need storage
        storage_tests = [item for item in items if "storage" in item.fixturenames]
        if storage_tests:
            print(f"\n⚠ WARNING: FalkorDB not available - {len(storage_tests)} tests SKIPPED")
            print("  Set VESTIG_REQUIRE_BACKEND=1 to fail instead of skip\n")


@pytest.fixture(params=get_backends())
def storage(request) -> DatabaseInterface:
    """
    Parametrized fixture providing FalkorDB storage backend.

    Usage:
        def test_something(storage):
            storage.store_entity(...)

    Environment variables:
        VESTIG_FALKORDB_HOST    - FalkorDB host (default: 192.168.20.4)
        VESTIG_FALKORDB_PORT    - FalkorDB port (default: 6379)
        VESTIG_FALKORDB_GRAPH   - Graph name (default: random per test)
    """
    from vestig.core.config import load_config
    from vestig.core.db_falkordb import FalkorDBDatabase

    falkor_config = get_falkordb_config()
    # Load full config for embedding dimension
    vestig_config = load_config("config_test.yaml")
    # Use unique graph name per test to avoid conflicts
    graph_name = f"vestig_test_{uuid.uuid4().hex[:8]}"

    db = FalkorDBDatabase(
        host=falkor_config["host"],
        port=falkor_config["port"],
        graph_name=graph_name,
        config=vestig_config,
    )
    yield db

    # Cleanup: drop the test graph
    try:
        db._graph.delete()
    except Exception:
        pass  # Best effort cleanup
    db.close()


# FalkorDB only


@pytest.fixture
def falkordb_storage() -> DatabaseInterface:
    """Fixture for tests that specifically need FalkorDB only."""
    if not falkordb_available():
        pytest.skip("FalkorDB not available")

    from vestig.core.config import load_config
    from vestig.core.db_falkordb import FalkorDBDatabase

    falkor_config = get_falkordb_config()
    # Load full config for embedding dimension
    vestig_config = load_config("config_test.yaml")
    graph_name = f"vestig_test_{uuid.uuid4().hex[:8]}"

    db = FalkorDBDatabase(
        host=falkor_config["host"],
        port=falkor_config["port"],
        graph_name=graph_name,
        config=vestig_config,
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
