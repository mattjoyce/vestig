"""Test for Issue #3: CLI Simplification with Source node creation."""

import tempfile

from vestig.core.cli import build_runtime
from vestig.core.commitment import commit_memory
from vestig.core.config import load_config
from vestig.core.models import SourceNode


def test_memory_add_creates_source_node():
    """Test that memory add creates Source nodes as per Issue #3."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        config = load_config("config_test.yaml")
        config["storage"]["db_path"] = db_path

        storage, embedding_engine, event_storage, _ = build_runtime(config)

        # Simulate what cmd_add() does now
        agent_name = "claude-code"
        session_id = "test-session-123"

        # Create Source node (new behavior)
        source_node = SourceNode.from_agent(
            agent=agent_name,
            session_id=session_id,
            metadata={"command": "memory add"},
        )
        source_id = storage.store_source(source_node)

        # Commit memory with source_id
        outcome = commit_memory(
            content="Test memory for issue #3",
            storage=storage,
            embedding_engine=embedding_engine,
            source="manual",
            event_storage=event_storage,
            m4_config=config.get("m4", {}),
            source_id=source_id,
        )

        # Verify outcome
        assert outcome.outcome == "INSERTED_NEW"
        memory_id = outcome.memory_id

        # Verify Source node was created
        retrieved_source = storage.get_source(source_id)
        assert retrieved_source is not None
        assert retrieved_source.source_type == "agentic"
        assert retrieved_source.agent == agent_name
        assert retrieved_source.session_id == session_id

        # Verify Memory is linked to Source
        memory = storage.get_memory(memory_id)
        assert memory is not None
        assert memory.source_id == source_id

        # Verify we can query by agent
        sources_by_agent = storage.get_sources_by_agent(agent_name)
        assert len(sources_by_agent) == 1
        assert sources_by_agent[0].source_id == source_id

        # Verify we can query by session
        sources_by_session = storage.get_sources_by_session(session_id)
        assert len(sources_by_session) == 1
        assert sources_by_session[0].source_id == source_id

        print("✓ Test passed: memory add creates Source nodes correctly")

        storage.close()

    finally:
        import os

        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    test_memory_add_creates_source_node()
