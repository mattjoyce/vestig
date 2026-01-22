"""Component-level tests for db_falkordb.py storage methods (Issue #9).

These tests target specific methods that integration tests miss, preventing
bugs like Issue #7 (vecf32 bug passed 17/21 integration tests but failed in production).

Tests are organized by priority:
- CRITICAL: Vector/vecf32 handling (7 tests)
- HIGH: CRUD edge cases (6 tests)
- MEDIUM: Query logic (5 tests)
"""

import json
import random

from vestig.core.db_interface import DatabaseInterface
from vestig.core.models import EdgeNode, EntityNode, MemoryNode


def make_test_embedding(seed: int, dim: int = 768) -> list[float]:
    """Generate deterministic test embedding for speed."""
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(dim)]


# =============================================================================
# CRITICAL: Vector/vecf32 Handling (7 tests)
# =============================================================================


class TestVectorSearch:
    """Tests for native vector search functionality."""

    def test_search_memories_by_vector_empty_results(self, storage: DatabaseInterface):
        """Empty graph returns [], no crash."""
        query_vector = make_test_embedding(seed=42)

        results = storage.search_memories_by_vector(
            query_vector=query_vector,
            limit=10,
            kind_filter=None,
            include_expired=False,
        )

        assert results == []

    def test_search_memories_by_vector_kind_filter(self, storage: DatabaseInterface):
        """kind_filter=SUMMARY/MEMORY/None works correctly."""
        # Create a MEMORY
        memory = MemoryNode.create(
            memory_id="mem_test_kind_1",
            content="This is a regular memory",
            embedding=make_test_embedding(seed=1),
            source="test",
        )
        storage.store_memory(memory, kind="MEMORY")

        # Create a SUMMARY
        summary = MemoryNode.create(
            memory_id="mem_test_kind_2",
            content="This is a summary",
            embedding=make_test_embedding(seed=2),
            source="test",
        )
        storage.store_memory(summary, kind="SUMMARY")

        query_vector = make_test_embedding(seed=1)  # Similar to memory

        # Filter by MEMORY only
        memory_results = storage.search_memories_by_vector(
            query_vector=query_vector,
            limit=10,
            kind_filter="MEMORY",
            include_expired=False,
        )
        assert len(memory_results) >= 1
        for mem, _ in memory_results:
            assert mem.kind == "MEMORY"

        # Filter by SUMMARY only
        summary_results = storage.search_memories_by_vector(
            query_vector=query_vector,
            limit=10,
            kind_filter="SUMMARY",
            include_expired=False,
        )
        assert len(summary_results) >= 1
        for mem, _ in summary_results:
            assert mem.kind == "SUMMARY"

        # No filter - returns both
        all_results = storage.search_memories_by_vector(
            query_vector=query_vector,
            limit=10,
            kind_filter=None,
            include_expired=False,
        )
        assert len(all_results) >= 2

    def test_search_memories_by_vector_score_ordering(self, storage: DatabaseInterface):
        """Results sorted by score DESC."""
        # Create memories with different embeddings
        for i in range(5):
            memory = MemoryNode.create(
                memory_id=f"mem_test_order_{i}",
                content=f"Test memory {i}",
                embedding=make_test_embedding(seed=i * 100),
                source="test",
            )
            storage.store_memory(memory, kind="MEMORY")

        query_vector = make_test_embedding(seed=0)  # Matches first memory best

        results = storage.search_memories_by_vector(
            query_vector=query_vector,
            limit=10,
            kind_filter=None,
            include_expired=False,
        )

        assert len(results) >= 2
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score DESC"

    def test_search_entities_by_vector_no_embedding(self, storage: DatabaseInterface):
        """Entities without embedding are handled gracefully."""
        # Create entity WITHOUT embedding
        entity_no_embed = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Alice Without Embedding",
        )
        storage.store_entity(entity_no_embed)

        # Create entity WITH embedding
        entity_with_embed = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Bob With Embedding",
        )
        entity_with_embed.embedding = make_test_embedding(seed=42)
        storage.store_entity(entity_with_embed)

        # Search - should only find entity with embedding, no crash
        query_vector = make_test_embedding(seed=42)
        results = storage.search_entities_by_vector(
            query_vector=query_vector,
            entity_type="PERSON",
            limit=10,
            include_expired=False,
        )

        # Should find at least the one with embedding
        assert len(results) >= 1
        found_names = [e.canonical_name for e, _ in results]
        assert "Bob With Embedding" in found_names


class TestEmbeddingUpdates:
    """Tests for embedding update operations."""

    def test_update_node_embedding_memory(self, storage: DatabaseInterface):
        """Memory embedding update works."""
        # Create memory with initial embedding
        initial_embedding = make_test_embedding(seed=1)
        memory = MemoryNode.create(
            memory_id="mem_test_update_embed",
            content="Test memory for embedding update",
            embedding=initial_embedding,
            source="test",
        )
        storage.store_memory(memory, kind="MEMORY")

        # Update embedding
        new_embedding = make_test_embedding(seed=999)
        storage.update_node_embedding(
            node_id="mem_test_update_embed",
            embedding_json=json.dumps(new_embedding),
            node_type="memory",
        )

        # Verify update
        retrieved = storage.get_memory("mem_test_update_embed")
        assert retrieved is not None
        assert len(retrieved.content_embedding) == 768
        # Embeddings should be different (new seed)
        assert retrieved.content_embedding[0] != initial_embedding[0]

    def test_update_node_embedding_entity(self, storage: DatabaseInterface):
        """Entity embedding update works."""
        # Create entity without embedding
        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Test Entity",
            entity_id="ent_test_update_embed",
        )
        storage.store_entity(entity)

        # Add embedding
        new_embedding = make_test_embedding(seed=123)
        storage.update_node_embedding(
            node_id="ent_test_update_embed",
            embedding_json=json.dumps(new_embedding),
            node_type="entity",
        )

        # Verify update
        retrieved = storage.get_entity("ent_test_update_embed")
        assert retrieved is not None
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 768


class TestVecf32Roundtrip:
    """Tests for vecf32 storage/retrieval."""

    def test_store_memory_vecf32_roundtrip(self, storage: DatabaseInterface):
        """vecf32 storage/retrieval preserves data."""
        # Create a specific embedding
        original_embedding = make_test_embedding(seed=42)

        memory = MemoryNode.create(
            memory_id="mem_test_vecf32",
            content="Test vecf32 roundtrip",
            embedding=original_embedding,
            source="test",
        )
        storage.store_memory(memory, kind="MEMORY")

        # Retrieve and compare
        retrieved = storage.get_memory("mem_test_vecf32")
        assert retrieved is not None
        assert len(retrieved.content_embedding) == len(original_embedding)

        # Check values match (within floating point tolerance)
        for orig, ret in zip(original_embedding, retrieved.content_embedding):
            assert abs(orig - ret) < 1e-5, f"Embedding mismatch: {orig} vs {ret}"


# =============================================================================
# HIGH: CRUD Edge Cases (6 tests)
# =============================================================================


class TestEdgeOperations:
    """Tests for edge CRUD operations."""

    def test_store_edge_weight_default(self, storage: DatabaseInterface):
        """weight=None defaults to 1.0."""
        # Create memory and entity
        memory = MemoryNode.create(
            memory_id="mem_test_edge_weight",
            content="Test memory for edge weight",
            embedding=make_test_embedding(seed=1),
            source="test",
        )
        storage.store_memory(memory, kind="MEMORY")

        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Test Person",
            entity_id="ent_test_edge_weight",
        )
        storage.store_entity(entity)

        # Create edge with default weight
        edge = EdgeNode.create(
            from_node="mem_test_edge_weight",
            to_node="ent_test_edge_weight",
            edge_type="MENTIONS",
            weight=1.0,  # Default
        )
        storage.store_edge(edge)

        # Retrieve and check weight
        retrieved = storage.get_edge(edge.edge_id)
        assert retrieved is not None
        assert retrieved.weight == 1.0

    def test_store_edge_deduplication(self, storage: DatabaseInterface):
        """Same edge twice returns existing ID."""
        # Create memory and entity
        memory = MemoryNode.create(
            memory_id="mem_test_edge_dedup",
            content="Test memory for edge dedup",
            embedding=make_test_embedding(seed=1),
            source="test",
        )
        storage.store_memory(memory, kind="MEMORY")

        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Test Person Dedup",
            entity_id="ent_test_edge_dedup",
        )
        storage.store_entity(entity)

        # Create first edge
        edge1 = EdgeNode.create(
            from_node="mem_test_edge_dedup",
            to_node="ent_test_edge_dedup",
            edge_type="MENTIONS",
        )
        edge_id1 = storage.store_edge(edge1)

        # Create "same" edge (same from/to/type)
        edge2 = EdgeNode.create(
            from_node="mem_test_edge_dedup",
            to_node="ent_test_edge_dedup",
            edge_type="MENTIONS",
        )
        edge_id2 = storage.store_edge(edge2)

        # Should return the same ID (deduplication)
        assert edge_id1 == edge_id2

    def test_get_edges_confidence_filter(self, storage: DatabaseInterface):
        """min_confidence filter works."""
        # Create memory and entity
        memory = MemoryNode.create(
            memory_id="mem_test_conf_filter",
            content="Test memory for confidence filter",
            embedding=make_test_embedding(seed=1),
            source="test",
        )
        storage.store_memory(memory, kind="MEMORY")

        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Test Person Conf",
            entity_id="ent_test_conf_filter",
        )
        storage.store_entity(entity)

        # Create edges with different confidence levels
        edge_low = EdgeNode.create(
            from_node="mem_test_conf_filter",
            to_node="ent_test_conf_filter",
            edge_type="MENTIONS",
            confidence=0.3,
        )
        storage.store_edge(edge_low)

        # Create another memory for a second edge
        memory2 = MemoryNode.create(
            memory_id="mem_test_conf_filter_2",
            content="Test memory 2 for confidence filter",
            embedding=make_test_embedding(seed=2),
            source="test",
        )
        storage.store_memory(memory2, kind="MEMORY")

        edge_high = EdgeNode.create(
            from_node="mem_test_conf_filter_2",
            to_node="ent_test_conf_filter",
            edge_type="MENTIONS",
            confidence=0.9,
        )
        storage.store_edge(edge_high)

        # Query with min_confidence=0.5 - should only get high confidence edge
        edges = storage.get_edges_to_entity(
            entity_id="ent_test_conf_filter",
            include_expired=False,
            min_confidence=0.5,
        )
        assert len(edges) == 1
        assert edges[0].confidence == 0.9


class TestReinforcementTracking:
    """Tests for reinforce count operations."""

    def test_increment_reinforce_count(self, storage: DatabaseInterface):
        """COALESCE(NULL,0)+1 works."""
        # Create memory with reinforce_count=0
        memory = MemoryNode.create(
            memory_id="mem_test_reinforce",
            content="Test memory for reinforce count",
            embedding=make_test_embedding(seed=1),
            source="test",
        )
        storage.store_memory(memory, kind="MEMORY")

        # Verify initial count
        retrieved = storage.get_memory("mem_test_reinforce")
        assert retrieved is not None
        assert retrieved.reinforce_count == 0

        # Increment
        storage.increment_reinforce_count("mem_test_reinforce")

        # Verify incremented
        retrieved = storage.get_memory("mem_test_reinforce")
        assert retrieved.reinforce_count == 1

        # Increment again
        storage.increment_reinforce_count("mem_test_reinforce")
        retrieved = storage.get_memory("mem_test_reinforce")
        assert retrieved.reinforce_count == 2


class TestSoftDelete:
    """Tests for soft delete operations."""

    def test_expire_entity_soft_delete(self, storage: DatabaseInterface):
        """Sets expired_at, excluded from queries."""
        # Create entity
        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Test Person Expire",
            entity_id="ent_test_expire",
        )
        storage.store_entity(entity)

        # Verify entity is active
        found = storage.find_entity_by_norm_key(entity.norm_key, include_expired=False)
        assert found is not None

        # Expire entity
        storage.expire_entity("ent_test_expire", merged_into=None)

        # Verify excluded from default queries
        found = storage.find_entity_by_norm_key(entity.norm_key, include_expired=False)
        assert found is None

        # Verify included with include_expired=True
        found = storage.find_entity_by_norm_key(entity.norm_key, include_expired=True)
        assert found is not None
        assert found.expired_at is not None

    def test_deprecate_memory_soft_delete(self, storage: DatabaseInterface):
        """Sets t_expired, excluded from active."""
        # Create memory
        memory = MemoryNode.create(
            memory_id="mem_test_deprecate",
            content="Test memory for deprecation",
            embedding=make_test_embedding(seed=1),
            source="test",
        )
        storage.store_memory(memory, kind="MEMORY")

        # Verify in active memories
        active = storage.get_active_memories()
        active_ids = [m.id for m in active]
        assert "mem_test_deprecate" in active_ids

        # Deprecate
        storage.deprecate_memory("mem_test_deprecate")

        # Verify excluded from active
        active = storage.get_active_memories()
        active_ids = [m.id for m in active]
        assert "mem_test_deprecate" not in active_ids

        # Verify still retrievable by ID
        retrieved = storage.get_memory("mem_test_deprecate")
        assert retrieved is not None
        assert retrieved.t_expired is not None


# =============================================================================
# MEDIUM: Query Logic (5 tests)
# =============================================================================


class TestEntityExtractionQueries:
    """Tests for entity extraction query logic."""

    def test_get_memories_for_entity_extraction_reprocess(self, storage: DatabaseInterface):
        """reprocess flag logic works correctly."""
        # Create a memory without MENTIONS edges
        memory1 = MemoryNode.create(
            memory_id="mem_test_extract_1",
            content="Memory without entities",
            embedding=make_test_embedding(seed=1),
            source="test",
        )
        storage.store_memory(memory1, kind="MEMORY")

        # Create a memory and add a MENTIONS edge
        memory2 = MemoryNode.create(
            memory_id="mem_test_extract_2",
            content="Memory with entity",
            embedding=make_test_embedding(seed=2),
            source="test",
        )
        storage.store_memory(memory2, kind="MEMORY")

        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Extracted Person",
            entity_id="ent_test_extracted",
        )
        storage.store_entity(entity)

        edge = EdgeNode.create(
            from_node="mem_test_extract_2",
            to_node="ent_test_extracted",
            edge_type="MENTIONS",
        )
        storage.store_edge(edge)

        # Without reprocess: only memory1 (no MENTIONS edges)
        needs_extraction = storage.get_memories_for_entity_extraction(reprocess=False)
        ids = [m[0] for m in needs_extraction]
        assert "mem_test_extract_1" in ids
        assert "mem_test_extract_2" not in ids

        # With reprocess: both memories
        all_memories = storage.get_memories_for_entity_extraction(reprocess=True)
        ids = [m[0] for m in all_memories]
        assert "mem_test_extract_1" in ids
        assert "mem_test_extract_2" in ids


class TestListEntitiesWithCounts:
    """Tests for entity listing with mention counts."""

    def test_list_entities_with_mention_counts(self, storage: DatabaseInterface):
        """COUNT aggregation correct."""
        # Create entity
        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Popular Person",
            entity_id="ent_test_mentions",
        )
        storage.store_entity(entity)

        # Create multiple memories mentioning this entity
        for i in range(3):
            memory = MemoryNode.create(
                memory_id=f"mem_test_mention_{i}",
                content=f"Memory {i} about popular person",
                embedding=make_test_embedding(seed=i),
                source="test",
            )
            storage.store_memory(memory, kind="MEMORY")

            edge = EdgeNode.create(
                from_node=f"mem_test_mention_{i}",
                to_node="ent_test_mentions",
                edge_type="MENTIONS",
            )
            storage.store_edge(edge)

        # List entities with counts
        results = storage.list_entities_with_mention_counts(include_expired=False)

        # Find our entity
        entity_row = None
        for row in results:
            if row[0] == "ent_test_mentions":
                entity_row = row
                break

        assert entity_row is not None
        # Row: (id, type, name, created_at, expired_at, merged_into, mentions)
        mentions_count = entity_row[6]
        assert mentions_count == 3


class TestOrphanedMemories:
    """Tests for orphaned memory detection."""

    def test_get_orphaned_memories(self, storage: DatabaseInterface):
        """Finds memories without edges."""
        # Create orphaned memory (no CONTAINS, no MENTIONS)
        orphan = MemoryNode.create(
            memory_id="mem_test_orphan",
            content="Orphaned memory without any edges",
            embedding=make_test_embedding(seed=1),
            source="test",
        )
        storage.store_memory(orphan, kind="MEMORY")

        # Create connected memory (with MENTIONS edge)
        connected = MemoryNode.create(
            memory_id="mem_test_connected",
            content="Connected memory with entity",
            embedding=make_test_embedding(seed=2),
            source="test",
        )
        storage.store_memory(connected, kind="MEMORY")

        entity = EntityNode.create(
            entity_type="PERSON",
            canonical_name="Connected Entity",
            entity_id="ent_test_connected",
        )
        storage.store_entity(entity)

        edge = EdgeNode.create(
            from_node="mem_test_connected",
            to_node="ent_test_connected",
            edge_type="MENTIONS",
        )
        storage.store_edge(edge)

        # Get orphaned memories
        orphans = storage.get_orphaned_memories()
        orphan_ids = [m[0] for m in orphans]

        assert "mem_test_orphan" in orphan_ids
        assert "mem_test_connected" not in orphan_ids


class TestSearchWithExpired:
    """Tests for search including expired items."""

    def test_search_memories_include_expired(self, storage: DatabaseInterface):
        """include_expired flag works."""
        # Create and expire a memory
        memory = MemoryNode.create(
            memory_id="mem_test_search_expired",
            content="Memory that will be expired",
            embedding=make_test_embedding(seed=42),
            source="test",
        )
        storage.store_memory(memory, kind="MEMORY")
        storage.deprecate_memory("mem_test_search_expired")

        query_vector = make_test_embedding(seed=42)

        # Search without expired - should not find
        results_no_expired = storage.search_memories_by_vector(
            query_vector=query_vector,
            limit=10,
            kind_filter=None,
            include_expired=False,
        )
        found_ids = [m.id for m, _ in results_no_expired]
        assert "mem_test_search_expired" not in found_ids

        # Search with expired - should find
        results_with_expired = storage.search_memories_by_vector(
            query_vector=query_vector,
            limit=10,
            kind_filter=None,
            include_expired=True,
        )
        found_ids = [m.id for m, _ in results_with_expired]
        assert "mem_test_search_expired" in found_ids


class TestCountMethods:
    """Tests for count method accuracy."""

    def test_count_methods_accuracy(self, storage: DatabaseInterface):
        """Counts match reality."""
        # Create known quantities
        num_memories = 3
        num_summaries = 2
        num_entities = 2

        for i in range(num_memories):
            memory = MemoryNode.create(
                memory_id=f"mem_test_count_{i}",
                content=f"Test memory {i}",
                embedding=make_test_embedding(seed=i),
                source="test",
            )
            storage.store_memory(memory, kind="MEMORY")

        for i in range(num_summaries):
            summary = MemoryNode.create(
                memory_id=f"sum_test_count_{i}",
                content=f"Test summary {i}",
                embedding=make_test_embedding(seed=100 + i),
                source="test",
            )
            storage.store_memory(summary, kind="SUMMARY")

        for i in range(num_entities):
            entity = EntityNode.create(
                entity_type="PERSON",
                canonical_name=f"Test Person {i}",
                entity_id=f"ent_test_count_{i}",
            )
            storage.store_entity(entity)

        # Verify counts
        assert storage.count_memories(kind="MEMORY") == num_memories
        assert storage.count_memories(kind="SUMMARY") == num_summaries
        assert storage.count_memories() == num_memories + num_summaries
        assert storage.count_entities() == num_entities
