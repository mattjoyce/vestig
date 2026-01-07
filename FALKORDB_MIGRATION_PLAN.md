# FalkorDB Migration Plan

## Strategy: Abstraction First, Clean Later

**Key Insight:** Don't waste time cleaning SQLite code we're going to delete anyway.

**Approach:**
1. Create database abstraction layer
2. Wrap existing SQLite code as-is (don't touch it)
3. Implement FalkorDB adapter
4. Switch backends via config
5. Delete SQLite entirely once FalkorDB works

**Current State:**
- SQLite has CREATE/ALTER/migration code scattered everywhere
- 84 direct SQL calls in storage.py
- SQL scattered across 4 files
- **Don't fix any of this - just wrap it**

## Objectives

1. **Create database abstraction layer** - Clean interface hiding implementation
2. **Wrap SQLite as-is** - Adapter pattern, no cleanup needed
3. **Implement FalkorDB** - Native graph database with Cypher
4. **Migrate data** - Tool to export SQLite → FalkorDB
5. **Delete SQLite** - Remove all legacy code once FalkorDB works

---

## Phase 1: Database Abstraction Layer

### 1.1 Design Abstraction Interface

**Create:** `src/vestig/core/db_interface.py`

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

class DatabaseInterface(ABC):
    """Abstract interface for graph database operations"""

    # Node operations
    @abstractmethod
    def create_node(self, node_type: str, properties: dict) -> str:
        """Create node, return ID"""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[dict]:
        """Get node by ID"""
        pass

    @abstractmethod
    def update_node(self, node_id: str, properties: dict) -> None:
        """Update node properties"""
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete node"""
        pass

    # Edge operations
    @abstractmethod
    def create_edge(self, from_id: str, to_id: str, edge_type: str, properties: dict) -> str:
        """Create edge, return edge ID"""
        pass

    @abstractmethod
    def get_edges_from(self, node_id: str, edge_type: Optional[str] = None) -> List[dict]:
        """Get outgoing edges from node"""
        pass

    @abstractmethod
    def get_edges_to(self, node_id: str, edge_type: Optional[str] = None) -> List[dict]:
        """Get incoming edges to node"""
        pass

    # Graph traversal
    @abstractmethod
    def traverse(self, start_id: str, pattern: str, params: dict) -> List[dict]:
        """Execute graph traversal query"""
        pass

    # Vector search
    @abstractmethod
    def vector_search(self, embedding: List[float], limit: int, filters: dict) -> List[Tuple[dict, float]]:
        """Semantic search by embedding similarity"""
        pass

    # Transaction management
    @abstractmethod
    def begin_transaction(self) -> None:
        pass

    @abstractmethod
    def commit_transaction(self) -> None:
        pass

    @abstractmethod
    def rollback_transaction(self) -> None:
        pass
```

### 1.2 SQLite Adapter (Legacy Wrapper)

**Create:** `src/vestig/core/db_sqlite.py`

**IMPORTANT:** This is a thin wrapper around existing MemoryStorage. Don't refactor anything!

```python
class SQLiteDatabase(DatabaseInterface):
    """SQLite adapter - wraps existing MemoryStorage as-is (legacy, will be deleted)"""

    def __init__(self, db_path: str):
        # Import existing storage
        from vestig.core.storage import MemoryStorage
        self.storage = MemoryStorage(db_path)

    def create_node(self, node_type: str, properties: dict) -> str:
        # Delegate to existing methods
        if node_type == "Memory":
            node = MemoryNode(**properties)
            return self.storage.store_memory(node)
        elif node_type == "Entity":
            node = EntityNode(**properties)
            return self.storage.store_entity(node)
        # ... delegate to existing MemoryStorage methods

    def traverse(self, start_id: str, pattern: str, params: dict) -> List[dict]:
        # Limited Cypher-like patterns mapped to existing methods
        # e.g., "(c:Chunk)-[:CONTAINS]->(m:Memory)" → storage.get_memories_in_chunk()
        pass
```

**Key:** Don't touch MemoryStorage internals. Just wrap it.

### 1.3 FalkorDB Adapter (Target)

**Create:** `src/vestig/core/db_falkordb.py`

```python
from falkordb import FalkorDB
from ollama import Client as OllamaClient

class FalkorDBDatabase(DatabaseInterface):
    """FalkorDB graph database adapter with direct Ollama integration"""

    def __init__(self, host: str, port: int, graph_name: str, ollama_host: str = None):
        self.client = FalkorDB(host=host, port=port)
        self.graph = self.client.select_graph(graph_name)

        # Direct Ollama client for embeddings
        if ollama_host:
            self.ollama = OllamaClient(host=ollama_host)
        else:
            self.ollama = None

    def create_node(self, node_type: str, properties: dict) -> str:
        query = f"CREATE (n:{node_type} $props) RETURN id(n)"
        result = self.graph.query(query, {"props": properties})
        return result.result_set[0][0]

    def traverse(self, start_id: str, pattern: str, params: dict) -> List[dict]:
        # Native Cypher support
        query = f"MATCH {pattern} WHERE id(start) = $start_id RETURN ..."
        result = self.graph.query(query, {"start_id": start_id, **params})
        return self._parse_results(result)

    def generate_embedding(self, text: str, model: str = "embeddinggemma:latest") -> List[float]:
        """Generate embedding using Ollama directly"""
        if not self.ollama:
            raise RuntimeError("Ollama client not configured")

        response = self.ollama.embeddings(model=model, prompt=text)
        return response['embedding']

    def vector_search(self, embedding: List[float], limit: int, filters: dict) -> List[Tuple[dict, float]]:
        """
        Vector search in FalkorDB
        NOTE: Vector index syntax to be determined - FalkorDB docs needed
        """
        # Placeholder - need to verify FalkorDB vector search syntax
        query = """
        // Vector search syntax TBD based on FalkorDB documentation
        MATCH (m:Memory)
        // WHERE vector.similarity(m.content_embedding, $embedding) > threshold
        RETURN m, score
        LIMIT $limit
        """
        result = self.graph.query(query, {"embedding": embedding, "limit": limit})
        return self._parse_vector_results(result)
```

**Key Architecture Decision:**
- NO GraphRAG SDK - direct FalkorDB + Ollama integration
- Keep Vestig's specialized ingestion/retrieval logic
- Use Ollama for embeddings only (via ollama Python client)
- Use existing LLM integration for fact extraction (keep current prompts)

---

## Phase 2: FalkorDB Schema Design

### 2.1 Node Types

```cypher
// Memory nodes
(:Memory {
  id: STRING,           // mem_<uuid>
  content: STRING,
  content_hash: STRING,
  created_at: STRING,
  metadata: MAP,
  kind: STRING,         // "MEMORY" | "SUMMARY"

  // M3: Temporal
  t_valid: STRING,
  t_invalid: STRING,
  t_created: STRING,
  t_expired: STRING,
  temporal_stability: STRING,

  // M3: Reinforcement
  last_seen_at: STRING,
  reinforce_count: INTEGER,

  // Vector index
  content_embedding: VECTOR(384)  // or dimension from config
})

// Entity nodes
(:Entity {
  id: STRING,           // ent_<uuid>
  entity_type: STRING,  // PERSON | ORG | SYSTEM | ...
  canonical_name: STRING,
  norm_key: STRING,
  created_at: STRING,
  expired_at: STRING,
  merged_into: STRING,
  embedding: VECTOR(384)
})

// Chunk nodes (Hub)
(:Chunk {
  id: STRING,           // chunk_<uuid>
  file_id: STRING,
  start: INTEGER,
  length: INTEGER,
  sequence: INTEGER,
  created_at: STRING
})

// File nodes
(:File {
  id: STRING,           // file_<uuid>
  path: STRING,
  created_at: STRING,
  ingested_at: STRING,
  file_hash: STRING,
  metadata: MAP
})

// Event nodes
(:Event {
  id: STRING,           // evt_<uuid>
  event_type: STRING,
  occurred_at: STRING,
  source: STRING,
  actor: STRING,
  artifact_ref: STRING,
  payload: MAP
})
```

### 2.2 Edge Types

```cypher
// Hub-and-spoke (Chunk as central hub)
(:Chunk)-[:CONTAINS {weight: FLOAT}]->(:Memory)
(:Chunk)-[:LINKED {weight: FLOAT, confidence: FLOAT, evidence: STRING}]->(:Entity)
(:Chunk)-[:SUMMARIZED_BY {weight: FLOAT}]->(:Memory {kind: "SUMMARY"})

// Secondary relationships
(:Memory)-[:MENTIONS {weight: FLOAT, confidence: FLOAT, evidence: STRING}]->(:Entity)
(:Memory)-[:RELATED {weight: FLOAT, similarity: FLOAT}]->(:Memory)

// Provenance
(:File)-[:CONTAINS_CHUNK]->(:Chunk)
(:Event)-[:AFFECTS]->(:Memory)
```

### 2.3 Indexes

```cypher
// Vector indexes for semantic search
CREATE VECTOR INDEX memory_embedding FOR (m:Memory) ON (m.content_embedding)
  OPTIONS {dimension: 384, similarity: 'cosine'}

CREATE VECTOR INDEX entity_embedding FOR (e:Entity) ON (e.embedding)
  OPTIONS {dimension: 384, similarity: 'cosine'}

// Unique constraints
CREATE CONSTRAINT unique_content_hash FOR (m:Memory) REQUIRE m.content_hash IS UNIQUE
CREATE CONSTRAINT unique_norm_key FOR (e:Entity) REQUIRE e.norm_key IS UNIQUE

// Range indexes
CREATE INDEX memory_created FOR (m:Memory) ON (m.created_at)
CREATE INDEX memory_kind FOR (m:Memory) ON (m.kind)
CREATE INDEX entity_type FOR (e:Entity) ON (e.entity_type)
CREATE INDEX chunk_file FOR (c:Chunk) ON (c.file_id, c.sequence)
```

### 2.4 Schema File

**Create:** `src/vestig/core/schema_falkor.cypher`

```cypher
// Vestig FalkorDB Schema v1.0
// Graph-native schema for memory system

// ===== Node Type Definitions =====

// Memory nodes (atomic facts and summaries)
CREATE CONSTRAINT unique_memory_id FOR (m:Memory) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT unique_content_hash FOR (m:Memory) REQUIRE m.content_hash IS UNIQUE;
CREATE INDEX memory_kind FOR (m:Memory) ON (m.kind);
CREATE INDEX memory_created FOR (m:Memory) ON (m.created_at);
CREATE INDEX memory_expired FOR (m:Memory) ON (m.t_expired);
CREATE VECTOR INDEX memory_embedding FOR (m:Memory) ON (m.content_embedding)
  OPTIONS {dimension: 384, similarity: 'cosine'};

// Entity nodes (canonical entities)
CREATE CONSTRAINT unique_entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT unique_norm_key FOR (e:Entity) REQUIRE e.norm_key IS UNIQUE;
CREATE INDEX entity_type FOR (e:Entity) ON (e.entity_type);
CREATE INDEX entity_expired FOR (e:Entity) ON (e.expired_at);
CREATE VECTOR INDEX entity_embedding FOR (e:Entity) ON (e.embedding)
  OPTIONS {dimension: 384, similarity: 'cosine'};

// Chunk nodes (provenance hubs)
CREATE CONSTRAINT unique_chunk_id FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE INDEX chunk_file FOR (c:Chunk) ON (c.file_id, c.sequence);

// File nodes (source documents)
CREATE CONSTRAINT unique_file_id FOR (f:File) REQUIRE f.id IS UNIQUE;
CREATE INDEX file_path FOR (f:File) ON (f.path);
CREATE INDEX file_ingested FOR (f:File) ON (f.ingested_at);

// Event nodes (audit trail)
CREATE CONSTRAINT unique_event_id FOR (evt:Event) REQUIRE evt.id IS UNIQUE;
CREATE INDEX event_type FOR (evt:Event) ON (evt.event_type);
CREATE INDEX event_occurred FOR (evt:Event) ON (evt.occurred_at);

// ===== Edge Type Definitions =====

// No explicit edge constraints in FalkorDB, but enforce at application layer:
// - CONTAINS: Chunk → Memory
// - LINKED: Chunk → Entity (1st class, from original text)
// - SUMMARIZED_BY: Chunk → Summary (Memory with kind="SUMMARY")
// - MENTIONS: Memory → Entity (2nd class, from extracted facts)
// - RELATED: Memory → Memory (semantic relationships)
// - CONTAINS_CHUNK: File → Chunk
// - AFFECTS: Event → Memory
```

---

## Phase 3: Migration Strategy

### 3.1 Data Export from SQLite

**Create:** `src/vestig/tools/export_to_falkor.py`

```python
"""Export SQLite database to FalkorDB"""

def export_nodes(sqlite_db: str, falkor_db: FalkorDBDatabase):
    """Export all nodes"""
    conn = sqlite3.connect(sqlite_db)

    # Export memories
    cursor = conn.execute("SELECT * FROM memories")
    for row in cursor:
        memory = parse_memory_row(row)
        falkor_db.create_node("Memory", memory)

    # Export entities
    cursor = conn.execute("SELECT * FROM entities")
    for row in cursor:
        entity = parse_entity_row(row)
        falkor_db.create_node("Entity", entity)

    # Export chunks, files, events...

def export_edges(sqlite_db: str, falkor_db: FalkorDBDatabase):
    """Export all edges"""
    conn = sqlite3.connect(sqlite_db)

    cursor = conn.execute("SELECT * FROM edges")
    for row in cursor:
        edge = parse_edge_row(row)
        falkor_db.create_edge(
            from_id=edge.from_node,
            to_id=edge.to_node,
            edge_type=edge.edge_type,
            properties={"weight": edge.weight, "confidence": edge.confidence, ...}
        )

def migrate_database(sqlite_path: str, falkor_config: dict):
    """Full migration from SQLite to FalkorDB"""
    falkor = FalkorDBDatabase(**falkor_config)

    print("Exporting nodes...")
    export_nodes(sqlite_path, falkor)

    print("Exporting edges...")
    export_edges(sqlite_path, falkor)

    print("Verifying migration...")
    verify_migration(sqlite_path, falkor)
```

### 3.2 Configuration

**Update:** `config.yaml`

```yaml
storage:
  backend: "falkordb"  # or "sqlite" for testing

  # FalkorDB config
  falkordb:
    host: "192.168.20.4"
    port: 6379
    graph_name: "vestig"

  # SQLite config (legacy)
  sqlite:
    db_path: "vestig.db"

# LLM Configuration
llm:
  # Ollama for embeddings and generation
  ollama:
    host: "http://192.168.20.8:11434"
    embedding_model: "embeddinggemma:latest"
    generative_model: "gpt-oss:latest"
    embedding_dimension: 384  # embeddinggemma dimension
```

### 3.3 Runtime Selection

**File:** `src/vestig/core/cli.py`

```python
def build_runtime(config):
    """Build database backend from config"""
    backend = config["storage"]["backend"]

    if backend == "falkordb":
        db = FalkorDBDatabase(
            host=config["storage"]["falkordb"]["host"],
            port=config["storage"]["falkordb"]["port"],
            graph_name=config["storage"]["falkordb"]["graph_name"]
        )
    elif backend == "sqlite":
        db = SQLiteDatabase(
            db_path=config["storage"]["sqlite"]["db_path"]
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    storage = MemoryStorage(db)
    # ... rest of runtime setup
```

---

## Phase 4: Testing Strategy

### 4.1 Abstraction Tests

**Create:** `test/test_db_interface.py`

```python
"""Test database abstraction layer with both backends"""

import pytest
from vestig.core.db_interface import DatabaseInterface
from vestig.core.db_sqlite import SQLiteDatabase
from vestig.core.db_falkordb import FalkorDBDatabase

@pytest.fixture(params=["sqlite", "falkordb"])
def db(request):
    if request.param == "sqlite":
        return SQLiteDatabase(":memory:")
    else:
        return FalkorDBDatabase("localhost", 6379, "test")

def test_create_node(db: DatabaseInterface):
    node_id = db.create_node("Memory", {
        "content": "test",
        "created_at": "2025-01-01"
    })
    assert node_id.startswith("mem_")

def test_create_edge(db: DatabaseInterface):
    chunk_id = db.create_node("Chunk", {"file_id": "test"})
    memory_id = db.create_node("Memory", {"content": "test"})

    edge_id = db.create_edge(chunk_id, memory_id, "CONTAINS", {"weight": 1.0})

    edges = db.get_edges_from(chunk_id, "CONTAINS")
    assert len(edges) == 1
    assert edges[0]["to_node"] == memory_id
```

### 4.2 Migration Validation

**Create:** `test/test_migration.py`

```python
def test_data_integrity_after_migration():
    """Verify all data migrated correctly"""
    sqlite_db = SQLiteDatabase("test.db")
    falkor_db = FalkorDBDatabase("localhost", 6379, "test")

    # Export
    export_nodes("test.db", falkor_db)
    export_edges("test.db", falkor_db)

    # Verify counts
    assert falkor_db.count_nodes("Memory") == sqlite_db.count_nodes("Memory")
    assert falkor_db.count_edges("CONTAINS") == sqlite_db.count_edges("CONTAINS")

    # Verify specific records
    for memory_id in sqlite_db.get_all_memory_ids():
        sqlite_memory = sqlite_db.get_node(memory_id)
        falkor_memory = falkor_db.get_node(memory_id)
        assert sqlite_memory == falkor_memory
```

---

## Implementation Order

### Step 1: Abstraction Layer (No Changes to Existing Code)
1. Create `db_interface.py` - Abstract interface
2. Create `db_sqlite.py` - Thin wrapper around existing MemoryStorage
3. Add config: `storage.backend = "sqlite" | "falkordb"`
4. Update CLI to use adapter pattern
5. **Test**: Verify existing functionality unchanged

### Step 2: FalkorDB Implementation
1. Create `db_falkordb.py` - FalkorDB adapter
2. Create `schema_falkor.cypher` - Graph schema
3. Implement all DatabaseInterface methods for FalkorDB
4. **Test**: Feature parity with SQLite

### Step 3: Migration Tools
1. Create `export_to_falkor.py` - SQLite → FalkorDB data migration
2. Test with real databases
3. Document migration process
4. **Test**: Data integrity validation

### Step 4: Switch Default Backend
1. Update default config: `backend: "falkordb"`
2. Update docs: FalkorDB is now primary
3. Mark SQLite adapter as deprecated
4. **Test**: Full test suite on FalkorDB

### Step 5: Remove SQLite Code (Future)
1. Delete `src/vestig/core/storage.py` (entire legacy file)
2. Delete `src/vestig/core/event_storage.py`
3. Delete `src/vestig/core/schema.sql`
4. Delete `db_sqlite.py` adapter
5. Only FalkorDB remains

---

## Critical Files

**New Files:**
- `src/vestig/core/db_interface.py` - Abstract interface
- `src/vestig/core/db_sqlite.py` - SQLite wrapper (thin, delegates to existing MemoryStorage)
- `src/vestig/core/db_falkordb.py` - FalkorDB adapter
- `src/vestig/core/schema_falkor.cypher` - FalkorDB schema
- `src/vestig/tools/export_to_falkor.py` - Migration tool
- `test/test_db_interface.py` - Abstraction tests
- `test/test_migration.py` - Migration validation

**Modified Files:**
- `src/vestig/core/cli.py` - Use adapter pattern, backend selection
- `config.yaml` - Add `storage.backend` option

**Untouched Files (keep as-is):**
- `src/vestig/core/storage.py` - Wrapped by SQLiteDatabase, never modified
- `src/vestig/core/event_storage.py` - Used by SQLite adapter
- `src/vestig/core/schema.sql` - Used by SQLite adapter

**Files to Delete (later):**
- All SQLite-specific code once FalkorDB proven

---

## Success Criteria

✓ Zero CREATE/ALTER statements in Python code
✓ All schema in schema.sql (SQLite) or schema_falkor.cypher (FalkorDB)
✓ Database abstraction layer allows backend swap via config
✓ FalkorDB backend fully functional
✓ Migration tool successfully exports SQLite → FalkorDB
✓ All tests pass on both backends
✓ Performance improvement with graph queries
✓ Vector search leverages FalkorDB native indexes

---

## Key Principles

1. **Don't touch SQLite code** - It works, leave it alone, just wrap it
2. **Adapter pattern** - Thin wrapper, no refactoring
3. **Parallel development** - FalkorDB built alongside, not as replacement of SQLite
4. **Config-based switching** - Easy to toggle backends
5. **Delete, don't refactor** - Once FalkorDB works, delete SQLite entirely

## Risks and Mitigations

**Risk:** Abstraction overhead slows SQLite
- **Mitigation:** Wrapper is thin, just delegates to existing methods

**Risk:** FalkorDB feature gaps
- **Mitigation:** Keep SQLite working until FalkorDB feature-complete

**Risk:** Migration data loss
- **Mitigation:** Extensive validation, keep SQLite DB as backup

**Risk:** Cypher query complexity
- **Mitigation:** FalkorDB adapter doesn't need to support every pattern, just common ones

---

## Questions for User

1. **Timeline:** Should we do this in one shot or phased rollout?
2. **Backward compatibility:** Acceptable to break old DBs and force schema.sql recreation?
3. **FalkorDB hosting:** Self-hosted or cloud? Docker compose setup?
4. **Testing:** Do you have existing databases we should test migration against?
5. **Vector dimensions:** Confirm embedding size (384 for gemma?)?
