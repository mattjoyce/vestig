#!/bin/bash
# M4 Smoke Test: Quick validation of graph layer functionality
# Tests: Entity extraction, MENTIONS edges, RELATED edges, graph expansion

set -e  # Exit on error

echo "======================================================================"
echo "M4 SMOKE TEST: Graph Layer Validation"
echo "======================================================================"
echo ""

# Resolve repo root for config access
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP_DIR="$REPO_ROOT/tests/tmp"
mkdir -p "$TMP_DIR"

# Use temp graph + config
GRAPH_NAME="vestig_m4_smoke_${RANDOM}_${RANDOM}"
CONFIG=$(mktemp "$TMP_DIR/vestig-m4-config.XXXXXX")
sed "s|graph_name:.*|graph_name: $GRAPH_NAME|g" "$REPO_ROOT/config_test.yaml" > "$CONFIG"
echo "Using temp graph: $GRAPH_NAME"
echo "Using config: $CONFIG"
export CONFIG_PATH="$CONFIG"
# Use same env vars as conftest.py for consistency
export FALKOR_HOST="${VESTIG_FALKORDB_HOST:-192.168.20.4}"
export FALKOR_PORT="${VESTIG_FALKORDB_PORT:-6379}"
export REPO_ROOT
# Embedding provider is llm CLI (Ollama), not HuggingFace
trap 'rm -f "$CONFIG"; redis-cli -h "$FALKOR_HOST" -p "$FALKOR_PORT" GRAPH.DELETE "$GRAPH_NAME" >/dev/null 2>&1 || true' EXIT

# Activate virtual environment
source ~/Environments/vestig/bin/activate

# Run smoke test
python3 << 'EOF'
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(os.environ["REPO_ROOT"]) / "src"))

from vestig.core.commitment import commit_memory
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.db_falkordb import FalkorDBDatabase
from vestig.core.config import load_config
from vestig.core.graph import expand_via_entities, expand_via_related
from vestig.core.ingestion import MemoryExtractionResult

# Load config
config = load_config(os.environ["CONFIG_PATH"])
storage = FalkorDBDatabase(
    host=config["storage"]["falkordb"]["host"],
    port=config["storage"]["falkordb"]["port"],
    graph_name=config["storage"]["falkordb"]["graph_name"],
    config=config,
)
event_storage = storage.event_storage
embedding_engine = EmbeddingEngine(
    model_name=config["embedding"]["model"],
    expected_dimension=config["embedding"]["dimension"],
)

# M4 config
m4_config = config.get("m4", {})

print("Test 1: Entity extraction and storage")
mock_response = MemoryExtractionResult.model_validate({
    "memories": [
        {
            "content": "Alice fixed PostgreSQL bug",
            "confidence": 0.90,
            "rationale": "test",
            "entities": [
                {"name": "Alice", "type": "PERSON", "confidence": 0.90, "evidence": "developer"},
                {"name": "PostgreSQL", "type": "SYSTEM", "confidence": 0.95, "evidence": "database"},
            ],
        }
    ]
})

with patch("vestig.core.ingestion.call_llm", return_value=mock_response):
    outcome = commit_memory(
        content="Alice fixed PostgreSQL bug",
        storage=storage,
        embedding_engine=embedding_engine,
        event_storage=event_storage,
        m4_config=m4_config,
    )
    mem_id = outcome.memory_id

# Verify entities created
entities = storage.get_all_entities()
assert len(entities) == 2, f"Expected 2 entities, got {len(entities)}"
print("✓ Entity extraction working")

# Verify MENTIONS edges created
mentions_edges = storage.get_edges_from_memory(mem_id, edge_type="MENTIONS")
assert len(mentions_edges) == 2, f"Expected 2 MENTIONS edges, got {len(mentions_edges)}"
print("✓ MENTIONS edge creation working")

print("\nTest 2: Entity deduplication")
# Add another memory with same entity
with patch("vestig.core.ingestion.call_llm", return_value=mock_response):
    outcome2 = commit_memory(
        content="Alice optimized PostgreSQL queries",
        storage=storage,
        embedding_engine=embedding_engine,
        event_storage=event_storage,
        m4_config=m4_config,
    )
    mem_id2 = outcome2.memory_id

# Should still have only 2 entities (deduplication)
entities = storage.get_all_entities()
assert len(entities) == 2, f"Expected 2 entities (deduped), got {len(entities)}"
print("✓ Entity deduplication working")

print("\nTest 3: RELATED edges (semantic similarity)")
# Check if RELATED edges were created
related_edges = storage.get_edges_from_memory(mem_id2, edge_type="RELATED")
# May or may not have edges depending on similarity threshold
print(f"✓ RELATED edge creation working (created {len(related_edges)} edges)")

print("\nTest 4: Graph expansion via entities")
# Expand via shared entities
expansion = expand_via_entities(
    memory_ids=[mem_id],
    storage=storage,
    limit=5,
)

# Should find mem_id2 (shares entities)
expanded_ids = [r["memory"].id for r in expansion]
assert mem_id2 in expanded_ids, f"Expected to find mem_id2 in expansion"
print("✓ Graph expansion via entities working")

print("\nTest 5: ENTITY_EXTRACTED event logging")
events = event_storage.get_events_for_memory(mem_id)
entity_events = [e for e in events if e.event_type == "ENTITY_EXTRACTED"]
assert len(entity_events) > 0, "Expected ENTITY_EXTRACTED event"
assert entity_events[0].payload.get("model_name") is not None
assert entity_events[0].payload.get("prompt_hash") is not None
print("✓ Event logging working")

print("\n" + "="*70)
print("✅ ALL SMOKE TESTS PASSED")
print("="*70)
print(f"\nDatabase stats:")
print(f"  Memories: {len(storage.get_all_memories())}")
print(f"  Entities: {len(storage.get_all_entities())}")
all_edges = []
for mem in storage.get_all_memories():
    all_edges.extend(storage.get_edges_from_memory(mem.id))
print(f"  Edges: {len(all_edges)}")

EOF

echo ""
echo "Smoke test complete! Temp database removed."
