#!/bin/bash
# M4 Demo: Graph Layer - Entity Extraction & Graph Traversal
# Demonstrates LLM-based entity extraction, MENTIONS/RELATED edges, and graph expansion

set -e  # Exit on error

echo "======================================================================"
echo "M4 DEMO: Graph Layer - Entities, Edges, and Traversal"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Use temp database
DB=$(mktemp -t vestig-m4-demo.XXXXXX)
echo -e "${BLUE}Using temp database: $DB${NC}"
echo ""

# Activate virtual environment
source ~/Environments/vestig/bin/activate

# Test Python script for M4 demo
python3 << 'EOF'
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vestig.core.commitment import commit_memory
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.storage import MemoryStorage
from vestig.core.event_storage import MemoryEventStorage
from vestig.core.config import load_config
from vestig.core.graph import expand_via_entities, expand_via_related, expand_with_graph

# Color codes
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

def print_section(title):
    print(f"\n{BLUE}{'='*70}")
    print(f"{title}")
    print(f"{'='*70}{NC}\n")

def print_success(msg):
    print(f"{GREEN}✓ {msg}{NC}")

def print_info(msg):
    print(f"{YELLOW}→ {msg}{NC}")

# Get temp DB path from environment
db_path = os.environ.get('DB_PATH')

# Load config
config = load_config("config.yaml")
storage = MemoryStorage(db_path)
event_storage = MemoryEventStorage(storage.conn)
embedding_engine = EmbeddingEngine(
    model_name=config["embedding"]["model"],
    expected_dimension=config["embedding"]["dimension"],
)

# M4 config
m4_config = config.get("m4", {})

print_section("PART 1: Entity Extraction with LLM")
print_info("Committing memories with entity extraction enabled...")

# Mock LLM responses for demo (in production, uses real Anthropic API)
mock_responses = {
    "Alice fixed PostgreSQL": json.dumps({
        "entities": [
            {"name": "Alice", "type": "PERSON", "confidence": 0.92, "evidence": "developer who fixed the bug"},
            {"name": "PostgreSQL", "type": "SYSTEM", "confidence": 0.95, "evidence": "database system"}
        ]
    }),
    "Bob optimized Redis": json.dumps({
        "entities": [
            {"name": "Bob", "type": "PERSON", "confidence": 0.90, "evidence": "engineer who optimized"},
            {"name": "Redis", "type": "SYSTEM", "confidence": 0.93, "evidence": "cache system"}
        ]
    }),
    "Alice deployed Kubernetes": json.dumps({
        "entities": [
            {"name": "Alice", "type": "PERSON", "confidence": 0.91, "evidence": "engineer who deployed"},
            {"name": "Kubernetes", "type": "SYSTEM", "confidence": 0.94, "evidence": "orchestration platform"}
        ]
    }),
}

memories = []

# Memory 1: Alice + PostgreSQL
with patch("vestig.core.entity_extraction.call_llm", return_value=mock_responses["Alice fixed PostgreSQL"]):
    outcome = commit_memory(
        content="Alice fixed the PostgreSQL replication bug in production",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        event_storage=event_storage,
        m4_config=m4_config,
    )
    memories.append(("Alice + PostgreSQL", outcome.memory_id))
    print_success(f"Memory 1 committed: {outcome.memory_id[:16]}...")
    print(f"   Content: 'Alice fixed the PostgreSQL replication bug'")

# Memory 2: Bob + Redis
with patch("vestig.core.entity_extraction.call_llm", return_value=mock_responses["Bob optimized Redis"]):
    outcome = commit_memory(
        content="Bob optimized Redis query performance by 40%",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        event_storage=event_storage,
        m4_config=m4_config,
    )
    memories.append(("Bob + Redis", outcome.memory_id))
    print_success(f"Memory 2 committed: {outcome.memory_id[:16]}...")
    print(f"   Content: 'Bob optimized Redis query performance'")

# Memory 3: Alice + Kubernetes
with patch("vestig.core.entity_extraction.call_llm", return_value=mock_responses["Alice deployed Kubernetes"]):
    outcome = commit_memory(
        content="Alice deployed the new Kubernetes cluster for microservices",
        storage=storage,
        embedding_engine=embedding_engine,
        source="manual",
        event_storage=event_storage,
        m4_config=m4_config,
    )
    memories.append(("Alice + Kubernetes", outcome.memory_id))
    print_success(f"Memory 3 committed: {outcome.memory_id[:16]}...")
    print(f"   Content: 'Alice deployed Kubernetes cluster'")

print_section("PART 2: Entity Deduplication & MENTIONS Edges")
print_info("Verifying entity deduplication and MENTIONS edge creation...")

# Check entities
all_entities = storage.get_all_entities()
print_success(f"Total entities extracted: {len(all_entities)}")

for entity in all_entities:
    print(f"   - {entity.canonical_name} ({entity.entity_type}) | norm_key: {entity.norm_key}")

# Check MENTIONS edges from memory 1
mem1_id = memories[0][1]
mentions_edges = storage.get_edges_from_memory(mem1_id, edge_type="MENTIONS")
print_success(f"Memory 1 has {len(mentions_edges)} MENTIONS edges")
for edge in mentions_edges:
    entity = storage.get_entity(edge.to_node)
    print(f"   → {entity.canonical_name} (confidence: {edge.confidence:.2f})")

# Verify Alice appears in multiple memories (deduplication)
alice_norm = "PERSON:alice"
alice_entity = storage.find_entity_by_norm_key(alice_norm)
if alice_entity:
    alice_edges = storage.get_edges_to_entity(alice_entity.id)
    print_success(f"Entity 'Alice' appears in {len(alice_edges)} memories (deduplicated!)")

print_section("PART 3: RELATED Edges (Semantic Similarity)")
print_info("Checking RELATED edges between semantically similar memories...")

# Check RELATED edges from memory 1
related_edges = storage.get_edges_from_memory(mem1_id, edge_type="RELATED")
if len(related_edges) > 0:
    print_success(f"Memory 1 has {len(related_edges)} RELATED edges")
    for edge in related_edges:
        target = storage.get_memory(edge.to_node)
        if target:
            print(f"   → {target.content[:50]}... (similarity: {edge.weight:.3f})")
else:
    print_info("No RELATED edges (similarity threshold not met)")

print_section("PART 4: Graph Expansion - Via Entities")
print_info("Expanding from Memory 1 via shared entities...")

# Expand via entities
expansion = expand_via_entities(
    memory_ids=[mem1_id],
    storage=storage,
    limit=5,
)

if len(expansion) > 0:
    print_success(f"Found {len(expansion)} memories via shared entities:")
    for result in expansion:
        print(f"   → {result['memory'].content[:50]}...")
        print(f"     Shared entities: {len(result['shared_entities'])}, Score: {result['expansion_score']}")
else:
    print_info("No expansion results (no shared entities)")

print_section("PART 5: Graph Expansion - Via RELATED Edges")
print_info("Expanding from Memory 1 via RELATED edges...")

# Expand via RELATED
expansion = expand_via_related(
    memory_ids=[mem1_id],
    storage=storage,
    limit=5,
)

if len(expansion) > 0:
    print_success(f"Found {len(expansion)} memories via RELATED edges:")
    for result in expansion:
        print(f"   → {result['memory'].content[:50]}...")
        print(f"     Similarity: {result['similarity_score']:.3f}")
else:
    print_info("No expansion results (no RELATED edges)")

print_section("PART 6: Combined Graph Expansion")
print_info("Performing both entity and RELATED expansion together...")

combined = expand_with_graph(
    memory_ids=[mem1_id],
    storage=storage,
    entity_limit=3,
    related_limit=3,
)

print_success(f"Combined expansion results:")
print(f"   Via entities: {len(combined['via_entities'])} memories")
print(f"   Via RELATED:  {len(combined['via_related'])} memories")

print_section("PART 7: Entity Extraction Events")
print_info("Checking ENTITY_EXTRACTED events for auditability...")

events = event_storage.get_events_for_memory(mem1_id)
entity_events = [e for e in events if e.event_type == "ENTITY_EXTRACTED"]

if len(entity_events) > 0:
    event = entity_events[0]
    print_success(f"ENTITY_EXTRACTED event found:")
    print(f"   Model: {event.payload.get('model_name')}")
    print(f"   Prompt hash: {event.payload.get('prompt_hash')}")
    print(f"   Entities extracted: {event.payload.get('entity_count')}")
    print(f"   Min confidence: {event.payload.get('min_confidence')}")

print_section("M4 DEMO COMPLETE!")
print(f"{GREEN}{'='*70}")
print("✓ Entity extraction with LLM")
print("✓ Entity deduplication via norm_key")
print("✓ MENTIONS edges (Memory → Entity)")
print("✓ RELATED edges (Memory → Memory)")
print("✓ Graph expansion (via entities and RELATED)")
print("✓ Event logging for auditability")
print(f"{'='*70}{NC}")
print("")
print(f"Database stats:")
print(f"  Memories: {len(storage.get_all_memories())}")
print(f"  Entities: {len(storage.get_all_entities())}")
print("")

EOF

# Cleanup
rm -f "$DB"
echo -e "${GREEN}Demo complete! Temp database removed.${NC}"
