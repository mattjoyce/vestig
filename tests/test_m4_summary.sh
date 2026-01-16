#!/bin/bash
# M4 Summary Nodes Smoke Test

set -e  # Exit on error

# Suppress huggingface progress bars and warnings
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONWARNINGS="ignore"

echo "========================================="
echo "M4: Summary Nodes Smoke Test"
echo "========================================="
echo ""

# Setup
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR="$REPO_ROOT/tests/tmp"
TEST_DOC="$TEST_DIR/test_summary_doc.txt"
CONFIG_TEMPLATE="$REPO_ROOT/config_test.yaml"
CONFIG="$TEST_DIR/test_m4_summary.yaml"
GRAPH_NAME="vestig_m4_summary_${RANDOM}_${RANDOM}"
export FALKOR_HOST=localhost
export FALKOR_PORT=6379
trap 'redis-cli -h "$FALKOR_HOST" -p "$FALKOR_PORT" GRAPH.DELETE "$GRAPH_NAME" >/dev/null 2>&1 || true' EXIT

# Clean up previous test
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# Create test document with 7+ facts (ensures ≥5 memories)
echo "Creating test document with 7+ facts..."
cat > "$TEST_DOC" << 'EOF'
Technical Discussion: Team Structure and Project Status

Alice Johnson works at Acme Corp as the Senior Backend Engineer. She has been leading the API refactoring project since March 2024.

Bob Smith is a Python developer who joined the team in January 2024. He specializes in data pipeline development and has experience with Apache Airflow.

Charlie Rodriguez manages the DevOps team and is responsible for the CI/CD infrastructure. The team recently migrated to Kubernetes in production.

Diana Lee leads product design and has been working on the new user dashboard. The dashboard redesign is scheduled for release in Q2 2024.

Eve Taylor handles customer support and maintains the knowledge base documentation. She reports that the most common support tickets are related to authentication issues.

Frank Chen writes technical documentation and created the onboarding guide for new engineers. The documentation is hosted on Confluence.

The team meets every Monday at 10 AM for sprint planning. They use Jira for project tracking and Slack for team communication.
EOF

echo "✅ Test document created"
echo ""

# Build config with unique graph name
sed "s|graph_name:.*|graph_name: $GRAPH_NAME|g" "$CONFIG_TEMPLATE" > "$CONFIG"

# Set PYTHONPATH
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

# Test 1: Ingest document (should create summary)
echo "Test 1: Ingest document with summary generation"
echo "-----------------------------------------------"
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
python3 << EOF
import sys
from pathlib import Path
from vestig.core.db_falkordb import FalkorDBDatabase
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.ingestion import ingest_document
import yaml

# Load config
with open('$CONFIG') as f:
    config = yaml.safe_load(f)

# Initialize components
storage = FalkorDBDatabase(
    host=config['storage']['falkordb']['host'],
    port=config['storage']['falkordb']['port'],
    graph_name=config['storage']['falkordb']['graph_name'],
    config=config,
)
embedding_config = config['embedding']
embedding_engine = EmbeddingEngine(
    model_name=embedding_config['model'],
    expected_dimension=embedding_config['dimension'],
    normalize=embedding_config.get('normalize', True)
)
event_storage = storage.event_storage

# Ingest document
result = ingest_document(
    document_path='$TEST_DOC',
    storage=storage,
    embedding_engine=embedding_engine,
    extraction_model='claude-haiku-4.5',
    event_storage=event_storage,
    m4_config=config.get('m4', {}),
    verbose=True,
)

print(f"\n📊 Ingestion Results:")
print(f"   Memories extracted: {result.memories_extracted}")
print(f"   Memories committed: {result.memories_committed}")
print(f"   Memories deduplicated: {result.memories_deduplicated}")
print(f"   Entities created: {result.entities_created}")

if result.memories_committed < 5:
    print(f"\n⚠️  Warning: Only {result.memories_committed} memories committed (need ≥5 for summary)")
    sys.exit(1)

storage.close()
print("\n✅ Test 1 passed: Document ingested")
EOF

echo ""

# Test 2: Verify summary was created
echo "Test 2: Verify summary exists in database"
echo "------------------------------------------"
python3 << EOF
import sys
from vestig.core.db_falkordb import FalkorDBDatabase
import yaml

# Load config
with open('$CONFIG') as f:
    config = yaml.safe_load(f)

storage = FalkorDBDatabase(
    host=config['storage']['falkordb']['host'],
    port=config['storage']['falkordb']['port'],
    graph_name=config['storage']['falkordb']['graph_name'],
    config=config,
)

# Check for summary with kind=SUMMARY
summaries = [memory for memory in storage.get_all_memories() if memory.kind == 'SUMMARY']

if len(summaries) == 0:
    print("❌ No summary found in database")
    sys.exit(1)

if len(summaries) > 1:
    print(f"⚠️  Warning: Found {len(summaries)} summaries (expected 1)")

summary = summaries[0]
summary_id = summary.id
content = summary.content
metadata = summary.metadata or {}
print(f"✅ Found summary: {summary_id}")
print(f"\nSummary content preview:")
print(content[:200] + "...")

print(f"\nMetadata:")
print(f"   Title: {metadata.get('title')}")
print(f"   Themes: {metadata.get('themes')}")
print(f"   Memory count: {metadata.get('memory_count')}")

storage.close()
print("\n✅ Test 2 passed: Summary exists")
EOF

echo ""

# Test 3: Verify SUMMARIZES edges
echo "Test 3: Verify SUMMARIZES edges exist"
echo "--------------------------------------"
python3 << EOF
import sys
from vestig.core.db_falkordb import FalkorDBDatabase
import yaml

# Load config
with open('$CONFIG') as f:
    config = yaml.safe_load(f)

storage = FalkorDBDatabase(
    host=config['storage']['falkordb']['host'],
    port=config['storage']['falkordb']['port'],
    graph_name=config['storage']['falkordb']['graph_name'],
    config=config,
)

# Get summary ID
summaries = [memory for memory in storage.get_all_memories() if memory.kind == 'SUMMARY']
summary_id = summaries[0].id if summaries else None
if not summary_id:
    print("❌ No summary found in database")
    sys.exit(1)

# Get SUMMARIZES edges
edges = storage.get_edges_from_memory(summary_id, edge_type='SUMMARIZES')

if len(edges) == 0:
    print("❌ No SUMMARIZES edges found")
    sys.exit(1)

print(f"✅ Found {len(edges)} SUMMARIZES edges")
print(f"   Summary → Memory edges: {len(edges)}")

# Verify edges point to actual memories
for edge in edges[:3]:  # Show first 3
    mem = storage.get_memory(edge.to_node)
    if mem:
        content_preview = mem.content[:50] + "..." if len(mem.content) > 50 else mem.content
        print(f"   - {edge.to_node}: {content_preview}")

storage.close()
print("\n✅ Test 3 passed: SUMMARIZES edges exist")
EOF

echo ""

# Test 4: Verify SUMMARY_CREATED event
echo "Test 4: Verify SUMMARY_CREATED event logged"
echo "--------------------------------------------"
python3 << EOF
import sys
from vestig.core.db_falkordb import FalkorDBDatabase
import yaml

# Load config
with open('$CONFIG') as f:
    config = yaml.safe_load(f)

storage = FalkorDBDatabase(
    host=config['storage']['falkordb']['host'],
    port=config['storage']['falkordb']['port'],
    graph_name=config['storage']['falkordb']['graph_name'],
    config=config,
)
event_storage = storage.event_storage

# Check for SUMMARY_CREATED event
summaries = [memory for memory in storage.get_all_memories() if memory.kind == 'SUMMARY']
if not summaries:
    print("❌ No summary found in database")
    sys.exit(1)

summary_id = summaries[0].id
events = [
    event
    for event in event_storage.get_events_for_memory(summary_id)
    if event.event_type == 'SUMMARY_CREATED'
]

if len(events) == 0:
    print("❌ No SUMMARY_CREATED event found")
    sys.exit(1)

event = events[0]
event_id = event.event_id
memory_id = event.memory_id
print(f"✅ Found SUMMARY_CREATED event: {event_id}")
print(f"   Memory ID: {memory_id}")

payload = event.payload
print(f"   Payload: {payload}")

storage.close()
print("\n✅ Test 4 passed: Event logged")
EOF

echo ""

# Test 5: Test idempotency (re-ingest should not create duplicate)
echo "Test 5: Test idempotency (re-ingest)"
echo "------------------------------------"
python3 << EOF
import sys
from pathlib import Path
from vestig.core.db_falkordb import FalkorDBDatabase
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.ingestion import ingest_document
import yaml

# Load config
with open('$CONFIG') as f:
    config = yaml.safe_load(f)

# Initialize components
storage = FalkorDBDatabase(
    host=config['storage']['falkordb']['host'],
    port=config['storage']['falkordb']['port'],
    graph_name=config['storage']['falkordb']['graph_name'],
    config=config,
)
embedding_config = config['embedding']
embedding_engine = EmbeddingEngine(
    model_name=embedding_config['model'],
    expected_dimension=embedding_config['dimension'],
    normalize=embedding_config.get('normalize', True)
)
event_storage = storage.event_storage

# Count summaries before re-ingest
count_before = len([memory for memory in storage.get_all_memories() if memory.kind == 'SUMMARY'])

# Re-ingest same document
result = ingest_document(
    document_path='$TEST_DOC',
    storage=storage,
    embedding_engine=embedding_engine,
    extraction_model='claude-haiku-4.5',
    event_storage=event_storage,
    m4_config=config.get('m4', {}),
    verbose=False,
)

# Count summaries after re-ingest
count_after = len([memory for memory in storage.get_all_memories() if memory.kind == 'SUMMARY'])

print(f"Summaries before: {count_before}")
print(f"Summaries after:  {count_after}")

if count_after != count_before:
    print(f"❌ Duplicate summary created! Expected {count_before}, got {count_after}")
    sys.exit(1)

storage.close()
print("\n✅ Test 5 passed: Idempotency works")
EOF

echo ""

# Test 6: Verify summary can be retrieved by artifact_ref
echo "Test 6: Test get_summary_for_artifact() helper"
echo "-----------------------------------------------"
python3 << EOF
import sys
from pathlib import Path
from vestig.core.db_falkordb import FalkorDBDatabase
import yaml

# Load config
with open('$CONFIG') as f:
    config = yaml.safe_load(f)

storage = FalkorDBDatabase(
    host=config['storage']['falkordb']['host'],
    port=config['storage']['falkordb']['port'],
    graph_name=config['storage']['falkordb']['graph_name'],
    config=config,
)

# Get summary using helper method
artifact_ref = Path('$TEST_DOC').name
summary = storage.get_summary_for_artifact(artifact_ref)

if summary is None:
    print(f"❌ get_summary_for_artifact('{artifact_ref}') returned None")
    sys.exit(1)

print(f"✅ Retrieved summary: {summary.id}")
print(f"   Artifact ref: {summary.metadata.get('artifact_ref')}")
print(f"   Title: {summary.metadata.get('title')}")

storage.close()
print("\n✅ Test 6 passed: Helper method works")
EOF

echo ""

# All tests passed
echo "========================================="
echo "✅ ALL M4 TESTS PASSED!"
echo "========================================="
echo ""
echo "Summary:"
echo "- Graph initialization: ✅"
echo "- Summary generation: ✅"
echo "- SUMMARIZES edges: ✅"
echo "- SUMMARY_CREATED event: ✅"
echo "- Idempotency: ✅"
echo "- Helper methods: ✅"
echo ""
echo "M4 Summary Nodes implementation complete! 🎉"
