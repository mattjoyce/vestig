# Quick Start: Ingest Session & Query

## Step 1: Create Test Config

Create `config_test.yaml`:
```yaml
# Test config - uses Haiku for cost efficiency
storage:
  db_path: "test_vestig.db"  # Fresh database

embedding:
  model: "BAAI/bge-m3"
  dimension: 1024
  normalize: true

hygiene:
  min_chars: 12
  max_chars: 4000
  normalize_whitespace: true
  reject_exact_duplicates: true
  near_duplicate:
    enabled: true
    threshold: 0.92
    skip_manual_source: true

m3:
  temporal:
    enabled: true
  events:
    enabled: true
  tracerank:
    enabled: true
    tau_days: 21.0
    cooldown_hours: 24.0
    burst_discount: 0.2
    k: 0.35

m4:
  entity_types:
    allowed_types:
      - PERSON
      - ORG
      - SYSTEM
      - PROJECT
      - PLACE

  entity_extraction:
    enabled: true
    mode: llm
    llm:
      model: claude-haiku-4.5            # Fast & cheap for testing
      max_entities_per_memory: 10
      min_confidence: 0.75
      store_low_confidence: false
      max_evidence_length: 200
    heuristics:
      strip_titles: true
      normalize_org_suffixes: true
      reject_garbage: true

  edge_creation:
    mentions:
      enabled: true
      confidence_gated: true
    related:
      enabled: true
      similarity_threshold: 0.6
      max_edges_per_memory: 10
```

## Step 2: Set Anthropic API Key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Step 3: Ingest Session File

```bash
# Ingest using config (model, chunk size, etc. from config_test.yaml)
python -m vestig.core.cli --config config_test.yaml ingest your_session.txt

# Ingest a Claude Code session export (JSONL)
python -m vestig.core.cli --config config_test.yaml ingest session.jsonl \
  --format claude-session

# Ingest multiple session files with a glob
python -m vestig.core.cli --config config_test.yaml ingest "session*.jsonl" \
  --format claude-session

# Force a project entity onto every memory
python -m vestig.core.cli --config config_test.yaml ingest session.jsonl \
  --format claude-session \
  --force-entity PROJECT:vestig

# Or override config with CLI args
python -m vestig.core.cli --config config_test.yaml ingest your_session.txt \
  --model claude-haiku-4.5 \
  --min-confidence 0.6
```

**What happens:**
- Document chunked by ~5k tokens
- Haiku extracts discrete memories from each chunk
- Each memory committed with entity extraction
- Entities linked via MENTIONS edges
- Similar memories linked via RELATED edges

## Step 4: Query Memories

```bash
# Search by semantic similarity
python -m vestig.core.cli --config config_test.yaml memory search "PostgreSQL optimization"

# Show specific memory
python -m vestig.core.cli --config config_test.yaml memory show mem_<id>

# Show all memories (raw)
sqlite3 test_vestig.db "SELECT id, substr(content, 1, 60) FROM memories LIMIT 10;"

# Show all entities
sqlite3 test_vestig.db "SELECT canonical_name, entity_type FROM entities;"

# Show MENTIONS edges
sqlite3 test_vestig.db "
  SELECT m.content, e.canonical_name, e.entity_type
  FROM edges ed
  JOIN memories m ON ed.from_node = m.id
  JOIN entities e ON ed.to_node = e.id
  WHERE ed.edge_type = 'MENTIONS'
  LIMIT 10;
"
```

## Step 5: Inspect Results

```bash
# Count what was created
echo "Memories: $(sqlite3 test_vestig.db 'SELECT COUNT(*) FROM memories;')"
echo "Entities: $(sqlite3 test_vestig.db 'SELECT COUNT(*) FROM entities;')"
echo "Edges: $(sqlite3 test_vestig.db 'SELECT COUNT(*) FROM edges;')"
```

## Cleanup

```bash
rm test_vestig.db
```
