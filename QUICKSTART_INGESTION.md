# Quick Start: Ingest Session & Query

## Step 1: Create Test Config

Create `config_test.yaml`:
```yaml
# Test config - uses Haiku for cost efficiency
storage:
  backend: falkordb
  falkordb:
    host: localhost
    port: 6379
    graph_name: vestig_test

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

# Ingest recursively with a glob
python -m vestig.core.cli --config config_test.yaml ingest "sessions/**/*.jsonl" \
  --format claude-session \
  --recurse

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

# Recall with LLM-ready formatting
python -m vestig.core.cli --config config_test.yaml memory recall "PostgreSQL optimization"

# Recall with explanation for each result
python -m vestig.core.cli --config config_test.yaml memory recall "PostgreSQL optimization" --explain

# Show specific memory
python -m vestig.core.cli --config config_test.yaml memory show mem_<id>

# Set FalkorDB env to match config_test.yaml
export FALKOR_HOST=localhost
export FALKOR_PORT=6379
export FALKOR_GRAPH=vestig_test

# Show all memories (raw)
scripts/falkor "MATCH (m:Memory) RETURN m.id, substring(m.content, 0, 60) LIMIT 10"

# Show all entities
scripts/falkor "MATCH (e:Entity) RETURN e.canonical_name, e.entity_type LIMIT 10"

# Show MENTIONS edges
scripts/falkor "
  MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
  RETURN m.content, e.canonical_name, e.entity_type
  LIMIT 10
"
```

## Step 5: Inspect Results

```bash
# Count what was created
echo "Memories: $(scripts/falkor 'MATCH (m:Memory) RETURN COUNT(m)' | tail -1)"
echo "Entities: $(scripts/falkor 'MATCH (e:Entity) RETURN COUNT(e)' | tail -1)"
echo "Edges: $(scripts/falkor 'MATCH ()-[r]->() RETURN COUNT(r)' | tail -1)"
```

## Cleanup

```bash
# Drop the test graph if needed
# redis-cli -h "$FALKOR_HOST" -p "$FALKOR_PORT" GRAPH.DELETE "$FALKOR_GRAPH"
```
