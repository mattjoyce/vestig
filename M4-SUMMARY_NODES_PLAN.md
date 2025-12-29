# M4: Summary Nodes Implementation Plan

## Overview

Add first-class SUMMARY nodes to Vestig that automatically summarize extracted memories during ingestion. Summaries will:
- Be created when ≥5 memories are extracted from a single ingest run
- Summarize extracted memories (not raw content) via LLM
- Link to constituent memories via SUMMARIZES edges
- Support idempotent re-runs (no duplicates)
- Enable token budget control in retrieval

## Core Design Principle

**Reuse existing Memory table with a `kind` discriminator field** ("MEMORY" | "SUMMARY") rather than creating a separate table. This keeps the data model simple and leverages existing bi-temporal, embedding, and event infrastructure.

**Grouping key**: Use `artifact_ref` (filename/session ID in events) to identify which memories belong to a single ingest run.

## Critical Files to Modify

### 1. `/src/vestig/core/storage.py`
**Schema Migration** (in `_init_schema()` around line 80):
- Add `kind TEXT DEFAULT 'MEMORY'` column to memories table
- Create index on `kind` field for fast summary queries
- Update `store_memory()` to accept optional `kind` parameter
- Add helper method `get_summary_for_artifact(artifact_ref)` to check for existing summaries

### 2. `/src/vestig/core/models.py`
**Model Updates**:
- Add "SUMMARIZES" to allowed edge types in `EdgeNode.create()` validation (line 202)
- Add "SUMMARY_CREATED" to event type enum/docstring (line 238)

### 3. `/src/vestig/core/prompts.yaml`
**New Prompt**:
```yaml
summarize_memories: |
  Analyze these memories extracted from a document and create a concise summary.

  GUIDELINES:
  - Create 3-7 bullet points capturing key themes and insights
  - Be factual and grounded in the provided memories
  - Focus on what matters: technical decisions, key facts, important entities
  - Do NOT invent information not present in the memories
  - Reference specific memory IDs when highlighting important details

  MEMORIES TO SUMMARIZE:
  {{memories_list}}

  OUTPUT (valid JSON only):
  {
    "summary_bullets": ["First insight (ref: mem_xxx)", "Second insight", ...],
    "confidence": 0.0-1.0,
    "key_themes": ["theme1", "theme2", "theme3"]
  }
```

### 4. `/src/vestig/core/ingestion.py`
**Add Summary Generation Logic**:

**New Pydantic Schema** (after line 130):
```python
class SummaryResult(BaseModel):
    summary_bullets: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    key_themes: list[str] = Field(default_factory=list)
```

**New Function** `generate_summary()`:
- Takes list of MemoryNode objects + model + artifact_ref
- Formats memories for prompt (truncate to 500 chars each for efficiency)
- Calls LLM with `summarize_memories` prompt and SummaryResult schema
- Returns (summary_content, confidence, metadata)

**New Function** `commit_summary()`:
- **Idempotency check**: Query for existing summary with matching `artifact_ref` in metadata
- If found, return existing ID (skip creation)
- If not found, create summary MemoryNode with `kind='SUMMARY'`
- Generate embedding for summary content
- Create SUMMARIZES edges from summary to each memory
- Log SUMMARY_CREATED event
- Atomic transaction for all operations

**Integration Point** (after line 522, before return):
```python
# M4: Generate summary if ≥5 memories committed
if memories_committed >= 5:
    # Query events to get all memories from this ingest (by artifact_ref)
    # Call generate_summary() with extracted memories
    # Call commit_summary() to persist summary + edges + event
```

### 5. `/src/vestig/core/retrieval.py` (Optional Enhancement)
**Add Summary-Aware Retrieval**:
- Add `use_summaries` boolean parameter to `search_memories()`
- Add `summary_top_k` parameter (default 3) for detail memory count
- When enabled:
  - Separate summaries from regular memories in results
  - For each top summary, expand SUMMARIZES edges to get detail memories
  - Return: summary + top-k detail memories per summary
  - Provides token budget control

### 6. `/src/vestig/core/cli.py` (Optional)
**CLI Support**:
- Add `--use-summaries` flag to `cmd_recall()` (default false)
- Add `--summary-top-k` argument (default 3)
- Pass flags to `search_memories()`

## Implementation Sequence

### Phase 1: Schema & Models (30 min)
1. Add `kind` column to memories table with migration
2. Update EdgeNode to allow SUMMARIZES edge type
3. Update EventNode to include SUMMARY_CREATED
4. Test: Verify schema migration works on existing DBs

### Phase 2: Summary Generation (60 min)
1. Add `summarize_memories` prompt to prompts.yaml
2. Create SummaryResult Pydantic schema
3. Implement `generate_summary()` function
4. Implement `commit_summary()` with idempotency
5. Test: Unit test summary generation with mock LLM

### Phase 3: Ingest Integration (30 min)
1. Add summary generation logic to `ingest_document()`
2. Query events by artifact_ref to get memories from current run
3. Call generate/commit only if ≥5 memories
4. Test: Smoke test with real document

### Phase 4: Retrieval Enhancement (45 min - Optional)
1. Add `use_summaries` parameter to `search_memories()`
2. Implement summary expansion logic (traverse SUMMARIZES edges)
3. Add CLI flags
4. Test: Verify summary + detail retrieval works

## Idempotency Strategy

**Problem**: Re-ingesting the same document should not create duplicate summaries.

**Solution**: Use `artifact_ref` (stored in metadata) as natural key:
1. Before creating summary, query: `SELECT id FROM memories WHERE kind='SUMMARY'` and check metadata['artifact_ref']
2. If match found → return existing ID (idempotent success)
3. If not found → create new summary

**Edge Case**: If re-ingest has all duplicates (0 new memories), threshold not met, no summary created. Previous summary remains valid.

## Testing Plan

### Unit Tests
- `test_summary_generation()`: Test LLM prompt formatting and response parsing
- `test_summary_idempotency()`: Verify duplicate summaries not created
- `test_summarizes_edges()`: Verify edge creation and traversal

### Integration Test
- `test_ingest_with_summary.py`:
  ```python
  # Ingest document with 7 memories
  result = ingest_document(path, ...)
  assert result.memories_committed == 7

  # Verify summary created
  summary = storage.get_summary_for_artifact(path.name)
  assert summary is not None
  assert summary.metadata['memory_count'] == 7

  # Verify edges
  edges = storage.get_edges_from_memory(summary.id, edge_type='SUMMARIZES')
  assert len(edges) == 7

  # Re-ingest (idempotency check)
  result2 = ingest_document(path, ...)
  summary2 = storage.get_summary_for_artifact(path.name)
  assert summary2.id == summary.id  # Same summary, not duplicate
  ```

### Smoke Test
```bash
# Create test document
echo "Alice works at Acme Corp. Bob is a Python developer. Charlie manages the DevOps team. Diana leads product design. Eve handles customer support. Frank writes documentation." > test_doc.txt

# Ingest
vestig ingest test_doc.txt --verbose

# Verify summary in DB
sqlite3 test/test_vestig.db "SELECT id, content FROM memories WHERE kind='SUMMARY';"

# Test recall
vestig memory recall "team structure" --use-summaries

# Re-ingest (should be idempotent)
vestig ingest test_doc.txt --verbose  # Should print "Summary already exists"
```

## Configuration

Add to `test/config.yaml`:
```yaml
m4:
  summaries:
    enabled: true
    min_memories_for_summary: 5
    summary_model: claude-haiku-4.5  # Or same as ingestion model
```

## Edge Cases & Risks

### Risk 1: LLM Failure
- **Mitigation**: Wrap summary generation in try/except, log error, continue ingest
- Summary can be regenerated manually later if needed

### Risk 2: Summary Quality
- **Mitigation**: Confidence threshold (0.7), prompt forbids hallucination, grounded references

### Risk 3: Large Ingest (100+ memories)
- **Mitigation**: Truncate memory content to 500 chars each in prompt, stays within LLM limits

### Edge Case: Entity Extraction on Summaries
- **Decision**: Skip entity extraction for summaries (check `source='summary_generation'` in commit flow)
- Summaries are meta-content, entity extraction from them may introduce noise

## Success Criteria

✅ Ingest with ≥5 memories creates exactly 1 summary node
✅ Summary has `kind='SUMMARY'` and valid embedding
✅ SUMMARIZES edges link summary → memories
✅ SUMMARY_CREATED event logged
✅ Re-ingest is idempotent (same summary ID returned)
✅ Retrieval with `--use-summaries` returns summary + detail memories
✅ Existing ingests without summaries continue to work

## Non-Goals (Out of Scope)

- Multi-level hierarchy (chunk summaries → doc summaries → persona summaries)
- Contradiction resolution / supersession logic for summaries
- Summary of summaries / periodic consolidation
- Automatic re-summarization when memories are deprecated

These can be added in future milestones if needed.
