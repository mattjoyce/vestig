# Temporal Extraction Architecture - Implementation Plan

## Goal
Implement "When-Learned vs When-It-Was" distinction by having parsers extract temporal hints that flow through the ingestion pipeline to MemoryNode initialization.

## User Requirements
- **Parser-specific extraction**: Each parser extracts temporal hints from domain-specific sources
- **Priority**: Record-level timestamps > file-level metadata > current time
- **Enabled by default**: Always attempt extraction; graceful fallback to t_valid=now
- **Solve upstream**: Extend data structures so parsers pass temporal context forward

## Architecture Overview

```
File Input (with metadata)
    ↓
Parser extracts temporal hints (record timestamp / file mtime / now)
    ↓
normalize_document_text() → (text, format, TemporalHints)
    ↓
extract_memories_from_chunk() → list[ExtractedMemory] (with temporal fields)
    ↓
commit_memory() → receives temporal hints
    ↓
MemoryNode.create() → uses hints for t_valid initialization
    ↓
Database: t_valid ≠ t_created (When-It-Was ≠ When-Learned)
```

## Implementation Phases

### Phase 1: Data Structure Enhancement

**File**: `src/vestig/core/ingestion.py` (after line 28)

1. **Add `TemporalHints` dataclass** (~80 lines):
   - Fields: `t_valid`, `stability`, `extraction_method`, `evidence`
   - Factory methods: `from_now()`, `from_file_mtime(path)`, `from_timestamp(ts, evidence)`
   - Purpose: Container for parser-extracted temporal metadata

2. **Extend `ExtractedMemory` dataclass** (lines 21-28):
   - Add fields: `t_valid_hint: str | None = None`
   - Add fields: `temporal_stability_hint: str | None = None`
   - Add fields: `temporal_evidence: str | None = None`
   - Backward compatible: all optional with None defaults

### Phase 2: Parser Implementation

**File**: `src/vestig/core/ingest_sources.py`

1. **Update `normalize_document_text()` signature** (lines 19-39):
   - Current: `-> tuple[str, str]` (text, format)
   - New: `-> tuple[str, str, TemporalHints]` (text, format, temporal_hints)

2. **Implement `extract_claude_session_text_with_temporal()`** (new function ~70 lines):
   - Extract `timestamp` field from JSONL events
   - Use **earliest** timestamp as t_valid (conversation start time)
   - Graceful fallback: if no timestamps → `TemporalHints.from_now()`
   - Return: `(normalized_text, temporal_hints)`

3. **Plain text parser** (handled in normalize_document_text):
   - If `path` available → `TemporalHints.from_file_mtime(path)`
   - Otherwise → `TemporalHints.from_now()`

**Temporal Hint Precedence**:
- claude-session: JSONL event timestamp (earliest)
- plain: file mtime
- fallback: current time

### Phase 3: Commitment Pipeline Integration

**File**: `src/vestig/core/ingestion.py` (lines 229-427)

1. **Update `ingest_document()` flow** (line 278):
   ```python
   # OLD: text, resolved_format = normalize_document_text(...)
   # NEW:
   text, resolved_format, document_temporal_hints = normalize_document_text(
       text, source_format=source_format, format_config=format_config, path=path
   )
   ```

2. **Pass temporal hints to extraction** (line 310):
   ```python
   extracted = extract_memories_from_chunk(
       chunk, model=extraction_model, min_confidence=min_confidence,
       temporal_hints=document_temporal_hints  # NEW
   )
   ```

3. **Update `extract_memories_from_chunk()` signature** (lines 157-227):
   - Add parameter: `temporal_hints: TemporalHints | None = None`
   - Attach temporal fields to each `ExtractedMemory` object
   - If hints provided: populate `t_valid_hint`, `temporal_stability_hint`, `temporal_evidence`

4. **Pass temporal hints to commit** (line 361):
   ```python
   outcome = commit_memory(
       content=memory.content,
       # ... existing params ...
       temporal_hints=memory,  # NEW: Pass ExtractedMemory with temporal fields
   )
   ```

**File**: `src/vestig/core/commitment.py` (lines 88-324)

5. **Update `commit_memory()` signature** (line 88):
   - Add parameter: `temporal_hints: ExtractedMemory | None = None`

6. **Extract temporal fields** (after line 175):
   ```python
   t_valid_hint = None
   temporal_stability_hint = None
   if temporal_hints and isinstance(temporal_hints, ExtractedMemory):
       t_valid_hint = temporal_hints.t_valid_hint
       temporal_stability_hint = temporal_hints.temporal_stability_hint
   ```

7. **Pass to MemoryNode.create()** (line 255):
   ```python
   node = MemoryNode.create(
       memory_id=memory_id, content=normalized, embedding=embedding,
       source=source, tags=tags, content_hash=content_hash,
       t_valid_hint=t_valid_hint,  # NEW
       temporal_stability_hint=temporal_stability_hint,  # NEW
   )
   ```

**File**: `src/vestig/core/models.py` (lines 33-84)

8. **Update `MemoryNode.create()` signature** (line 34):
   - Add parameters: `t_valid_hint: str | None = None`, `temporal_stability_hint: str | None = None`

9. **Use temporal hints in initialization** (lines 67-81):
   ```python
   now = datetime.now(timezone.utc).isoformat()

   # Use hints if provided, otherwise default to now
   t_valid = t_valid_hint if t_valid_hint else now
   temporal_stability = temporal_stability_hint if temporal_stability_hint else "unknown"

   return cls(
       # ...
       t_valid=t_valid,  # Uses hint or now
       t_created=now,    # Transaction time always now
       temporal_stability=temporal_stability,  # Uses hint or "unknown"
   )
   ```

### Phase 4: Testing & Validation

**Create**: `tests/test_temporal_extraction.py` (~150 lines)

1. **Unit tests**:
   - `test_temporal_hints_from_now()` - Fallback to current time
   - `test_temporal_hints_from_file_mtime()` - Extract from file metadata
   - `test_claude_session_temporal_extraction()` - JSONL timestamp extraction
   - `test_claude_session_no_timestamps()` - Graceful fallback
   - `test_extracted_memory_with_temporal_hints()` - Data structure
   - `test_memory_node_create_with_temporal_hints()` - Initialization with hints
   - `test_memory_node_create_without_temporal_hints()` - Backward compatibility

2. **Integration tests**:
   - `test_ingest_claude_session_with_temporal()` - End-to-end JSONL ingestion
   - `test_ingest_plain_text_with_file_mtime()` - Plain text with file metadata

3. **Manual validation**:
   - Ingest session.jsonl → verify `t_valid` from JSONL timestamps
   - Ingest plain text → verify `t_valid` from file mtime
   - Check database: `SELECT id, t_valid, t_created WHERE t_valid != t_created`

## Critical Files to Modify

1. **`src/vestig/core/ingestion.py`** (lines 21-427)
   - Add TemporalHints class
   - Extend ExtractedMemory
   - Update extract_memories_from_chunk()
   - Update ingest_document() flow

2. **`src/vestig/core/ingest_sources.py`** (lines 19-151)
   - Update normalize_document_text() signature
   - Add extract_claude_session_text_with_temporal()
   - Implement parser temporal extraction

3. **`src/vestig/core/models.py`** (lines 33-84)
   - Update MemoryNode.create() signature
   - Use temporal hints in initialization

4. **`src/vestig/core/commitment.py`** (lines 88-324)
   - Add temporal_hints parameter
   - Thread hints to MemoryNode.create()

5. **`tests/test_temporal_extraction.py`** (new file)
   - Comprehensive test suite

## Backward Compatibility

- ✅ All new parameters optional with sensible defaults
- ✅ Existing callers don't break (graceful fallback to `now`)
- ✅ No database schema changes (temporal fields already exist)
- ✅ No migration required

## Implementation Sequence

1. Add `TemporalHints` class and extend `ExtractedMemory` in `ingestion.py`
2. Update `MemoryNode.create()` in `models.py` to accept hints
3. Implement parser temporal extraction in `ingest_sources.py`
4. Thread hints through `ingest_document()` and `commit_memory()`
5. Add tests and validate end-to-end

## Expected Outcome

**Before**:
```sql
SELECT id, t_valid, t_created FROM memories LIMIT 3;
-- All rows have t_valid = t_created (learned now about something that just happened)
```

**After**:
```sql
SELECT id, t_valid, t_created FROM memories LIMIT 3;
-- claude-session: t_valid = JSONL timestamp, t_created = ingestion time
-- plain text: t_valid = file mtime, t_created = ingestion time
-- manual add: t_valid = t_created = now (unchanged)
```

**Success Criteria**:
- ✅ Claude-session ingestion extracts JSONL timestamps for t_valid
- ✅ Plain text ingestion uses file mtime for t_valid
- ✅ Manual `vestig add` command still works (fallback to now)
- ✅ Database shows `t_valid != t_created` for ingested documents
- ✅ No breaking changes to existing code

## Estimated Effort
- Implementation: 3-4 hours
- Testing: 2 hours
- Total: 5-6 hours
