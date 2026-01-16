# Vestig Testing Strategy: Component Test Implementation Plan

## Context

**Problem Identified:** Production bug (Issue #7 vecf32() wrapper) passed 17/21 tests but failed immediately in production.

**Root Cause:** Zero component-level tests for storage layer methods. Integration tests used high-level APIs that bypassed the broken code path.

**Decision:** Implement Option B (Strategic Component Tests) - hybrid 3-tier testing pyramid.

---

## 3-Tier Testing Pyramid

```
        Integration Tests (Few)
           /            \
    Component Tests (More)  ← WE ARE HERE
         /                  \
  Pure Unit Tests (Most)
```

### Tier 1: Pure Unit Tests (Fast, Isolated)
- **What:** Test pure functions and algorithms
- **No database required**
- **Examples:** Algorithm logic (TraceRank ✓), model creation, utility functions
- **Status:** Good coverage for TraceRank, norm_key computation

### Tier 2: Component Tests (Storage & Core Logic) ← **GAP TO FILL**
- **What:** Test ONE method/component at a time against real dependencies
- **Use real FalkorDB** - no mocking
- **Examples:** Vector search methods, CRUD operations, entity matching
- **Status:** Missing for critical paths

### Tier 3: Integration Tests (E2E Workflows)
- **What:** Test complete user scenarios
- **Examples:** commit → reinforce → search → verify ranking
- **Status:** Good coverage for major workflows

---

## Phase 1: Close Critical Gaps (High Priority)

### 1.1 Storage Layer Component Tests

**File:** `tests/test_storage_methods.py`

**Purpose:** Test db_falkordb.py methods in isolation against real FalkorDB

**Test Coverage:**

#### Vector Search Methods (Critical - missed vecf32 bug)
- `test_search_memories_by_vector_basic()`
  - Create memory with known embedding
  - Search with query vector
  - Verify results structure, scores (0-1 range), sorting

- `test_search_memories_by_vector_with_kind_filter()`
  - Create MEMORY and SUMMARY nodes
  - Filter by kind
  - Verify only requested kind returned

- `test_search_memories_by_vector_with_expired_filter()`
  - Create expired and valid memories
  - Test include_expired=True/False
  - Verify filtering works

- `test_search_entities_by_vector_basic()`
  - Create entities with embeddings
  - Search with query vector
  - Verify results and scores

- `test_search_entities_by_vector_with_type_filter()`
  - Create multiple entity types (PERSON, ORG, SYSTEM)
  - Filter by entity_type
  - Verify only requested type returned

#### CRUD Operations
- `test_store_and_retrieve_memory()`
  - Store memory with vecf32 embedding
  - Retrieve by ID
  - Verify all fields preserved

- `test_store_and_retrieve_entity()`
  - Store entity with embedding
  - Retrieve by ID
  - Verify norm_key, embedding format

- `test_find_entity_by_norm_key()`
  - Store entity
  - Find by normalized key
  - Verify deduplication logic

#### Edge Operations
- `test_create_mentions_edge()`
  - Create memory and entity
  - Create MENTIONS edge
  - Verify edge properties (confidence, weight)

- `test_create_related_edge()`
  - Create two memories with similar embeddings
  - Create RELATED edge
  - Verify similarity_score stored correctly

#### Query Building & Filtering
- `test_memory_query_with_multiple_filters()`
  - Combine kind, expired, date range filters
  - Verify Cypher WHERE clauses work correctly

**Estimated:** 15-20 tests, ~300-400 lines

---

### 1.2 Embedding Engine Component Tests

**File:** `tests/test_embeddings.py`

**Purpose:** Test embeddings.py in isolation with real embedding models

**Test Coverage:**

- `test_embedding_dimension_validation()`
  - Configure expected dimension (e.g., 768)
  - Generate embedding
  - Verify dimension matches, raise error if mismatch

- `test_embedding_normalization()`
  - Generate embedding with normalize=True
  - Verify L2 norm ≈ 1.0
  - Test with normalize=False (no normalization)

- `test_embedding_consistency()`
  - Embed same text twice
  - Verify embeddings are identical (deterministic)

- `test_embedding_model_failure_handling()`
  - Mock LLM provider failure
  - Verify error handling (retry logic, fallback, clear error messages)

- `test_batch_embedding_generation()`
  - Generate embeddings for multiple texts
  - Verify batch processing works
  - Check all dimensions match

**Estimated:** 8-10 tests, ~200 lines

---

### 1.3 Entity Ontology Component Tests

**File:** `tests/test_entity_ontology.py`

**Purpose:** Test entity type matching, synonym resolution, tier logic

**Test Coverage:**

- `test_entity_type_exact_match()`
  - Load ontology
  - Match exact type names (PERSON, ORG, SYSTEM)
  - Verify correct entity definition returned

- `test_entity_type_synonym_match()`
  - Match "organization" → ORG
  - Match "company" → ORG
  - Match "individual" → PERSON
  - Verify synonym resolution works

- `test_entity_type_case_insensitive()`
  - Match "person", "PERSON", "Person"
  - Verify all resolve to same type

- `test_entity_tier_logic()`
  - Verify tier 1 (core entities: PERSON, ORG, SYSTEM)
  - Verify tier 2 (knowledge entities: PRINCIPLE, HEURISTIC)
  - Verify tier 3 (meta: METALEARNING)

- `test_canonical_name_matching()`
  - Test example entities match their types
  - "Matt Joyce" → PERSON
  - "Vestig" → SYSTEM

- `test_unknown_entity_type_handling()`
  - Query unknown type "FOOBAR"
  - Verify graceful failure (None or error)

**Estimated:** 6-8 tests, ~150 lines

---

### 1.4 Model Validation Component Tests (Extend existing)

**File:** `tests/test_m4_item1.py` (extend)

**Current Coverage:** ✓ compute_norm_key, basic entity creation

**Add Coverage:**

- `test_memory_node_validation()`
  - Create MemoryNode with invalid fields
  - Verify validation logic

- `test_entity_node_embedding_format()`
  - Verify embedding is list[float], not JSON string
  - Test with None embedding (should be allowed)

**Estimated:** 3-5 additional tests

---

## Phase 2: Opportunistic Testing (Test When Modified)

### Strategy
- **Rule:** When modifying a module, add component tests for the changed logic
- **Target modules:** retrieval.py, commitment.py, graph.py

### Guidelines

#### retrieval.py
- If adding new ranking/scoring functions → component test for function
- If modifying entity-based retrieval → test entity matching logic
- Keep integration tests for end-to-end recall

#### commitment.py
- If changing deduplication logic → component test for hash computation
- If modifying memory merge → test merge rules in isolation

#### graph.py
- If adding traversal algorithms → component test for path finding
- If modifying expansion logic → test expansion rules

---

## Phase 3: Testing Guidelines for Future Development

### New Feature Testing Checklist

When adding new functionality:

1. **Pure function?** → Write unit test (no DB)
2. **Storage method?** → Write component test (real DB, isolated call)
3. **User workflow?** → Write integration test (end-to-end)

### Code Review Checklist

- [ ] Does this PR add new storage methods? → Requires component tests
- [ ] Does this PR modify critical paths (vector search, retrieval)? → Requires tests
- [ ] Do all tests pass in isolation? → Run `pytest tests/test_<module>.py`
- [ ] Do all tests pass together? → Run `pytest tests/`

### Testing Decision Tree

```
Does the module have complex logic?
├─ NO → Skip (or minimal smoke test)
└─ YES → Is it pure functions or isolated components?
    ├─ NO (orchestration/LLM/UI) → Integration tests only
    └─ YES → Is it currently causing bugs or frequently changing?
        ├─ NO → Test when you modify it
        └─ YES → Write component tests NOW (HIGH PRIORITY)
```

---

## Module Priority Matrix

| Module | Priority | Test Type | Rationale |
|--------|----------|-----------|-----------|
| db_falkordb.py | 🔴 HIGH | Component | Critical storage, complex queries, missed bug |
| embeddings.py | 🔴 HIGH | Component | Critical path, dimension validation |
| entity_ontology.py | 🔴 HIGH | Component | Type matching, synonym resolution |
| models.py | 🟡 MED | Unit | Partially done, extend validation tests |
| retrieval.py | 🟡 MED | Integration | Good coverage, test when modified |
| graph.py | 🟡 MED | Component | Test traversal algorithms in isolation |
| commitment.py | 🟡 MED | Integration | Well-covered, test dedup logic if changed |
| tracerank.py | 🟢 LOW | ✓ Done | Excellent unit test coverage |
| ingestion.py | 🟢 LOW | Integration | Document workflows, hard to isolate |
| entity_extraction.py | 🟢 LOW | Integration | LLM-based, non-deterministic |
| cli.py | 🟢 LOW | Integration | UI layer, better tested end-to-end |
| config.py | 🟢 LOW | Skip | Simple config loading, no complex logic |

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `tests/test_storage_methods.py` exists with 15-20 component tests
- [ ] `tests/test_embeddings.py` exists with 8-10 component tests
- [ ] `tests/test_entity_ontology.py` exists with 6-8 component tests
- [ ] All tests pass independently and together
- [ ] Test coverage increased by ~30-40 tests
- [ ] Future vecf32-style bugs would be caught before production

### Overall Success Metrics:
- **Bug Prevention:** Critical storage methods have direct test coverage
- **Fast Feedback:** Component tests run in <30 seconds
- **Maintainability:** No mocking (test against real FalkorDB)
- **Clear Failures:** Test failures pinpoint exact method/line causing issue

---

## Timeline Estimate

- **Phase 1.1 (Storage):** 3-4 hours (highest value)
- **Phase 1.2 (Embeddings):** 1-2 hours
- **Phase 1.3 (Entity Ontology):** 1 hour
- **Phase 1.4 (Model Validation):** 30 minutes
- **Total Phase 1:** ~6-8 hours

**Phase 2 & 3:** Ongoing as code evolves

---

## Notes

- All component tests should use real FalkorDB via conftest.py fixtures
- No mocking - test actual database operations
- Each test should be independent (use unique graph names)
- Follow existing patterns from test_native_vector_search.py
- Tests should be fast (<1 second per test when possible)

---

## References

- Issue #7: Native vector search implementation (exposed testing gap)
- test_native_vector_search.py: Example of good component testing
- test_tracerank.py: Example of excellent pure unit testing
- test_tracerank_retrieval.py: Example of integration testing
