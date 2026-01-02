# Hybrid Retrieval Test Plan

## Overview

We've implemented hybrid semantic + entity-based retrieval (M5). This document describes how to create comprehensive test cases to validate the implementation.

**What we're testing:** Whether entity-based retrieval correctly:
1. Extracts entities from queries
2. Matches them to database entities via embedding similarity
3. Retrieves memories that mention those entities (via MENTIONS edges)
4. Combines semantic + entity scores to improve results

**Not a Q&A system:** This is context expansion, not answer finding. We're testing whether the system retrieves comprehensive relevant context when entities are mentioned.

## Current Status

**Implementation complete:** End-to-end hybrid retrieval working behind feature flag

**Initial testing (4 ad-hoc queries):**
- ✅ 2 successful: Entity matches found, scores boosted
- ❌ 2 failed: Entity extraction ran but no matches above threshold

**Success rate: 50%** - Promising but needs systematic validation.

## Test Case Generation Strategy

### Reverse Engineering Approach

Instead of guessing queries, **derive test cases from actual database content**:

#### Step 1: Query Memories with Entities

```sql
-- Find memories that have entity mentions
SELECT
    m.id as memory_id,
    m.content,
    e.canonical_name as entity_name,
    e.entity_type,
    edge.confidence
FROM memories m
JOIN edges edge ON edge.from_node = m.id AND edge.edge_type = 'MENTIONS'
JOIN entities e ON e.id = edge.to_node
WHERE edge.confidence >= 0.75
ORDER BY e.entity_type, e.canonical_name;
```

This gives you memories that definitely mention specific entities with high confidence.

#### Step 2: Create Test Queries

For each entity-memory pair, create a query that mentions the entity:

**Example:**
```
Memory: "Matt is the project lead for Project Bronte."
Entities: Matt Joyce (PERSON), Project Bronte (PROJECT)

Test queries:
1. "Matt Joyce on Project Bronte" (both entities, exact names)
2. "Matt working on Bronte" (both entities, partial names)
3. "Who is leading Project Bronte" (one entity, natural question)
4. "Matt" (single entity, minimal context)
```

#### Step 3: Document Expected Behavior

For each test query, record:
```yaml
test_cases:
  - query: "Matt Joyce on Project Bronte"

    # Entity extraction expectations
    expected_extracted_entities:
      - name: "Matt Joyce"
        type: "PERSON"
      - name: "Project Bronte"
        type: "PROJECT"

    # Entity matching expectations
    expected_db_matches:
      - entity_id: "ent_abc123"
        canonical_name: "Matt Joyce"
        similarity: ">0.95"  # Should be near-exact match
      - entity_id: "ent_xyz789"
        canonical_name: "Project Bronte"
        similarity: ">0.95"

    # Retrieval expectations
    expected_retrieved_memories:
      - memory_id: "mem_72ff630a..."
        reason: "Mentions both Matt Joyce and Project Bronte"
        expected_in_top_n: 3

    # Scoring expectations
    expected_score_boost:
      semantic_only_rank: 1
      hybrid_rank: 1
      score_increase: ">0.2"  # Entity path should boost score
```

## Test Categories

### Category 1: Exact Entity Matches (High Confidence)

**Goal:** Verify entity path works for clear, unambiguous entity mentions

**Example queries:**
- "Matt Joyce on Project Bronte" → Both entities exact match
- "Mimaso SaaS platform development" → Entity exact match
- "Riverina Drug and Alcohol services" → Org exact match

**Expected behavior:**
- Entities extracted from query: ✅
- Entities matched to DB (similarity >0.9): ✅
- Memories retrieved via MENTIONS edges: ✅
- Score boost applied: ✅

**Success criteria:**
- Entity match rate: 100%
- Retrieval coverage: All memories mentioning the entity appear in top-10
- Score improvement: Memories with entity mentions rank higher than semantic-only

### Category 2: Partial Entity Matches (Medium Confidence)

**Goal:** Test fuzzy matching when query uses variations of entity names

**Example queries:**
- "Matt on Bronte" → "Matt" should match "Matt Joyce" (via embedding similarity)
- "John working on project" → "John" might match "John Sutherland" (if similarity >0.7)
- "Mimaso platform" → Should match "Mimaso SaaS platform"

**Expected behavior:**
- Entities extracted: ✅
- Entities matched (similarity 0.7-0.9): ✅ (if close enough)
- Memories retrieved: ✅ (if matched)
- Score boost: ✅ (if matched)

**Success criteria:**
- Entity match rate: >70%
- Coverage: Most memories with matching entities appear
- No false positives: "John Smith" shouldn't match "John Sutherland"

### Category 3: No Entity Matches (Negative Cases)

**Goal:** Verify graceful fallback to semantic-only when entity path fails

**Example queries:**
- "working with someone" → Generic, no specific entities
- "how to optimize performance" → Conceptual query, no entities
- "Alice Smith on unknown project" → Entities don't exist in DB

**Expected behavior:**
- Entities extracted: Maybe (LLM might extract "Alice Smith")
- Entities matched: ❌ (not in DB)
- Memories retrieved via entities: ❌
- Fallback to semantic: ✅
- Performance: Should not slow down significantly

**Success criteria:**
- No errors or crashes
- Results same as semantic-only retrieval
- Total time <5s (entity extraction overhead acceptable)

### Category 4: Multi-Entity Queries (Complex Cases)

**Goal:** Test when query mentions multiple entities

**Example queries:**
- "Matt and Bojan working on Mimaso" → 3 entities
- "Project Bronte and ProcessX comparison" → 2 projects

**Expected behavior:**
- Multiple entities extracted: ✅
- All entities matched (if they exist): ✅
- Memories retrieved for each entity: ✅
- Score aggregation: Memories mentioning multiple matched entities rank highest

**Success criteria:**
- All entities extracted and matched
- Memories mentioning multiple entities score higher
- No duplicate results

## Test Harness Structure

### Suggested Implementation

```python
#!/usr/bin/env python3
"""
Hybrid retrieval test harness

Usage:
    python test_hybrid_retrieval.py --config config.yaml --test-cases test_cases.yaml
"""

class HybridRetrievalTest:
    def __init__(self, config, test_cases):
        self.config = config
        self.test_cases = test_cases
        self.results = []

    def run_test(self, test_case):
        """Run single test case"""
        query = test_case["query"]

        # Run with entity path enabled
        hybrid_results = search_memories(
            query=query,
            entity_config={"enabled": True, "entity_weight": 0.5},
            ...
        )

        # Run with entity path disabled (baseline)
        semantic_results = search_memories(
            query=query,
            entity_config={"enabled": False},
            ...
        )

        # Validate results
        validation = {
            "entity_extraction": self.check_entity_extraction(test_case, ...),
            "entity_matching": self.check_entity_matching(test_case, ...),
            "retrieval_coverage": self.check_coverage(test_case, hybrid_results),
            "score_improvement": self.check_score_delta(semantic_results, hybrid_results),
            "performance": self.check_timing(...),
        }

        return {
            "test_case": test_case,
            "hybrid_results": hybrid_results,
            "semantic_results": semantic_results,
            "validation": validation,
            "passed": all(validation.values())
        }

    def check_entity_extraction(self, test_case, ...):
        """Did LLM extract expected entities from query?"""
        expected = test_case.get("expected_extracted_entities", [])
        # Compare with actual extracted entities
        # Return True if matches expectations
        pass

    def check_entity_matching(self, test_case, ...):
        """Did extracted entities match DB entities above threshold?"""
        expected_matches = test_case.get("expected_db_matches", [])
        # Compare with actual matched entities
        # Check similarity scores
        pass

    def check_coverage(self, test_case, results):
        """Did we retrieve expected memories?"""
        expected_memories = test_case.get("expected_retrieved_memories", [])
        retrieved_ids = [m.id for m, score in results]

        coverage = len(set(expected_memories) & set(retrieved_ids)) / len(expected_memories)
        return coverage >= test_case.get("min_coverage", 0.8)

    def check_score_delta(self, semantic_results, hybrid_results):
        """Did entity path improve scores?"""
        # Compare rankings and scores
        # Entity-matched memories should rank higher in hybrid
        pass

    def generate_report(self):
        """Generate test report with metrics"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])

        print(f"Hybrid Retrieval Test Results")
        print(f"=" * 60)
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        print()

        # Category breakdown
        for category in ["exact_match", "partial_match", "no_match", "multi_entity"]:
            category_results = [r for r in self.results if r["test_case"]["category"] == category]
            category_passed = sum(1 for r in category_results if r["passed"])
            print(f"{category}: {category_passed}/{len(category_results)} passed")

        # Detailed failures
        print("\nFailed Tests:")
        for result in self.results:
            if not result["passed"]:
                print(f"  - {result['test_case']['query']}")
                print(f"    Reason: {result['validation']}")
```

### Test Case Format (YAML)

```yaml
test_cases:
  - id: "exact_match_001"
    category: "exact_match"
    query: "Matt Joyce on Project Bronte"

    expected_extracted_entities:
      - name: "Matt Joyce"
        type: "PERSON"
      - name: "Project Bronte"
        type: "PROJECT"

    expected_db_matches:
      - canonical_name: "Matt Joyce"
        min_similarity: 0.95
      - canonical_name: "Project Bronte"
        min_similarity: 0.95

    expected_retrieved_memories:
      - memory_id: "mem_72ff630a-68b8-4b54-a083-abdd836fdddd"
        should_rank_in_top: 5

    min_coverage: 0.8
    expected_score_boost: true

  - id: "partial_match_001"
    category: "partial_match"
    query: "Matt on Bronte"

    expected_extracted_entities:
      - name: "Matt"
        type: "PERSON"
      - name: "Bronte"
        type: "PROJECT"

    expected_db_matches:
      - canonical_name: "Matt Joyce"
        min_similarity: 0.7
      - canonical_name: "Project Bronte"
        min_similarity: 0.7

    min_coverage: 0.6  # Lower bar for partial matches

  - id: "no_match_001"
    category: "no_match"
    query: "how to optimize performance"

    expected_extracted_entities: []
    expected_db_matches: []
    fallback_to_semantic: true
    max_total_time_ms: 5000
```

## Generating Test Cases from Database

### Script to Generate Test Cases

```python
#!/usr/bin/env python3
"""
Generate test cases by reverse engineering from database

Usage:
    python generate_test_cases.py --config config.yaml --output test_cases.yaml
"""

def generate_test_cases_from_db(storage, limit=50):
    """Generate test cases from memories with entities"""

    # Query memories with entity mentions
    cursor = storage.conn.execute("""
        SELECT DISTINCT
            e.id as entity_id,
            e.canonical_name,
            e.entity_type,
            COUNT(DISTINCT edge.from_node) as mention_count
        FROM entities e
        JOIN edges edge ON edge.to_node = e.id AND edge.edge_type = 'MENTIONS'
        WHERE e.embedding IS NOT NULL
          AND edge.confidence >= 0.75
        GROUP BY e.id
        HAVING mention_count >= 2  -- Entities mentioned in multiple memories
        ORDER BY mention_count DESC
        LIMIT ?
    """, (limit,))

    test_cases = []

    for entity_id, canonical_name, entity_type, mention_count in cursor.fetchall():
        # Get memories that mention this entity
        memories = storage.conn.execute("""
            SELECT m.id, m.content
            FROM memories m
            JOIN edges e ON e.from_node = m.id AND e.to_node = ?
            WHERE e.edge_type = 'MENTIONS'
            LIMIT 5
        """, (entity_id,)).fetchall()

        # Generate test queries
        test_cases.append({
            "id": f"entity_{entity_type.lower()}_{canonical_name.replace(' ', '_')}",
            "category": "exact_match",
            "query": f"information about {canonical_name}",
            "expected_extracted_entities": [
                {"name": canonical_name, "type": entity_type}
            ],
            "expected_db_matches": [
                {"canonical_name": canonical_name, "min_similarity": 0.95}
            ],
            "expected_retrieved_memories": [
                {"memory_id": m[0], "should_rank_in_top": 10}
                for m in memories
            ],
            "min_coverage": 0.8,
            "notes": f"Entity mentioned in {mention_count} memories"
        })

    return test_cases
```

## Success Metrics

### Overall Metrics

- **Entity extraction rate**: % of queries where entities were successfully extracted
- **Entity match rate**: % of extracted entities that matched DB entities above threshold
- **Retrieval success rate**: % of queries where entity path retrieved relevant memories
- **Score improvement rate**: % of queries where entity path improved ranking
- **Performance**: p50, p95, p99 latency for hybrid retrieval

### Target Goals

- Entity extraction rate: >80% (for queries mentioning entities)
- Entity match rate: >70% (for extracted entities)
- Retrieval success rate: >75% (when entities matched)
- Score improvement: Memories with entity mentions rank higher
- Performance: p95 latency <5s

### Current Status (Ad-hoc Testing)

- Entity extraction rate: 100% (4/4 queries)
- Entity match rate: 50% (2/4 queries had matches)
- Retrieval success rate: 50% (2/4 queries)
- Performance: 3-5s total (entity extraction is bottleneck at 2-4s)

## Next Steps

1. **Generate test cases**: Run `generate_test_cases.py` to create initial test suite
2. **Run baseline**: Test with entity path disabled to establish baseline
3. **Run hybrid**: Test with entity path enabled
4. **Analyze failures**: Understand why entity matching fails
5. **Tune thresholds**: Adjust similarity_threshold if needed (currently 0.7)
6. **Iterate**: Refine entity extraction, matching, or scoring based on results

## Known Issues to Investigate

1. **Why did "working with John" fail?**
   - Was "John" extracted as an entity?
   - Did it match any "John X" in the DB?
   - What was the similarity score?

2. **Why did "Riverina services" fail?**
   - Was "Riverina" extracted?
   - "Riverina Drug and Alcohol services" exists in DB
   - Similarity threshold too high for partial matches?

3. **Entity extraction latency**
   - Currently 2-4 seconds (55-82% of total time)
   - Can we batch extract for multiple queries?
   - Can we cache extracted entities for repeated queries?

## Deliverables

1. **Test cases file** (`test_cases.yaml`): 50+ test cases covering all categories
2. **Test harness** (`test_hybrid_retrieval.py`): Automated testing script
3. **Baseline results**: Performance with entity path disabled
4. **Hybrid results**: Performance with entity path enabled
5. **Analysis report**: Success rates, failure patterns, recommendations
