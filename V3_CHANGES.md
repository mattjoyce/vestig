# Vestig v3 Changes

## Overview

Version 3 database uses the v4 extraction prompt with tiered entity hierarchy and quality gates.
Focus: improving entity extraction quality and adding rate limiting support for LLM providers.

## Key Changes

### 1. Enhanced Entity Extraction Guidelines

**Problem:** Previous extraction created 612 CONCEPT entities (54% of all entities), many were generic/low-value:
- Dates: "September 26, 2025", "2025-09-26"
- Generic concepts: "storage", "legal", "aged care"
- Document artifacts: "SOW Number: 2"
- Overly abstract: "reassurance and emotional connection"

**Solution:** Created `extract_memories_v4` prompt with tiered hierarchy and quality gates:

**Tiered Extraction Priority:**
- **Tier 1** (always extract): PERSON, ORG, SYSTEM, PROJECT
  - Must be named entities with full canonical forms
  - "Matt Joyce", "Calvary Health Care", "Heidi Health AI Scribe", "AI Adoption Survey"

- **Tier 2** (extract when significant): TOOL, PLACE, SKILL, FILE
  - Must be specific, contextually relevant
  - "Docker", "Enfield Intermodal", "acoustic monitoring", "prompts.yaml"

- **Tier 3** (high bar): CONCEPT
  - MUST be multi-word domain-specific phrases
  - YES: "ambient voice AI", "therapeutic robotics", "bioacoustic analysis"
  - NO: "AI", "storage", "training", "technology" (too generic)

**Quality Gates** (ALL must pass):
1. Specificity: Proper noun OR multi-word specialized term?
2. Discriminability: Distinguishes specific context from generic discussion?
3. Retrievability: Would searching help find relevant memories?
4. Consistency: Name is stable and can be normalized?

**Normalization Rules:**
- Full names: "Matt" → "Matt Joyce"
- Canonical forms: "Calvary" → "Calvary Health Care"
- Preserve specificity: "Heidi Health AI Scribe" not "Heidi"

### 2. Rate Limiting Configuration

**Problem:** Cerebras API has rate limits (30 RPM / 60k TPM for tier 1, 900 RPM / 1M TPM for tier 2)

**Solution:** Added rate_limits config to ingestion section:

```yaml
ingestion:
  model: cerebras-gpt-oss-120b
  rate_limits:
    requests_per_minute: 30      # Tier 1
    tokens_per_minute: 60000     # 60k TPM
```

**Implementation needed:**
- [ ] Rate limiter class with token bucket algorithm
- [ ] Track both RPM and TPM limits
- [ ] Integrate into LLM client (sleep/retry when limits hit)
- [ ] Per-model rate limit configuration
- [ ] Separate limits for ingestion vs entity extraction

### 3. New Database

**Database:** `cerebras_v3.db`
**Config:** `test/config-cerebras-v3.yaml`

Cleanly separates v3 data from previous versions for comparison.

### 4. TraceRank Configuration

**Learning from testing:** Graph connectivity boost (graph_k) hurts Q&A accuracy but may help contextual retrieval

**v3 default:** Disabled graph boost for testing
```yaml
m3:
  tracerank:
    k: 0.35                        # Temporal boost enabled
    graph_connectivity_enabled: false  # Graph boost disabled
    graph_k: 0.0
```

## Expected Improvements

### Entity Quality
- Reduce CONCEPT entities from 612 to ~100-150
- Eliminate date/artifact noise
- Better entity normalization (fewer duplicates)
- Higher signal-to-noise ratio in entity graph

### Retrieval Performance
With cleaner entities:
- More accurate entity-based context retrieval
- Less noise in graph expansion
- Better entity similarity matching

## Future Considerations

### Entity Embeddings (Post-v3)

**Idea:** Use smaller embedding model specifically for entities

**Benefits:**
- Reduce storage: entities don't need 768d embeddings
- Faster entity similarity search
- Entity-specific semantic space
- Easier to detect duplicate entities via embedding similarity

**Implementation approach:**
```yaml
m4:
  entity_embeddings:
    enabled: true
    model: all-minilm         # Small, fast (384d)
    dimension: 384
    similarity_threshold: 0.85  # Auto-merge similar entities

  entity_consolidation:
    enabled: true
    schedule: daily           # Periodic consolidation
    min_similarity: 0.90      # High bar for auto-merge
    require_same_type: true   # Only merge entities of same type
```

**Consolidation process:**
1. Embed all entities with small model
2. Find high-similarity pairs (>0.90) of same type
3. Propose merges (or auto-merge with human review)
4. Update edges to point to canonical entity
5. Mark deprecated entities as merged_into

**Benefits for current problem:**
- "Matt" and "Matt Joyce" would have similarity >0.95 → auto-merge
- "AI Survey" and "AI Adoption Survey" → merge
- "Calvary" variants → normalize to canonical form

### Rate Limiting Implementation

**Token bucket algorithm:**
```python
class RateLimiter:
    def __init__(self, rpm: int, tpm: int):
        self.rpm_bucket = TokenBucket(rpm, window=60)
        self.tpm_bucket = TokenBucket(tpm, window=60)

    async def acquire(self, tokens: int):
        # Wait for both RPM and TPM capacity
        await self.rpm_bucket.acquire(1)
        await self.tpm_bucket.acquire(tokens)
```

**Per-model configuration:**
```yaml
llm_providers:
  cerebras:
    rate_limits:
      tier1:
        rpm: 30
        tpm: 60000
      tier2:
        rpm: 900
        tpm: 1000000

  openai:
    rate_limits:
      tier1:
        rpm: 500
        tpm: 30000
```

## Migration Path

### Testing v3

1. **Create fresh database:**
   ```bash
   cd test
   rm -f cerebras_v3.db  # Start clean
   ```

2. **Ingest test data:**
   ```bash
   vestig --config config-cerebras-v3.yaml memory add \
     --source "test-ingestion" \
     test_data.txt
   ```

3. **Verify entity quality:**
   ```bash
   sqlite3 cerebras_v3.db \
     "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type;"
   ```

   Expected: CONCEPT count much lower (~100-150 vs 612)

4. **Sample entities:**
   ```bash
   sqlite3 cerebras_v3.db \
     "SELECT canonical_name FROM entities
      WHERE entity_type='CONCEPT'
      ORDER BY RANDOM() LIMIT 20;"
   ```

   Should see: domain-specific terms, no dates/artifacts

5. **Compare to v2:**
   ```bash
   # Count entities by type
   for db in matt_cerebras2.db cerebras_v3.db; do
     echo "=== $db ==="
     sqlite3 $db "SELECT entity_type, COUNT(*)
                  FROM entities GROUP BY entity_type
                  ORDER BY COUNT(*) DESC;"
   done
   ```

### Full Migration

Once v3 entity quality is validated:

1. Re-ingest all source data with v3 config
2. Compare retrieval quality (entity-based and semantic)
3. If improved, deprecate v2 databases
4. Update default configs to use v3 settings

## Success Metrics

### Entity Quality
- [ ] CONCEPT entities reduced by >60% (612 → <250)
- [ ] Zero date entities extracted
- [ ] Zero document artifact entities
- [ ] Spot check: 20 random CONCEPTs are all domain-specific

### Retrieval Quality
- [ ] Entity-based context retrieval covers more relevant memories
- [ ] Fewer irrelevant memories in graph expansion
- [ ] Better performance on contextual tasks (vs Q&A)

### Rate Limiting
- [ ] No 429 errors from Cerebras API
- [ ] Ingestion throughput matches configured limits
- [ ] Graceful degradation when limits hit

## Notes

### Entity Embedding Timing
- Don't implement entity embeddings yet
- First validate v3 extraction quality
- Entity consolidation can be manual initially
- Add embeddings in M5 or M6 if entity duplication remains an issue

### Contextual Testing
- Current QA harness tests Q&A (find THE answer)
- Need new test for contextual retrieval (find ALL context)
- Design contextual test harness before claiming retrieval improvements
