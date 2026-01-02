# TraceRank Test Failure - Root Cause Analysis

**Date:** 2026-01-01
**Issue:** All TraceRank configuration variants produced identical results
**Impact:** Full comparison test suite (21 configurations) was invalid

---

## Executive Summary

The full embedding comparison test ran successfully but **all TraceRank variants (full, no-graph, off) produced identical results** because the sed-based configuration modification script failed silently on macOS. The configs were not actually different, invalidating the TraceRank tuning experiment.

**Root Cause:** BSD sed (macOS) incompatibility with GNU sed regex syntax
**Fix:** Replaced sed-based YAML modification with proper Python YAML library

---

## Timeline of Events

1. **Test Design:** Created `run_full_comparison.sh` to test 7 models × 3 TraceRank configs
2. **Test Execution:** Ran full suite (~35 minutes, 840 queries)
3. **Results Analysis:** All models showed **identical scores across all TraceRank variants**
4. **Investigation:** Configs were found to have identical k values (all 0.35)
5. **Root Cause:** sed regex `[0-9.]\+` doesn't work in BSD sed
6. **Fix:** Created Python-based config generator

---

## Detailed Analysis

### What Should Have Happened

Three config variants per model:

| Variant | k value | graph_connectivity_enabled | graph_k | Purpose |
|---------|---------|---------------------------|---------|---------|
| full | 0.35 | true | 0.15 | Full TraceRank with graph boost |
| no-graph | 0.35 | false | 0.0 | TraceRank without graph connectivity |
| off | **0.0** | false | 0.0 | Pure embeddings, no TraceRank |

### What Actually Happened

All configs had the same values:

```yaml
config-embeddinggemma-full.yaml:     k: 0.35
config-embeddinggemma-no-graph.yaml: k: 0.35
config-embeddinggemma-off.yaml:      k: 0.35  # ❌ WRONG! Should be 0.0
```

### Test Results (Invalid)

All variants produced identical accuracy:
```
embeddinggemma_full:      62.5%
embeddinggemma_no-graph:  62.5%  (same!)
embeddinggemma_off:       62.5%  (same!)
```

**This invalidated the entire TraceRank comparison.**

---

## Root Cause

### Technical Details

**File:** `run_full_comparison.sh`, line 66

**Failing Code:**
```bash
sed -i.bak "s/k: [0-9.]\+/k: $k_value/" "$output_config"
```

**Problem:**
- `\+` is a GNU sed extended regex quantifier (meaning "one or more")
- macOS uses BSD sed, which doesn't recognize `\+`
- BSD sed treated `\+` as a literal character, not a quantifier
- Pattern never matched, so k value was never replaced
- **No error was reported** - sed silently failed

### Why It Wasn't Caught Earlier

1. **No validation test:** Should have run one model with 3 configs first
2. **No --explain flag:** Would have shown TraceRank multiplier breakdown in output
3. **No config verification:** Didn't check generated configs before running tests
4. **Silent failure:** sed didn't report any errors

---

## Impact Assessment

### Tests That Were Invalid ❌

- **TraceRank tuning comparison** (primary goal of test)
- **Graph connectivity impact analysis**
- **TraceRank value assessment**

### Tests That Remained Valid ✅

- **Embedding model comparison** (all models tested with same TraceRank settings)
- **Performance/speed comparison**
- **ada-002 failure** (separate issue, unrelated to TraceRank)

### Wasted Resources

- **Time:** ~35 minutes runtime
- **Compute:** 840 queries across 21 configurations
- **OpenAI API:** ~$0.004 for ada-002 queries
- **Analysis time:** Investigation and RCA

---

## Fix Implemented

### Solution: Python-based YAML Modification

**New Script:** `create_tracerank_variants.py`

**Advantages:**
- ✅ Uses Python PyYAML library (platform-independent)
- ✅ Properly parses and modifies YAML
- ✅ Type-safe (k: 0.0 vs k: 0.35)
- ✅ Preserves YAML structure and comments
- ✅ Clear error messages if something fails

**Usage:**
```bash
python3 create_tracerank_variants.py config-embeddinggemma.yaml
```

**Output:**
```
✓ Created config-embeddinggemma-full.yaml
   k=0.35, graph_enabled=True, graph_k=0.15
✓ Created config-embeddinggemma-no-graph.yaml
   k=0.35, graph_enabled=False, graph_k=0.0
✓ Created config-embeddinggemma-off.yaml
   k=0.0, graph_enabled=False, graph_k=0.0  ← NOW CORRECT
```

---

## Verification

**Before Fix:**
```bash
$ grep "k:" config-embeddinggemma-off.yaml
    k: 0.35  # ❌ WRONG
```

**After Fix:**
```bash
$ grep "k:" config-embeddinggemma-off.yaml
    k: 0.0   # ✅ CORRECT
```

---

## Lessons Learned

### Testing Best Practices

1. **✅ Validate with one sample first** before running full suite
   - Test 1 model × 3 configs before 7 models × 3 configs

2. **✅ Use --explain flag** for TraceRank tests
   - Shows semantic_score, tracerank_multiplier breakdown
   - Would have immediately shown identical multipliers

3. **✅ Verify generated configs** before running tests
   - Check that modifications were actually applied
   - Diff configs to confirm differences

4. **✅ Fail fast on errors**
   - Don't suppress stderr with `2>&1 || true`
   - Let errors bubble up immediately

### Code Quality

5. **✅ Use proper YAML libraries** instead of sed/grep
   - Platform-independent
   - Type-safe
   - Better error handling

6. **✅ Avoid silent failures**
   - Check return codes
   - Validate outputs
   - Log what you're doing

7. **✅ Test on target platform**
   - macOS BSD tools differ from Linux GNU tools
   - Don't assume GNU sed/awk/grep

---

## Next Steps

### Immediate Actions

1. ✅ **Fixed:** Created Python-based config generator
2. ⏭️ **TODO:** Re-run test with corrected configs
3. ⏭️ **TODO:** Add --explain flag to test harness calls
4. ⏭️ **TODO:** Add config validation step to test runner

### Updated Test Plan

**Quick Validation Test (5 minutes):**
```bash
# Test just embeddinggemma with 3 configs + --explain
python3 create_tracerank_variants.py test/config-embeddinggemma.yaml
# Run 3 configs × 2 methods = 6 tests
# Verify TraceRank multipliers are different
```

**Full Test (if validation passes):**
```bash
# Generate all variants for all models
# Run full suite with --explain
# Analyze results
```

---

## Secondary Issue: ada-002 Failure

**Separate RCA needed** for ada-002's 5% accuracy (vs expected >60%)

**Possible causes:**
- Wrong embedding dimensions in database
- Embedding generation failure
- Database corruption during copy
- API issues during regeneration

**Investigation needed:**
```bash
sqlite3 matt_ada_002.db "SELECT LENGTH(content_embedding) FROM memories LIMIT 5;"
# Should show ~12288 bytes (1536 dims × 8 bytes)
```

---

## Summary

**Root Cause:** BSD sed incompatibility with GNU sed regex syntax
**Impact:** TraceRank comparison invalid, embedding comparison still valid
**Fix:** Python-based YAML modification
**Prevention:** Validate samples before full runs, use proper libraries
**Status:** Fix implemented, ready for re-test with validation
