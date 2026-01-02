# TraceRank Analysis Summary

Date: 2025-12-31
Test dataset: 20 Q&A pairs from matterbase projects

## Key Findings

### 1. Performance Comparison: Search vs Recall

| Metric | Search | Recall | Change |
|--------|--------|--------|--------|
| Relevant answers | 45% (9/20) | 65% (13/20) | +20% ✓ |
| Avg query time | 15.6s | 14.9s | -0.7s |
| Basic facts | 70% | 80% | +10% ✓ |
| Detailed content | 0% | 40% | +40% ✓ |
| Contextual | 0% | 100% | +100% ✓ |

**Recall is significantly better than search for accuracy**, but performance is equally terrible.

---

## 2. TraceRank Problems Identified

### Problem A: Cross-Project Contamination

**Example: Q4 "What problem is Heidi Health trying to solve?"**

Top 3 results returned:
1. Score 0.7301, TraceRank 1.44x: "Direct cost for Heidi Health pilot is $9,570" (WRONG - just cost)
2. Score 0.7245, TraceRank 1.10x: "Commitment to Heidi Health as sole provider is out of scope" (WRONG - just scope)
3. **Score 0.6970, TraceRank 1.72x: "Abi is designed to augment care staff..."** ❌

The 3rd result is about **Abi robot** (DIFFERENT PROJECT), has the:
- **LOWEST semantic score** (0.6970)
- **HIGHEST TraceRank boost** (1.72x)
- **Most connections** (12 connections)
- **Reinforced** (1x)

**TraceRank is boosting wrong answers from wrong projects because they're highly connected.**

---

### Problem B: Reinforcement Amplifies Incorrect Information

**Example: Q1 "Who is the project lead for the AI Adoption Survey?"**

Results include:
- Correct: "Matt Joyce is the project lead" - Score 0.9598, TraceRank 1.21x
- **WRONG: "The project lead is Bojan"** - Score 0.8200, TraceRank **1.50x** (reinforced)

The wrong answer about Bojan has:
- Lower semantic score
- **Higher TraceRank boost** (1.50x vs 1.21x)
- Been reinforced (likely from a different project)

**Reinforcement is boosting incorrect cross-project information above correct answers.**

---

### Problem C: Temporal Features Not Utilized

All memories show:
- `age=10h` (all ingested at same time)
- `stability=static` (no dynamic vs static differentiation working)
- No temporal discrimination

**The bi-temporal system isn't providing value - everything is treated equally.**

---

## 3. TraceRank Statistics

From 97 memories across 20 queries:

| Metric | Value |
|--------|-------|
| Boost range | 1.10x - 1.72x |
| Average boost | 1.26x |
| Connections range | 1 - 12 |
| Average connections | 3.4 |
| Reinforced memories | 19.6% |

**The boost range is significant (1.10x to 1.72x)**, enough to reorder results substantially.

---

## 4. Performance Analysis

**ALL queries are extremely slow:**
- Fast (<1s): 0/20
- Medium (1-10s): 0/20
- **Slow (>10s): 20/20** ❌

Average: ~15 seconds per query

**No observability yet to understand WHERE the time goes:**
- Embedding generation?
- SQLite queries?
- Vector similarity calculation?
- TraceRank computation?
- Graph traversal?

---

## 5. Conclusions

### TraceRank Issues:

1. **Cross-project contamination**: Highly connected entities from wrong projects get boosted
2. **Reinforcement backfire**: Incorrect information that was repeated gets higher scores
3. **No project/context boundaries**: Graph connections ignore semantic relevance boundaries
4. **Overpowers semantic similarity**: A 1.72x boost can overcome a 0.07 semantic score difference

### Recommended Actions:

#### Immediate:
1. **Add observability** - Instrument timing at every stage
2. **Test with TraceRank disabled** - Compare k=0 vs current k=0.35
3. **Examine graph connections** - Are cross-project edges being created?

#### Short-term:
4. **Add project/context scoping** - Limit graph traversal to relevant projects
5. **Adjust TraceRank parameters**:
   - Reduce k from 0.35 to something smaller (0.1?)
   - Increase cooldown to reduce burst effects
   - Consider connection quality not just quantity

#### Long-term:
6. **Rethink graph strategy**:
   - Should entities be project-scoped?
   - Need "relevance-weighted" connections not just counts?
   - Consider semantic similarity for graph expansion

7. **Temporal features**:
   - Why aren't they working? (all showing static)
   - Need better event time detection?

---

## 6. Next Steps

**Priority 1: Observability**
- Add timing instrumentation
- Identify performance bottlenecks
- 15 seconds per query is unacceptable

**Priority 2: TraceRank Experiment**
- Run tests with k=0 (disable TraceRank)
- Compare accuracy with/without
- Determine if it's helping or hurting overall

**Priority 3: Fix or Remove**
- If TraceRank hurts more than helps → disable it
- If it helps in some cases → fix the cross-contamination
- Consider simpler alternatives (recency only?)
