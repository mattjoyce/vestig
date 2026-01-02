# Embedding Model Comparison - Final Results

**Test Date:** 2026-01-01
**Database:** matt_cerebras2.db (766 memories)
**Test Suite:** 20 Q&A pairs across 7 categories

---

## Overall Rankings (by Accuracy)

| Rank | Model | Accuracy | Search | Recall | Avg Speed | Match Ratio | Size |
|------|-------|----------|--------|--------|-----------|-------------|------|
| 🥇 1 | **embeddinggemma** | **62.5%** | 55.0% | 70.0% | 2516ms | 47.4% | 621 MB |
| 🥈 2 | **all-minilm** | **57.5%** | 45.0% | 70.0% | 2212ms | 39.5% | 45 MB |
| 🥉 3 | **bge-m3** | **55.0%** | 45.0% | 65.0% | 2564ms | 43.2% | 1.2 GB |
| 4 | **mxbai-embed-large** | **45.0%** | 35.0% | 55.0% | 2399ms | 36.6% | 669 MB |
| 5 | **granite-embedding** | **42.5%** | 35.0% | 50.0% | 2303ms | 33.1% | 62 MB |
| 6 | **nomic-embed-text** | **42.5%** | 35.0% | 50.0% | 2317ms | 37.9% | 274 MB |

---

## Key Findings

### 🏆 Best Overall: embeddinggemma (62.5%)
- **Strengths:**
  - Best accuracy across all models
  - Excels at basic facts (90%)
  - Strong keyword matching (47.4%)
  - 8K context window (no truncation needed)
- **Weaknesses:**
  - Medium speed (2.5s avg)
  - Largest model tested (621 MB)
- **Recommendation:** Use for production where quality matters most

### 🚀 Best Efficiency: all-minilm (57.5%)
- **Strengths:**
  - Second-best accuracy despite smallest size
  - **Fastest** at 2.2s avg
  - Tiny footprint (45 MB)
  - Strong recall performance (70%)
- **Weaknesses:**
  - Small context (256 tokens, needs truncation)
  - Lowest dimensions (384)
- **Recommendation:** Best speed/quality trade-off for resource-constrained environments

### 📊 Solid Middle: bge-m3 (55.0%)
- **Strengths:**
  - Good match ratio (43.2%)
  - Large 8K context window
  - 1024 dimensions
- **Weaknesses:**
  - Slowest at 2.6s avg
  - Largest download (1.2 GB)
  - Only 55% accuracy despite size
- **Recommendation:** Skip - all-minilm is faster and nearly as accurate

### ⚠️ Underperformers

**mxbai-embed-large (45.0%)**, **granite-embedding (42.5%)**, **nomic-embed-text (42.5%)**
- All perform below expectations
- nomic-embed-text surprisingly weak despite being popular
- granite-embedding matches nomic despite being 1/4 the size

---

## Performance Analysis

### Speed Comparison
```
all-minilm:         2212ms  ████████████████████ (fastest)
granite-embedding:  2303ms  ████████████████████▌
nomic-embed-text:   2317ms  ████████████████████▌
mxbai-embed-large:  2399ms  █████████████████████▌
embeddinggemma:     2516ms  ██████████████████████▌
bge-m3:             2564ms  ███████████████████████ (slowest)
```

**Performance is relatively flat** - only 16% difference between fastest and slowest. All models are in the 2.2-2.6s range for average query time.

### Quality vs Speed

```
embeddinggemma:  62.5% quality, 2.5s speed  [best quality]
all-minilm:      57.5% quality, 2.2s speed  [best balance]
bge-m3:          55.0% quality, 2.6s speed
mxbai:           45.0% quality, 2.4s speed
granite/nomic:   42.5% quality, 2.3s speed
```

**Sweet spot:** all-minilm offers 92% of embeddinggemma's quality at 88% of the speed, in 7% of the download size.

---

## Category Performance Analysis

### Categories All Models Excel At:
- **historical_context:** 100% accuracy (all models)
- **basic_fact:** 60-90% accuracy

### Categories All Models Struggle With:
- **cross_project:** 0% (all models failed)
- **problem_analysis:** 0% (all models failed)
- **detailed_content:** 0-60% (highly variable)

### Recall vs Search:
**Recall consistently outperforms Search by 10-20%** across all models. This suggests TraceRank and temporal ranking significantly improve results.

---

## Model Specifications

| Model | Dimensions | Context | Parameters | Architecture |
|-------|-----------|---------|-----------|--------------|
| embeddinggemma | 768 | 8192 | ? | gemma |
| all-minilm | 384 | 256 | ? | bert |
| bge-m3 | 1024 | 8192 | ? | bert |
| mxbai-embed-large | 1024 | 512 | 334M | bert |
| granite-embedding | 384 | 512 | 30M | bert |
| nomic-embed-text | 768 | 8192 | ? | bert |

---

## Recommendations

### For Production (Quality-First)
✅ **embeddinggemma** - Best accuracy (62.5%)

### For Development/Testing (Speed-First)
✅ **all-minilm** - Best speed/quality balance (57.5%, 2.2s)

### For Constrained Environments
✅ **granite-embedding** - Smallest working model (62 MB, 42.5%)

### Not Recommended
❌ **bge-m3** - Too slow for the accuracy gained
❌ **mxbai-embed-large** - Underperforms despite size
❌ **nomic-embed-text** - Below expectations (42.5%)

---

## Technical Notes

### Truncation Applied
Models with small context windows were truncated:
- **all-minilm:** 1000 chars (256 token limit)
- **mxbai-embed-large:** 2000 chars (512 token limit)
- **granite-embedding:** 2000 chars (512 token limit)

Max memory length in dataset: 2694 chars (~675 tokens)

### Warmup Strategy
All models were warmed up with `llm embed -c "warmup" -m <model>` before benchmarking to ensure fair performance comparisons (excludes initial model load time).

---

## Issues Encountered

1. **granite-embedding dimension mismatch** (RESOLVED)
   - Config specified 768 dimensions
   - Actual model outputs 384 dimensions
   - Fix: Updated config to correct dimension

2. **Context length errors** (RESOLVED)
   - Added max_length truncation support to embeddings.py
   - Configured per-model limits in YAML configs

---

## Test Configuration

- **Test harness:** test_qa_harness.py
- **Automation:** test_embedding_models.sh
- **Analysis:** analyze_embedding_results.py
- **QA Dataset:** test/qa_matterbase_projects.json (20 questions)
- **Methods tested:** search, recall
- **Database:** 766 memories from matterbase projects
