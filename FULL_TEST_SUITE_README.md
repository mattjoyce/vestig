# Full Embedding + TraceRank Comparison Suite

## What This Tests

**21 Configurations Total:**
- 7 embedding models (embeddinggemma, all-minilm, bge-m3, mxbai, granite, nomic, ada-002)
- 3 TraceRank variants per model:
  1. **Full TraceRank** (k=0.35, graph_k=0.15) - current production
  2. **No Graph Boost** (k=0.35, graph off) - temporal signals only
  3. **No TraceRank** (k=0.0) - pure embedding quality

**Total Tests:** 42 test runs (21 configs × 2 methods)
**Total Queries:** 840 queries (20 questions × 42 runs)
**Estimated Time:** ~35 minutes
**Cost:** ~$0.004 (ada-002 queries only)

## Running the Full Suite

```bash
cd /Users/mattjoyce/Projects/vestig
./run_full_comparison.sh
```

The script will:
1. Auto-generate all 21 config file variations
2. Run search + recall tests for each
3. Save results to `test/full_comparison_<timestamp>/`
4. Show progress (e.g., "[5/21] Testing...")

**You can leave it running and come back!**

## Analyzing Results

After tests complete:

```bash
python3 analyze_full_comparison.py ./test/full_comparison_<timestamp>/
```

This generates a comprehensive report with:

### 1. Overall Rankings
Top 10 configurations ranked by accuracy

### 2. Best Config Per Model
Which TraceRank setting works best for each embedding model

### 3. TraceRank Impact Analysis
For each model, shows:
- Graph connectivity impact (Full vs No-Graph)
- TraceRank value (No-Graph vs Off)
- Whether TraceRank helps or hurts

### 4. Performance Comparison
Speed rankings across all configurations

### 5. Commercial vs Open Source
How ada-002 stacks up against free local models

## Expected Insights

**Key Questions Answered:**

1. **Does graph connectivity boost help or hurt?**
   - Compare "Full" vs "No-Graph" for each model
   - If No-Graph wins → graph connectivity was biasing results

2. **Is TraceRank valuable at all?**
   - Compare "No-Graph" vs "Off" for each model
   - Shows if temporal signals improve retrieval

3. **Best model overall?**
   - Which embedding model + TraceRank config gives best results
   - Whether small models (all-minilm) can beat large ones

4. **Is ada-002 worth paying for?**
   - Does commercial model beat free alternatives?
   - Cost/benefit analysis

## Files Created

- `run_full_comparison.sh` - Main test runner
- `analyze_full_comparison.py` - Results analyzer
- Auto-generated configs: `config-{model}-{full|no-graph|off}.yaml`
- Results: `test/full_comparison_<timestamp>/`
  - 42 result JSON files (search + recall for each config)
  - 42 log files
  - `full_comparison_report.txt` - Final analysis

## Output Format

The report will look like:

```
1. OVERALL RANKINGS
🥇 embeddinggemma_no-graph     67.5%   60.0%   75.0%
🥈 all-minilm_off              62.5%   55.0%   70.0%
🥉 ada-002_full                60.0%   55.0%   65.0%
...

2. BEST CONFIGURATION PER MODEL
embeddinggemma    no-graph    67.5%
all-minilm        off         62.5%
...

3. TRACERANK IMPACT ANALYSIS
embeddinggemma:
  Full TraceRank:     62.5%
  No Graph Boost:     67.5%  📈 Graph impact: +5.0%
  No TraceRank:       60.0%  📉 TraceRank impact: -7.5%
...
```

## Next Steps After Results

Based on findings, you can:

1. **Update production config** to use best TraceRank settings
2. **Choose embedding model** with best quality/speed/cost trade-off
3. **Document findings** for Mentor with data-driven recommendations
4. **Fine-tune further** if needed (adjust k values, graph_k, etc.)

## Monitoring Progress

The script outputs progress as it runs:
```
[1/21] Preparing: embeddinggemma - TraceRank-Full
Running search tests...
  ✓ Search results saved
Running recall tests...
  ✓ Recall results saved
  ✓ embeddinggemma_full complete!
```

Check `test/full_comparison_*/` to see results accumulating as tests run.
