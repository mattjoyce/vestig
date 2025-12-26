# Embedding Model Decision

**Date**: 2025-12-26
**Decision**: Use BAAI/bge-m3 as primary embedding model

## Why BGE-m3?

### 1. Production Validation
- **Zep** (arXiv 2501.13956) uses BGE-m3 in production
- Achieved **71.2% accuracy** on LongMemEval (115k token conversations)
- **90% latency reduction** (2.58s vs 28.9s baseline)
- Reduced context from 115k → 1.6k tokens

### 2. State-of-the-Art Performance
- Beats OpenAI text-embedding-3-large on many benchmarks
- Top performer on MTEB (Massive Text Embedding Benchmark)
- Strong zero-shot performance across domains
- Multi-lingual support (not critical for us, but nice)

### 3. Technical Specifications
- **Dimension**: 1024 (sweet spot)
- **Max sequence length**: 8192 tokens
- **Normalization**: Built-in L2 normalization
- **Multi-functionality**:
  - Dense retrieval (embeddings)
  - Sparse retrieval (BM25-style)
  - Multi-vector retrieval

### 4. Practical Benefits
- **Open source**: No API costs
- **Local deployment**: Can run on-device
- **Well-supported**: Active development, good docs
- **Easy integration**: Works with sentence-transformers, llm CLI

### 5. Storage Efficiency
**Per memory node** (with dual embeddings + hypothetical queries):
- Content embedding: 1024 × 4 bytes = 4KB
- Trigger embedding: 1024 × 4 bytes = 4KB
- 2 hypothetical query embeddings: 8KB
- **Total**: ~16KB per memory

**Scaling**:
- 1,000 memories: ~16MB
- 10,000 memories: ~160MB
- 100,000 memories: ~1.6GB

All very reasonable for sqlite-graph.

## Alternative Models Considered

### BGE-small-en-v1.5
- **Dimension**: 384
- **Pros**: 3x faster, smaller storage
- **Cons**: Lower retrieval quality
- **Use case**: Speed-critical applications
- **Our verdict**: Quality > speed for personal memory

### nomic-embed-text-v1.5
- **Dimension**: 768
- **Pros**: Good balance, open source
- **Cons**: Slightly below BGE-m3 performance
- **Use case**: Middle ground
- **Our verdict**: If BGE-m3 is too slow, this is fallback

### OpenAI text-embedding-3-small
- **Dimension**: 1536 (configurable)
- **Pros**: Easy API, good quality
- **Cons**: Costs money, API dependency, privacy
- **Use case**: Prototyping without local setup
- **Our verdict**: Avoid for personal memory system

### PubMedBERT (Mnemosyne's choice)
- **Dimension**: 768 (likely)
- **Pros**: Domain-specific for healthcare
- **Cons**: Not general-purpose
- **Use case**: Healthcare-specific memory
- **Our verdict**: Too specialized for our use case

## Implementation Plan

### Phase 1: BGE-m3 via sentence-transformers
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')
embeddings = model.encode([
    "This is a test sentence",
    "Another sentence"
], normalize_embeddings=True)
```

### Phase 2: Integrate with llm CLI
```bash
# Install plugin
llm install llm-sentence-transformers

# Embed via CLI
llm embed -m sentence-transformers/BAAI/bge-m3 -i input.txt
```

### Phase 3: Configuration
```yaml
# config.yaml
embedding:
  model: "BAAI/bge-m3"
  dimension: 1024
  normalize: true
  batch_size: 10
  device: "cpu"  # or "cuda" if available
```

## Benchmarks to Validate

Once implemented, we should benchmark:

1. **Embedding speed**: Time to embed 100 memories
2. **Search quality**: Precision@K on synthetic queries
3. **Storage size**: Actual DB size with 1k memories
4. **Memory usage**: RAM footprint during embedding

## Future Considerations

### If BGE-m3 is too slow:
- Switch to `bge-small-en-v1.5` (384-dim)
- Use GPU acceleration
- Batch embedding operations
- Cache frequently accessed embeddings

### If we need better quality:
- Use `bge-large-en-v1.5` (1024-dim, larger model)
- Fine-tune on our domain (problem-solving, programming)
- Ensemble multiple models

### If we need multi-modal:
- Consider CLIP-style models
- Would enable image/code/text embeddings
- Probably overkill for v1

## Decision Matrix

| Model | Quality | Speed | Storage | Cost | Local | Verdict |
|-------|---------|-------|---------|------|-------|---------|
| BGE-m3 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Free | ✅ | **CHOSEN** |
| BGE-small | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Free | ✅ | Fallback |
| nomic-embed | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | ✅ | Alternative |
| OpenAI-small | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | N/A | $$ | ❌ | Avoid |
| PubMedBERT | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Free | ✅ | Too specific |

## References

- **BGE-m3 Paper**: https://arxiv.org/abs/2402.03216
- **FlagEmbedding GitHub**: https://github.com/FlagOpen/FlagEmbedding
- **MTEB Leaderboard**: https://huggingface.co/spaces/mteb/leaderboard
- **Zep Paper**: arXiv 2501.13956 (production validation)

## Conclusion

**BGE-m3 is the right choice** because:
1. Proven in production (Zep)
2. SOTA quality
3. Reasonable 1024-dim size
4. Open source, local deployment
5. Strong community support

We specify alternatives in config for flexibility, but default to BGE-m3 for best retrieval quality.
