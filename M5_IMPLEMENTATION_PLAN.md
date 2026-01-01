# M5: HyDE (Hypothetical Document Embeddings) Implementation Plan

## Overview

HyDE transforms queries into hypothetical answers before embedding, bridging the semantic gap between question-style queries and statement-style memories.

**Current best performance:** embeddinggemma_no-graph at 80% recall accuracy

**Target improvement areas:**
- detailed_content: 60% → target 80%+
- problem_analysis: 0% → target 60%+
- cross_project: 0% → target 40%+

## Architecture

### Core Components

```
src/vestig/core/hyde.py
├── HyDEConfig (dataclass)
│   ├── enabled: bool
│   ├── model: str
│   ├── num_variants: int
│   ├── hybrid_weight: float
│   └── prompt_template: str
│
└── HyDEGenerator (class)
    ├── __init__(config, llm_client)
    ├── generate(query: str) -> str | list[str]
    ├── _generate_single(query: str) -> str
    └── _apply_hybrid(query_emb, hyde_emb, weight) -> ndarray
```

### Integration Points

**1. retrieval.py modifications:**
```python
class MemoryRetrieval:
    def __init__(self, ..., hyde_config: HyDEConfig | None = None):
        self.hyde = HyDEGenerator(hyde_config) if hyde_config else None

    def search(self, query: str, ...) -> list[Memory]:
        # Apply HyDE transformation if enabled
        if self.hyde and self.hyde.config.enabled:
            query_embedding = self._get_hyde_embedding(query)
        else:
            query_embedding = self.embedding.embed(query)

        # Rest of search logic unchanged
        ...

    def _get_hyde_embedding(self, query: str) -> ndarray:
        # Generate hypothetical answer(s)
        hyde_text = self.hyde.generate(query)
        hyde_emb = self.embedding.embed(hyde_text)

        # Optional: hybrid with original query
        if self.hyde.config.hybrid_weight > 0.0:
            query_emb = self.embedding.embed(query)
            return self.hyde._apply_hybrid(
                query_emb, hyde_emb,
                self.hyde.config.hybrid_weight
            )

        return hyde_emb
```

**2. cli.py configuration loading:**
```python
# In load_config()
hyde_cfg = config.get("m5", {}).get("hyde", {})
hyde_config = HyDEConfig(
    enabled=hyde_cfg.get("enabled", False),
    model=hyde_cfg.get("model", "cerebras-gpt-oss-120b"),
    num_variants=hyde_cfg.get("num_variants", 1),
    hybrid_weight=hyde_cfg.get("hybrid_weight", 0.0),
    prompt_template=hyde_cfg.get("prompt_template", "hyde_generate"),
)

# Pass to MemoryRetrieval
retrieval = MemoryRetrieval(
    ...,
    hyde_config=hyde_config
)
```

**3. prompts.py additions:**
```python
HYDE_GENERATE = """You are helping improve memory search by generating hypothetical answers.

Given a question, write a concise, factual answer as if you were recalling it from memory.

Requirements:
- Write in statement form, not question form
- Be specific and factual
- Keep it under 100 words
- If the question asks for multiple items, include them
- Don't add speculation or uncertainty

Question: {query}

Hypothetical answer:"""

HYDE_GENERATE_MULTI = """You are helping improve memory search by generating hypothetical answers.

Given a question, write 3 different hypothetical answers that might exist in a memory system.

Requirements:
- Write in statement form, not question form
- Each variant should emphasize different aspects
- Be specific and factual
- Keep each under 100 words

Question: {query}

Hypothetical answers:
1."""
```

## Configuration Schema

```yaml
m5:
  hyde:
    # Enable/disable HyDE transformation
    enabled: false

    # LLM model for generating hypothetical answers
    # Recommendations:
    #   - cerebras-gpt-oss-120b: fast, cheap, good quality
    #   - claude-haiku-4.5: higher quality, slower
    model: cerebras-gpt-oss-120b

    # Number of hypothetical answer variants to generate
    # 1 = single answer (faster)
    # 3 = multiple variants (more diverse, slower)
    num_variants: 1

    # Hybrid weight: blend HyDE with original query
    # 0.0 = pure HyDE (only use hypothetical answer)
    # 0.5 = balanced blend
    # 1.0 = pure query (no HyDE, original behavior)
    hybrid_weight: 0.0

    # Prompt template to use
    prompt_template: hyde_generate

    # Optional: cache hypothetical answers
    cache_enabled: false
```

## Implementation Phases

### Phase 1: Core Implementation (2-3 hours)

**Files to create:**
- `src/vestig/core/hyde.py` - HyDEGenerator and HyDEConfig

**Files to modify:**
- `src/vestig/core/retrieval.py` - integrate HyDE into search/recall
- `src/vestig/core/cli.py` - load HyDE config
- `src/vestig/prompts.py` - add HyDE prompts

**Tasks:**
1. Create HyDEConfig dataclass with validation
2. Implement HyDEGenerator class
   - Single variant generation
   - LLM integration (reuse existing llm_client pattern)
   - Hybrid embedding blending
3. Modify MemoryRetrieval to use HyDE
4. Add config loading in cli.py
5. Create prompt templates

### Phase 2: Testing Infrastructure (1 hour)

**Files to create:**
- `test/config-embeddinggemma-hyde.yaml` - test config with HyDE enabled
- `test/test_hyde.sh` - automated test script

**Tasks:**
1. Create test configs:
   - hyde-pure (hybrid_weight=0.0)
   - hyde-blend (hybrid_weight=0.5)
   - hyde-off (enabled=false, baseline)
2. Run QA harness with each config
3. Compare results

### Phase 3: Tuning & Optimization (2-3 hours)

**Experiments to run:**

1. **Model comparison:**
   - cerebras-gpt-oss-120b (fast, cheap)
   - claude-haiku-4.5 (quality)
   - gemini-flash-2.0 (if available)

2. **Hybrid weight tuning:**
   - Test: 0.0, 0.25, 0.5, 0.75, 1.0
   - Find optimal blend

3. **Num variants:**
   - Test: 1 vs 3 variants
   - Measure quality vs latency tradeoff

4. **Prompt engineering:**
   - Test different prompt styles
   - Few-shot vs zero-shot

**Success metrics:**
- Recall accuracy improvement (target: 80% → 85%+)
- Category-specific improvements (detailed_content, problem_analysis)
- Latency impact (should stay under 5s per query)

## Testing Strategy

### Unit Tests
```python
# tests/test_hyde.py
def test_hyde_generation():
    """Test basic HyDE generation"""

def test_hyde_hybrid_blending():
    """Test embedding blending with different weights"""

def test_hyde_disabled():
    """Test that disabled HyDE doesn't affect results"""
```

### Integration Tests

**Baseline:** embeddinggemma_no-graph (80% recall)

**Test matrix:**
```
Model: embeddinggemma
TraceRank: no-graph (k=0.35, graph_connectivity_enabled=false)
HyDE variants:
  1. hyde-off (baseline)
  2. hyde-pure (hybrid_weight=0.0)
  3. hyde-blend-25 (hybrid_weight=0.25)
  4. hyde-blend-50 (hybrid_weight=0.5)
  5. hyde-blend-75 (hybrid_weight=0.75)
```

**Commands:**
```bash
cd test

# Generate HyDE config variants
python3 create_hyde_variants.py config-embeddinggemma-no-graph.yaml

# Run tests
for config in config-embeddinggemma-no-graph-hyde-*.yaml; do
  python3 ../test_qa_harness.py recall "$config" qa_matterbase_projects.json
done

# Analyze results
python3 analyze_hyde_results.py
```

### Expected Results

**Conservative estimate:**
- Recall: 80% → 82-85% (modest improvement)
- detailed_content: 60% → 70-80% (significant improvement)
- problem_analysis: 0% → 40-60% (major improvement)
- Latency: +500-1000ms per query (acceptable)

**Optimistic estimate:**
- Recall: 80% → 85-90% (strong improvement)
- detailed_content: 60% → 80-90%
- problem_analysis: 0% → 60-80%

## Implementation Details

### HyDE Generation Logic

```python
class HyDEGenerator:
    def generate(self, query: str) -> str | list[str]:
        """Generate hypothetical answer(s) for query"""

        if self.config.num_variants == 1:
            return self._generate_single(query)
        else:
            return self._generate_multi(query)

    def _generate_single(self, query: str) -> str:
        """Generate single hypothetical answer"""

        prompt = PROMPTS[self.config.prompt_template].format(query=query)

        # Use llm_client to generate
        response = self.llm_client.generate(
            prompt=prompt,
            model=self.config.model,
            max_tokens=200,
            temperature=0.3,  # low temp for consistency
        )

        return response.strip()

    def _generate_multi(self, query: str) -> list[str]:
        """Generate multiple hypothetical answer variants"""

        prompt = PROMPTS["hyde_generate_multi"].format(query=query)

        response = self.llm_client.generate(
            prompt=prompt,
            model=self.config.model,
            max_tokens=400,
            temperature=0.5,  # higher temp for diversity
        )

        # Parse numbered list
        variants = []
        for line in response.split('\n'):
            if re.match(r'^\d+\.\s', line):
                variants.append(re.sub(r'^\d+\.\s', '', line).strip())

        return variants[:self.config.num_variants]
```

### Hybrid Embedding Blending

```python
def _apply_hybrid(
    self,
    query_emb: np.ndarray,
    hyde_emb: np.ndarray,
    weight: float
) -> np.ndarray:
    """Blend query and HyDE embeddings

    Args:
        query_emb: Original query embedding
        hyde_emb: HyDE hypothetical answer embedding
        weight: How much to weight query (0.0=pure HyDE, 1.0=pure query)

    Returns:
        Blended embedding (normalized)
    """

    # Weighted blend
    blended = (1 - weight) * hyde_emb + weight * query_emb

    # Re-normalize
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm

    return blended
```

### Multi-variant Handling

```python
def _get_hyde_embedding_multi(self, query: str) -> np.ndarray:
    """Get embedding using multiple HyDE variants"""

    # Generate multiple hypothetical answers
    variants = self.hyde.generate(query)  # returns list[str]

    # Embed each variant
    variant_embeddings = [
        self.embedding.embed(variant)
        for variant in variants
    ]

    # Average embeddings
    avg_emb = np.mean(variant_embeddings, axis=0)

    # Re-normalize
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm

    # Optional: hybrid with query
    if self.hyde.config.hybrid_weight > 0.0:
        query_emb = self.embedding.embed(query)
        return self.hyde._apply_hybrid(
            query_emb, avg_emb,
            self.hyde.config.hybrid_weight
        )

    return avg_emb
```

## Rollout Plan

### Stage 1: Development (Days 1-2)
- Implement core HyDE functionality
- Unit tests
- Basic integration test with one config

### Stage 2: Testing (Day 3)
- Run full test matrix (5 configs × 20 questions = 100 queries)
- Analyze results
- Identify best configuration

### Stage 3: Tuning (Day 4)
- Test different models if needed
- Optimize prompts if quality is low
- Test multi-variant approach if single variant disappoints

### Stage 4: Documentation (Day 5)
- Update README with M5 HyDE feature
- Document configuration options
- Add examples and use cases
- Write completion report

## Risk Analysis

### Risks & Mitigations

**Risk 1: HyDE doesn't improve quality**
- Likelihood: Low-Medium
- Impact: High (wasted effort)
- Mitigation: Quick prototype test with 5 questions before full implementation
- Fallback: Disable HyDE, keep as optional feature

**Risk 2: Latency becomes unacceptable (>5s)**
- Likelihood: Medium
- Impact: Medium
- Mitigation: Use fast model (cerebras), cache if needed
- Fallback: Make HyDE opt-in only for specific query types

**Risk 3: HyDE helps some categories but hurts others**
- Likelihood: Medium
- Impact: Low-Medium
- Mitigation: Hybrid blending allows balancing
- Fallback: Category-specific HyDE enablement

**Risk 4: Prompt engineering is difficult**
- Likelihood: Low
- Impact: Medium
- Mitigation: Start with simple prompts, iterate based on results
- Fallback: Use few-shot examples from actual memories

## Success Criteria

### Must Have
- [ ] HyDE implementation integrated into retrieval.py
- [ ] Configuration working correctly
- [ ] No regression on baseline queries (80% recall maintained)
- [ ] Tests run successfully

### Should Have
- [ ] Recall accuracy improves to 82%+
- [ ] detailed_content improves to 70%+
- [ ] problem_analysis improves to 40%+
- [ ] Latency stays under 5s per query

### Nice to Have
- [ ] Recall accuracy improves to 85%+
- [ ] Multi-variant support working
- [ ] Category-specific HyDE enablement
- [ ] Cache implementation for repeated queries

## Future Enhancements (Post-M5)

1. **Query classification:** Detect query type, apply HyDE selectively
2. **Adaptive HyDE:** Learn which queries benefit from HyDE
3. **Multi-step HyDE:** Generate → search → refine → search again
4. **HyDE with examples:** Include few-shot examples from top memories
5. **Reverse HyDE:** For ingestion, generate likely queries for each memory

## References

- Original HyDE paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)
- Implementation inspiration: LangChain HyDE, LlamaIndex HyDE
- vestig architecture: M3 (TraceRank), M4 (Entity Graph)
