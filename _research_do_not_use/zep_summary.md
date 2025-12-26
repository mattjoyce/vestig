# Zep Paper Summary

**Paper**: "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"
**Authors**: Rasmussen et al., Zep AI
**arXiv**: 2501.13956

## Core Innovation

**Temporal Knowledge Graph** with bi-temporal modeling and edge invalidation for dynamic, evolving knowledge.

## Three-Tier Graph Architecture

### 1. Episode Subgraph (Ge)
- **Purpose**: Non-lossy raw data store
- **Nodes**: Episodes containing messages, text, or JSON
- **Edges**: Connect episodes to semantic entities they reference
- **Key feature**: Bidirectional indices (episode ↔ semantic artifacts)

### 2. Semantic Entity Subgraph (Gs)
- **Purpose**: Extracted knowledge
- **Nodes**: Entities extracted and resolved from episodes
- **Edges**: Facts/relationships between entities
- **Key feature**: Entity resolution with reflexion technique

### 3. Community Subgraph (Gc)
- **Purpose**: High-level domain understanding
- **Nodes**: Clusters of strongly connected entities
- **Edges**: Connect communities to member entities
- **Key feature**: Label propagation for dynamic updates

## Bi-Temporal Modeling (Novel!)

**Two timelines**:

1. **T (Event timeline)**: When things actually happened
   - `t_valid`: When fact became true
   - `t_invalid`: When fact stopped being true

2. **T' (Transaction timeline)**: When we learned about it
   - `t_created`: When data entered system
   - `t_expired`: When data was invalidated

**Benefits**:
- Track both "when it happened" and "when we learned it"
- Handle relative dates ("next Thursday", "two weeks ago")
- Audit trail for data ingestion
- Novel advancement in LLM-based KG construction

## Temporal Extraction

**Capabilities**:
- Absolute timestamps: "June 23, 1912"
- Relative timestamps: "two weeks ago", "next Thursday"
- Reference timestamp (t_ref) for message context
- Converts relative → absolute using t_ref

**Edge metadata**:
```python
{
  "fact": "...",
  "t_valid": "when relationship started",
  "t_invalid": "when relationship ended",
  "t_created": "when we ingested this",
  "t_expired": "when we invalidated this"
}
```

## Edge Invalidation (Key Feature!)

**Purpose**: Handle contradictions and updates

**Process**:
1. New edge arrives
2. LLM compares against semantically similar existing edges
3. If temporal overlap + contradiction detected:
   - Set `t_invalid` of old edge = `t_valid` of new edge
   - Prioritize new information (transaction timeline)

**Example**:
- Old: "Alice works for Company X" (valid: 2020-01-01 → ∞)
- New: "Alice works for Company Y" (valid: 2024-01-01 → ∞)
- Result: Old edge t_invalid = 2024-01-01

## Entity Extraction & Resolution

### Extraction
- Process current message + last n messages (n=4)
- Speaker automatically extracted as entity
- Reflexion technique to minimize hallucinations
- Generate entity summary for resolution/retrieval

### Resolution
- Embed entity name → 1024-dim vector
- Cosine similarity search on existing entities
- Full-text search on names/summaries
- LLM resolution prompt with candidates + episode context
- Generate updated name/summary if duplicate

### Fact Extraction
- Extract facts (relationships) between entities
- Same fact can appear multiple times (hyper-edges)
- Deduplication via hybrid search
- Constrained to same entity pairs (reduces complexity)

## Community Detection

**Algorithm**: Label propagation (not Leiden)

**Why label propagation**:
- Straightforward dynamic extension
- Add node → survey neighbor communities → assign to plurality
- Delays need for complete refresh
- Lower latency, lower LLM costs

**Community nodes**:
- High-level summaries (map-reduce style)
- Community names with key terms
- Embedded names for cosine similarity search

## Retrieval Architecture

**Pipeline**: Search → Rerank → Construct

### 1. Search (φ)
Three methods (high recall):

**a) Cosine similarity (φ_cos)**:
- Semantic search on embeddings
- Fields: fact (edges), entity name (entities), community name (communities)

**b) BM25 full-text (φ_bm25)**:
- Keyword/lexical search
- Same fields as cosine

**c) Breadth-first search (φ_bfs)**:
- Graph traversal (n-hops)
- Can use recent episodes as seeds
- Contextual similarity (graph proximity = conversational context)

**Multi-faceted coverage**:
- Full-text → word similarities
- Cosine → semantic similarities
- BFS → contextual similarities

### 2. Rerank (ρ)
Increase precision from high-recall candidates:

**Options**:
- Reciprocal Rank Fusion (RRF)
- Maximal Marginal Relevance (MMR)
- Episode-mentions reranker (frequency-based)
- Node distance reranker (from centroid)
- Cross-encoders (LLM relevance scoring - expensive)

### 3. Construct (χ)
Format as text context:

```
FACTS and ENTITIES represent relevant context.

format: FACT (Date range: from - to)
<FACTS>
{facts with t_valid, t_invalid}
</FACTS>

ENTITY_NAME: entity summary
<ENTITIES>
{entity summaries}
</ENTITIES>
```

## Performance Benchmarks

### Deep Memory Retrieval (DMR)
- **Zep**: 94.8% (gpt-4-turbo), 98.2% (gpt-4o-mini)
- **MemGPT**: 93.4% (gpt-4-turbo)
- **Full context**: 94.4% (gpt-4-turbo)

**Limitation**: Only 60 messages per conversation, fits in context window

### LongMemEval (115k token conversations)
**Zep with gpt-4o**:
- Accuracy: 71.2% (vs 60.2% baseline) → **+18.5%**
- Latency: 2.58s (vs 28.9s baseline) → **90% reduction**
- Context: 1.6k tokens (vs 115k baseline)

**Strongest improvements**:
- Single-session-preference: +184%
- Temporal-reasoning: +38.4%
- Multi-session: +30.7%
- Knowledge-update: +6.5%

**Weakness**:
- Single-session-assistant: -17.7% (needs work)

## Technical Details

**Models**:
- Embeddings/Reranking: BGE-m3 (BAAI)
- Graph construction: gpt-4o-mini
- Response generation: gpt-4o-mini or gpt-4o

**Storage**: Neo4j (graph database)

**Search**: Lucene (via Neo4j)

## Key Design Principles

1. **Non-lossy storage**: Episodes preserve raw data
2. **Bi-temporal modeling**: Event time ≠ ingestion time
3. **Dynamic updates**: Edge invalidation handles contradictions
4. **Hierarchical organization**: Episodes → Entities → Communities
5. **Multi-faceted search**: Semantic + lexical + graph
6. **Temporal awareness**: Facts have validity periods
7. **Production-focused**: Latency and cost optimization

## Comparison with Other Systems

| Feature | Zep | Mnemosyne | MemGPT |
|---------|-----|-----------|--------|
| Graph structure | 3-tier (episodes/entities/communities) | 2-tier (memory/entity) | Archival |
| Temporal model | Bi-temporal (event + transaction) | Single (created_at + decay) | No |
| Edge invalidation | Yes (LLM-based) | No | No |
| Communities | Label propagation | No | No |
| Temporal decay | No (but has validity periods) | Yes (forgetting curve) | No |
| Redundancy handling | Deduplication | Boosting | No |
| Search methods | 3 (cosine + BM25 + BFS) | Cosine similarity | Vector |
| Benchmarks | DMR: 94.8%, LME: 71.2% | LoCoMo: 54.6% | DMR: 93.4% |

## Novel Contributions

1. **Bi-temporal modeling** for LLM memory
2. **Edge invalidation** via LLM contradiction detection
3. **Episode-based non-lossy storage** with semantic extraction
4. **Temporal extraction** (absolute + relative dates)
5. **Multi-faceted retrieval** (semantic + lexical + graph)
6. **Production benchmarks** (latency + cost)

## Relevance to Our Project

### Direct applications:
- **Bi-temporal model**: Track event time vs ingestion time
- **Edge invalidation**: Handle contradictory memories
- **Episode storage**: Non-lossy raw data preservation
- **Temporal extraction**: Parse relative dates
- **BFS search**: Graph-based contextual retrieval
- **Multiple search methods**: Hybrid retrieval

### Considerations:
- Do we need bi-temporal? (might be overkill for personal use)
- Edge invalidation vs boosting (different philosophies)
- Communities via label propagation (simpler than Leiden)
- Production focus (latency matters even for hobby code)

### Questions:
- Should we store raw episodes separately?
- How do we handle contradictions? (invalidate vs boost vs merge)
- Do we need transaction timeline for audit?
- Label propagation vs k-means for communities?
