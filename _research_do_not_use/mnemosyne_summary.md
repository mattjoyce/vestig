# Mnemosyne Paper Summary

**Paper**: "Mnemosyne: An Unsupervised, Human-Inspired Long-Term Memory Architecture for Edge-Based LLMs"
**Authors**: Jonelagadda et al., Kaliber AI
**arXiv**: 2510.08601

## Core Architecture

**Graph-based memory system** with:
- **Memory Nodes**: conversation summaries with embeddings
- **Entity Nodes**: extracted entities (people, tools, concepts) with embeddings
- **Relationships**: `RELATED_TO` edges weighted by similarity

## Key Mechanisms

### 1. Commitment (Memory Creation)
- **Substance Filter**: LLM judges if conversation is important enough to store
- **Redundancy Filter**: Detects duplicate/similar memories using:
  - Mutual Information + Jaccard similarity
  - If redundant → pairs with existing node, discards newer of pair
- **Boosting/Rewind**: Redundant memories get temporal boost (stay relevant longer)
- **Graph Construction**: Creates edges to similar nodes based on similarity threshold

### 2. Recall (Memory Retrieval)
- **Start Node Selection**: Hybrid score combining:
  - Semantic similarity (query vs hypothetical query)
  - Metadata similarity (temporal language like "last week", keywords)
- **Probabilistic Traversal**: DFS-style graph walk with:
  - Edge weight × temporal decay × exploration factor
  - Random selection based on probability
- **Temporal Decay**: Reverse sigmoid function (human forgetting curve)
- **Output**: Set of recalled nodes formatted as LLM context

### 3. Core Summary
- Fixed-length subset of central nodes
- Captures user personality/traits
- Always injected into LLM context
- Scored by: connectivity + boost + recency + information density
- K-means clustering ensures diversity

### 4. Pruning (Optional)
- Removes low-probability nodes when approaching memory limits
- Based on selection probability without exploration term

## Human-Inspired Features

1. **Forgetting Curve**: Temporal decay modeled after Ebbinghaus
2. **Primacy-Recency**: Older of redundant pair kept (primacy effect)
3. **Memory Consolidation**: Repeated exposure strengthens memory (boosting)
4. **Temporal Language**: "last week" vs absolute timestamps
5. **Probabilistic Recall**: Not deterministic like traditional RAG

## Technical Details

- **Storage**: Redis (in-memory graph database)
- **Embeddings**: PubMedBERT (domain-specific)
- **LLM**: Mistral-7B for filters/generation
- **Edge-compatible**: Runs on constrained devices
- **Unsupervised**: No training required

## Node Schema

**Memory Node:**
```python
{
  "id": "mem_123",
  "content": "...",
  "content_embedding": [...],
  "trigger": "problem-solution: ...",  # Why created
  "trigger_embedding": [...],
  "created_at": "timestamp",
  "metadata": {...}
}
```

**Entity Node:**
```python
{
  "id": "ent_456",
  "content": "Python",
  "content_embedding": [...],
  "created_at": "timestamp"
}
```

## Performance

- **LoCoMo Benchmark**: 54.55% overall, best in temporal reasoning (60.4%)
- **Human Eval**: 65.8% win rate vs 31.1% for RAG baseline
- Beats Mem0, MemGPT, Zep on most metrics
- Second only to Memory-R1 (which requires RL training)

## Key Innovations

1. **Hypothetical Queries**: Generated during commit, used during recall (HyDE technique)
2. **Dual Filtering**: Substance + redundancy gates
3. **Temporal Boosting**: Sigmoid rewind function for repeated memories
4. **Naturalized Time**: "last week" in similarity calculation
5. **Core Summary**: Fixed-size personality extraction

## Relevance to Our Project

**Direct Applications:**
- Substance filter before committing memories
- Redundancy detection with boosting mechanism
- Temporal decay functions
- Core summary for user traits/patterns
- Hypothetical query generation

**Differences from Our Approach:**
- They use Redis; we're using sqlite-graph
- They focus on healthcare dialogues
- They have more sophisticated temporal mechanics
- We focus on problem-solving/learning patterns
- We have explicit trigger types (problem-solution, feedback, etc.)

**Questions to Consider:**
- Should we implement temporal decay?
- Do we need a substance filter?
- How do we handle redundancy? (boost vs merge vs dedupe)
- Core summary vs distributed entity knowledge?
- Probabilistic traversal vs deterministic retrieval?
