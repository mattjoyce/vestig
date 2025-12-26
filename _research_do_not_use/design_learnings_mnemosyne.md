# Design Learnings from Mnemosyne Paper

**Date**: 2025-12-25
**Source**: arXiv 2510.08601

## Decisions Made

1. **Temporal Mechanics**: Fade memories through temporal decay + PageRank-style scoring
2. **Redundancy**: Mnemosyne-style boosting (strengthen original when similar memory arrives)
3. **Substance Filter**: YES - keep noise low, filter before committing
4. **User Entity/Core Summary**: YES - user is an entity, maintain core summary
5. **Hypothetical Queries**: YES but discerning - useful but potential noise source

---

## 1. Graph Schema Design

### Node Types (Both have embeddings)

**Memory Node:**
```python
{
  "id": uuid,
  "node_type": "memory",
  "content": text,              # Main content
  "content_embedding": vector,
  "trigger": text,              # Why created (free text)
  "trigger_embedding": vector,
  "hypothetical_query": text,   # LEARNING: What question might this answer?
  "hypothetical_embedding": vector,
  "created_at": timestamp,
  "last_accessed": timestamp,   # LEARNING: Track access for PageRank
  "access_count": int,          # LEARNING: Reinforcement through recall
  "boost_value": float,         # LEARNING: Accumulated boost from redundancy
  "last_boosted": timestamp,
  "metadata": {
    "source": "hook|batch|manual",
    "keywords": [...]           # LEARNING: For Jaccard similarity
  }
}
```

**Entity Node:**
```python
{
  "id": uuid,
  "node_type": "entity",
  "entity_subtype": "user|person|tool|concept|error",  # LEARNING: Typed entities
  "content": text,
  "content_embedding": vector,
  "created_at": timestamp,
  "core_summary": text,         # LEARNING: For user entity, maintain summary
  "core_summary_updated": timestamp
}
```

### Edges
```python
{
  "from_node": id,
  "to_node": id,
  "relationship": "RELATED_TO",
  "weight": float,              # Similarity score (0-1)
  "boost": float,               # LEARNING: Temporal boost value
  "created_at": timestamp,
  "last_boosted": timestamp
}
```

---

## 2. Commitment Pipeline

### Stage 1: Substance Filter (LEARNING)
**Purpose**: Prevent noise from cluttering memory
**Method**: LLM-based binary classifier

```yaml
# prompts.yaml
substance_filter: |
  You are a memory importance judge. Determine if this content is substantial enough to remember.

  Substantial content includes:
  - Problem-solution patterns
  - Learning successes or failures
  - Feedback (positive or corrective) with reasons
  - Significant decisions or insights
  - Technical discoveries or errors

  Non-substantial content:
  - Routine greetings or small talk
  - Trivial confirmations
  - Repeated mundane updates

  User Entity Summary: {{1}}
  Content: {{2}}

  Output only: 1 (substantial) or 0 (not substantial)
```

**Implementation**:
- Call LLM with content + user core summary as baseline
- If 0 → discard immediately, no storage
- If 1 → proceed to redundancy filter

### Stage 2: Entity Extraction (LEARNING)
**Before redundancy check**, extract entities to enable similarity calculation

```yaml
# prompts.yaml
extract_entities: |
  Extract significant entities from this text.

  Entity types:
  - person: Names, roles, @mentions
  - tool: Technologies, libraries, frameworks, commands
  - concept: Abstract ideas, patterns, methodologies
  - error: Error types, failure modes

  Return JSON only:
  {
    "entities": [
      {"text": "...", "type": "..."},
      ...
    ]
  }

  Text: {{1}}
```

### Stage 3: Redundancy Filter (LEARNING - Mnemosyne Style)
**Purpose**: Strengthen memories through repetition, avoid duplicates

**Redundancy Score**:
```python
# Combination of:
# 1. Mutual Information (MI) of embeddings
# 2. Jaccard similarity of keywords

RS(new, existing) = α_NMI * MI(emb_new, emb_existing) +
                    (1 - α_NMI) * Jaccard(keywords_new, keywords_existing)

# Config: α_NMI = 0.6 (weight MI more than keywords)
# Threshold: RS > 0.25 = redundant
```

**Actions**:
```python
if RS(new, existing) > threshold:
    # REDUNDANT - Apply Mnemosyne boosting
    if existing.is_paired is None:
        # Case 1: First redundancy
        store_new()
        pair(new, existing)
        boost_edge(new, existing, calculate_boost(time_delta))
    else:
        # Case 2: Already has pair
        older = select_older(existing, existing.paired_node)
        newer = select_newer(existing, existing.paired_node)
        remove(newer)  # Primacy effect
        store_new()
        pair(new, older)
        boost_edge(new, older, calculate_boost(time_delta))
else:
    # UNIQUE - Store normally
    store_new()
```

**Boost Function** (LEARNING - Human memory reinforcement):
```python
# Sigmoid function that plateaus
# Discourages frequent updates, rewards spaced repetition

Δ_boost(t) = Δ_max * (1 / (1 + exp(-t + t_last_boost + t_crit)))

# Where:
# - t = current time
# - t_last_boost = when edge was last boosted
# - t_crit = time to plateau (e.g., 7 days)
# - Δ_max = maximum boost value (e.g., 30 days)
```

### Stage 4: Hypothetical Query Generation (LEARNING - HyDE Technique)
**Purpose**: Bridge query-memory gap during recall

```yaml
# prompts.yaml
generate_hypothetical_query: |
  Given this memory content, generate 1-2 questions that this memory would answer.
  Make them specific and natural, as a user would ask.

  Memory content: {{1}}
  Trigger: {{2}}

  Output format:
  Q1: ...
  Q2: ...
```

**Strategy to Reduce Noise**:
- Only generate for `problem-solution` and `learning-failure` triggers
- Skip for routine feedback or observations
- Limit to 2 queries maximum per memory

### Stage 5: Graph Construction
**Create edges to similar nodes**:
```python
similarity(new, existing) = α_key * cosine(emb_new, emb_existing) +
                           (1 - α_key) * Jaccard(keywords_new, keywords_existing)

# Config: α_key = 0.3 (semantic > keywords)
# Edge created if similarity > 0.5
```

---

## 3. Temporal Mechanics (LEARNING)

### Decay Function (Human Forgetting Curve)
**Purpose**: Old memories fade naturally

```python
# Reverse sigmoid with linear correction for early values

def temporal_decay(effective_age):
    """
    Returns probability multiplier (0 to 1) based on memory age

    effective_age = current_time - created_at - boost_value
    """
    if effective_age >= c:
        # Sigmoid region
        return (1 - d) / (1 + exp((effective_age - a) / b))
    else:
        # Linear correction region (ensures = 1 at t=0)
        return 1 - (effective_age * (1 - decay(c)) / c)

# Parameters (config.yaml):
# a = 28 days (sigmoid midpoint)
# b = 7 days (steepness)
# c = 2 days (linear correction cutoff)
# d = 0.05 (floor - never fully forgotten)
```

### PageRank-Style Scoring (OUR ADDITION)
**Purpose**: Important/central memories stay relevant

```python
# Combine temporal decay with graph centrality
recall_score(node) = temporal_decay(node.age - node.boost) *
                    pagerank(node) *
                    access_count_factor(node)

# Where:
# - pagerank(node) = centrality in graph (0-1)
# - access_count_factor = log(1 + access_count) / log(max_access_count)
```

---

## 4. Retrieval Strategy (LEARNING)

### Start Node Selection (Hybrid Scoring)
**Purpose**: Find best entry point for graph traversal

```python
# LEARNING: Combine semantic + metadata scoring

# 1. Metadata Score (temporal language + keywords)
S_meta = cosine(query_emb,
                concat(naturalized_time_delta, domain_state, keywords))

# naturalized_time_delta examples:
# - "just now" (< 1 hour)
# - "earlier today" (< 24 hours)
# - "last week" (< 7 days)
# - "last month" (< 30 days)
# - "a while ago" (> 30 days)

# 2. Query Score (semantic + hypothetical)
S_query = cosine(query_emb,
                concat(hypothetical_query, content))

# 3. Combined Score
S_total = α_meta * S_meta + (1 - α_meta) * S_query

# Config: α_meta = 0.6 (favor metadata for old memories)
# Select top-K starts (K=3)
```

### Probabilistic Graph Traversal (LEARNING)
**Purpose**: Natural, human-like memory activation spread

```python
# LEARNING: Neuron activation propagation model

def probabilistic_recall(start_node, max_nodes=5):
    """
    DFS-style traversal with probabilistic selection
    Signal decays as it spreads through graph
    """
    recalled = []
    visited = set()

    def traverse(current, signal_strength):
        if len(recalled) >= max_nodes:
            return

        visited.add(current)
        recalled.append(current)

        for neighbor in current.neighbors:
            if neighbor in visited:
                continue

            # LEARNING: Probability = edge_weight × decay × boost × signal × exploration
            edge = get_edge(current, neighbor)
            p_select = (edge.weight *
                       temporal_decay(neighbor.effective_age) *
                       (1 + edge.boost) *
                       signal_strength *
                       μ)  # exploration factor

            if random() < p_select:
                # Signal decays as it propagates (neuron model)
                new_signal = signal_strength * edge.weight
                traverse(neighbor, new_signal)

    traverse(start_node, signal_strength=1.0)
    return recalled
```

---

## 5. User Entity & Core Summary (LEARNING)

### Concept
- User is a special entity node
- Maintains evolving "core summary" of personality/patterns
- Always injected into context alongside recalled memories

### Core Summary Update Strategy
**Fixed-size subset approach** (LEARNING):

```python
# Periodically (e.g., every 10 new memories) update core summary

def select_core_subset(graph, max_nodes=20):
    """
    Select most representative nodes for summarization
    LEARNING: Multi-factor scoring
    """
    # 1. K-means clustering for diversity (k=5)
    clusters = kmeans(graph.memory_nodes, k=5)

    # 2. Score each node
    for node in graph.memory_nodes:
        score = (θ_conn * connectivity_score(node) +      # How central?
                θ_boost * boost_score(node) +             # How reinforced?
                θ_recency * recency_score(node) +         # How recent?
                θ_entropy * entropy_score(node))          # How informative?

    # 3. Select top node from each cluster
    core_nodes = [top_node(cluster) for cluster in clusters]

    # 4. Add top-K global nodes not yet selected
    core_nodes += top_k_remaining(all_nodes - core_nodes, k=5)

    return core_nodes[:max_nodes]

# Config (sum to 1.0):
# θ_conn = 0.3 (connectivity)
# θ_boost = 0.3 (reinforcement)
# θ_recency = 0.2 (recency)
# θ_entropy = 0.2 (information density)
```

### Core Summary Generation
```yaml
# prompts.yaml
generate_core_summary: |
  Based on these memories, create a concise summary of what you've learned about the user.

  Focus on:
  - Recurring patterns in problems/solutions
  - Learning preferences and strengths
  - Common mistakes or challenges
  - Preferred tools, technologies, approaches
  - Working style and methodology

  Previous summary (if exists): {{1}}
  Recent memories: {{2}}

  Generate updated summary (max 500 words):
```

---

## 6. Configuration Structure

```yaml
# config.yaml

embedding:
  model: "default"  # llm default
  batch_size: 10

storage:
  db_path: "./data/memory.db"
  in_memory: false  # Set true for speed (like Mnemosyne's Redis)

commitment:
  substance_filter:
    enabled: true
    min_confidence: 0.7

  redundancy:
    enabled: true
    threshold: 0.25
    alpha_nmi: 0.6  # Weight mutual information vs Jaccard
    boost_max: 2592000  # 30 days in seconds
    boost_crit: 604800  # 7 days critical point

  entity_extraction:
    enabled: true
    types: ["person", "tool", "concept", "error"]

  hypothetical_queries:
    enabled: true
    triggers: ["problem-solution", "learning-failure"]  # Selective
    max_queries: 2

retrieval:
  start_node:
    top_k: 3
    alpha_meta: 0.6  # Weight metadata vs semantic

  traversal:
    max_nodes: 5
    exploration_factor: 2.0  # μ
    signal_decay: 0.8  # Per hop

  temporal_decay:
    enabled: true
    midpoint_days: 28  # a
    steepness_days: 7   # b
    linear_cutoff_days: 2  # c
    floor: 0.05         # d - never fully forgotten

  pagerank:
    enabled: true
    weight: 0.3  # vs temporal decay

core_summary:
  enabled: true
  update_frequency: 10  # Every N new memories
  max_subset_nodes: 20
  clustering_k: 5
  theta_connectivity: 0.3
  theta_boost: 0.3
  theta_recency: 0.2
  theta_entropy: 0.2

pruning:
  enabled: false  # Only when approaching limits
  threshold: 0.01  # Minimum recall probability to keep
```

---

## 7. Prompts Structure

```yaml
# prompts.yaml

# All prompts use {{N}} token substitution

substance_filter: |
  [See Stage 1 above]

extract_entities: |
  [See Stage 2 above]

generate_trigger: |
  Generate a concise trigger phrase explaining why this memory is being created.
  Format: "<category>: <specific reason>"
  Suggested categories: problem-solution, positive-feedback, corrective-feedback,
                        learning-success, learning-failure

  Memory content: {{1}}

  Trigger:

generate_hypothetical_query: |
  [See Stage 4 above]

generate_core_summary: |
  [See Section 5 above]

naturalize_time_delta: |
  Convert this time difference to natural language.

  Time delta: {{1}} seconds

  Output one of: "just now", "earlier today", "yesterday", "last week",
                "last month", "a few months ago", "a while ago"
```

---

## 8. Key Design Principles (LEARNINGS)

### From Mnemosyne Paper:

1. **Human-Inspired > Optimization-Driven**
   - Temporal decay mirrors forgetting curves
   - Repetition strengthens memory (boosting)
   - Probabilistic recall (not deterministic top-K)
   - Natural language for time ("last week" not timestamps)

2. **Modular Filtering Pipeline**
   - Substance → Redundancy → Extraction → Storage
   - Each gate is independent and configurable
   - Fail early (discard at substance filter)

3. **Dual Representation**
   - Content embeddings (what it says)
   - Hypothetical query embeddings (what it answers)
   - Bridges semantic gap in retrieval

4. **Graph Centrality Matters**
   - Central nodes = important concepts
   - PageRank-style scoring
   - Used for core summary selection

5. **Temporal Context is Critical**
   - Not just "when" but "how long ago"
   - Naturalized time in similarity calculation
   - Decay + boost creates dynamic relevance

6. **Fixed-Size Summaries Scale**
   - Core summary from subset, not full graph
   - K-means clustering ensures diversity
   - Multi-factor scoring (connectivity + recency + boost + entropy)

7. **Edge-Compatible Design**
   - Small, local models (Mistral-7B class)
   - Efficient graph operations
   - Avoid expensive operations in hot path

### Our Additions:

8. **PageRank Integration**
   - Graph centrality as relevance signal
   - Combines with temporal decay
   - Prevents important nodes from fading

9. **Discerning Hypothetical Queries**
   - Only for specific trigger types
   - Limits noise generation
   - Quality > quantity

10. **User as Entity**
    - Consistent with graph model
    - Core summary attached to user node
    - Enables user-centric retrieval

---

## 9. Implementation Priority

### Phase 1: Core Graph (MVP)
- [ ] Node/edge schema
- [ ] Basic storage (sqlite-graph)
- [ ] Embedding generation (llm)
- [ ] Simple commit + recall

### Phase 2: Commitment Pipeline
- [ ] Substance filter
- [ ] Entity extraction
- [ ] Redundancy detection
- [ ] Boosting mechanism
- [ ] Edge creation

### Phase 3: Temporal Mechanics
- [ ] Decay function
- [ ] Effective age calculation
- [ ] Boost value application
- [ ] Temporal language generation

### Phase 4: Advanced Retrieval
- [ ] Hypothetical query generation
- [ ] Hybrid start node selection
- [ ] Probabilistic traversal
- [ ] PageRank scoring

### Phase 5: User Entity
- [ ] Core summary generation
- [ ] Periodic update mechanism
- [ ] Fixed-size subset selection
- [ ] K-means clustering

---

## 10. Noise Mitigation Strategies (CRITICAL)

**Learning**: Mnemosyne's biggest risk is LLM-generated noise

### Our Safeguards:

1. **Substance Filter**: Gate at entry
2. **Selective Hypotheticals**: Only for problem-solving triggers
3. **Redundancy Threshold**: High bar (0.25) to avoid over-merging
4. **Max Queries**: Limit to 2 per memory
5. **Core Summary Size**: Fixed at 20 nodes max
6. **Entity Type Constraints**: Only 4 types (person, tool, concept, error)
7. **Temporal Floor**: Memories never fully forgotten (0.05 floor)
8. **Pruning Threshold**: Only remove very low probability (<0.01)

### Monitoring:
- Track substance filter rejection rate
- Monitor graph size growth
- Measure retrieval latency
- Audit core summary changes

---

## Summary

**We're adopting Mnemosyne's core innovations**:
- ✅ Graph-based memory with dual embeddings
- ✅ Modular commitment pipeline (substance + redundancy)
- ✅ Temporal decay with boosting (human-inspired)
- ✅ Hypothetical queries (HyDE technique)
- ✅ Probabilistic graph traversal
- ✅ Core summary from fixed-size subset
- ✅ Natural language temporal references

**With our enhancements**:
- ✅ PageRank-style centrality scoring
- ✅ User entity with core summary
- ✅ Discerning hypothetical generation (noise control)
- ✅ Explicit trigger types for memory categorization
- ✅ sqlite-graph instead of Redis (portability)

**Next**: Update SPEC.md with these learnings, then begin implementation.
