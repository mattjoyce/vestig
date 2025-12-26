# Vestig: LLM Agent Memory System

**Updated**: 2025-12-26 (Post-MemGPT, Zep, and Mnemosyne review - cognitive features)

## Overview
A memory system for LLM agents that enables contextual recall through graph-based RAG with human-inspired temporal mechanics. The system stores memories as a graph with embeddings, temporal decay, and reinforcement learning from repetition. Advanced cognitive features include working sets for recency bias, lateral thinking for creative associations, and daydream mode for insight discovery.

**Inspired by**:
- Mnemosyne (arXiv 2510.08601) - Human-inspired long-term memory architecture
- Zep (arXiv 2501.13956) - Bi-temporal modeling and edge invalidation
- MemGPT (arXiv 2310.08560) - Virtual context management and working memory

## Goals
- **Personal scale**: Thousands of memories, small and portable codebase
- **Learning-focused**: Capture problem-solving, feedback, and learning experiences
- **Contextual retrieval**: Find relevant memories using semantic search, graph relationships, and temporal dynamics
- **Agent integration**: CLI tool usable via hooks, MCP, or direct API
- **Human-like memory**: Temporal decay, reinforcement through repetition, probabilistic recall

## Memory Types (Triggers)
Memories are created with a free-text `trigger` explaining why the memory was created. Common patterns (steered via LLM prompts):
- `problem-solution`: Specific problem and its solution
- `positive-feedback`: Why positive feedback was given
- `corrective-feedback`: Why corrective feedback was given
- `learning-success`: Learning how to do something well
- `learning-failure`: When/why things failed
- Meta-learning patterns across multiple experiences

## Architecture

### Node Types

**Memory Node:**
```python
{
  "id": "mem_<uuid>",
  "node_type": "memory",

  # Content
  "content": "<main memory text>",
  "content_embedding": [vector],

  # Trigger (why created)
  "trigger": "<free text: why this memory was created>",
  "trigger_embedding": [vector],

  # Hypothetical queries (HyDE technique)
  "hypothetical_queries": ["<query1>", "<query2>"],
  "hypothetical_embeddings": [[vector1], [vector2]],

  # Bi-Temporal Modeling (Zep-inspired)
  # T (Event timeline): When it happened
  "t_valid": "<ISO 8601 timestamp>",      # When fact became true
  "t_invalid": null,                      # When fact stopped being true (null = still valid)
  "t_ref": "<ISO 8601 timestamp>",        # Reference time for relative date conversion

  # T' (Transaction timeline): When we learned it
  "t_created": "<ISO 8601 timestamp>",    # When data entered system (was: created_at)
  "t_expired": null,                      # When data was invalidated (edge invalidation)

  # Derived temporal fields
  "last_accessed": "<ISO 8601 timestamp>",
  "access_count": 0,
  "learning_lag": 0,                      # t_created - t_valid (seconds)

  # Fact mutability (time-invariant vs time-variant)
  "temporal_stability": "static|dynamic|unknown",  # How likely to change

  # Reinforcement (from redundancy)
  "boost_value": 0.0,           # Accumulated boost in seconds
  "last_boosted": null,
  "is_paired": null,            # ID of paired redundant node

  # Metadata
  "metadata": {
    "source": "hook|batch|manual|daydream",
    "session": "<optional session id>",
    "keywords": ["<extracted>", "<keywords>"],
    "retrieval_reason": null  # "lateral_association" if from lateral thinking
  }
}
```

**Working Set (Runtime Only - Not Stored in DB):**
```python
{
  "memories": [
    {
      "node_id": "<node_id>",
      "access_time": "<timestamp>",
      "embedding": [vector]
    }
  ],
  "max_size": 10,
  "current_size": 0
}
```

**Entity Node:**
```python
{
  "id": "ent_<uuid>",
  "node_type": "entity",
  "entity_subtype": "user|person|tool|concept|error",

  # Content
  "content": "<entity text: name, noun, concept>",
  "content_embedding": [vector],

  # Bi-Temporal (entities have simpler temporal model)
  "t_created": "<ISO 8601 timestamp>",    # When entity first extracted
  "t_expired": null,                      # When entity was merged/invalidated

  # Special: User entity has core summary
  "core_summary": "<evolving summary of user patterns>",  # Only for user entity
  "core_summary_updated": "<timestamp>"                    # Only for user entity
}
```

### Relationships
```python
{
  "from_node": "<node_id>",
  "to_node": "<node_id>",
  "relationship": "RELATED_TO",

  # Weight
  "weight": 0.0-1.0,            # Similarity score

  # Reinforcement
  "boost": 0.0,                 # Accumulated boost from redundancy
  "last_boosted": null,

  # Bi-Temporal (for facts/relationships between entities)
  # T (Event timeline)
  "t_valid": "<ISO 8601 timestamp>",      # When relationship became true
  "t_invalid": null,                      # When relationship ended (null = still valid)

  # T' (Transaction timeline)
  "t_created": "<ISO 8601 timestamp>",    # When we learned about this relationship
  "t_expired": null,                      # When edge was invalidated (contradiction)

  # Edge invalidation metadata
  "invalidated_by": null,       # ID of edge that invalidated this one
  "invalidation_reason": null   # LLM explanation of contradiction
}
```

### Graph Structure
```
Memory --RELATED_TO--> Entity (extracted names, nouns, concepts)
Memory --RELATED_TO--> Memory (associative links, similar memories)
Entity --RELATED_TO--> Entity (implicit via shared memories)

Special: User Entity (stores core summary of patterns/personality)
```

## Commitment Pipeline

Memory creation follows a 6-stage pipeline to maintain quality and reduce noise:

### Stage 1: Substance Filter
**Purpose**: Discard unimportant memories before storage

**Method**: LLM binary classifier using user core summary as baseline

**Actions**:
- ✅ Substantial → proceed to temporal extraction
- ❌ Not substantial → discard immediately

### Stage 2: Temporal Extraction (Bi-Temporal Modeling)
**Purpose**: Extract event timeline and determine fact mutability

**Method**: LLM extracts temporal information from content

**Extracted Information**:
- **Absolute timestamps**: "June 23, 1912", "2024-01-15"
- **Relative timestamps**: "two weeks ago", "next Thursday", "yesterday"
- **Reference timestamp** (t_ref): Current message timestamp for relative conversion
- **Temporal stability**: Classify as `static` (unlikely to change) or `dynamic` (can change)

**Examples**:
- "I was born on June 23, 1912" → t_valid=1912-06-23, stability=static
- "My favorite color is blue" → t_valid=t_ref, stability=dynamic
- "Alice joined Company X two weeks ago" → t_valid=t_ref - 14 days, stability=dynamic

**Conversion Logic**:
- Relative dates → absolute using t_ref
- If no temporal info → t_valid = t_created (default: we learned it now, assume it's current)
- t_invalid = null (assume still valid unless contradicted later)

**Output**:
- `t_valid`: ISO 8601 timestamp
- `t_invalid`: null (initially)
- `t_ref`: Message timestamp
- `temporal_stability`: "static"|"dynamic"|"unknown"
- `learning_lag`: 0 if no lag, otherwise t_created - t_valid (seconds)

### Stage 3: Entity Extraction
**Purpose**: Extract significant entities for graph linking

**Method**: LLM extracts typed entities (person, tool, concept, error)

**Output**: List of entities with types

### Stage 4: Redundancy Detection & Boosting
**Purpose**: Strengthen memories through repetition, avoid duplicates

**Method**:
- Calculate redundancy score: `α_NMI * MI(emb_new, emb_existing) + (1-α_NMI) * Jaccard(keywords)`
- Threshold: 0.25 (configurable)

**Actions**:
- **Not redundant** → store as new memory
- **Redundant, first time** → store new, pair with existing, boost edge
- **Redundant, already paired** → keep older node (primacy effect), discard newer, pair new with older, boost edge

**Boost Function** (Human memory reinforcement):
```python
# Sigmoid that plateaus, discourages frequent updates
Δ_boost(t) = Δ_max / (1 + exp(-t + t_last_boost + t_crit))

# Config:
# Δ_max = 30 days (max boost)
# t_crit = 7 days (critical point)
```

### Stage 5: Hypothetical Query Generation (Selective)
**Purpose**: Bridge query-memory gap during recall (HyDE technique)

**Method**: LLM generates 1-2 questions this memory would answer

**Strategy to reduce noise**:
- Only for `problem-solution` and `learning-failure` triggers
- Max 2 queries per memory
- Skip for routine feedback/observations

### Stage 6: Graph Construction & Edge Invalidation
**Purpose**: Link new memory to similar existing nodes, handle contradictions

**Graph Construction**:
- Similarity score: `α_key * cosine(emb) + (1-α_key) * Jaccard(keywords)`
- Create edge if similarity > 0.5

**Edge Invalidation (Contradiction Detection)**:
When creating new edges, check for contradictions with existing edges:

1. **Find semantically similar edges** involving same entities
2. **LLM contradiction check**: Does new edge contradict existing edge?
3. **Temporal overlap check**: Do validity periods overlap?
   - Old: [t_valid_old, t_invalid_old]
   - New: [t_valid_new, t_invalid_new]
   - If overlap AND contradiction → invalidate old edge
4. **Invalidation action**:
   - Set old edge's `t_invalid` = new edge's `t_valid` (end old relationship)
   - Set old edge's `t_expired` = current time (mark as superseded)
   - Set old edge's `invalidated_by` = new edge ID
   - Store LLM explanation in `invalidation_reason`

**Skepticism for Static Facts**:
- If old edge has `temporal_stability="static"`, require higher confidence threshold
- LLM must explicitly confirm contradiction with stronger evidence
- Examples: DOB changes are highly suspicious, preferences can change easily

**Example**:
```
Old edge: (Alice) --[works_for]--> (CompanyX)
  t_valid: 2020-01-01, t_invalid: null, stability: dynamic

New edge: (Alice) --[works_for]--> (CompanyY)
  t_valid: 2024-01-01, t_invalid: null

Action:
  Old edge.t_invalid = 2024-01-01
  Old edge.t_expired = 2025-12-26 (now)
  Old edge.invalidated_by = new_edge_id
```

## Temporal Mechanics

### Decay Function (Human Forgetting Curve)
**Purpose**: Old memories naturally fade

**Formula**:
```python
# Reverse sigmoid with linear correction
def temporal_decay(effective_age):
    # effective_age = current_time - created_at - boost_value

    if effective_age >= c:
        # Sigmoid region
        return (1 - d) / (1 + exp((effective_age - a) / b))
    else:
        # Linear correction (ensures = 1 at t=0)
        return 1 - (effective_age * (1 - decay(c)) / c)

# Parameters (config.yaml):
# a = 28 days (midpoint)
# b = 7 days (steepness)
# c = 2 days (linear cutoff)
# d = 0.05 (floor - never fully forgotten)
```

### MemRank (Graph Centrality Scoring)
**Purpose**: Graph centrality indicates importance

**Concept**: MemRank is conceptually based on PageRank but adapted for memory graphs. While PageRank measures importance of web pages via link structure, MemRank measures importance of memories via their position in the associative graph. Central memories that are referenced by many other memories have higher MemRank scores.

**Algorithm**: Standard PageRank computation on memory graph with damping factor (default 0.85).

**Combined Recall Score**:
```python
recall_score(node) = (
    temporal_decay(node.effective_age) *
    memrank(node) *
    log(1 + node.access_count) / log(max_access_count) *
    learning_lag_confidence(node)
)
```

**MemRank Properties**:
- **Centrality**: Well-connected memories score higher
- **Reference quality**: Links from high-MemRank memories boost score more
- **Prevents decay dominance**: Important old memories stay retrievable

### Learning Lag & Salience (Late Learning Uncertainty)
**Purpose**: Memories learned long after events occurred have lower confidence

**Concept**: The gap between when something happened (t_valid) and when we learned it (t_created) introduces uncertainty. Late learning suggests:
- Information came through secondary sources (hearsay)
- Memory reconstruction, not direct experience
- Higher chance of inaccuracy or distortion

**Learning Lag Calculation**:
```python
learning_lag = t_created - t_valid  # in seconds

# Examples:
# - Event today, learned today: lag = 0 (high confidence)
# - Event 1 year ago, learned today: lag = 31536000 (lower confidence)
# - Event in future (planned): lag < 0 (speculative)
```

**Confidence Modifier**:
```python
def learning_lag_confidence(node):
    """Reduce salience for late-learned memories"""
    lag_days = node.learning_lag / 86400

    if lag_days <= 0:
        # Learned at time of event, or future event (speculative)
        return 1.0 if lag_days == 0 else 0.8

    # Decay confidence with learning lag
    # 1 day lag: ~0.95
    # 7 days lag: ~0.85
    # 30 days lag: ~0.70
    # 365 days lag: ~0.40
    return 1.0 / (1 + lag_days / 30)

# Config parameters:
# lag_decay_factor = 30 days (controls how quickly confidence drops)
```

**Special Cases**:
- **Static facts with long lag**: DOB learned years after birth is still high confidence
  - If `temporal_stability="static"`, multiply confidence by 1.5 (capped at 1.0)
- **Dynamic facts with long lag**: Preferences from 5 years ago are less reliable
  - If `temporal_stability="dynamic"`, use formula as-is
- **Future events**: Treat as speculative (0.8 confidence)

**Integration**:
- Learning lag confidence is multiplied into recall score
- Affects both retrieval ranking and salience display
- Makes the system appropriately skeptical of second-hand information

## Retrieval Strategy

### 1. Start Node Selection (Hybrid Scoring)
**Purpose**: Find best entry points for graph traversal

**Method**:
```python
# Metadata score (temporal language + keywords)
S_meta = cosine(query_emb,
                concat(naturalized_time_delta, domain_state, keywords))

# Query score (semantic + hypothetical)
S_query = cosine(query_emb,
                concat(hypothetical_queries, content))

# Combined
S_total = α_meta * S_meta + (1 - α_meta) * S_query

# Select top-K starts (K=3)
```

**Naturalized Time Deltas**:
- "just now" (< 1 hour)
- "earlier today" (< 24 hours)
- "yesterday" (1 day)
- "last week" (< 7 days)
- "last month" (< 30 days)
- "a while ago" (> 30 days)

### 2. Probabilistic Graph Traversal
**Purpose**: Human-like memory activation spread (neuron model)

**Method**:
```python
def probabilistic_recall(start_node, max_nodes=5):
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

            edge = get_edge(current, neighbor)

            # Probability = edge × decay × boost × signal × exploration
            p_select = (
                edge.weight *
                temporal_decay(neighbor.effective_age) *
                (1 + edge.boost) *
                signal_strength *
                μ  # exploration factor (config: 2.0)
            )

            if random() < p_select:
                # Signal decays as it propagates
                new_signal = signal_strength * edge.weight
                traverse(neighbor, new_signal)

    traverse(start_node, signal_strength=1.0)
    return recalled
```

### 3. Context Formatting
**Output**: Recalled memories + user core summary formatted for LLM context

## Advanced Cognitive Features

### Working Set (Recency-Based Saliency)
**Inspired by**: MemGPT's working context, human working memory, CPU cache

**Concept**: Maintain a runtime cache of recently accessed memories that influences future retrieval through recency bias (temporal locality).

**Purpose**:
- Create context coherence across consecutive recalls
- Boost retrieval of memories semantically similar to recently used ones
- Model human working memory effects (recency bias)
- Enable "train of thought" continuity

**Implementation**:
```python
class WorkingSet:
    """Recently accessed memory cache (runtime only, not persisted)"""

    def __init__(self, max_size=10):
        self.memories = []  # [(node_id, access_time, embedding), ...]
        self.max_size = max_size

    def add(self, node_id, embedding):
        """Add memory to working set on access"""
        self.memories.append((node_id, time.time(), embedding))

        # Evict oldest if over capacity (FIFO)
        if len(self.memories) > self.max_size:
            self.memories.pop(0)

    def get_saliency_boost(self, candidate_embedding):
        """Calculate saliency boost for candidate based on working set similarity"""
        if not self.memories:
            return 0.0

        # Average cosine similarity to all working set memories
        similarities = [
            cosine(candidate_embedding, ws_emb)
            for _, _, ws_emb in self.memories
        ]
        return np.mean(similarities)

    def clear(self):
        """Manually clear working set (new task/context)"""
        self.memories = []
```

**Retrieval Integration**:
```python
def calculate_recall_score_with_saliency(node, query_emb, working_set, config):
    """Enhanced recall score with working set boost"""

    # Base score (existing)
    base_score = (
        temporal_decay(node.effective_age) *
        memrank(node) *
        log(1 + node.access_count) / log(max_access_count) *
        learning_lag_confidence(node) *
        cosine(query_emb, node.embedding)
    )

    # Saliency boost from working set
    saliency = working_set.get_saliency_boost(node.embedding)
    recency_weight = config.working_set.recency_weight  # e.g., 0.3

    # Boost similar-to-recent memories
    return base_score * (1 + recency_weight * saliency)
```

**Lifecycle**:
- Updated on every `memory recall` operation
- Persists across multiple recalls in same session
- Can be manually cleared via CLI
- Not persisted to database (runtime state only)

**Use Cases**:
- Maintaining context during multi-turn problem solving
- Following related memories ("tell me more about X")
- Building on previous recalls naturally

---

### Lateral Thinking ("That-Reminds-Me-Of")
**Concept**: Probabilistically inject semantically distant but graph-connected memories to spark creative associations and unexpected insights.

**Purpose**:
- Enable serendipitous discovery
- Surface non-obvious connections
- Model human associative memory jumps
- Break out of semantic similarity echo chambers

**Method**:
1. During recall, with probability P (default 15%), activate lateral mode
2. Pick random start from recalled memories
3. Perform random walk 2-3 hops away on graph
4. Apply quality filters:
   - Memory must be recent enough (temporal decay > threshold)
   - Must be semantically diverse from main results
   - Must pass diversity threshold
5. Add lateral memory to results with special annotation

**Algorithm**:
```python
def add_lateral_memory(recalled_memories, graph, working_set, config):
    """Inject a semantically distant but graph-connected memory"""

    # Probabilistic trigger
    if random.random() > config.lateral_thinking.probability:
        return recalled_memories

    # Start from random recalled or working set memory
    start_candidates = recalled_memories + [get_node(id) for id, _, _ in working_set.memories]
    if not start_candidates:
        return recalled_memories

    start_node = random.choice(start_candidates)

    # Random walk 2-3 hops away
    current = start_node
    hops = random.randint(config.min_hops, config.max_hops)

    for _ in range(hops):
        neighbors = graph.get_neighbors(current)
        if not neighbors:
            break

        # Weighted random choice (favor stronger edges)
        weights = [edge.weight for edge in neighbors]
        current = random.choices(neighbors, weights=weights)[0]

    lateral_node = current

    # Quality filter 1: Must be recent enough
    if temporal_decay(lateral_node.effective_age) < config.min_decay_score:
        return recalled_memories  # Too decayed, skip

    # Quality filter 2: Must be semantically diverse
    similarities = [
        cosine(lateral_node.embedding, m.embedding)
        for m in recalled_memories
    ]

    if max(similarities) > (1 - config.diversity_threshold):
        return recalled_memories  # Too similar, not lateral enough

    # Annotate and add
    lateral_node.metadata["retrieval_reason"] = "lateral_association"
    lateral_node.metadata["lateral_path_from"] = start_node.id
    lateral_node.metadata["lateral_hops"] = hops

    recalled_memories.append(lateral_node)

    return recalled_memories
```

**Context Formatting**:
```python
def format_lateral_memory(memory):
    """Special formatting for lateral associations"""
    path_from_id = memory.metadata.get("lateral_path_from")
    hops = memory.metadata.get("lateral_hops", "?")

    return f"""
[Lateral Association - {hops} hops]
This reminds me of: {memory.content}

Trigger: {memory.trigger}
Created: {naturalize_time_delta(time.time() - memory.t_created)}
"""
```

**Configuration**:
- Probability: 15% (tunable)
- Min hops: 2
- Max hops: 3
- Diversity threshold: 0.3 (must be <70% similar)
- Min decay score: 0.2 (don't surface forgotten memories)

**Use Cases**:
- "This problem reminds me of that other time..."
- Analogical reasoning across domains
- Creative problem solving through distant connections
- Breaking fixation on immediate semantic matches

---

### Daydream Mode (Creative Synthesis) 🔮
**Status**: Experimental, opt-in feature

**Concept**: Offline memory consolidation - synthesize working set memories into creative narratives to discover patterns and insights, without automatically storing outputs.

**Purpose**:
- Pattern recognition across disparate experiences
- Meta-learning (recognizing recurring patterns in problems/solutions)
- Analogical reasoning (problem X similar to problem Y)
- Creative recombination for insight discovery
- Models human sleep/offline consolidation

**Important**: Daydream outputs are **NOT** automatically stored as memories to prevent noise and speculation from polluting the memory graph. User approval required for insights.

**Triggers**:
1. **Manual**: `memory daydream` command
2. **Scheduled**: Background processing (e.g., hourly)
3. **Automatic**: When working set reaches threshold size (e.g., 8 memories)

**Process**:
```python
def daydream(working_set, config):
    """Synthesize working set into creative narrative"""

    # Need sufficient material
    if len(working_set.memories) < 3:
        return None

    # Gather memory contents
    memory_nodes = [get_node(node_id) for node_id, _, _ in working_set.memories]
    memory_texts = [node.content for node in memory_nodes]

    # LLM synthesis prompt
    synthesis_result = llm_synthesize(
        memories=memory_texts,
        style=config.narrative_style,  # problem_solving, pattern_recognition, analogical
        max_length=config.max_synthesis_length
    )

    return DaydreamResult(
        narrative=synthesis_result.narrative,
        insights=synthesis_result.insights,
        source_memory_ids=[node.id for node in memory_nodes],
        timestamp=time.time()
    )
```

**Insight Detection**:
```python
def process_daydream_insights(daydream_result, config):
    """Evaluate and optionally store insights as memories"""

    if not daydream_result.insights or daydream_result.insights == "none":
        return

    # LLM judges insight quality
    quality_score = evaluate_insight_substance(
        insight=daydream_result.insights,
        source_memories=daydream_result.source_memory_ids
    )

    if quality_score > config.insight_threshold:
        if config.prompt_user:
            # Ask user permission
            response = prompt_user_for_insight_approval(daydream_result)
            if response == "approve":
                create_insight_memory(daydream_result)

        elif config.auto_create_memory:
            create_insight_memory(daydream_result)

def create_insight_memory(daydream_result):
    """Convert approved insight to memory"""
    return commit_memory(
        content=daydream_result.insights,
        trigger=f"daydream_insight: synthesized from {len(daydream_result.source_memory_ids)} memories",
        metadata={
            "source": "daydream",
            "source_memory_ids": daydream_result.source_memory_ids,
            "narrative": daydream_result.narrative,
            "discovery_time": daydream_result.timestamp
        }
    )
```

**Narrative Styles**:
- **problem_solving**: Find connections between problems and solutions
- **pattern_recognition**: Identify recurring themes/mistakes
- **analogical**: Draw analogies between different domains

**Configuration**:
- Enabled: false (opt-in)
- Trigger mode: manual | scheduled | working_set_threshold
- Working set threshold: 8 memories
- Schedule interval: 3600 seconds (1 hour)
- Insight threshold: 0.7 (confidence)
- Prompt user: true (don't auto-store)

**CLI Output Example**:
```
$ memory daydream

🔮 Daydream Mode (3 memories synthesized)

NARRATIVE:
The pattern I'm noticing is that authentication errors tend to occur
after dependency updates, particularly with JWT libraries. The first
memory shows a problem with token expiration after upgrading to v5,
the second shows similar issues with refresh tokens, and the third
involves session handling changes...

INSIGHTS DISCOVERED:
Before upgrading auth-related dependencies, always check for breaking
changes in token validation and session management. Create test cases
for token expiry edge cases before upgrade.

Quality Score: 0.85
Store as memory? (y/n):
```

**Use Cases**:
- End-of-day reflection on work session
- Finding meta-patterns across multiple problem-solving sessions
- Discovering analogies between different technical domains
- Recognizing recurring mistakes or anti-patterns
- Generating hypotheses for testing

## User Entity & Core Summary

### Concept
- **User** is a special entity node in the graph
- Maintains evolving **core summary** of patterns/personality
- Always injected into LLM context alongside recalled memories

### Core Summary Update (Fixed-Size Subset)
**Trigger**: Every N new memories (e.g., N=10)

**Method**:
```python
1. K-means clustering on memory embeddings (k=5)
2. Score each node:
   score = (θ_conn * connectivity_score +      # How central?
           θ_boost * boost_score +             # How reinforced?
           θ_recency * recency_score +         # How recent?
           θ_entropy * entropy_score)          # How informative?
3. Select top node from each cluster
4. Add top-K global nodes not yet selected
5. Limit to max 20 nodes
6. Generate summary via LLM
```

**Scores**:
- `θ_conn = 0.3` (connectivity)
- `θ_boost = 0.3` (reinforcement)
- `θ_recency = 0.2` (recency)
- `θ_entropy = 0.2` (information density)

## Technology Stack
- **Storage**: [sqlite-graph](https://github.com/agentflare-ai/sqlite-graph) with Python bindings
- **Embeddings**: BGE-m3 (1024-dim) via `llm` Python module or sentence-transformers
- **Language**: Python 3.x
- **Interface**: CLI (extensible to MCP later)

**Embedding Model Choice**:
- **Primary**: BAAI/bge-m3 (used in Zep's production system)
- **Dimension**: 1024 (good balance of quality vs. size)
- **Performance**: SOTA on retrieval benchmarks
- **Storage**: ~16KB per memory node, ~160MB for 10k memories

## Project Structure
```
vestig/
├── core/
│   ├── __init__.py
│   ├── cli.py                # CLI commands
│   ├── storage.py            # sqlite-graph wrapper
│   ├── embeddings.py         # llm module wrapper
│   ├── commitment.py         # Commitment pipeline (6 stages)
│   ├── retrieval.py          # Probabilistic recall
│   ├── temporal.py           # Decay/boost functions
│   ├── models.py             # Node/relationship schemas
│   ├── graph.py              # Graph operations
│   ├── working_set.py        # Working set management
│   ├── lateral.py            # Lateral thinking logic
│   └── daydream.py           # Daydream mode synthesis
├── data/                      # gitignored runtime data
│   └── memory.db
├── research/                  # exploration work (preserved)
│   ├── mnemosyne_summary.md
│   ├── design_learnings_mnemosyne.md
│   └── 2310.08560v2.pdf      # MemGPT paper
├── config.yaml               # Configuration
├── prompts.yaml              # All LLM prompts with token substitution
├── requirements.txt
├── setup.py                  # pip install -e . → `memory` command
├── README.md
└── .gitignore
```

## CLI Interface

All commands accept `--config <path>` to override default config.yaml.

### Commands (v1)
```bash
# Add memory
memory add <text> --trigger <trigger_text>
memory add <text>  # auto-generate trigger via LLM

# Batch processing
memory batch <file>  # JSON/JSONL file with memories

# Retrieval
memory search <query> --limit N  # semantic search
memory recall <query>  # formatted for agent context
memory recall <query> --lateral  # force lateral association

# Working set
memory working-set show           # display current working set
memory working-set clear           # clear working set (new context)

# Daydream mode
memory daydream                    # trigger daydream on working set
memory daydream --show-narrative   # show full narrative
memory daydream --auto-commit      # auto-create insights as memories

# Inspection
memory list [--type memory|entity]
memory show <id>  # show node details + related nodes
memory graph <id>  # visualize relationships

# User entity
memory user-summary  # show current core summary
memory user-update   # force core summary update
```

### Error Handling
- **Fail hard**: Reasonable error checking, but OK to crash on invalid config, DB errors, LLM failures
- **No extensive validation**: Hobby code, keep it simple

## Configuration

### config.yaml
```yaml
# Embedding configuration
embedding:
  model: "BAAI/bge-m3"  # 1024-dim, SOTA retrieval (used in Zep)
  dimension: 1024
  normalize: true       # Important for cosine similarity
  batch_size: 10

  # Alternative models for different use cases
  alternatives:
    fast: "BAAI/bge-small-en-v1.5"      # 384-dim, faster inference
    balanced: "nomic-ai/nomic-embed-text-v1.5"  # 768-dim, good balance
    api: "openai/text-embedding-3-small"  # API-based, 1536-dim

# Storage
storage:
  db_path: ./data/memory.db
  in_memory: false  # Set true for speed (like Mnemosyne's Redis)

# Commitment Pipeline
commitment:
  # Stage 1: Substance filter
  substance_filter:
    enabled: true
    min_confidence: 0.7

  # Stage 2: Temporal extraction (bi-temporal modeling)
  temporal_extraction:
    enabled: true
    extract_dates: true           # Extract absolute/relative dates
    classify_stability: true      # Classify as static/dynamic/unknown
    static_confidence_boost: 1.5  # Multiplier for static facts with long lag

  # Stage 3: Entity extraction
  entity_extraction:
    enabled: true
    types: ["person", "tool", "concept", "error"]

  # Stage 4: Redundancy detection
  redundancy:
    enabled: true
    threshold: 0.25           # Redundancy score threshold
    alpha_nmi: 0.6           # Weight MI vs Jaccard
    boost_max: 2592000       # 30 days in seconds
    boost_crit: 604800       # 7 days critical point

  # Stage 5: Hypothetical queries
  hypothetical_queries:
    enabled: true
    triggers: ["problem-solution", "learning-failure"]  # Selective
    max_queries: 2

  # Stage 6: Graph construction & edge invalidation
  graph_construction:
    edge_threshold: 0.5        # Min similarity to create edge
    alpha_key: 0.3            # Weight keywords vs semantic

  # Edge invalidation (contradiction detection)
  edge_invalidation:
    enabled: true
    static_fact_threshold: 0.9  # Higher confidence required for static facts
    check_temporal_overlap: true

# Retrieval
retrieval:
  # Start node selection
  start_node:
    top_k: 3                # Number of start nodes
    alpha_meta: 0.6         # Weight metadata vs semantic

  # Probabilistic traversal
  traversal:
    max_nodes: 5            # Max memories to recall
    exploration_factor: 2.0  # μ - higher = more exploration
    signal_decay: 0.8       # Per-hop signal decay

  # Temporal decay
  temporal_decay:
    enabled: true
    midpoint_days: 28       # a - sigmoid midpoint
    steepness_days: 7       # b - sigmoid steepness
    linear_cutoff_days: 2   # c - linear correction cutoff
    floor: 0.05            # d - never fully forgotten

  # MemRank scoring (PageRank-based graph centrality)
  memrank:
    enabled: true
    damping_factor: 0.85   # PageRank damping factor
    weight: 0.3            # vs temporal decay

  # Learning lag confidence (late learning uncertainty)
  learning_lag:
    enabled: true
    lag_decay_factor: 30   # Days - controls confidence drop rate
    future_event_confidence: 0.8  # Confidence for future events (speculative)

  # Working set (recency-based saliency)
  working_set:
    enabled: true
    max_size: 10           # Max memories in working set
    recency_weight: 0.3    # Boost factor for similar-to-recent memories
    persist_across_sessions: false  # Runtime only by default

  # Lateral thinking
  lateral_thinking:
    enabled: true
    probability: 0.15      # 15% chance per recall
    min_hops: 2            # Minimum graph distance
    max_hops: 3            # Maximum graph distance
    diversity_threshold: 0.3  # Must be <70% similar to main results
    min_decay_score: 0.2   # Don't surface completely forgotten memories
    traversal_method: "random_walk"  # or "bridging_node"

# User Entity / Core Summary
core_summary:
  enabled: true
  update_frequency: 10       # Every N new memories
  max_subset_nodes: 20      # Fixed-size subset
  clustering_k: 5           # K-means clusters
  theta_connectivity: 0.3
  theta_boost: 0.3
  theta_recency: 0.2
  theta_entropy: 0.2

# Daydream Mode (Creative Synthesis)
daydream:
  enabled: false            # Opt-in experimental feature

  # Triggers
  trigger_mode: "manual"    # manual|scheduled|working_set_threshold
  working_set_threshold: 8  # Auto-trigger when working set reaches this size
  schedule_interval: 3600   # Seconds between scheduled daydreams (1 hour)

  # Synthesis parameters
  narrative_style: "problem_solving"  # problem_solving|pattern_recognition|analogical
  max_synthesis_length: 500  # tokens

  # Insight detection
  detect_insights: true
  insight_threshold: 0.7    # LLM confidence threshold for quality

  # Actions on insights
  prompt_user: true         # Ask user before storing insights
  auto_create_memory: false # Don't auto-store (keep memory clean)

# Pruning (future)
pruning:
  enabled: false
  threshold: 0.01           # Min recall probability to keep
```

### prompts.yaml
All LLM prompts stored here with token substitution ({{1}}, {{2}}, etc.)

```yaml
# Stage 1: Substance Filter
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

  User Summary: {{1}}
  Content: {{2}}

  Output only: 1 (substantial) or 0 (not substantial)

# Stage 2: Temporal Extraction
extract_temporal_info: |
  Extract temporal information from this text for bi-temporal modeling.

  Reference timestamp (message time): {{1}}
  Content: {{2}}

  Extract:
  1. Event time (t_valid): When did the fact/event become true?
     - Absolute: "June 23, 1912", "2024-01-15"
     - Relative: "two weeks ago", "yesterday", "next Thursday"
     - Convert relative to absolute using reference timestamp
     - If no temporal info, assume t_valid = reference timestamp

  2. Temporal stability: How likely is this fact to change?
     - static: Unlikely to change (DOB, birthplace, historical events)
     - dynamic: Can change (job, preferences, current status)
     - unknown: Not clear

  Return JSON only:
  {
    "t_valid": "ISO 8601 timestamp",
    "temporal_stability": "static|dynamic|unknown",
    "explanation": "brief explanation of temporal reasoning"
  }

# Stage 3: Entity Extraction
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

# Stage 5: Hypothetical Queries (HyDE)
generate_hypothetical_query: |
  Given this memory content, generate 1-2 questions that this memory would answer.
  Make them specific and natural, as a user would ask.

  Memory content: {{1}}
  Trigger: {{2}}

  Output format:
  Q1: ...
  Q2: ...

# Stage 6: Edge Contradiction Detection
detect_edge_contradiction: |
  Determine if these two relationship facts contradict each other.

  Old relationship:
  - From: {{1}}
  - To: {{2}}
  - Fact: {{3}}
  - Valid period: {{4}} to {{5}}
  - Temporal stability: {{6}}

  New relationship:
  - From: {{7}}
  - To: {{8}}
  - Fact: {{9}}
  - Valid period: {{10}} to {{11}}
  - Temporal stability: {{12}}

  Consider:
  1. Do these facts contradict each other?
  2. Is temporal overlap present?
  3. If old fact is "static", require high confidence (>0.9) to contradict
  4. Examples:
     - "Alice works for X" vs "Alice works for Y" → contradiction (job change)
     - "Born June 23" vs "Born July 15" → contradiction (requires high confidence, suspicious)
     - "Favorite color blue" vs "Favorite color red" → contradiction (preference change)

  Return JSON only:
  {
    "is_contradiction": true|false,
    "confidence": 0.0-1.0,
    "explanation": "brief reasoning"
  }

# Trigger generation
generate_trigger: |
  Generate a concise trigger phrase explaining why this memory is being created.
  Format: "<category>: <specific reason>"
  Suggested categories: problem-solution, positive-feedback, corrective-feedback,
                        learning-success, learning-failure

  Memory content: {{1}}

  Trigger:

# Core summary generation
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

# Time naturalization
naturalize_time_delta: |
  Convert this time difference to natural language.

  Time delta: {{1}} seconds

  Output one of: "just now", "earlier today", "yesterday", "last week",
                "last month", "a few months ago", "a while ago"

# Daydream synthesis
daydream_synthesize: |
  You are performing creative memory synthesis. Your task is to find connections,
  patterns, and insights across these recent memories.

  Narrative Style: {{1}}  # problem_solving, pattern_recognition, or analogical

  Recent memories from working set:
  {{2}}

  Instructions based on style:
  - problem_solving: Connect problems to solutions, find recurring solution patterns
  - pattern_recognition: Identify recurring themes, mistakes, or anti-patterns
  - analogical: Draw analogies between different domains or problem types

  Generate:
  1. NARRATIVE: A creative synthesis exploring connections (max 300 words)
  2. INSIGHTS: Concrete, actionable insights discovered (or "none")

  Format:
  NARRATIVE:
  [Your synthesis here]

  INSIGHTS:
  [Your insights here, or "none"]

# Daydream insight evaluation
evaluate_daydream_insight: |
  Evaluate the quality and substance of this daydream insight.

  Source memories (count): {{1}}
  Insight: {{2}}

  Criteria for substantial insight:
  - Actionable (can be applied in future)
  - Non-obvious (not just restating existing memories)
  - Generalizable (applies beyond specific instance)
  - Novel connection or pattern recognition

  Return JSON only:
  {
    "quality_score": 0.0-1.0,
    "is_substantial": true|false,
    "reasoning": "brief explanation"
  }
```

## Integration Options

### Option 1: CLI (Bash/Shell)
- Agent invokes via shell commands: `memory recall "authentication error"`
- Works with any LLM agent that can execute bash commands
- Simple, universal compatibility

### Option 2: MCP Server
- MCP server exposing memory operations as tools
- Direct tool calling without shell wrapper
- Native integration for MCP-compatible agents

### Option 3: Direct API (future)
- Python library importable into agent code
- Programmatic access to all memory operations
- Suitable for custom agent implementations

## Lifecycle

### Memory Creation (Commitment Pipeline)
1. **Substance filter** → discard if not important
2. **Temporal extraction** → extract event time, determine mutability
3. **Entity extraction** → extract typed entities
4. **Redundancy check** → boost if duplicate, otherwise store
5. **Hypothetical queries** → generate (selective)
6. **Graph construction & edge invalidation** → link to similar nodes, handle contradictions

**Sources**:
- **From messages**: Hook captures conversation, creates memories
- **Batch processing**: Import from files (JSON/JSONL)
- **Manual**: Direct CLI invocation

### Memory Recall (Retrieval)
1. **Start node selection** → hybrid scoring (semantic + metadata + working set saliency)
2. **Probabilistic traversal** → DFS with decay/boost
3. **Lateral thinking** → probabilistic injection of distant associations (15% chance)
4. **Working set update** → add recalled memories to working set
5. **Context formatting** → recalled memories + user core summary

### Daydream (Optional/Scheduled)
1. **Trigger check** → manual command, scheduled interval, or working set threshold
2. **Synthesis** → LLM combines working set into creative narrative
3. **Insight detection** → LLM evaluates discovered patterns/insights
4. **User approval** → prompt user to approve storing insights as memories

### User Core Summary
- **Update trigger**: Every N new memories (configurable)
- **Method**: Fixed-size subset via k-means + multi-factor scoring
- **Storage**: Attached to user entity node

## Design Decisions

### Human-Inspired Principles
**From Mnemosyne**:
- ✅ **Temporal decay**: Memories fade naturally (forgetting curve)
- ✅ **Reinforcement**: Repetition strengthens memory (boosting)
- ✅ **Probabilistic recall**: Not deterministic top-K
- ✅ **Natural language time**: "last week" not timestamps
- ✅ **Primacy effect**: Older of redundant pair kept
- ✅ **Fixed-size summaries**: Core summary from subset, scales

**From Zep**:
- ✅ **Bi-temporal modeling**: Track event time vs learning time
- ✅ **Edge invalidation**: Handle contradictions with temporal precision
- ✅ **Late learning uncertainty**: Reduce confidence for second-hand info
- ✅ **Fact mutability awareness**: Static vs dynamic facts

**From MemGPT**:
- ✅ **Working set / active context**: Recent memories influence retrieval (recency bias)
- ✅ **Iterative retrieval patterns**: Multi-step search refinement capability
- ⚠️ **Not adopting**: FIFO queue for conversation history (out of scope)
- ⚠️ **Not adopting**: Archival vs recall split (all memories treated equally in graph)

**Novel Additions (Human Cognition)**:
- ✅ **Lateral thinking**: Probabilistic distant associations (creativity)
- ✅ **Daydream mode**: Offline consolidation and insight discovery
- ✅ **Recency-based saliency**: Working set boosts similar memories
- ✅ **Serendipitous discovery**: Breaking semantic similarity echo chambers

### What We're NOT Doing (Initially)
- ❌ Speed optimization
- ❌ Observability/logging of retrievals
- ❌ Async operations
- ❌ Unit tests (hobby code)

### What We ARE Doing
- ✅ Simple, clean graph schema with bi-temporal fields
- ✅ Dual embeddings (content + trigger)
- ✅ Hypothetical queries (HyDE technique)
- ✅ 6-stage commitment pipeline
- ✅ Substance filter (noise reduction)
- ✅ Temporal extraction (absolute + relative dates)
- ✅ Fact mutability classification (static/dynamic)
- ✅ Redundancy detection with boosting
- ✅ Edge invalidation for contradictions
- ✅ Temporal decay + MemRank + learning lag scoring
- ✅ Probabilistic graph traversal
- ✅ User entity with core summary
- ✅ LLM-guided extraction
- ✅ Configurable prompts
- ✅ Graph-based associations
- ✅ Working set with recency bias (saliency)
- ✅ Lateral thinking for creative associations
- ✅ Daydream mode for insight discovery (opt-in)

### Noise Mitigation Strategies
1. **Substance filter**: Gate at entry
2. **Selective hypotheticals**: Only for problem-solving triggers
3. **Redundancy threshold**: High bar (0.25)
4. **Max queries**: Limit to 2 per memory
5. **Core summary size**: Fixed at 20 nodes
6. **Entity type constraints**: Only 4 types
7. **Temporal floor**: Never fully forgotten (0.05)
8. **Pruning threshold**: Only remove very low probability (<0.01)
9. **Lateral thinking filters**: Diversity + decay thresholds prevent noise
10. **Daydream user approval**: Insights require explicit user consent before storage
11. **Working set size limit**: Max 10 memories (prevents unbounded growth)

## Implementation Phases

### Phase 1: Core Graph (MVP)
- [ ] Node/edge schema
- [ ] Basic storage (sqlite-graph)
- [ ] Embedding generation (llm)
- [ ] Simple commit + recall

### Phase 2: Commitment Pipeline
- [ ] Substance filter
- [ ] Temporal extraction (bi-temporal)
- [ ] Entity extraction
- [ ] Redundancy detection
- [ ] Boosting mechanism
- [ ] Edge creation
- [ ] Edge invalidation (contradiction detection)

### Phase 3: Temporal Mechanics
- [ ] Decay function
- [ ] Effective age calculation
- [ ] Boost value application
- [ ] Learning lag confidence calculation
- [ ] Temporal language generation

### Phase 4: Advanced Retrieval
- [ ] Hypothetical query generation
- [ ] Hybrid start node selection
- [ ] Probabilistic traversal
- [ ] MemRank scoring (PageRank-based)

### Phase 5: User Entity
- [ ] Core summary generation
- [ ] Periodic update mechanism
- [ ] Fixed-size subset selection
- [ ] K-means clustering

### Phase 6: Advanced Cognitive Features
- [ ] Working set implementation
- [ ] Working set integration with retrieval scoring
- [ ] Lateral thinking random walk
- [ ] Lateral thinking quality filters
- [ ] Daydream mode synthesis
- [ ] Daydream insight detection
- [ ] Daydream user approval flow

## Success Criteria
1. Agent can recall relevant past problems/solutions
2. System learns from positive and corrective feedback
3. Memories surface based on semantic similarity + temporal relevance + graph centrality
4. Redundant memories strengthen (not duplicate)
5. Old but important memories don't fully fade (floor + MemRank)
6. User core summary captures evolving patterns
7. Working set creates context coherence across consecutive recalls
8. Lateral thinking occasionally surfaces unexpected but useful connections
9. Daydream mode discovers actionable insights (with user approval)
10. Easy to integrate with any LLM agent workflow (CLI, MCP, or API)
11. Codebase stays small and maintainable

## References

### Papers
- **Mnemosyne**: arXiv 2510.08601 (Jonelagadda et al., 2025) - Human-inspired temporal memory architecture
- **Zep**: arXiv 2501.13956 (Rasmussen et al., 2025) - Temporal knowledge graph for agent memory
- **MemGPT**: arXiv 2310.08560 (Packer et al., 2024) - Virtual context management and OS-inspired LLM memory
- **HyDE**: Hypothetical Document Embeddings (Gao et al., 2023)
- **BGE-m3**: arXiv 2402.03216 (Chen et al., 2024) - Multi-lingual, multi-functionality embeddings

### Technologies
- **sqlite-graph**: https://github.com/agentflare-ai/sqlite-graph
- **BGE Embeddings**: https://github.com/FlagOpen/FlagEmbedding
- **llm CLI**: https://github.com/simonw/llm
