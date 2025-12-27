# Sprint 1 Backlog — Vestig M1: MVP Core Loop

**Goal**: Working CLI that can add a memory, persist it, and recall top-K relevant memories.

---

## Work Items (9 total)

### 1. Project Scaffold & Configuration
**Objective**: Establish project structure, dependencies, and minimal config

**File/Module Boundary**:
- `pyproject.toml` (packaging + dependencies + tool config)
- `src/vestig/core/__init__.py` (src/ layout)
- `config.yaml`
- `.gitignore`
- `data/` directory (gitignored)

**Acceptance Checks**:
- `pip install -e .` succeeds and registers `memory` CLI command
- `memory --help` shows usage without errors
- `config.yaml` exists with `embedding.model`, `embedding.dimension`, and `storage.db_path` settings
- Empty `data/` directory is created and gitignored
- `pyproject.toml` includes: project metadata, dependencies, `[project.scripts]` entry point, ruff config

**Non-goals**:
- No prompts.yaml yet (no LLM calls in M1)
- No complex config validation beyond basic schema
- No logging/observability infrastructure

---

### 2. Embeddings Module (Minimal)
**Objective**: Wrap embedding generation for text → vector with dimension validation

**File/Module Boundary**:
- `src/vestig/core/embeddings.py`

**Acceptance Checks**:
- `embed_text(text) -> List[float]` returns vector of configured dimension
- Uses `sentence-transformers` library with BAAI/bge-m3 model
- Reads model and dimension from `config.yaml`
- Validates actual embedding length matches `config.embedding.dimension`
- Raises clear error if dimension mismatch: `"Embedding dimension mismatch: expected 1024, got {actual}"`
- Batch embedding works: `embed_batch(texts) -> List[List[float]]`
- Vectors are normalized (for cosine similarity)

**Non-goals**:
- No caching of embeddings
- No dual embeddings (content + trigger) — single embedding only
- No hypothetical query embeddings
- No alternative embedding backends (defer to M2+)

---

### 3. Storage Layer (Minimal Schema)
**Objective**: Persist and retrieve memory nodes with embeddings

**File/Module Boundary**:
- `src/vestig/core/storage.py`
- `src/vestig/core/models.py` (data classes)

**Acceptance Checks**:
- Initializes sqlite-graph database at `config.storage.db_path`
- `MemoryNode` dataclass includes: id, content, content_embedding, created_at, metadata
- `store_memory(node) -> str` persists node and returns ID
- `get_memory(id) -> MemoryNode | None` retrieves by ID
- `get_all_memories() -> List[MemoryNode]` loads all memories (for brute-force search)
- Database file persists across Python sessions

**Non-goals**:
- No graph edges/relationships
- No entity nodes
- No temporal fields (t_valid, t_invalid, learning_lag)
- No boost_value, access_count, or reinforcement fields
- No trigger field (M2)
- No vector index optimization (brute-force is fine for M1)

---

### 4. CLI Skeleton & Error Handling (Minimal)
**Objective**: CLI entry point with minimal error handling

**File/Module Boundary**:
- `src/vestig/core/cli.py`

**Acceptance Checks**:
- `memory` command routes to subcommands correctly
- `memory add`, `memory search`, `memory recall`, `memory show` subcommands registered
- Invalid commands show argparse/click default errors
- Missing config.yaml raises clear error with helpful message
- Invalid/missing db_path in config raises clear error
- `--config <path>` flag accepted globally (optional)

**Non-goals**:
- No fancy CLI framework (argparse is fine, click if preferred)
- No progress bars or rich formatting
- No interactive prompts
- **Only handle common footguns** (missing config, bad db_path) — everything else crashes with traceback (fail hard)

---

### 5. `memory add` Command
**Objective**: Add a memory from CLI

**File/Module Boundary**:
- `src/vestig/core/cli.py` (add subcommand handler)
- `src/vestig/core/commitment.py` (simple commit logic)

**Acceptance Checks**:
- `memory add "text here"` stores memory and prints ID
- Embedding is generated automatically
- `created_at` timestamp is set to current time (ISO 8601)
- Metadata includes `source: "manual"`
- Success message shows memory ID
- Empty content raises error

**Non-goals**:
- No `--trigger` flag (M2)
- No substance filter (M2)
- No entity extraction (M2)
- No redundancy detection (M2)
- No graph edge creation

---

### 6. `memory search` Command
**Objective**: Retrieve top-K memories by semantic similarity (brute-force)

**File/Module Boundary**:
- `src/vestig/core/cli.py` (search subcommand handler)
- `src/vestig/core/retrieval.py` (brute-force retrieval logic)

**Acceptance Checks**:
- `memory search "query text" --limit 5` returns top-5 matches
- Default limit is 5 if not specified
- Output shows: memory ID, content (truncated to 100 chars), created timestamp, similarity score
- Query embedding is generated on-the-fly
- Brute-force algorithm: load all embeddings, compute cosine similarity in Python (numpy), return top-K sorted descending
- Results ranked by cosine similarity (descending)

**Non-goals**:
- No graph traversal
- No probabilistic recall
- No temporal decay weighting
- No MemRank
- No working set
- No lateral thinking
- No vector index optimization (brute-force is intentional for M1)

---

### 7. `memory recall` Command (Formatted Output)
**Objective**: Same as search but formatted for agent context

**File/Module Boundary**:
- `src/vestig/core/cli.py` (recall subcommand handler)
- `src/vestig/core/retrieval.py` (formatting function)

**Acceptance Checks**:
- `memory recall "query text"` returns formatted output suitable for LLM context
- Format: plain text, each memory separated by `---`
- Each memory shows: content (full), created timestamp (human-readable, e.g., "2 hours ago")
- Default limit is 5
- Output is clean (no table borders, just readable text blocks)

**Non-goals**:
- No user core summary injection (M2+)
- No "why retrieved" explanation
- No trigger display (doesn't exist in M1)

---

### 8. `memory show <id>` Command (Inspection)
**Objective**: Inspect a single memory by ID for observability

**File/Module Boundary**:
- `src/vestig/core/cli.py` (show subcommand handler)

**Acceptance Checks**:
- `memory show <id>` displays full memory details
- Output shows: ID, content (full), created_at, metadata (formatted as JSON or key-value)
- Output shows: embedding length (e.g., "Embedding: 1024 dimensions")
- If ID not found, show clear error message
- Optionally show first 5 values of embedding vector for debugging

**Non-goals**:
- No visualization of embedding vector
- No related memories (that's graph traversal, M4+)
- Just raw inspection of stored data

---

### 9. Demo Script & Acceptance Test
**Objective**: Prove M1 works end-to-end

**File/Module Boundary**:
- `demo_m1.sh`

**Acceptance Checks**:
- Script adds 10–12 varied memories (tech problems, solutions, preferences)
- Script runs 3–5 test queries using `memory search` and `memory recall`
- Script inspects at least one memory using `memory show <id>`
- For each query, top results are semantically relevant (manual verification)
- Script runs twice without errors (proves persistence)
- Output shows all commands and results clearly

**Non-goals**:
- No automated assertion-based tests (manual verification OK for M1)
- No unit tests (as per SPEC.md design decisions)
- Just a demo that shows it works

---

## Parking Lot (M2+)

**Deferred to M2 (Quality Firewall)**:
- Substance filter (LLM-based importance classifier)
- Trigger extraction and trigger embeddings
- Redundancy detection and pairing
- Boost mechanism
- prompts.yaml infrastructure

**Deferred to M3 (Temporal & Truth)**:
- Bi-temporal fields (t_valid, t_invalid, t_created, t_expired, t_ref)
- Temporal stability classification (static/dynamic)
- Learning lag calculation
- Temporal decay function
- Effective age calculations

**Deferred to M4 (Graph)**:
- Entity nodes and entity extraction
- Graph edges/relationships
- Graph operations and queries
- Edge invalidation and contradiction detection

**Deferred to M5 (Advanced Retrieval)**:
- Hypothetical query generation (HyDE)
- Hybrid start node selection
- Probabilistic graph traversal
- MemRank (PageRank-based scoring)
- Multi-factor recall scoring

**Deferred to M6 (Cognitive Features)**:
- Working set
- Lateral thinking
- Daydream mode
- User entity and core summary

**Other Deferred Items**:
- `memory list` command
- Batch import (`memory batch <file>`)
- Advanced error recovery and validation
- Vector index optimization (acceptable to defer until scaling issues arise)

---

## Demo Script (End-to-End Proof)

```bash
#!/bin/bash
# demo_m1.sh - Proves M1 works end-to-end

set -e  # Exit on error

echo "=== Vestig M1 Demo ==="
echo ""

echo "Step 1: Adding memories..."
echo ""

# Capture first ID for later inspection
FIRST_ID=$(memory add "Solved authentication bug by checking JWT token expiry in middleware" | grep -oE 'mem_[a-f0-9-]+' | head -1)

memory add "User prefers dark mode and minimal UI"
memory add "Fixed database migration error by running migrations in correct order"
memory add "Learned that Redis cache invalidation needs explicit TTL settings"
memory add "User gave positive feedback on fast API response times"
memory add "Debugging tip: always check logs before assuming code issue"
memory add "React useState hook caused infinite loop when missing dependency array"
memory add "User likes to work in short focused sprints, not long sessions"
memory add "PostgreSQL index on user_id column improved query speed 10x"
memory add "Error handling: always return user-friendly messages, not stack traces"

echo ""
echo "Step 2: Testing retrieval..."
echo ""

echo "Query 1: 'authentication problems'"
memory search "authentication problems" --limit 3
echo ""

echo "Query 2: 'database performance'"
memory search "database performance" --limit 3
echo ""

echo "Query 3: 'user preferences'"
memory search "user preferences" --limit 3
echo ""

echo "Query 4: 'React debugging' (formatted for agent)"
memory recall "React debugging"
echo ""

echo "Step 3: Inspecting a memory..."
echo ""
echo "Showing first memory added (ID: $FIRST_ID):"
memory show "$FIRST_ID"
echo ""

echo "=== Demo Complete ==="
echo "Run this script again to verify persistence across sessions."
```

**Expected Results**:
- Query 1 surfaces JWT/authentication memory
- Query 2 surfaces PostgreSQL index and Redis cache memories
- Query 3 surfaces dark mode and sprint preferences
- Query 4 surfaces React useState memory with formatted context output
- `memory show` displays full memory with metadata and embedding length
- Second run adds 10 more memories (20 total) and recall still works

---

## Sprint 1 Definition of Done

Sprint 1 is **complete** when all of the following are true:

1. ✅ **End-to-end proof**: `demo_m1.sh` runs twice without errors (proves persistence)
2. ✅ **Agent-ready output**: `memory recall` produces clean text blocks separated by `---`
3. ✅ **Observability**: `memory show <id>` lets you inspect stored memories (content, metadata, embedding length)
4. ✅ **Semantic retrieval works**: Manual verification shows top results are contextually relevant for test queries
5. ✅ **No scope creep**: Zero LLM calls, zero graph edges, zero temporal mechanics in M1 code

---

## Sprint 1 Summary

**What we're building**:
- 9 focused work items
- ~500-900 lines of Python (estimate)
- 4 core modules: embeddings, storage, retrieval, cli (in `src/vestig/core/`)
- 4 commands: `memory add`, `memory search`, `memory recall`, `memory show`
- 1 demo script proving it works
- `pyproject.toml` packaging (src/ layout)

**What we're earning**:
- Stable CLI interface (`memory add`, `memory search/recall`, `memory show`)
- Working persistence layer (sqlite-graph)
- Brute-force semantic retrieval that proves the concept
- Observability without logging infrastructure
- Foundation for all future milestones

**What we're NOT doing** (yet):
- Any LLM calls
- Any graph relationships
- Any temporal mechanics
- Any quality filtering
- Any fancy features
- Any vector index optimization (brute-force is intentional)

**Complexity earned**: The minimum viable loop that proves the core claim: "I can store and recall memories semantically."

---

## Implementation Notes

### Project Structure & Packaging
- **Layout**: `src/` layout (modern Python best practice)
- **Packaging**: `pyproject.toml` (not setup.py + requirements.txt)
- **Why**: Aligns with Ruff-first style guide, reduces churn in later sprints
- **Entry point**: `[project.scripts]` in pyproject.toml → `memory` command

### Embedding Backend (Locked for M1)
- **Choice**: `sentence-transformers` library (not `llm` CLI tool)
- **Model**: BAAI/bge-m3 (1024-dim)
- **Why**: Standard Python library, no CLI dependency, boring and reliable
- **Defer**: Alternative backends (OpenAI, Cohere, etc.) to M2+

### Brute-Force Search Rationale (M1)
- Load all memory embeddings into Python
- Compute cosine similarity using numpy
- Sort and return top-K
- **Why**: Keeps M1 shippable; don't fight sqlite-graph vector indexing yet
- **When to optimize**: M2+ when you have >1000 memories and notice slowness

### Error Handling Philosophy (M1) — FAIL HARD
- **Only handle**: Missing config.yaml, invalid db_path
- **Everything else**: Let it crash with traceback (fail hard)
- **Why**: Avoid polish rabbit holes; focus on proving the core loop
- **No drift**: Don't let agents add "helpful" error messages or validation beyond these two cases

### Dimension Validation
- Read `embedding.dimension` from config.yaml
- Validate actual embedding matches expected dimension
- Raise clear error on mismatch: `"Embedding dimension mismatch: expected 1024, got {actual}"`
- **Why**: Prevents silent corruption if you swap models later

### Demo Script ID Capture
- First `memory add` captures ID via grep: `FIRST_ID=$(memory add "..." | grep -oE 'mem_[a-f0-9-]+' | head -1)`
- Use captured ID for `memory show "$FIRST_ID"`
- **Why**: No hardcoded placeholder IDs that break on first run

---

Ready to start implementation?
