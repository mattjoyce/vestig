# Research Archive

This folder contains all exploratory work and prototypes for the Vestig Agent Memory System.

## Organization

```
research/
├── core/              # Core implementation files
│   ├── agent_memory.py        # SQLite storage + embeddings + RAG
│   ├── session_parser.py      # Naive extraction (regex-based)
│   └── discerned_extractor.py # Discerned extraction (LLM-based)
│
├── demos/             # Working examples and demos
│   ├── rag_demo.py            # RAG system demonstration
│   ├── test_extraction.py     # Naive vs discerned comparison
│   ├── query_combined_example.py  # SQL + semantic queries
│   ├── import_all_sessions.py # Batch import all sessions
│   └── show_discerned.py      # Display discerned facts
│
├── docs/              # Documentation and design notes
│   ├── CLAUDE.md                      # Main project documentation
│   ├── RAG_SYSTEM.md                  # RAG implementation guide
│   ├── DISCERNED_EXTRACTION.md        # Discerned extraction guide
│   ├── GRAPH_INTEGRATION_PLAN.md      # sqlite-graph integration plan
│   ├── naive_vs_sophisticated.md      # Extraction approach comparison
│   └── example_datalog_reasoning.md   # Datalog examples (future)
│
├── tests/             # Test data
│   └── test_session.jsonl     # Sample session for testing
│
└── databases/         # Research databases
    ├── vestig_memory.db       # Full import of all sessions
    ├── rag_demo.db           # RAG demo database
    ├── test_combined.db      # Combined query testing
    └── example_agent_memory.db  # Early prototype
```

## Key Findings

### What Works

1. **Dual Extraction** (Naive + Discerned)
   - Naive: Fast regex extraction (hosts, paths, commands, tools)
   - Discerned: LLM semantic analysis (intent, workflows, feedback)
   - Both stored together with `extraction_method` flag

2. **Session Deduplication**
   - SHA256 hash prevents re-processing
   - Automatic duplicate detection

3. **Semantic Search (RAG)**
   - Embeddings via `llm` Python API (ada-002)
   - Cosine similarity for relevance ranking
   - Works well for "fuzzy" concept matching

4. **Hybrid Querying**
   - SQL for exact relationship queries
   - Embeddings for semantic "what's relevant?"
   - Both can be combined

### What Didn't Work (Yet)

1. **Datalog Reasoning**
   - Prepared (export function exists)
   - Not implemented
   - RAG embeddings solved similar problem differently

2. **Graph Database**
   - Considered sqlite-graph integration
   - EAV pattern sufficient for current needs
   - Could add later if path-finding becomes important

## Architecture Evolution

### Phase 1: Basic Extraction
- Parse JSONL sessions
- Extract facts as triples (subject-predicate-object)
- Store in SQLite EAV pattern

### Phase 2: Dual Extraction
- Added LLM-based "discerned" extraction
- Captures semantic insights (workflow, intent, feedback)
- Flags facts with extraction_method

### Phase 3: RAG System
- Added embeddings for semantic search
- Deduplication via file hashing
- find_similar_facts() for retrieval

### Future Directions
- sqlite-graph for proper graph operations?
- Datalog for logical reasoning?
- Active learning (track useful retrievals)?

## Running the Code

All code requires the venv:
```bash
source ~/Environments/vestig/bin/activate
```

### Process Sessions
```bash
python demos/import_all_sessions.py
```

### Test RAG System
```bash
python demos/rag_demo.py
```

### Compare Extraction Methods
```bash
python demos/test_extraction.py
```

## Dependencies

```bash
pip install numpy llm
```

## Database Schema

See `docs/RAG_SYSTEM.md` for full schema documentation.

Key tables:
- `facts` - All extracted facts (naive + discerned)
- `sessions` - Session metadata + file hashes
- `embeddings` - Vector embeddings for semantic search

## Lessons Learned

1. **Simplicity wins** - SQLite + embeddings > complex graph DB for this use case
2. **Hybrid extraction** - Combining regex + LLM gives best results
3. **Deduplication is critical** - Don't re-process same sessions
4. **Python API > subprocess** - Use `llm` module directly, not CLI
5. **EAV pattern** - Flexible enough to handle diverse fact types

## Next Steps (If Continuing)

1. Benchmark performance at scale (10k+ sessions)
2. Evaluate sqlite-graph vs current EAV approach
3. Add temporal decay (older facts matter less)
4. Cross-session pattern mining
5. Multi-modal embeddings (code + text)
