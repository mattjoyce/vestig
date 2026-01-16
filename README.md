# Vestig — Agent Memory System (Local‑First)

Vestig is a **local memory store + recall CLI** for coding agents.
It turns session artifacts (notes, transcripts, JSONL exports) into **searchable memories** so an agent can reuse what it learned last week instead of re-deriving it.

This repository is the **front door** for new contributors and new agents joining the project.

---

## Status: Progressive Maturation (M1 → M6)

We build Vestig in **progressive maturation** steps. Each maturity level is a *complete, usable slice* with a clear scope boundary.

- **M1 — Core Loop (Complete):** add → embed → persist → recall (top‑K). Fail fast. Minimal scaffolding. ✓
- **M2 — Quality Firewall (Complete):** de-duplication, content hygiene, basic ranking improvements, and controlled recall formatting. ✓
- **M3 — Time & Truth (Complete):** bi-temporal tracking (t_valid, t_created), temporal stability classification, event storage, and decay mechanics. ✓
- **M4 — Graph Layer (Complete):** entity extraction, entity/edge nodes, MENTIONS relationships, SUMMARY nodes with SUMMARIZES edges, graph operations, and knowledge graph foundation. ✓
- **M5 — Advanced Retrieval (In Progress):** TraceRank, graph-based retrieval, hybrid scoring, and sophisticated recall strategies.
- **M6 — Productisation:** stable interfaces, hardening, packaging, and clean integration patterns for agent ecosystems.

See:
- `ROADMAP.md` — the maturation roadmap (M1→M6) + guiding preamble and principles
- `ARCHITECTURE.md` — current implementation documentation and technical contract
- `M4_Completion_Report.md` — detailed M4 milestone report
- `docs/archive/SPEC_RESEARCH.md` — archived research vision (aspirational features)

**Current Status:** M1–M4 are complete and operational. **M5 is in progress** (TraceRank implemented, hybrid retrieval underway). Anything not required for the current milestone is intentionally deferred.

---

## What Vestig Does (In One Sentence)

**Given a query, return the most relevant stored memories (with metadata) suitable for direct insertion into an LLM context.**

---

## Non‑Goals (Current Phase)

To prevent scope creep, we maintain clear boundaries for each maturity level.

**Completed in M1–M2:**
- ✓ Basic store and recall (M1)
- ✓ Content hygiene and deduplication (M2)

**Not Yet Implemented (M3+):**
- Temporal awareness and decay mechanics (planned for M3)
- Graph relationships / entity resolution (M4)
- Complex ranking, reranking, or hybrid retrieval (M4–M5)
- Long-running background jobs (M5)

We earn complexity progressively. Each feature waits for its maturity level.

---

## Quick Start (Dev)

### 1) Setup environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Run the CLI
```bash
vestig --help
vestig memory add "Solved auth bug by checking JWT expiry handling"
vestig memory recall "jwt expiry" --limit 5
```

### 3) Validate before PR
```bash
ruff format .
ruff check .
bash demos/demo_m1.sh
```

> Note: we bias toward **rapid iteration**. It is OK to fail hard during development,
> but code should be readable, consistent, and lint-clean.

---

## CLI Reference

Top-level:
- `vestig --config <path>` (default: `config.yaml`)

Ingest:
- `vestig ingest <document>`
- Options: `--format auto|plain|claude-session`, `--force-entity TYPE:Name` (repeatable),
  `--chunk-size N`, `--chunk-overlap N`, `--model <name>`, `--min-confidence F`,
  `--verbose`, `--timing`, `--recurse` (or `-r`) for recursive globbing (enables `**` patterns)

Memory:
- `vestig memory add "<text>"` with `--agent <name>`, `--session-id <id>`, `--tags a,b,c`
- `vestig memory recall "<query>"` with `--limit N`, `--explain`
- `vestig memory show <id>` with `--expand 0|1`, `--include-expired`
- `vestig memory list` with `--limit N`, `--snippet-len N`, `--include-expired`
- `vestig memory deprecate <id>` with `--reason "<text>"`, `--t-invalid <ISO8601>`
  - Tip: add `--explain` to include per-result scoring and TraceRank details.

Entity:
- `vestig entity list` with `--limit N`, `--include-expired`
- `vestig entity show <id>` with `--expand 0|1`, `--include-expired`

Edge:
- `vestig edge list` with `--limit N`, `--type ALL|MENTIONS|RELATED`, `--snippet-len N`,
  `--include-expired`
- `vestig edge show <id>`

---

## Architecture (Current)

```
Input text (manual for now)
        ↓
Embedding (single vector per memory)
        ↓
FalkorDB graph storage
        ↓
Semantic similarity retrieval
        ↓
Recall formatting for agent context
```

Storage is handled by FalkorDB, a graph database optimized for relationships and vector similarity.

---

## Repository Layout

Typical structure (may evolve slightly as we implement):

- `src/vestig/` — implementation package (includes core/prompts.yaml)
- `test/` — test configuration files (config.yaml, test-specific prompts.yaml)
- `tests/` — test scripts and smoke tests
- `demos/` — demo scripts (demo_m1.sh, demo_m4.sh, etc.)
- `benchmarks/` — performance benchmarking scripts
- `ARCHITECTURE.md` — technical architecture and implementation documentation
- `ROADMAP.md` — maturation roadmap and milestone planning
- `docs/archive/` — archived research and design documents

---

## Configuration

Vestig is designed to be **local-first** and configurable. Typical config includes:

- FalkorDB connection details (host, port, graph name)
- embedding model selection
- optional import paths for session artifacts (future)

If configuration is present in the repo, prefer:
- `config.yaml` checked in as an **example**
- `config.local.yaml` for developer overrides (gitignored)

---

## Storage Backend

Vestig uses **FalkorDB**, a graph database, as its storage backend.

**Graph Structure:**
- Memory nodes with vector embeddings
- Entity nodes (PERSON, ORG, SYSTEM, etc.)
- Source nodes for provenance tracking
- Edges for relationships (MENTIONS, RELATED, SUMMARIZES)

**Configuration:**
- FalkorDB connection is configured in `config.yaml`
- Default: `localhost:6379` with graph name `vestig`
- Test graph names are auto-generated to avoid conflicts

**Implementation:**
- `src/vestig/core/db_falkordb.py` — FalkorDB backend implementation
- Graph schema is defined implicitly through node and edge creation

---

## Guiding Principles (Rapid Mode)

These are the rules we use to move quickly without creating chaos:

1. **Earn complexity.** Start simple; add structure only when forced by reality.
2. **Fail fast, fail loudly.** Prefer obvious crashes over silent corruption.
3. **Test the slice, not the atom.** Smoke/integration tests over exhaustive unit tests.
4. **Readable > clever.** This is a long game; future-us must understand it.
5. **One thing per commit.** Small, reviewable PRs are how we keep velocity.

---

## Contributing

### Branching & PR hygiene
- Keep PRs small and vertical (a usable slice)
- Include a short “demo” section in the PR description:
  - what you ran (`demo_m1.sh`, CLI commands)
  - what output changed
- Update docs when you change interfaces

### Style
- Python formatting and linting via **ruff**
- Docstrings where it helps the next reader
- Prefer type hints where they clarify intent

---

## Project Vocabulary

- **Memory**: a stored item (text + metadata + embedding)
- **Summary**: a derived memory node (kind=SUMMARY) created after an ingest run (currently when >=5 memories are committed) and linked to source memories via SUMMARIZES edges
- **Recall**: retrieval + formatting suitable for agent context
- **Artifact**: a source input (session transcript, JSONL, note file, etc.)
- **Maturity level (M1–M6)**: a bounded capability slice in the roadmap

---

## Roadmap Snapshot

- **Sprint 1 (Complete):** ✓ M1 real, boring, and reliable (CLI + graph storage + embeddings + recall)
- **Sprint 2 (Complete):** ✓ M2 "Quality Firewall" (dedupe + hygiene + better recall formatting)
- **Sprint 3 (Next):** M3 "Time & Truth" (temporal awareness, decay/refresh, provenance) — awaiting green light
- **Sprint 4+:** M4–M6 as per `PLAN.md`

If you're new: start by running the demo and exploring the M2 quality firewall features.

---

## License

TBD (add once decided).
