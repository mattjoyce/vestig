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
- `archive/SPEC_RESEARCH.md` — archived research vision (aspirational features)

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

**Not Yet Implemented (M5+):**
- Service/daemon mode and external integrations (Discord, webhooks, API)
- Background jobs for continuous maintenance (scheduled housekeeping)
- Advanced traversal and multi-hop retrieval beyond chunk expansion

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
bash tests/test_m2_smoke.sh
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
  `--verbose`, `--timing`, `--no-entities`,
  `--recurse` (or `-r`) for recursive globbing (enables `**` patterns)

Memory:
- `vestig memory add "<text>"` with `--agent <name>`, `--session-id <id>`, `--tags a,b,c`, `--source <name>`
- `vestig memory recall "<query>"` with `--limit N`, `--explain`, `--timing`
- `vestig memory show <id>` with `--expand 0|1`, `--include-expired`
- `vestig memory list` with `--limit N`, `--snippet-len N`, `--include-expired`
- `vestig memory deprecate <id>` with `--reason "<text>"`, `--t-invalid <ISO8601>`
- `vestig memory regen-embeddings` with `--model <name>`, `--batch-size N`, `--limit N`

Entity:
- `vestig entity list` with `--limit N`, `--include-expired`
- `vestig entity show <id>` with `--expand 0|1`, `--include-expired`
- `vestig entity extract` with `--reprocess`, `--batch-size N`, `--verbose`
- `vestig entity purge --force`
- `vestig entity regen-embeddings` with `--model <name>`, `--batch-size N`, `--limit N`

Edge:
- `vestig edge list` with `--limit N`, `--type ALL|MENTIONS|RELATED`, `--snippet-len N`,
  `--include-expired`
- `vestig edge show <id>`

Config:
- `vestig config show`

Housekeeping:
- `vestig housekeeping report`
- `vestig housekeeping entity-backfill` with `--batch-size N`, `--verbose`
- `vestig housekeeping orphans` with `--fix`, `--verbose`

---

## Architecture (Current)

```
Input text (manual or ingested artifacts)
        ↓
Embedding (single vector per memory)
        ↓
FalkorDB graph storage
        ↓
Semantic similarity retrieval + TraceRank
        ↓
Recall formatting with chunk expansion
```

Storage is handled by FalkorDB, a graph database optimized for relationships and vector similarity.

---

## Repository Layout

Typical structure (may evolve slightly as we implement):

- `src/vestig/` — implementation package (includes core/prompts.yaml)
- `tests/` — test scripts and smoke tests
- `tests/` — smoke tests (`tests/test_m2_smoke.sh`, `tests/test_m4_smoke.sh`, etc.)
- `benchmarks/` — performance benchmarking scripts
- `ARCHITECTURE.md` — technical architecture and implementation documentation
- `ROADMAP.md` — maturation roadmap and milestone planning
- `archive/` — archived research and design documents

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
- Memory nodes with vector embeddings (including summaries via `kind=SUMMARY`)
- Entity nodes (PERSON, ORG, SYSTEM, etc.)
- Source nodes for provenance tracking
- Chunk nodes for positional provenance
- Edges for relationships (PRODUCED, HAS_CHUNK, CONTAINS, LINKED, MENTIONS, RELATED, SUMMARIZES, SUMMARIZED_BY, AFFECTS)

**Configuration:**
- FalkorDB connection is configured in `config.yaml`
- Default: `localhost:6379` with graph name `vestig`
- Tests and fixtures create unique graph names to avoid conflicts

**Implementation:**
- `src/vestig/core/db_falkordb.py` — FalkorDB backend implementation
- Constraints and indexes are initialized on startup; reference schema lives in `src/vestig/core/schema_falkor.cypher`

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
- **Summary**: a derived memory node (kind=SUMMARY) created per chunk (when >=2 memories are committed) and linked to source memories via SUMMARIZES edges
- **Recall**: retrieval + formatting suitable for agent context
- **Artifact**: a source input (session transcript, JSONL, note file, etc.)
- **Maturity level (M1–M6)**: a bounded capability slice in the roadmap

---

## Roadmap Snapshot

- **Sprint 1 (Complete):** ✓ M1 real, boring, and reliable (CLI + graph storage + embeddings + recall)
- **Sprint 2 (Complete):** ✓ M2 "Quality Firewall" (dedupe + hygiene + better recall formatting)
- **Sprint 3 (Complete):** ✓ M3 "Time & Truth" (temporal awareness, decay/refresh, provenance)
- **Sprint 4 (Complete):** ✓ M4 "Graph Layer" (entities, edges, summaries)
- **Sprint 5 (In Progress):** M5 advanced retrieval (hybrid scoring, chunk expansion, tracing)

If you're new: start by running the demo and exploring the M2 quality firewall features.

---

## License

TBD (add once decided).
