# Vestig — Agent Memory System (Local‑First)

Vestig is a **local memory store + recall CLI** for coding agents.
It turns session artifacts (notes, transcripts, JSONL exports) into **searchable memories** so an agent can reuse what it learned last week instead of re-deriving it.

This repository is the **front door** for new contributors and new agents joining the project.

---

## Status: Progressive Maturation (M1 → M6)

We build Vestig in **progressive maturation** steps. Each maturity level is a *complete, usable slice* with a clear scope boundary.

- **M1 — Core Loop (Now):** add → embed → persist → recall (top‑K). Fail fast. Minimal scaffolding.
- **M2 — Quality Firewall:** de-duplication, content hygiene, basic ranking improvements, and controlled recall formatting.
- **M3 — Time & Truth:** temporal awareness, decay/refresh, provenance, and “is this still true?” mechanics.
- **M4 — Structure:** entities/edges (light graph), relationships, and richer retrieval.
- **M5 — Operations:** import pipelines, observability, performance, backup/restore, and automation hooks.
- **M6 — Productisation:** stable interfaces, hardening, packaging, and clean integration patterns for agent ecosystems.

See:
- `PLAN.md` — the maturation roadmap (M1→M6) + guiding preamble and principles
- `SPEC.md` — target design and technical contract (what we are building toward)

**Important:** We are implementing **M1 first**. Anything not required for M1 is intentionally deferred.

---

## What Vestig Does (In One Sentence)

**Given a query, return the most relevant stored memories (with metadata) suitable for direct insertion into an LLM context.**

---

## Non‑Goals (for M1)

To prevent scope creep, M1 explicitly does **not** include:

- LLM-based extraction/summarisation pipelines
- Graph relationships / entity resolution
- Complex ranking, reranking, or hybrid retrieval
- Long-running background jobs
- “Perfect” unit test suites (we prefer smoke/integration tests in this phase)

We will earn these later in M2–M6.

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
vestig memory search "jwt expiry" --limit 5
vestig memory recall "jwt expiry" --limit 5
```

### 3) Validate before PR
```bash
ruff format .
ruff check .
./demo_m1.sh
```

> Note: we bias toward **rapid iteration**. It is OK to fail hard during development,
> but code should be readable, consistent, and lint-clean.

---

## Core Commands (M1)

Expected CLI surface (subject to small refinements as M1 lands):

- `vestig memory add "<text>" [--tags ...] [--source ...]`
- `vestig memory search "<query>" --limit N`
- `vestig memory recall "<query>" --limit N` (LLM-ready formatting)
- `vestig memory show <id>`
- `vestig memory list [--recent N]`

---

## Architecture (Current / M1)

```
Input text (manual for now)
        ↓
Embedding (single vector per memory)
        ↓
SQLite persistence
        ↓
Brute-force similarity retrieval (M1)
        ↓
Recall formatting for agent context
```

M1 can use brute-force retrieval over a modest corpus. We optimise later.

---

## Repository Layout

Typical structure (may evolve slightly as we implement):

- `src/vestig/` — implementation package
- `tests/` — smoke / integration tests (keep them pragmatic)
- `data/` — local sqlite DB (gitignored)
- `config/` — config defaults / examples (if used)
- `demo_m1.sh` — end-to-end demo for M1
- `PLAN.md` — progressive maturation roadmap
- `SPEC.md` — technical spec / contract
- `research/` — preserved research artifacts (historical)

---

## Configuration

Vestig is designed to be **local-first** and configurable. Typical config includes:

- database path (SQLite)
- embedding model selection
- optional import paths for session artifacts (future)

If configuration is present in the repo, prefer:
- `config.yaml` checked in as an **example**
- `config.local.yaml` for developer overrides (gitignored)

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
- **Recall**: retrieval + formatting suitable for agent context
- **Artifact**: a source input (session transcript, JSONL, note file, etc.)
- **Maturity level (M1–M6)**: a bounded capability slice in the roadmap

---

## Roadmap Snapshot

- **Sprint 1:** make M1 real, boring, and reliable (CLI + SQLite + embeddings + recall)
- **Sprint 2:** M2 “Quality Firewall” (dedupe + hygiene + better recall formatting)
- **Sprint 3+:** M3+ as per `PLAN.md`

If you’re new: start by running the demo and adding one small improvement that strengthens M1.

---

## License

TBD (add once decided).
