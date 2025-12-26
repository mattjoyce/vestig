# M3 Supplemental — Event Nodes, Reinforcement, and TraceRank
*Date: 2025-12-25*  
*Applies to: Vestig Progressive Maturation (M3: Time & Truth)*

## Why this document exists
M2 introduces **duplicate detection** and basic hygiene.  
M3 will turn “duplication” into **signal** via an **event layer** (reinforcement history), enabling temporal ranking (TraceRank), provenance, and “is this still true?” mechanics.

This doc is **supplemental** to `PLAN.md` / `SPEC.md` and is intended to:
- define the **Event Node** concept
- describe how **reinforcement** works without polluting retrieval
- specify an **M2 dupe-detect hook** that captures data we can leverage in M3

---

## Core idea
Separate the *claim* from the *evidence*:

- **Memory Node** = the canonical thing we want to recall (stable ID, content, embedding).
- **Event Node** = an append-only record that the memory mattered again (or was added, imported, edited, deprecated, etc).

This prevents retrieval spam (top‑K full of duplicates) while preserving the “repetition is signal” property.

---

## M3 Goals (Time & Truth)
M3 adds:
1. **Reinforcement events** (the same memory recurring matters)
2. **Temporal semantics**: recency, spacing, and decay
3. **Provenance**: where did this come from? what session/artifact?
4. **Truth mechanics**: detect staleness, allow deprecations/supersession
5. **Ranking** that uses both:
   - semantic similarity
   - event history (TraceRank)

---

## Data model (M3 shape)

### Memory Node (canonical)
Fields (conceptual; SQLite schema may differ):
- `id` (e.g., `mem_<uuid>`)
- `content`
- `content_embedding`
- `created_at`
- `content_hash` (sha256 of normalized content)
- `metadata` (source, tags, etc.)
- optional convenience fields (can be derived from events later):
  - `last_seen_at`
  - `reinforce_count`

### Event Node (append-only)
Fields (minimum useful set):
- `event_id` (e.g., `evt_<uuid>`)
- `memory_id` (FK → memories.id)
- `event_type` (enum-ish string)
- `occurred_at` (UTC timestamp)
- `source` (manual/hook/import, etc.)
- `actor` (optional: user/agent name)
- `artifact_ref` (optional: session_id, filename, URL)
- `payload_json` (optional, structured details)

Recommended event types (start small):
- `ADD` — canonical memory created
- `REINFORCE_EXACT` — identical content re-seen (hash match)
- `REINFORCE_NEAR` — near-duplicate match (similarity above threshold)
- `IMPORT` — memory imported from an artifact
- `DEPRECATE` — memory marked stale/incorrect
- `SUPERSEDE` — memory replaced by another memory_id (payload holds new id)

---

## Retrieval contract (stays clean)
Retrieval returns **unique Memory Nodes**, not Event Nodes.

Event history influences ranking, but recall output stays “one block per memory.”

---

## TraceRank (temporal reinforcement ranking)
TraceRank is a multiplier or tie-breaker that adjusts similarity scores based on event history.

### Intuition
- “I added the same thing 5 times in 2 minutes” is **not** strong signal (burst noise).
- “I re-used the same idea every few weeks” **is** strong signal (durable relevance).

### Components (suggested)
1. **Recency decay**
   - weight recent events more than old events  
   - e.g. `w_recency = exp(-Δt / τ)` with a time constant τ (days/weeks)

2. **Diminishing returns / anti-burst**
   - apply discount to events that occur within a short window of previous events
   - e.g. a “cooldown” window: events within 24h contribute less

3. **Spacing bonus (optional)**
   - spaced reinforcement (days/weeks apart) can be valued more than bursts

### Example (conceptual)
Let events be ordered newest→oldest.
For each reinforcement event i:
- compute `w_recency(i)`
- compute `w_burst(i)` based on time since prior event
- contribution = `w_recency(i) * w_burst(i)`

Then:
- `trace = sum(contributions)`
- `final_score = similarity * (1 + k * log1p(trace))`

This keeps TraceRank bounded, interpretable, and hard to game.

> Note: M3 does not need a perfect formula. We need a **stable first implementation** that can be tuned in M4/M5.

---

## M2 requirement: Dupe Detect Hook (capture signal now)
In M2 we can keep storage simple, but we should expose a **hook** that reports dupe decisions in a structured way.

### Why the hook exists
- M2: prevent duplicate node pollution
- M3: treat duplicate attempts as reinforcement events with timestamps + distances

The hook is a *thin affordance*: it emits structured data, and does not decide what to do with it.

### Hook interface (recommended)
Create a small dataclass (or TypedDict) that represents the outcome of a commit attempt.

**Commit outcome states**
- `INSERTED_NEW` — a new Memory Node created
- `EXACT_DUPE` — hash match to existing memory
- `NEAR_DUPE` — semantic match above threshold to an existing memory
- `REJECTED_HYGIENE` — blocked by hygiene rule(s)

**Suggested payload fields**
- `outcome` (string enum)
- `memory_id` (the canonical ID that should be used downstream)
- `matched_memory_id` (if dupe/near-dupe)
- `query_score` (float; for near-dupe)
- `content_hash` (sha256 of normalized content)
- `hygiene_reasons` (list[str], if rejected)
- `thresholds` (near-dupe threshold used, etc.)
- `occurred_at` (UTC timestamp)
- `source` (manual/hook/import)
- `tags` (optional)
- `artifact_ref` (optional)

### Where to invoke the hook
In `commit_memory()` (or the commit pipeline), after deciding outcome, call:

- `on_commit(outcome: CommitOutcome) -> None`

If no hook is provided, do nothing.

### Persistence
M2 does **not** need to persist events yet.  
However, it should be easy in M3 to wire `on_commit` to a `MemoryEventStorage` implementation that writes `memory_events`.

---

## Migration notes (M2 → M3)
If M2 implements only:
- exact/near dupe detection
- optional `reinforce_count` / `last_seen_at` updates

Then M3 can:
- introduce `memory_events` table
- start writing new events immediately
- optionally backfill:
  - either **no backfill** (acceptable)
  - or create one synthetic event per existing memory (`ADD` at created_at)
  - and treat `reinforce_count` as a coarse prior (optional)

---

## Open questions (defer until M3 planning)
- Do we allow **multiple canonical memories** with same hash but different metadata? (probably no)
- When near-dupe matches, do we:
  - block insert?
  - insert and mark relation?
  - or “merge/suggest merge”?
- Do we treat “manual add” differently from “import” for reinforcement strength?
- How do we represent deprecation/supersession in recall output?

---

## What to tell the M2 team (one-liner)
**“Implement dupe detection now, and emit a structured CommitOutcome hook so M3 can convert duplicate attempts into reinforcement events and TraceRank.”**
