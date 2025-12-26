# Vestig — M3 Plan: Time & Truth (Event Layer + TraceRank)
*Date: 2025-12-25*  
*Milestone: M3 (Progressive Maturation)*

## 0) The point of M3 (in one sentence)
M3 makes Vestig **time-aware and truth-aware** by separating **canonical memories (claims)** from **events (evidence)**, then using event history to rank and manage staleness without duplicating nodes.

---

## 1) How to organise planning docs (recommended)
Keep two layers:

1. **`PLAN.md` (Roadmap, M1–M6)**  
   Short, stable, motivational. Defines scope boundaries per milestone and the “earn complexity” mantra.

2. **`PLAN_M3.md` (This document: milestone execution plan)**  
   Concrete work items, schema changes, acceptance checks, and migration steps.

Optional: **`SPEC.md` stays the contract** (schemas, interfaces, CLI contract).  
Plans can change; SPECs should change only when you’re intentionally changing a contract.

---

## 2) M3 scope
### 2.1 In scope (M3 must deliver)
1. **Event layer**
   - Append-only `memory_events` table.
   - Every write decision produces an event (`ADD`, `REINFORCE_*`, `IMPORT`, `DEPRECATE`, `SUPERSEDE`).

2. **Bi-temporal truth basics (minimal, but real)**
   - Capture **transaction time** (when we learned/stored) and **event time** (when it was true / happened).
   - Support marking a memory as no longer valid (deprecated/superseded) without deleting history.

3. **TraceRank v1**
   - Use reinforcement events to influence ranking.
   - Must discourage burst noise and reward spaced reinforcement.

4. **Retrieval contract stays clean**
   - Retrieval returns **unique canonical memories**, not duplicates, not events.

### 2.2 Explicitly out of scope (defer)
- Full entity graph / edges / contradiction resolution (M4).
- PageRank/MemRank, probabilistic traversal, HyDE (M5).
- Working set, lateral thinking, daydream (M6).

---

## 3) Dependencies from M2 (late ask, but small)
M2 should expose a **dupe detect hook** (structured `CommitOutcome`) so M3 can convert duplicate attempts into reinforcement events.

Minimum fields to emit:
- outcome: `INSERTED_NEW | EXACT_DUPE | NEAR_DUPE | REJECTED_HYGIENE`
- memory_id (canonical)
- matched_memory_id (for dupes)
- score (for near-dupe)
- content_hash
- occurred_at (UTC)
- source / artifact_ref (optional)
- hygiene_reasons (if rejected)

M3 can proceed without this, but it makes M3 cleaner and removes “retrofit pain.”

---

## 4) Conceptual model
### 4.1 Memory Node (canonical “claim”)
A stable thing you recall.

### 4.2 Event Node (append-only “evidence”)
A record that the memory was created, reinforced, imported, deprecated, or superseded.

**Key invariant:** *Events never overwrite; they only append.*

---

## 5) Storage model (relational first, graph later)
In M3 these are “nodes” conceptually, but implementation is just **separate tables** with FK links.

### 5.1 `memories` table (canonical)
Recommended fields (M3 minimum):
- `id TEXT PRIMARY KEY` (`mem_<uuid>`)
- `content TEXT NOT NULL`
- `content_embedding TEXT NOT NULL` (JSON list)
- `content_hash TEXT NOT NULL UNIQUE` (sha256 of normalized content)
- **Bi-temporal (minimal)**
  - `t_valid TEXT NULL` (event time; when it became true; may be null/unknown)
  - `t_invalid TEXT NULL` (event time; when it stopped being true; null = still valid)
  - `t_ref TEXT NULL` (reference time used for relative parsing; optional)
  - `t_created TEXT NOT NULL` (transaction time; replaces/aliases `created_at`)
  - `t_expired TEXT NULL` (transaction time; when we deprecated/superseded it)
- `temporal_stability TEXT NOT NULL DEFAULT 'unknown'` (`static|dynamic|unknown`)
- `metadata TEXT NOT NULL` (JSON)

Notes:
- You can keep `created_at` for backwards compatibility, but **prefer `t_created`** going forward.

### 5.2 `memory_events` table (append-only)
- `event_id TEXT PRIMARY KEY` (`evt_<uuid>`)
- `memory_id TEXT NOT NULL` (FK → memories.id)
- `event_type TEXT NOT NULL`
- `occurred_at TEXT NOT NULL` (UTC timestamp)
- `source TEXT NOT NULL`
- `actor TEXT NULL`
- `artifact_ref TEXT NULL`
- `payload_json TEXT NOT NULL` (JSON)

Indexes:
- `INDEX(memory_events.memory_id, memory_events.occurred_at)`
- `INDEX(memory_events.event_type)`

---

## 6) Event types (start small)
- `ADD` — canonical memory created
- `REINFORCE_EXACT` — identical content seen again (hash match)
- `REINFORCE_NEAR` — near-dupe match (similarity > threshold)
- `IMPORT` — stored from batch/source artifact
- `DEPRECATE` — marked stale/incorrect
- `SUPERSEDE` — replaced by another memory (payload includes `new_memory_id`)

**Rule:** M3 should *not* invent many event types. Keep the set tight.

---

## 7) Commitment rules (M3 behavior)
### 7.1 New memory
- Insert into `memories`
- Write event: `ADD`

### 7.2 Exact duplicate attempt
- Do **not** create another memory row
- Write event: `REINFORCE_EXACT` on the canonical memory
- Optionally update convenience fields on memory (derived anyway):
  - `last_seen_at` (if you keep it)
  - cached `reinforce_count`

### 7.3 Near duplicate attempt
- Do **not** create another canonical memory row (default)
- Write event: `REINFORCE_NEAR` with payload including:
  - `matched_memory_id`
  - `similarity_score`
  - thresholds used

> Design note: In M4, you may decide to create edges between near-dup memories instead. In M3 we keep it simple.

### 7.4 Hygiene rejection
- No memory insert
- Optional event stream (only if you want auditability):
  - Either do nothing
  - Or write `REJECTED` events to a separate table (recommended to defer)

---

## 8) TraceRank v1 (temporal reinforcement)
### 8.1 Goal
Use event history to boost durable, repeatedly reinforced memories, without letting burst noise dominate.

### 8.2 Components (M3 v1)
For each reinforcement event (exact/near) of a memory:
1. **Recency decay**
   - `w_recency = exp(-Δt / τ)` (τ in days/weeks)

2. **Anti-burst / cooldown**
   - events within a short window of previous events contribute less

Then:
- `trace = Σ (w_recency * w_burst)`
- `trace_mult = 1 + k * log1p(trace)`
- `final_score = semantic_similarity * trace_mult`

### 8.3 Config (suggested keys)
```yaml
retrieval:
  tracerank:
    enabled: true
    tau_days: 21          # recency time constant
    cooldown_hours: 24    # anti-burst window
    burst_discount: 0.2   # multiplier inside cooldown
    k: 0.35               # strength of TraceRank boost
```

**Implementation note:** keep TraceRank simple and interpretable. You will tune later.

---

## 9) Truth mechanics (minimal M3)
### 9.1 Deprecation
A memory can be marked stale/incorrect without deletion:
- `memories.t_invalid` set to the time the fact stopped being true (event time)
- `memories.t_expired` set to “now” (transaction time)
- Event: `DEPRECATE` (payload: reason, maybe replaced_by)

### 9.2 Supersession
When one memory replaces another:
- create the new memory (if needed)
- event `SUPERSEDE` on the old memory with payload: `new_memory_id`
- optional: mark old memory deprecated/expired

**Retrieval default:** prefer the newest non-expired memory, but allow showing superseded items on request.

---

## 10) Retrieval behavior changes in M3
### 10.1 Default ranking
- Base: cosine similarity(query, memory.content_embedding)
- Multiply by TraceRank multiplier (if enabled)
- Optionally apply freshness/stability weighting (small effect in M3; bigger later)

### 10.2 Filtering
- Default: exclude `t_expired != null` (deprecated/superseded) from recall,
  unless `--include-expired` flag is set.

### 10.3 Output formatting (agent context)
- Still “one block per memory”
- Add compact provenance/time hints (optional):
  - last reinforced date
  - reinforcement count
  - validity status (current vs expired)

---

## 11) CLI changes (minimal, but useful)
Recommended additions:
- `memory events <id> [--limit N]` → show recent events for a memory
- `memory deprecate <id> --reason "..." [--t-invalid <iso>]`
- `memory supersede <old_id> "<new content>" --reason "..."`

If you want to keep CLI surface area tiny, implement only:
- `memory show <id>` now includes a short “event summary” section.

---

## 12) Migration plan (M2 → M3)
1. Add `content_hash` to existing memories (backfill).
2. Add bi-temporal columns (can be nullable initially).
3. Create `memory_events` table.
4. Backfill synthetic events (choose one):
   - Option A (simplest): for each memory, create one `ADD` event at `t_created`.
   - Option B: no backfill; start event logging from M3 onwards (acceptable).
5. Start writing new events for every commit decision.

---

## 13) Work items (suggested backlog)
1. **Schema migration**
   - Add `content_hash`, bi-temporal columns to `memories`
   - Create `memory_events`
   - Acceptance: migration runs on existing DB; old data readable.

2. **Event storage module**
   - `MemoryEventStorage.add_event(...)`
   - Acceptance: can insert and query events by memory_id.

3. **Commit pipeline emits events**
   - Insert memory + `ADD` event
   - Duplicate → reinforcement event
   - Acceptance: manual test with 3 adds + 2 dupes produces expected events.

4. **TraceRank scoring**
   - Implement event aggregation and multiplier
   - Acceptance: a memory reinforced over weeks outranks an unrepeated one at equal similarity; burst duplicates don’t spike as much.

5. **Retrieval filtering for expired**
   - Default hide expired; add flag to include
   - Acceptance: deprecate a memory → it disappears unless flag used.

6. **CLI observability**
   - `memory events <id>` or enhanced `memory show`
   - Acceptance: you can explain why a memory is “strong” (event counts and recency).

---

## 14) Definition of Done (M3)
M3 is complete when:
1. The system records **event history** (ADD + reinforcement) in an append-only table.
2. Duplicate attempts **reinforce** without producing duplicate memory rows.
3. Recall ranking uses TraceRank and behaves sensibly (anti-burst works).
4. You can **deprecate/supersede** memories and retrieval respects it.
5. You can inspect a memory’s event history from the CLI.

---

## 15) Mentor’s warning (worth remembering)
Graph is **M4**.  
M3 is the foundation that prevents M4 from becoming a beautiful but wrong web of stale beliefs.

