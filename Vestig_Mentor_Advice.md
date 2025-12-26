# Vestig — Progressive Maturation Plan (M1–M6)

## Anchoring preamble: “Progressive maturation”
This project is ambitious *by design*. The only way it stays coherent (and enjoyable) is if we **treat capability as something we earn**: each stage must produce a *stable, usable* slice that becomes the foundation for the next stage.

A simple rule keeps you in charge while coding agents do the heavy lifting:

> **Build the smallest end-to-end loop that proves the next claim, then lock the interfaces, then deepen.**

The maturation path below is structured to:
- Keep **entropy** low (the real enemy).
- Make each stage **observable** (you can explain why the system did what it did).
- Preserve **sovereignty**: your CLI + schema become the rails that agents build within.

---

## Guiding principles (the mantra)
### The mantra
**Earn complexity.**

### What it means in practice
1. **Quality at the boundary beats cleverness in the middle.**  
   If junk gets in, no ranking algorithm rescues it.

2. **Stable interfaces = sovereignty.**  
   Guard the CLI contract and the schema shape. Everything else can evolve.

3. **One thin vertical slice, then deepen.**  
   Add → store → recall is always the backbone; everything else is an enhancement.

4. **Observability is a feature.**  
   Prefer “I can inspect why” over “it seems smart”.

5. **Explicit “done-ness” per milestone.**  
   Each milestone ends with acceptance checks you can run in minutes.

6. **Agents ship tasks, you ship direction.**  
   One agent task = one capability + one acceptance test + one boundary.

---

# Progressive Maturation (M1–M6)

## M1 — MVP Core Loop (“Store and recall without surprises”)
### Goal
A usable baseline: **store memories** and **recall relevant ones** from the CLI.

### In scope
- Minimal DB + schema
- `memory add` → persist memory
- `memory recall/search` → top‑K semantic retrieval
- Basic metadata capture (created time, source string)

### Explicitly out of scope
- Graph traversal, MemRank
- Contradiction/invalidation logic
- Working set, daydream
- Complex temporal truth mechanics

### Deliverables
- CLI works end-to-end
- A tiny “demo dataset” script (10–20 memories) for repeatable testing

### Acceptance checks
- Add 10 memories; recall returns the right 3–5 most of the time.
- No crashes; errors are readable.
- Data persists across runs.

---

## M2 — Quality Firewall (“Protect the corpus from entropy”)
### Goal
Prevent the system from becoming a junk drawer.

### In scope (recommended order)
1. **Substance filter**: store vs discard
2. **Trigger extraction**: “why it mattered” (plus optional trigger embedding)
3. **Redundancy detection**: link duplicates and treat as reinforcement

### Explicitly out of scope
- Sophisticated ranking (beyond simple multipliers)
- Full entity graphs

### Deliverables
- Ingestion pipeline that can refuse storage with a reason
- Redundancy pairing records (even if not used heavily yet)

### Acceptance checks
- Low-value notes are consistently rejected (with an explanation).
- Re-adding the same memory reinforces rather than duplicates.
- You can inspect “why stored” / “why rejected”.

---

## M3 — Temporal & Truth Mechanics (“Time-aware memory”)
### Goal
Make memory **age intelligently** and support “what was true then vs now”.

### In scope
- Bi-temporal fields (created vs valid vs learned, etc.)
- Stability classification (static/dynamic/unknown)
- Decay mechanics (effective age, freshness weighting)
- Learning-lag confidence (optional, coarse)

### Explicitly out of scope
- Graph-based contradiction resolution (save for M4)
- Fancy traversal

### Deliverables
- Stored records contain temporal fields (even if partially populated)
- Retrieval can optionally weight by recency/freshness/stability

### Acceptance checks
- Old-but-important items still surface when relevant.
- Dynamic facts feel “less certain” over time unless reinforced.
- You can inspect temporal fields for a memory.

---

## M4 — Graph Starts to Matter (“Entities + edges + invalidation”)
### Goal
Introduce structured relationships and handle contradictions without losing history.

### In scope
- Entity extraction (typed entities)
- Edge creation between memories/entities
- Contradiction detection as **edge invalidation** (not deletion)
- Minimal graph inspection commands (e.g., show edges for a node)

### Explicitly out of scope
- Full MemRank / probabilistic traversal (save for M5)
- Cognitive flourishes (save for M6)

### Deliverables
- Graph representation is real and queryable
- Invalidation records exist and are explainable

### Acceptance checks
- You can see entities extracted from a memory.
- You can see edges created (and why).
- Contradictory updates invalidate prior edges cleanly (history preserved).

---

## M5 — Agent‑Grade Retrieval (“Hybrid start nodes + traversal + MemRank”)
### Goal
Make recall feel **intentionally smart**, not accidentally lucky.

### In scope
- Hypothetical query generation (selective, gated)
- Hybrid start node selection (semantic + recency + reinforcement)
- Probabilistic traversal
- MemRank (or equivalent scoring model)

### Explicitly out of scope
- Daydream generation and speculative injection (save for M6)

### Deliverables
- Retrieval produces ranked results with a *reason trace* (even basic)
- Config flags to toggle advanced retrieval components

### Acceptance checks
- For a set of test queries, advanced retrieval beats simple top‑K.
- You can explain “why this memory surfaced” in one paragraph.
- No silent, uninspectable magic.

---

## M6 — Cognitive Flourishes (“Working set, lateral thinking, daydream — behind flags”)
### Goal
Delight and creativity **without polluting** the core memory store.

### In scope (recommended order)
1. **Working set**: short-term memory / recency bias for active contexts
2. **Lateral thinking hooks**: controlled “nearby” associations
3. **Daydream mode**: explicit, user‑approved synthesis only

### Safety constraints
- Default OFF for speculative generation features.
- Daydream output should be clearly labeled and never auto-committed.

### Deliverables
- Feature flags + modes
- “Session context” workflow (working set) that helps real tasks

### Acceptance checks
- Working set materially improves usefulness in an active project week.
- Lateral thinking produces occasionally novel but relevant connections.
- Daydream never contaminates the store without explicit approval.

---

# Agent tasking template (recommended)
Use this format when assigning work to coding agents:

- **Objective:** one capability only  
- **Boundary:** which files/modules may change  
- **Acceptance checks:** 3–5 bullet checks  
- **Non-goals:** explicit out-of-scope items  
- **Demo:** a minimal script or command sequence that proves it works

---

## Closing reminder
If you feel the temptation to “just add one more feature” mid-stage, pause and ask:

> **Have we earned this complexity yet?**
