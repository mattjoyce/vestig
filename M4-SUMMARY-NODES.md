A) Current understanding

You want summaries as first-class nodes: ingest content → extract multiple MEMORY nodes → synthesize a summary → store it as a SUMMARY node → link it to the extracted MEMORY nodes.

You’re aiming for this to be useful for retrieval/context building, not just “nice metadata.”

You want the smallest coherent contract that won’t paint us into a corner.

B) Recommended next step (one step only)

Implement one “document/session summary” per ingest run: summarize the extracted memories (not the raw content), store as a SUMMARY memory-kind, and create SUMMARIZES edges from the summary to each memory produced in that run.

C) Why this step now

Summarising memories is cheaper, cleaner, and safer than summarising raw text.

It creates an immediate payoff: a single node can stand in for 10–50 memories when building context, while still letting you “drill down” via edges.

D) Acceptance checks

After ingesting a source that yields ≥ N memories (pick N=5), the DB contains:

exactly 1 SUMMARY node for that ingest run (idempotent on re-run)

SUMMARIZES edges from SUMMARY → each extracted MEMORY

an audit event like SUMMARY_CREATED

A context builder path can request: “give me the summary + top-k linked memories” and it works deterministically.

E) Non-goals

No multi-level hierarchy yet (chunk summaries → doc summaries → persona summaries).

No contradiction resolution / supersession logic yet.

No “summary of summaries” / periodic consolidation jobs yet.

F) Risks & traps

Hallucinated summaries: mitigate by summarising only extracted memories and requiring “only use provided items.”

Idempotency drift: you need a stable key (e.g., analysis_run_id + summary_kind).

Edge explosion: start with one summary per run; avoid per-chunk summaries until you need them.

G) Agent packet (copy/paste)
Objective
Add “summary nodes” to ingest: create one SUMMARY per ingest run that summarises extracted MEMORY nodes; link with edges.

Files touched

DB schema/models: add memory_kind enum or similar (MEMORY, SUMMARY) if not present

Ingest pipeline: after memory extraction, call summarizer + write summary node + edges

Event log: add SUMMARY_CREATED

Boundaries

Summary is derived from extracted memories only (not raw content) for MVP.

One summary per ingest run (per source) only.

Must be idempotent (re-run doesn’t create duplicates).

Acceptance checks

Ingest produces 1 summary when ≥5 memories extracted

Edges exist and are traversable

Re-ingest produces the same summary node (or supersedes deterministically)

Concrete design (Vestig-shaped)
1) Data model: reuse “Memory” with a kind

Don’t invent a new table unless you must.

Memory fields (existing + minimal additions):

id

kind: "MEMORY" | "SUMMARY"

content: summary text (for SUMMARY)

source_id / analysis_run_id (whatever you already use to group an ingest)

t_created, t_invalid (bi-temporal as you already do)

meta: include summary_scope, memory_ids, prompt_version, etc.

2) Edge type

Create a single edge type:

SUMMARIZES: Summary → Memory

This direction is nice because expansion is “start at summary, drill down.”

(If you often start at a memory and want its summary, you can either reverse-query edges or optionally add the inverse later.)

3) How to generate the summary (MVP prompt strategy)

Summarise the extracted memories as the source of truth.

Input to summarizer:

A list of memory texts with their IDs (IDs are important for grounding)

Optional: a title/source label (filename, chat session name)

Output format recommendation (keep it rigid):

6–12 bullet points, each grounded in one or more memory IDs

Optional short “Open questions / uncertainties” section if you want

This gives you auditability (“which memories justify which bullets”) without copying raw text.

4) When to create a summary

Simple gating:

Only create a summary if num_memories >= 5 (or token count threshold)

Otherwise skip (don’t produce low-signal summaries)

5) Idempotency

Make the summary addressable with a deterministic key, e.g.:

summary_key = f"{analysis_run_id}:SUMMARY:DOC_V1"

On re-run:

If the same run ID is used, update/overwrite that summary node (or soft-delete old and insert new, but deterministically)

If a new run ID is created, it’s naturally a new summary.

6) Immediate payoff in retrieval / context building

A simple rule in your context builder:

If you have many memories from the same analysis_run_id/source_id, include:

the SUMMARY node content

top-k linked memories (by TraceRank / relevance)

omit the tail

This is where summaries stop being “metadata” and become token budget control.

2) Best way to use SUMMARY nodes
The core rule

SUMMARY nodes are navigation + compression, not replacement.

They’re most valuable when they:

reduce tokens (“what happened in this session/doc?”)

provide a stable handle to fan out to granular evidence

Retrieval pattern that works best (and stays simple)
Stage 1: Candidate retrieval (allow summaries)

Similarity search across both MEMORY and SUMMARY embeddings.

Apply a kind prior so summaries don’t dominate everything.

A simple deterministic prior:

kind_prior(SUMMARY) = -0.05

kind_prior(MEMORY) = 0

This means summaries can still win when they’re much more relevant, but they won’t casually crowd out granular items.

Stage 2: Expansion (summaries become gateways)

For each retrieved SUMMARY node:

expand via SUMMARIZES edges to the child MEMORY nodes

score those child memories directly against the query

then choose final context from:

(a) the summary itself (usually 1)

(b) top-k child memories (usually 3–8)

(c) any direct-hit memories from stage 1

This gives you “overview + receipts”.

Should SUMMARY nodes be boosted in TraceRank?

MVP answer: no special TraceRank boost.
Keep TraceRank semantics clean: it measures node utility over time via events and connectivity. Summaries will naturally become useful (and accumulate TraceRank) if they’re used.

If you want a safe boost later, do it as derived rank, not reinforcement:

summary_tracerank = aggregate(tracerank(children)) * 0.5 + tracerank(summary_events) * 0.5

Crucially: summary usage should not automatically reinforce all children, or you’ll create rank inflation.

A clean rule:

Viewing/using a summary reinforces the summary only.

Children are reinforced only when they’re actually selected into context or explicitly opened.

Are summaries the first-pass similarity search?

They can be, but with guardrails:

For broad queries (“what’s going on with X lately?”, “summarise my thinking about Y”), summaries are perfect first-pass hits.

For narrow queries (“what was the Felodipine dose?”, “which file path broke uv?”), summaries should not be the main answer—use them only as an expansion clue.

If you don’t want query classification yet, the two-stage approach above already behaves well: narrow queries will tend to score the specific memory higher than the summary once expansion happens.

If you want one crisp implementation decision:
Index summaries for similarity search, but always expand them and let granular memories compete for the final top-k. That’s the “gateway to granular” pattern you’re aiming for.
