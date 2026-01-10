# Vestig Roadmap

## Vision

A personal memory and knowledge base with affordances for AI access. The CLI is one client among many - Discord bots, webhooks, Claude Code integration. The core (graph storage, embeddings, retrieval) remains stable while an orchestration layer handles multiple input channels.

---

## Architecture Evolution

### Current: CLI-First
```
User → CLI → Core (storage, embeddings, retrieval)
```

### Future: Service-First
```
                    ┌─────────────────┐
                    │  Vestig Server  │
                    │  (daemon/API)   │
                    └────────┬────────┘
                             │
        ┌────────────┬───────┴───────┬────────────┐
        │            │               │            │
   Discord Bot   File Watch    Housekeeping     CLI
                                  Cron
```

### Real-time vs Batched Operations

| Operation | Mode | Rationale |
|-----------|------|-----------|
| Discord `!remember` | Real-time | User expects confirmation |
| Discord `!recall` | Real-time | Interactive query |
| Claude Code recall | Real-time | Agent is waiting |
| YouTube transcripts | Batched | Transcript fetch is slow |
| File ingestion | Batched | Could be large, no urgency |
| Entity backfill | Batched | Background housekeeping |

---

## Near-term Refactoring

### Source Abstraction
Unify provenance under a Source type with dual linking:
- Current: `File → Chunk → Memory`
- Proposed: `Source → Memory` (always) AND `Source → Chunk → Memory` (when chunked)

**Source Types:**
- `file` - Document ingestion
- `agentic` - AI agent contributions (with `agent` field: 'claude-code', 'codex', 'goose', etc.)
- `legacy` - Backfilled orphans from housekeeping

**Key Insight:** Source → Memory is always the primary provenance link. Chunk provides optional positional metadata (start, length, sequence) for chunked content.

Benefits:
- Unified provenance model for all content origins
- Session-based content (Claude Code, Discord) gets proper lineage
- Enables trust signals and quality assessment across source types
- Chunk becomes optional location metadata rather than required intermediary

### CLI Simplification
Consider deprecating:
- `memory add` - creates orphaned data without provenance
- `memory search` - redundant with `recall`

Keep ingestion as the primary path, with source type differentiation.

### Ad-hoc Memory Handling
For atomic memories that bypass the Source→Chunk chain:
- Entity extraction provides retrieval pathway via MENTIONS edges
- Consider `--entities` flag on `memory add` for explicit linking
- Periodic housekeeping can associate orphans with entities retroactively

---

## Housekeeping System

Scheduled batch jobs for graph maintenance:

1. **Entity Extraction Backfill** - Run extraction on memories lacking entity links
2. **Orphan Detection** - Find memories without provenance or entity connections
3. **TraceRank Decay** - Apply temporal decay to access scores
4. **Embedding Refresh** - Re-embed when model version changes
5. **Source Health Check** - Verify linked files still exist

Implementation: `vestig housekeeping` command with subcommands or `--task` flag.

---

## Integration Points

### Claude Code
- `/vestig-context` - Recall with conversation-aware query synthesis
- `/vestig-remember` - Bulk ingest conversation via stdin
- `/remember` - Agent-curated selective memory commits

### Discord (Future)
- `!remember <text>` - Ad-hoc memory, real-time confirmation
- `!recall <query>` - Interactive retrieval
- `!watch <youtube-url>` - Queue transcript for ingestion

### API (Future)
- REST/GraphQL endpoint for programmatic access
- Webhook receivers for external triggers
- SSE/WebSocket for real-time notifications

---

## Data Model Notes

### SUMMARY Nodes
`kind='SUMMARY'` memories are lightweight chunk representations optimized for retrieval. They capture semantic essence without full content, enabling efficient vector search while maintaining links to detailed Memory nodes.

### Provenance Chain (Revised)
```
Source (type=file|agentic|legacy)
    │
    ├──[PRODUCED]──────────────→ Memory (always, primary provenance)
    │                                │
    │                                └──[MENTIONS]──→ Entity
    ├──[PRODUCED]──────────────→ Summary
    │                                │
    │                                └──[MENTIONS]──→ Entity
    │
    └──[HAS_CHUNK]──→ Chunk (optional positional metadata)
                          │
                          └──[CONTAINS]──→ Memory (additional link for chunked content)
```

**Dual linking:** Memories have both Source (provenance) and optional Chunk (position) links.

### Entity Provenance & Trust
Entities can be extracted at different levels with varying trust:

| Extraction Path | Trust Level | Notes |
|-----------------|-------------|-------|
| Source → Memory → Entity | Medium | 1 LLM processing hop |
| Source → Summary → Entity | Medium | Same trust as Memory (1 LLM hop) |
| Source → Chunk → Entity | High (future) | Direct from raw text if implemented |

**Current implementation:** Entities extracted from Memory and Summary nodes. Both have equal trust as they're one LLM processing step from source.

**Open question:** Should Chunk remain a first-class node, or become edge/node metadata?

### Entity-based Discovery
Memories and Summaries connect to Entities via MENTIONS edges. This provides an alternative retrieval pathway independent of the provenance chain - critical for memories without source links (orphans).

---

## Technical Debt

- [ ] Conditional validation for backend-specific config (done: db_path)
- [ ] FalkorDB adapter edge cases (weight field nulls)
- [ ] Test coverage for dual-backend scenarios
- [ ] Embedding timeout configurability (done: `embedding.timeout`)

---

## Open Questions

### Phase 2 (Source Abstraction)
1. Should Chunk remain a first-class node, or become metadata on Memory/edges?
   - Pro keeping: Enables entity linking to document position
   - Pro flattening: Simpler model, chunk info as edge properties on PRODUCED
2. Should we implement direct Source → Chunk → Entity extraction (high trust)?
   - Would require entity extraction from raw chunks before memory extraction
3. Edge type naming: PRODUCED vs CREATED vs FROM for Source → Memory?
4. Should all three source types (file, agentic, legacy) support chunking?

### General
1. Should `memory add` require explicit entity specification to avoid orphans? ✓ Resolved: will create Source{type='agentic'}
2. How aggressive should orphan cleanup be? ✓ Partial: housekeeping can backfill entities
3. What's the right granularity for session-based Sources?
4. Should SUMMARY generation be optional/configurable per source type?
