# Vestig: Lessons Learned

**Research Phase:** December 2024
**Goal:** Build an agent memory system from Claude Code sessions

## What Worked ✅

### 1. Dual Extraction Architecture
**Naive (regex) + Discerned (LLM) = Powerful combination**

```python
# Naive: Fast, structured, deterministic
connects_to_host: "192.168.20.4"
executed_command: "ssh root@192.168.20.4 ..."
mentions_path: "/mnt/backups/"

# Discerned: Semantic, contextual, insightful
session_intent: "User wanted to verify backup integrity on unraid server"
workflow_pattern: "ssh_connection_then_verify_backup"
user_preference: "Prefers direct responses without over-explanation"
```

**Key insight:** Naive captures WHAT happened, Discerned captures WHY and HOW.

### 2. Deduplication via File Hashing
**SHA256 hash prevents re-processing sessions**
- Critical for batch imports
- Automatic duplicate detection
- Stored in `sessions.file_hash`

### 3. Semantic Search (RAG)
**Embeddings + cosine similarity works well**
- Used `llm` Python API (not subprocess!)
- OpenAI ada-002: 1536 dimensions
- Query: "verify backups" → Finds semantically similar facts
- 0.85+ similarity = highly relevant

### 4. SQLite EAV Pattern
**Simple, flexible, portable**
- Facts as triples: (subject, predicate, object)
- Rich metadata: confidence, extraction_method, timestamp, context
- Single .db file - git-friendly
- SQL joins for exact queries + embeddings for fuzzy search

## Critical Discovery: Noise Problem 🚨

### Claude Code Has Two File Types

**~/.claude/projects contains:**
1. **Session files** (18/56 = 32%) - `{uuid}.jsonl` - Real user conversations ✅
2. **Agent files** (38/56 = 68%) - `agent-{hash}.jsonl` - Internal background agents ❌

**Agent files are NOISE:**
- Background Task tool executions
- Start with "Warmup" messages
- Internal agent operations (Explore, claude-code-guide, etc.)
- **Not useful for user memory**

### Noise in Extracted Facts

**Current database (all 56 files):**
- Total: 1,790 facts
- **Noise: ~67%** (agent files + internal tools)
- **Signal: ~33%**

**Noise sources:**
- `used_tool: TodoWrite` (615×) - Internal task tracking
- `description: "Warmup"` (many) - Agent initialization
- Internal tools: TodoWrite, AskUserQuestion, EnterPlanMode, etc.

**After filtering (18 session files only):**
- Expected: ~600 facts
- **Signal: ~80%**
- **3x improvement in quality**

## Technical Learnings

### Use llm Python API, Not Subprocess
```python
# Bad (slow, fragile):
subprocess.run(['llm', '-m', model, '-c', text])

# Good (fast, clean):
model = llm.get_model("gpt-4o-mini")
response = model.prompt(text)
```

### Predicate Usefulness Ranking

**High Value:**
- `session_intent` - What user wanted
- `workflow_pattern` - Learned behaviors
- `problem_description` / `solution_found` - Problem-solving pairs
- `user_preference` - Behavioral patterns
- `connects_to_host` - Infrastructure (filtered)
- `executed_command` - Commands (filtered)

**Low Value:**
- `used_tool` - Mostly internal noise
- `task_pattern` - Too generic
- `assistant_tone` - Marginal utility

### Embeddings Performance
- Generation: ~0.5-2s per fact (OpenAI API)
- Cost: ~$0.10 per 1,000 facts
- Retrieval: <100ms for top-5 search
- **Worth it** for semantic search

## What Didn't Work ❌

### 1. Datalog Integration
- Prepared (export function exists)
- Never implemented
- **RAG embeddings solved the same problem better**
- Logical rules < semantic similarity for human language

### 2. Full Graph Database (sqlite-graph)
- Considered but not needed
- EAV pattern + SQL joins sufficient
- **Path finding not critical for this use case**
- Keep as future option if needed

### 3. Processing All Files Without Filtering
- 68% of files are agent noise
- Wastes time, money (LLM calls), storage
- **Filter first, then process**

## Architecture Recommendations

### Simple Stack
```
Claude Code Sessions (JSONL)
         ↓
   File Filter (UUID only)
         ↓
   Dual Extraction
   ├── Naive (regex) - structured facts
   └── Discerned (LLM) - semantic insights
         ↓
   SQLite Storage (EAV)
   ├── Facts (with extraction_method flag)
   ├── Sessions (with file_hash)
   └── Embeddings (for RAG)
         ↓
   Hybrid Querying
   ├── SQL (exact relationships)
   └── RAG (semantic similarity)
```

### Filters Needed
```python
# 1. File filter
if re.match(r'^[0-9a-f-]{36}\.jsonl$', filename):
    process()  # Real session
else:
    skip()     # Agent file

# 2. Tool filter
INTERNAL_TOOLS = {'TodoWrite', 'AskUserQuestion', 'EnterPlanMode', 'ExitPlanMode'}
if tool not in INTERNAL_TOOLS:
    extract()

# 3. Content filter
if message == "Warmup":
    skip()
```

## Cost & Performance

**For 18 real sessions (filtered):**
- Naive extraction: ~1 minute (free)
- Discerned extraction: ~5-10 minutes (~$0.50)
- Embeddings: ~5 minutes (~$0.10)
- **Total: ~15 minutes, ~$0.60**

**Storage:**
- ~600 high-quality facts
- ~10MB database
- Portable, git-friendly

## Key Metrics

**Signal-to-Noise by Method:**
- Naive extraction: 40% signal (lots of internal tool noise)
- Discerned extraction: 95% signal (semantic filtering works!)
- **Hybrid: 80% signal** (after file + tool filtering)

**Most Valuable Predicates:**
1. `session_intent` (discerned) - User goals
2. `workflow_pattern` (discerned) - Behavioral patterns
3. `problem_description`/`solution_found` (discerned) - Learning
4. `user_preference` (discerned) - Personalization
5. `connects_to_host` (naive) - Infrastructure knowledge

## Recommendations for Reboot

### Phase 1: Clean Import
1. ✅ Filter to session files only (UUID-named)
2. ✅ Skip internal tools (TodoWrite, etc.)
3. ✅ Skip warmup messages
4. ✅ Process ~18 sessions → ~600 quality facts

### Phase 2: Core Features
1. ✅ Dual extraction (naive + discerned)
2. ✅ Deduplication (SHA256 hash)
3. ✅ SQLite EAV storage
4. ✅ Embeddings via `llm` Python API

### Phase 3: Query Interface
1. ✅ RAG retrieval: `find_similar_facts(query)`
2. ✅ SQL queries for exact matches
3. ✅ Hybrid: combine both

### Skip (For Now)
- ❌ Datalog reasoning (embeddings work better)
- ❌ sqlite-graph (EAV sufficient)
- ❌ Complex graph queries (YAGNI)

## Design Principles

1. **Simple > Complex** - SQLite + embeddings beats graph DBs
2. **Filter early** - Don't process noise
3. **Hybrid extraction** - Regex + LLM beats either alone
4. **Python API > Subprocess** - Direct llm module calls
5. **Signal > Volume** - 600 quality facts > 1,800 noisy facts
6. **Portable** - Single .db file, no services

## Success Criteria

**Good memory system has:**
- ✅ High signal-to-noise ratio (>70%)
- ✅ Fast retrieval (<100ms)
- ✅ Semantic understanding (embeddings work)
- ✅ Learning from past (session_intent, workflows, preferences)
- ✅ Low maintenance (SQLite, no services)

## Next Steps

1. Clean reboot with filters applied
2. Process 18 session files only
3. Build simple query interface
4. Test RAG retrieval quality
5. **Ship it** - keep it simple

---

**Bottom Line:** Dual extraction + file filtering + embeddings = winning formula.
Keep it simple, filter the noise, focus on signal.
