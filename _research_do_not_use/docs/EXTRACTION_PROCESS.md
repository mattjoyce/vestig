# How Facts Are Extracted and Stored

## TL;DR

The system reads JSONL session files and applies **pattern-matching rules** to identify meaningful relationships. Each rule has a **confidence score** based on how reliable the extraction method is. Only relationships that match one of ~12 predefined patterns get stored.

---

## The Complete Process

### Step 1: Parse JSONL Messages
Each line in a `.jsonl` file is a JSON message with:
- **type**: "user" or "assistant"
- **uuid**: unique identifier
- **timestamp**: when it occurred
- **message.content**: the actual content (text or tool calls)

### Step 2: Apply Rule-Based Extraction

The system has **hard-coded predicate types** (not learned from data):

| Predicate | Confidence | Detection Method | Example |
|-----------|-----------|------------------|---------|
| `used_tool` | 1.0 | Direct from `tool_use` block | Used Bash tool |
| `executed_command` | 1.0 | Direct from `input.command` | `ssh user@host` |
| `connects_to_host` | 0.9 | Regex: `@([\w\.\-]+)` | 192.168.20.4 |
| `operation_type` | 0.8 | Keywords: unzip/tar/gzip | archive_manipulation |
| `mentions_path` | 0.8 | Regex: `/[\w/\-_.]+` | /mnt/user/data |
| `targets_system` | 0.7 | Keywords: unraid/server | unraid |
| `provides_solution` | 0.7 | Keywords: "solution", "fix" | (text snippet) |
| `discovery` | 0.6 | Keywords: "found", "discovered" | (text snippet) |
| `identifies_issue` | 0.7 | Keywords: "error", "failed" | (text snippet) |

### Step 3: Confidence-Based Filtering

Each fact gets a confidence score:

```
1.0 = Direct observation (tool calls, exact commands)
0.9 = Reliable regex extraction (SSH hosts)
0.8 = Pattern matching (archive operations)
0.7 = Keyword detection (system names, solutions)
0.6 = Natural language inference (discoveries)
```

When querying, you can filter by confidence:
```python
# Only high-confidence facts
facts = memory.query_facts(predicate='connects_to_host', min_confidence=0.8)
```

### Step 4: Storage in SQLite

Facts are stored in EAV (Entity-Attribute-Value) format:

```sql
CREATE TABLE facts (
    subject TEXT,      -- What entity (task_abc, action_xyz)
    predicate TEXT,    -- What relationship (used_tool, connects_to_host)
    object TEXT,       -- Related value (Bash, 192.168.20.4)
    confidence REAL,   -- How sure we are (0.0-1.0)
    ...
)
```

---

## What Qualifies for Storage?

### ✅ ALWAYS Stored (Confidence: 1.0)

**Structural facts** - the session skeleton:
- Every user message → creates a `task_id`
- Every assistant message → creates an `action_id`
- Every tool call → `used_tool(action_id, tool_name)`
- Every bash command → `executed_command(action_id, full_command)`

**Rationale**: These are direct observations from the JSONL structure. No interpretation needed.

### ✅ Usually Stored (Confidence: 0.8-0.9)

**Pattern-extracted facts**:
- SSH host from `ssh user@HOST` → `connects_to_host(action_id, host)`
- File paths from `/mnt/user/...` → `mentions_path(task_id, path)`
- Archive commands → `operation_type(action_id, 'archive_manipulation')`
- Verification flags → `operation_type(action_id, 'verification')`

**Rationale**: Regex and pattern matching are reliable but not perfect. False positives are possible.

### ⚠️ Conditionally Stored (Confidence: 0.6-0.7)

**Keyword-based extraction**:
- Text contains "unraid" → `targets_system(task_id, 'unraid')`
- Text contains "found that" → `discovery(action_id, text_snippet)`
- Text contains "solution" → `provides_solution(action_id, text_snippet)`

**Rationale**: Useful for semantic understanding, but keywords can appear in casual language. Lower confidence allows filtering later.

### ❌ NEVER Stored

**Filtered out as noise**:
- Generic words ("the", "a", "and")
- Empty or whitespace-only content
- Duplicate facts (database has UNIQUE constraint)
- Session metadata (stored in separate `sessions` table)

---

## Concrete Example

**Input JSONL:**
```json
{
  "type": "assistant",
  "uuid": "8bbd47e5",
  "message": {
    "content": [{
      "type": "tool_use",
      "name": "Bash",
      "input": {
        "command": "sshpass -p 'pwd' ssh root@192.168.20.4 'ls -la'"
      }
    }]
  }
}
```

**Facts Created:**

1. **used_tool** (confidence: 1.0)
   - Subject: `action_8bbd47e5`
   - Predicate: `used_tool`
   - Object: `Bash`
   - Method: Direct from `tool_use.name`

2. **executed_command** (confidence: 1.0)
   - Subject: `action_8bbd47e5`
   - Predicate: `executed_command`
   - Object: `sshpass -p 'pwd' ssh root@192.168.20.4 'ls -la'`
   - Method: Direct from `input.command`

3. **connects_to_host** (confidence: 0.9)
   - Subject: `action_8bbd47e5`
   - Predicate: `connects_to_host`
   - Object: `192.168.20.4`
   - Method: Regex `@([\w\.\-]+)` on command text

**Result**: 3 facts from 1 tool call!

---

## Why This Design?

### Advantages:

1. **Predictable**: Fixed set of predicates, not unbounded
2. **Queryable**: Can find all SSH connections with `WHERE predicate='connects_to_host'`
3. **Confidence-scored**: Can filter uncertain facts
4. **Extensible**: Add new rules by editing `session_parser.py`
5. **Interpretable**: Each fact has clear extraction logic

### Trade-offs:

1. **Manual rules**: New patterns require code changes (not learned automatically)
2. **Limited predicates**: Only ~12 types, may miss novel relationships
3. **Regex brittleness**: Edge cases can cause false positives/negatives

---

## Future Enhancements

### Use LLM for Extraction

Instead of regex/keywords, use an LLM to extract facts:

```python
prompt = f"""
Extract facts from this command:
{command}

Return JSON:
{{
  "hosts": [...],
  "operations": [...],
  "files": [...]
}}
"""
facts = llm.generate(prompt)
```

**Pros**: Better semantic understanding, fewer missed patterns
**Cons**: Slower, costs money, requires LLM API

### Learned Predicates

Let the system discover new predicate types from data:

```python
# Cluster similar relationships
similar_patterns = cluster_commands_by_similarity()
for cluster in similar_patterns:
    new_predicate = f"operation_{cluster.id}"
    # Automatically create new predicate type
```

**Pros**: Discovers domain-specific patterns
**Cons**: Harder to query, less interpretable

---

## Summary

**How it works:**
1. Parse JSONL → Extract messages
2. Apply rules → Pattern matching
3. Create facts → With confidence scores
4. Store in DB → EAV schema

**What gets stored:**
- Direct observations (1.0 confidence)
- Regex patterns (0.8-0.9)
- Keyword matches (0.6-0.7)

**What doesn't:**
- Noise, duplicates, metadata

**The key insight:** Predicates are **hard-coded patterns**, not learned. This trades flexibility for reliability and queryability.
