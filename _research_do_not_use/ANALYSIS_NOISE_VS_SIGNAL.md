# Analysis: Noise vs Signal in Extracted Facts

## File Types in ~/.claude/projects

Found **two types** of JSONL files:

### 1. Session Files (18 files) - **USEFUL**
```
076a5cf1-b3c8-4133-81fe-bd73f138b624.jsonl
2d667f23-24b3-4c79-9642-7a6ba610487a.jsonl
```
- UUID-named files
- Real user conversations
- Actual Claude Code interactions

### 2. Agent Files (38 files) - **MOSTLY NOISE**
```
agent-a2d11fd.jsonl
agent-a570f60.jsonl
```
- Background agent executions
- Internal Task tool calls (Explore, claude-code-guide, etc.)
- Start with "Warmup" messages
- **Not useful for user memory**

## Current Database Contents (vestig_memory.db)

### Extracted Facts by Predicate

```
Predicate              Method      Count   Assessment
─────────────────────────────────────────────────────────
used_tool              naive       615     NOISE (mostly TodoWrite, internal tools)
executed_command       naive       306     MIXED (some useful, some internal)
description            naive       144     MIXED (many "Warmup", some real)
requested_in_session   naive       144     STRUCTURAL (linking only)
operation_type         naive       77      USEFUL
connects_to_host       naive       73      USEFUL (SSH connections)
identifies_issue       naive       54      USEFUL
mentions_path          naive       53      USEFUL
session_intent         discerned   52      VERY USEFUL
provides_solution      naive       38      USEFUL
workflow_pattern       discerned   37      VERY USEFUL
knowledge_type         discerned   34      USEFUL
assistant_tone         discerned   33      MARGINAL
discovery              naive       20      USEFUL
problem_description    discerned   15      VERY USEFUL
solution_found         discerned   15      VERY USEFUL
targets_system         naive       13      USEFUL
user_preference        discerned   13      VERY USEFUL
involves_command       naive       10      USEFUL
task_pattern           naive       4       MARGINAL
```

### Noise Examples

**Internal tool usage (NOISE):**
```
subject: action_de028856 → used_tool: TodoWrite
subject: action_a258b0c9 → used_tool: TodoWrite
subject: action_72f679fb → used_tool: TodoWrite
```
TodoWrite is Claude Code's internal task tracking - not useful for user memory.

**Warmup messages (NOISE):**
```
subject: task_24572de9 → description: Warmup
subject: task_93605e42 → description: Warmup
subject: task_067dfdd5 → description: Warmup
```
Agent initialization messages - not real user tasks.

### Signal Examples

**Session intent (SIGNAL):**
```
"The user wanted to push a local repository to GitHub as a private repository"
"The user was trying to verify the capabilities and limitations of Claude Code"
"The user is trying to establish skills needed for effective support of the matterbase initiative"
```
These are useful! Real user goals.

**SSH connections (SIGNAL):**
```
connects_to_host: 192.168.20.4
connects_to_host: unraid.local
```
Real infrastructure interactions.

**Workflow patterns (SIGNAL):**
```
workflow_pattern: git_commit_and_push
workflow_pattern: verification_pattern
workflow_pattern: ssh_then_verify_backup
```
Learned patterns from user behavior.

## Noise Percentage Estimate

### Current State (All Files)
- **Total facts**: ~1,790
- **Noise (agent files + internal tools)**: ~60-70%
- **Signal (real user actions)**: ~30-40%

### If Filtering Agent Files
- **Agent files**: 38 (68% of files)
- **Session files**: 18 (32% of files)
- **Expected signal improvement**: 70-80% useful facts

## Recommendations

### 1. Filter Out Agent Files

```python
# Only process session files (UUID-named)
import re

session_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jsonl$')

for file in glob.glob("~/.claude/projects/**/*.jsonl"):
    filename = os.path.basename(file)
    if session_pattern.match(filename):
        # Process this - it's a real session
        process_session(file)
    else:
        # Skip - it's an agent file
        continue
```

### 2. Filter Internal Tools

Add to naive extraction:
```python
INTERNAL_TOOLS = {
    'TodoWrite',  # Task tracking
    'AskUserQuestion',  # Internal prompts
    'EnterPlanMode',  # Mode switching
    'ExitPlanMode',
    # Add others as discovered
}

def should_extract_tool_usage(tool_name):
    return tool_name not in INTERNAL_TOOLS
```

### 3. Filter Warmup Messages

```python
def is_noise_message(content):
    if content.strip() == "Warmup":
        return True
    if content.startswith("This is a warmup"):
        return True
    return False
```

### 4. Focus on High-Value Predicates

**Keep:**
- `session_intent` (discerned)
- `workflow_pattern` (discerned)
- `problem_description` / `solution_found` (discerned)
- `user_preference` (discerned)
- `connects_to_host` (naive)
- `mentions_path` (naive)
- `targets_system` (naive)
- `executed_command` (naive - filtered)

**Drop or deprioritize:**
- `used_tool` (unless filtering internal tools)
- `description` (unless filtering warmup)
- `task_pattern` (low value)
- `assistant_tone` (marginal value)

## Impact Assessment

### Current Database
```
Total: 1,790 facts
├── Noise: ~1,200 (67%)
└── Signal: ~590 (33%)
```

### After Filtering
```
Total: ~600 facts (from 18 session files only)
├── Noise: ~120 (20%)
└── Signal: ~480 (80%)
```

**Result**: ~80% signal-to-noise ratio vs current 33%

## Conclusion

**Main Issues:**
1. **Agent files dominate** (38/56 files = 68%)
2. **Internal tool tracking** (TodoWrite, etc.)
3. **Warmup messages** (agent initialization)

**Quick Wins:**
1. ✅ **Filter agent-* files** → Removes 68% of noise
2. ✅ **Skip internal tools** → Removes TodoWrite spam
3. ✅ **Skip warmup messages** → Removes agent init

**Expected Improvement:**
- From 33% signal to 80% signal
- From 1,790 facts to ~600 high-quality facts
- Much better RAG retrieval (less noise to search through)

## Next Steps

1. Update `import_all_sessions.py` to filter agent files
2. Add internal tool filter to session_parser.py
3. Add warmup message detection
4. Re-import with clean filters
5. Compare signal-to-noise ratio
