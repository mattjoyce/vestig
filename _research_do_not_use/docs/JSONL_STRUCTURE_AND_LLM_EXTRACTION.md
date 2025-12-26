# JSONL Structure & LLM-Based Extraction

## What's in the JSONL File

Claude Code session files contain **multiple message types**:

### Message Type Distribution (from real session)
```
assistant                 : 107 messages
user                      :  67 messages
system                    :   3 messages (internal)
file-history-snapshot     :   8 messages (metadata)
```

### 1. User Messages
User input, including special command messages:

```json
{
  "type": "user",
  "uuid": "36de2e6f-...",
  "timestamp": "2025-12-25T00:45:48.943Z",
  "message": {
    "role": "user",
    "content": "<command-message>init</command-message>\n<command-name>/init</command-name>"
  }
}
```

**Contains**:
- Actual user requests
- Command triggers (`/init`, `/commit`, etc.)
- Plain text questions and instructions

### 2. Assistant Messages
Claude's responses with multiple content blocks:

```json
{
  "type": "assistant",
  "timestamp": "2025-12-25T00:45:53.413Z",
  "message": {
    "content": [
      {
        "type": "text",
        "text": "I'll help you with that..."
      },
      {
        "type": "thinking",
        "thinking": "The user wants to... I should..."
      },
      {
        "type": "tool_use",
        "name": "Bash",
        "input": {
          "command": "ssh user@host",
          "description": "Connect to server"
        }
      }
    ]
  }
}
```

**Content Block Types** (from real session):
- `tool_use` : 61 blocks (tool calls with parameters)
- `thinking` : 36 blocks (Claude's internal reasoning)
- `text` : 10 blocks (responses to user)

### 3. System Messages
Internal Claude Code infrastructure:

```json
{
  "type": "system",
  "uuid": "7028605e-...",
  "message": {
    "content": "<system-reminder>The TodoWrite tool hasn't been used...</system-reminder>"
  }
}
```

**Contains**:
- `<system-reminder>` tags (10 occurrences in our session)
- Internal state management
- Hook execution results

### 4. File History Snapshots
Tracking file changes:

```json
{
  "type": "file-history-snapshot",
  "messageId": "...",
  "snapshot": {
    "trackedFileBackups": {...},
    "timestamp": "..."
  }
}
```

---

## Current Naive Extraction

**What it processes:**
- ✅ User messages → Extract task descriptions
- ✅ Assistant tool_use blocks → Extract tool usage
- ✅ Bash commands → Extract hosts, operations
- ❌ **IGNORES**: thinking blocks, system messages, file snapshots

**What it misses:**
- Claude's reasoning (`thinking` blocks have valuable context)
- System reminders (user preferences, patterns)
- File change patterns
- Cross-message relationships

---

## LLM-Based Extraction: The Upgrade

### Approach 1: Extract from Thinking Blocks

**Current**: Ignored completely
**LLM-based**: Mine for insights

```python
def extract_from_thinking(thinking_text: str) -> list[Fact]:
    prompt = f"""
    Analyze this internal reasoning from an AI assistant:

    {thinking_text}

    Extract facts as JSON:
    {{
      "user_preferences": ["prefers X", "usually does Y"],
      "task_understanding": "What the user really wants",
      "approach_rationale": "Why this approach was chosen",
      "concerns": ["potential issue 1", "edge case 2"]
    }}
    """

    response = llm.generate(prompt)
    facts = [
        Fact('session_123', 'user_preference', pref)
        for pref in response['user_preferences']
    ]
    return facts
```

**Example from real thinking block:**
```
Thinking: "The user wants me to check the code and test it with real data.
I should explore the .claude directory for session files..."
```

**LLM could extract:**
- `task_intent(task_456, "test_with_real_data")`
- `data_location(session, "~/.claude/projects")`
- `approach(task_456, "explore_then_test")`

### Approach 2: Multi-Message Context

**Current**: Each message processed independently
**LLM-based**: Understand conversation flow

```python
def extract_conversation_patterns(messages: list) -> list[Fact]:
    # Group related messages
    conversation = "\n".join([
        f"User: {msg['user_msg']}"
        f"Assistant: {msg['assistant_response']}"
        for msg in messages[:5]  # Last 5 exchanges
    ])

    prompt = f"""
    Analyze this conversation flow:

    {conversation}

    Identify:
    1. What pattern is the user following? (debugging, building, exploring)
    2. What problems were encountered and how were they solved?
    3. What did the user learn or accomplish?
    4. Are there recurring themes or preferences?

    Return JSON with extracted facts.
    """

    response = llm.generate(prompt)

    # Creates high-level facts:
    return [
        Fact('session_123', 'workflow_pattern', 'iterative_testing'),
        Fact('session_123', 'problem_solved', 'type_error_in_context_field'),
        Fact('session_123', 'learning', 'sqlite_eav_schema_design')
    ]
```

### Approach 3: Semantic Command Analysis

**Current**: Regex for SSH hosts
**LLM-based**: Full semantic understanding

```python
def analyze_command_semantically(command: str) -> list[Fact]:
    prompt = f"""
    Analyze this bash command and extract all meaningful information:

    Command: {command}

    Extract as JSON:
    {{
      "hosts": ["IP or hostname"],
      "operations": ["what it does semantically"],
      "risk_level": "safe|risky|destructive",
      "files_affected": ["paths"],
      "requires_auth": true/false,
      "dependencies": ["tools needed"],
      "intent": "what the user is trying to accomplish"
    }}
    """

    response = llm.generate(prompt)

    return [
        Fact(action_id, 'connects_to_host', host)
            for host in response['hosts']
    ] + [
        Fact(action_id, 'operation_intent', response['intent']),
        Fact(action_id, 'risk_level', response['risk_level']),
        # etc.
    ]
```

**Example:**
```bash
Command: "sshpass -p 'MarkerPen.59' ssh root@192.168.20.4 'cd /mnt/user && ls -la'"
```

**Naive extraction:**
- connects_to_host: 192.168.20.4

**LLM extraction:**
- connects_to_host: 192.168.20.4 (same)
- auth_method: password_based (NEW)
- user_role: root (NEW)
- operation_intent: list_directory_contents (NEW)
- working_directory: /mnt/user (NEW)
- risk_level: safe (NEW)

### Approach 4: System Message Mining

**Current**: Completely ignored
**LLM-based**: Extract user behavior patterns

```python
def mine_system_reminders(reminders: list[str]) -> list[Fact]:
    prompt = f"""
    These system reminders were shown to the AI during a session:

    {reminders}

    What do they reveal about:
    1. User's workflow preferences (uses TodoWrite? ignores it?)
    2. Common mistakes or oversights
    3. Tools the user relies on

    Return insights as facts.
    """

    # Creates facts like:
    return [
        Fact('user_profile', 'prefers_tool', 'TodoWrite'),
        Fact('user_profile', 'common_oversight', 'forgets_to_track_tasks'),
    ]
```

---

## Proposed Hybrid Architecture

```python
class EnhancedSessionParser:
    def __init__(self, use_llm=True):
        self.naive_parser = ClaudeCodeSessionParser()  # Fast baseline
        self.llm_enabled = use_llm

    def parse_session(self, filepath: str) -> dict:
        # Phase 1: Fast naive extraction
        facts = self.naive_parser.parse_session_file(filepath)

        if not self.llm_enabled:
            return facts

        # Phase 2: LLM-based enhancement
        messages = self._load_messages(filepath)

        # Extract from thinking blocks (high value)
        thinking_facts = self._llm_extract_thinking(messages)
        facts['facts'].extend(thinking_facts)

        # Analyze conversation flow (session-level)
        flow_facts = self._llm_analyze_flow(messages)
        facts['facts'].extend(flow_facts)

        # Semantic command analysis (selective - only complex commands)
        complex_commands = self._find_complex_commands(messages)
        for cmd in complex_commands:
            semantic_facts = self._llm_analyze_command(cmd)
            facts['facts'].extend(semantic_facts)

        return facts

    def _find_complex_commands(self, messages):
        """Only use LLM for commands that naive extraction struggles with"""
        # Simple commands: skip LLM
        # Complex commands: use LLM
        return [cmd for cmd in all_commands if len(cmd) > 100 or has_pipes(cmd)]
```

**Cost optimization:**
- Naive extraction: ~156 facts, <100ms, $0
- LLM thinking blocks: ~36 blocks × $0.001 = $0.036
- LLM flow analysis: 1 per session × $0.01 = $0.01
- LLM complex commands: ~5 commands × $0.001 = $0.005
- **Total: ~$0.05 per session** for 50% more insights

---

## Real-World Example: What We'd Gain

### Current Session Facts (Naive)
```
✓ used_tool(action_x, 'Bash') - 21 times
✓ connects_to_host(action_y, '192.168.20.4') - 20 times
✓ operation_type(action_z, 'verification') - 6 times
```

### Additional Facts (LLM Enhancement)
```
NEW: workflow_pattern(session, 'documentation_generation')
NEW: user_intent(session, 'create_CLAUDE_md_for_vestig_project')
NEW: problem_encountered(session, 'context_field_type_error')
NEW: solution_applied(session, 'added_type_conversion_in_parser')
NEW: learning_outcome(session, 'sqlite_eav_better_than_graph_db_for_agent_memory')
NEW: user_preference(user, 'prefers_minimal_verbose_documentation')
NEW: auth_pattern(user, 'uses_sshpass_with_plaintext_passwords')
NEW: iterative_approach(session, 'test_find_bug_fix_retest')
```

**Value**: These meta-facts enable:
- "How do I usually solve type errors?" → Query `solution_applied` predicates
- "What projects am I working on?" → Query `user_intent` by project
- "What's my debugging workflow?" → Query `workflow_pattern`

---

## Implementation Recommendation

**Start with:**
1. Keep naive extraction (fast, reliable baseline)
2. Add LLM analysis for **thinking blocks only** (highest ROI)
3. Add LLM **session summary** (one call per session)

**Cost per 100 sessions:** ~$5
**Additional facts extracted:** ~30-50 per session
**New insight types:** 5-10 new predicate types

**Later add:**
4. Semantic command analysis (for complex commands)
5. System message mining (user behavior patterns)

This gets you 80% of the value for 20% of the cost.
