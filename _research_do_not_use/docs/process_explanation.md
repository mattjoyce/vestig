# Agent Memory System: Fact Extraction Process

## Overview: From JSONL to Facts

```
JSONL Messages → Extract Metadata → Identify Events → Apply Rules → Create Facts → Store in DB
```

## Step 1: Parse JSONL Structure

Each Claude Code session is a JSONL file where each line is a JSON object representing:
- User messages
- Assistant messages (with tool calls)
- Tool results
- Session metadata

Example JSONL line:
```json
{
  "type": "assistant",
  "sessionId": "abc-123",
  "timestamp": "2025-12-25T10:00:00Z",
  "message": {
    "content": [
      {"type": "text", "text": "I'll help with that..."},
      {"type": "tool_use", "name": "Bash", "input": {"command": "ssh user@host"}}
    ]
  }
}
```

## Step 2: What Qualifies as a Fact?

A fact is created when we detect a **meaningful relationship** between entities.

### Qualification Criteria:

#### A. Direct Observations (Confidence: 1.0)
These are explicitly present in the data:
- ✅ Tool was invoked → `used_tool(action_id, 'Bash')`
- ✅ Command was executed → `executed_command(action_id, 'ssh ...')`
- ✅ User made a request → `requested_in_session(task_id, session_id)`
- ✅ Task description provided → `description(task_id, 'Help me...')`

#### B. High-Confidence Inferences (Confidence: 0.8-0.9)
These are derived from parsing command content:
- ✅ SSH command contains `@host` → `connects_to_host(action_id, 'host')`
- ✅ Command uses `unzip -t` → `operation_type(action_id, 'verification')`
- ✅ Text mentions `/path/to/file` → `mentions_path(task_id, '/path/to/file')`

#### C. Medium-Confidence Patterns (Confidence: 0.6-0.7)
These are detected from natural language:
- ✅ Text contains "unraid" → `targets_system(task_id, 'unraid')`
- ✅ Text says "found that..." → `discovery(action_id, 'text snippet')`
- ✅ Text says "the solution is..." → `provides_solution(action_id, 'solution')`

#### D. Low-Confidence Patterns (Confidence: 0.5)
These are workflow patterns:
- ✅ User asks → Assistant uses tools → `task_pattern(pattern_id, 'request->tools:Bash,Read')`

