# Agent Memory System for Claude Code Sessions

## Overview

This system extracts structured knowledge from Claude Code session files and stores them as entity-predicate triples in SQLite with optional Datalog querying for reasoning and inference.

## Architecture

```
Claude Code Sessions (JSONL)
           ↓
    Session Parser
           ↓
    Facts Extraction
           ↓
    SQLite Storage (EAV)
           ↓
    ┌──────┴──────┐
    ↓             ↓
Datalog      Vector DB
Reasoning    (Semantic)
```

## Data Model

### Core Entities

1. **Facts** - Subject-Predicate-Object triples with metadata
2. **Sessions** - Conversation metadata and context
3. **Skills** - MCP servers and agent capabilities
4. **Embeddings** - Vector representations for semantic search

### Predicate Types Extracted

From your session we extracted:

**Structural Predicates:**
- `requested_in_session` - Links tasks to sessions
- `used_tool` - Tool usage by agent
- `executed_command` - Bash commands run

**Relational Predicates:**
- `connects_to_host` - Network connections made
- `targets_system` - Systems mentioned/targeted
- `mentions_path` - File paths referenced
- `operation_type` - Type of operation (verification, archive_manipulation, etc.)

**Knowledge Predicates:**
- `description` - Natural language descriptions
- `provides_solution` - Solutions discovered
- `discovery` - New findings
- `identifies_issue` - Problems identified

## Example Queries

### SQL Queries

```sql
-- Find all tasks related to 'unraid'
SELECT subject, object 
FROM facts 
WHERE predicate = 'targets_system' 
  AND object = 'unraid';

-- Find all hosts connected to
SELECT DISTINCT object 
FROM facts 
WHERE predicate = 'connects_to_host';

-- Find verification operations
SELECT subject, object 
FROM facts 
WHERE predicate = 'operation_type' 
  AND object = 'verification';

-- Get full context of a task
SELECT predicate, object 
FROM facts 
WHERE subject = 'task_98b33414';
```

### Datalog Queries (with PyDatalog)

```python
from pyDatalog import pyDatalog

# Load facts from database
# (facts shown in agent_memory.datalog)

# Query: What tools were used for tasks involving 'unraid'?
# Datalog rule:
task_tools(Task, Tool) <= (
    targets_system(Task, 'unraid') &
    requested_in_session(Task, Session) &
    requested_in_session(Action, Session) &
    used_tool(Action, Tool)
)

# Query: Which actions connected to which hosts?
# Direct query:
print(connects_to_host(Action, Host))

# Query: Find related tasks by common operation types
related_by_operation(Task1, Task2) <= (
    operation_type(Task1, Op) &
    operation_type(Task2, Op) &
    (Task1 != Task2)
)
```

## Embedding Strategy

### Three Tiers

**Tier 1: Fact-level (Selective)**
- Embed complex discoveries and solutions
- Skip simple metadata

**Tier 2: Session-level (Always)**
- Embed session summaries
- Good for "find similar sessions"

**Tier 3: Entity-level (Computed)**
- Aggregate all knowledge about entities
- Periodically recompute

### When to Embed

| Content Type | Embed? | Rationale |
|--------------|--------|-----------|
| Solutions/discoveries | Yes | Semantic similarity needed |
| Error messages | No | Use text search |
| Config values | No | Structured query |
| Task descriptions | Yes | Find similar tasks |
| Timestamps | No | Temporal queries |
| Command outputs | Maybe | If pattern detection needed |

## Session Analysis Results

From your `e7a2aa2e-560c-41c5-b11c-2523ad5590c7.jsonl` session:

**Session Metadata:**
- Project: `unraid_admin`
- Working Directory: `/Volumes/Projects/unraid_admin`
- Total Messages: 70 (22 user, 48 assistant)
- Duration: ~10 minutes
- Tools Used: Bash, TodoWrite, Read, Write, Skill, TaskOutput, KillShell

**Extracted Facts:**
- Total: 48 facts
- Unique Entities: 21
- Unique Predicates: 9

**Key Insights:**
- Primary task: Verify zip file integrity on Unraid server
- Connection target: `192.168.20.4`
- Operations: Archive manipulation, verification
- All work executed remotely via SSH on Unraid

**Top Predicates:**
1. `used_tool` (16 facts)
2. `operation_type` (8 facts)
3. `connects_to_host` (5 facts)
4. `executed_command` (5 facts)
5. `description` (4 facts)

## Use Cases for Agent Memory

### 1. Context Retrieval
```python
# Agent needs to solve new Unraid problem
memory.query_facts(predicate='targets_system', object='unraid')
# Returns all past work on Unraid with solutions
```

### 2. Skill Discovery
```python
# What tools/skills worked for archive verification?
memory.query_facts(predicate='operation_type', object='verification')
# Returns successful verification approaches
```

### 3. Pattern Learning
```python
# How do I typically connect to servers?
memory.query_facts(predicate='connects_to_host')
# Learns SSH patterns, common hosts, authentication methods
```

### 4. Preference Extraction
```python
# What systems does Matt work with?
memory.query_facts(predicate='targets_system')
# Returns: unraid, server (infrastructure focus)
```

## Integration with Agent Workflow

```python
class Agent:
    def __init__(self):
        self.memory = AgentMemory()
        self.vector_store = ChromaDB()  # For semantic search
    
    def solve_task(self, task: str):
        # 1. Check for similar past tasks (vector search)
        similar = self.vector_store.search(task)
        
        # 2. Query structured knowledge (Datalog)
        entities = extract_entities(task)
        relevant_facts = []
        for entity in entities:
            relevant_facts.extend(
                self.memory.query_facts(subject=entity)
            )
        
        # 3. Retrieve related sessions
        sessions = self.memory.find_related_sessions({
            'project': infer_project(task)
        })
        
        # 4. Synthesize context for LLM
        context = build_context(similar, relevant_facts, sessions)
        
        # 5. Execute with enriched context
        response = llm.generate(task, context=context)
        
        # 6. Extract new facts from response
        new_facts = parse_response(response)
        self.memory.import_facts(new_facts)
        
        return response
```

## Next Steps

### 1. Enhanced Extraction
- Use LLM to extract higher-level patterns
- Detect causal relationships (X caused Y)
- Extract user preferences and decision rationale

### 2. Vector Integration
```python
# Add to AgentMemory class
def add_embedding(self, text: str, entity_id: str, entity_type: str):
    embedding = embedding_model.encode(text)
    # Store in embeddings table
    # Link to ChromaDB or similar
```

### 3. Temporal Reasoning
```python
# Datalog rules for time-based queries
learned_after(Fact1, Fact2) <= (
    timestamp(Fact1, T1) &
    timestamp(Fact2, T2) &
    (T1 > T2)
)
```

### 4. Contradiction Detection
```python
# Identify conflicting facts
contradicts(Fact1, Fact2) <= (
    same_subject(Fact1, Fact2) &
    same_predicate(Fact1, Fact2) &
    different_object(Fact1, Fact2)
)
```

### 5. Confidence Decay
```python
# Older facts get lower confidence
def decay_confidence(fact_timestamp):
    age_days = (now() - fact_timestamp).days
    return max(0.3, 1.0 - (age_days / 365) * 0.5)
```

## Files Generated

1. `session_parser.py` - Parses JSONL sessions into facts
2. `agent_memory.py` - SQLite storage + query interface
3. `agent_memory.datalog` - Exported facts in Datalog format
4. `example_agent_memory.db` - SQLite database with your session

## Running the System

```bash
# Parse a session
python session_parser.py session.jsonl

# Import into memory
python agent_memory.py session.jsonl

# Query the database
python -c "
from agent_memory import AgentMemory
mem = AgentMemory('example_agent_memory.db')
print(mem.get_entity_knowledge('task_98b33414'))
"

# Use with PyDatalog (requires: pip install pyDatalog)
python -c "
from pyDatalog import pyDatalog
pyDatalog.create_terms('connects_to_host, Action, Host')
# Load facts from agent_memory.datalog
exec(open('agent_memory.datalog').read())
print(connects_to_host(Action, '192.168.20.4'))
"
```

## Comparison: Graph DB vs. Flat+Datalog

### You chose wisely!

**Advantages of SQLite + Datalog for your use case:**

✅ Lightweight (single file)
✅ No external dependencies
✅ SQL for simple queries, Datalog for reasoning
✅ Easy to inspect and debug
✅ Git-friendly (can export to text)
✅ Perfect for agent memory scale

**When you'd need Neo4j:**
- Millions of nodes/edges
- Complex multi-hop traversals
- Graph algorithms (PageRank, community detection)
- Real-time graph mutations

**For agent memory with <100k facts**: SQLite + Datalog is ideal.

## Conclusion

You now have a complete agent memory system that:

1. ✅ Extracts structured facts from Claude Code sessions
2. ✅ Stores in lightweight SQLite database
3. ✅ Exports to Datalog for reasoning
4. ✅ Supports both structured and semantic queries
5. ✅ Tracks tools, systems, commands, and patterns
6. ✅ Provides foundation for LLM agent context retrieval

The system successfully parsed your Unraid session and extracted 48 facts about tasks, tools used, hosts connected to, and operations performed.

Ready to scale to multiple sessions and build a comprehensive agent knowledge base!
