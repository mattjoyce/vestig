# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Vestig** is an Agent Memory System that extracts structured knowledge from Claude Code session files (JSONL format) and stores them as entity-predicate triples in SQLite with optional Datalog querying for reasoning and inference.

The system enables LLM agents to:
- Learn from past Claude Code sessions
- Build a knowledge graph of tasks, tools, and solutions
- Query historical context to improve future responses
- Recognize patterns in user workflows and preferences

## Architecture

```
Claude Code Sessions (JSONL)
           ↓
    Session Parser (session_parser.py)
           ↓
    Facts Extraction
           ↓
    SQLite Storage (agent_memory.py)
           ↓
    ┌──────┴──────┐
    ↓             ↓
Datalog      Vector DB
Reasoning    (Future: Semantic Search)
```

### Core Components

1. **session_parser.py** - Parses JSONL session files and extracts structured facts
   - **Naive extraction** (default): Fast regex-based pattern matching
   - **Discerned extraction** (optional): LLM-powered semantic analysis
   - Extracts user intents, assistant actions, tool usage
   - Identifies entities (file paths, commands, systems, hosts)
   - Detects patterns (problem/solution pairs, discoveries)
   - Generates subject-predicate-object triples with extraction_method flag

2. **discerned_extractor.py** - LLM-based semantic fact extraction
   - Analyzes feedback dynamics (user→LLM, LLM→user)
   - Detects workflow patterns and user preferences
   - Identifies problem-solution relationships
   - Extracts session-level insights and intent
   - Uses Simon Willison's llm CLI for model access

3. **agent_memory.py** - SQLite storage with query interface
   - EAV (Entity-Attribute-Value) schema for facts
   - Session metadata tracking
   - Skills/MCP registry
   - Embeddings support (prepared for future vector search)
   - Export to Datalog format for PyDatalog integration
   - Tracks extraction_method (naive vs discerned) for all facts

4. **agent_example.py** - Demonstrates practical agent usage
   - Shows how to query memory for context retrieval
   - Examples of building memory-augmented prompts
   - Scenarios for learning from past sessions

## Data Model

### Core Tables

- **facts** - Subject-predicate-object triples with confidence scores
- **sessions** - Conversation metadata (project, tools used, timestamps)
- **skills** - MCP servers and agent capabilities registry
- **embeddings** - Vector representations for semantic search (future)

### Key Predicate Types

**Structural (Naive):**
- `requested_in_session` - Links tasks to sessions
- `used_tool` - Tool usage by agent
- `executed_command` - Bash commands executed

**Relational (Naive):**
- `connects_to_host` - Network connections made
- `targets_system` - Systems mentioned/targeted
- `mentions_path` - File paths referenced
- `operation_type` - Type of operation (verification, archive_manipulation, etc.)

**Knowledge (Naive):**
- `description` - Natural language task descriptions
- `provides_solution` - Solutions discovered
- `discovery` - New findings
- `identifies_issue` - Problems identified

**Semantic (Discerned):**
- `user_feedback_type` - User feedback style (corrective, approving, clarifying, rejecting)
- `assistant_tone` - LLM response tone (enthusiastic, cautious, direct, pedagogical)
- `session_intent` - Overall session goal/purpose
- `workflow_pattern` - Detected workflow (e.g., verify_then_cleanup, debug_cycle)
- `user_preference` - User preferences detected from behavior
- `problem_description` - Problems identified during session
- `solution_found` - Solutions to problems
- `knowledge_type` - Type of knowledge (procedural, declarative, troubleshooting)

## Common Development Commands

### Parsing Sessions

```bash
# Parse with naive extraction only (fast, free)
python session_parser.py path/to/session.jsonl

# Parse with discerned extraction (slow, insightful, requires LLM)
python session_parser.py path/to/session.jsonl --discerned

# Import session into memory database (naive only)
python agent_memory.py path/to/session.jsonl

# Import with discerned extraction
python agent_memory.py path/to/session.jsonl --discerned

# See DISCERNED_EXTRACTION.md for full documentation
```

### Querying Memory

```python
from agent_memory import AgentMemory

# Initialize memory
memory = AgentMemory("example_agent_memory.db")

# Query facts
facts = memory.query_facts(predicate='targets_system', object_value='unraid')

# Get entity knowledge
knowledge = memory.get_entity_knowledge('task_98b33414')

# Find related sessions
sessions = memory.find_related_sessions({'project': 'unraid_admin'})

# Get statistics
stats = memory.stats()
```

### Using Agent Memory

```bash
# Run practical examples
python agent_example.py

# Export to Datalog format
python -c "from agent_memory import AgentMemory; m = AgentMemory('example_agent_memory.db'); m.export_to_datalog('output.datalog')"
```

### Datalog Integration (requires PyDatalog)

```bash
# Install PyDatalog (optional)
pip install pyDatalog

# Use Datalog rules for reasoning (see agent_memory.py for examples)
```

## Code Architecture Notes

### Session Parser Design (session_parser.py)

The parser uses a two-phase extraction strategy:

**Phase 1: Naive Extraction (always runs)**
1. Extract session metadata (timestamps, tools used, project inference)
2. Extract user intents from user messages
3. Extract assistant actions from tool calls and responses
4. Analyze bash commands for deeper insights (SSH hosts, file operations)
5. Extract conversation patterns from user-assistant pairs

Entity extraction uses regex patterns for:
- File paths: `/[\w/\-_.]+`
- SSH hosts: `@([\w\.\-]+)`
- Commands: keyword matching for common tools

**Phase 2: Discerned Extraction (optional, --discerned flag)**
1. Sample significant conversation exchanges
2. Analyze feedback dynamics (user→LLM, LLM→user)
3. Extract session-level workflow patterns
4. Identify user preferences from behavior
5. Map problem-solution relationships
6. Classify session intent and knowledge type

See `naive_vs_sophisticated.md` for design rationale.

### Memory Storage Design (agent_memory.py)

Uses EAV pattern for flexibility:
- Subject: Entity ID (task_xxx, action_xxx, session_xxx, exchange_xxx)
- Predicate: Relationship type
- Object: Related value
- Metadata: confidence, source_session, timestamp, context, extraction_method

Indexes on subject, predicate, source_session, and timestamp enable fast queries.

All facts are tagged with `extraction_method` ('naive' or 'discerned') to distinguish data sources.

### Confidence Scoring

**Naive extraction:**
- 1.0: Direct extraction (tool usage, executed commands)
- 0.9: High-confidence inference (SSH host from command)
- 0.8: Medium-confidence patterns (operation types)
- 0.7: Entity mentions (systems, paths)
- 0.6: Discoveries from text analysis
- 0.5: Conversation patterns

**Discerned extraction:**
- 0.9: Session-level insights (intent, knowledge type)
- 0.85: Feedback dynamics, workflow patterns, problem-solution pairs
- 0.8: User preferences

Lower-confidence facts may need validation or decay over time.

## Future Extensions

The codebase is prepared for:
1. **Vector embeddings** - Table exists for semantic search integration
2. **PyDatalog reasoning** - Export format and example rules provided
3. **Multi-session learning** - Designed to aggregate across many sessions
4. **Confidence decay** - Temporal reasoning for fact aging
5. **Contradiction detection** - Identify conflicting facts

## Important Context

This system was built to enable LLM agents to maintain memory across sessions by extracting structured knowledge from Claude Code's JSONL session format. The design prioritizes:
- Lightweight storage (SQLite single file)
- Flexible schema (EAV pattern)
- Both structured queries (SQL) and reasoning (Datalog)
- Git-friendly export format
- No external service dependencies
