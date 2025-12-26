# Vestig RAG System

## Overview

Vestig now functions as a complete **Retrieval-Augmented Generation (RAG)** system, enabling agents to learn from past sessions and use semantic memory to improve responses.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Claude Code Session Files (JSONL)                   │
└──────────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│  Two-Pass Extraction                                  │
│  • Naive: Regex-based (fast, structured)             │
│  • Discerned: LLM-based (semantic, insights)         │
└──────────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│  Deduplication (SHA256 hash)                         │
│  • Skip already-processed sessions                    │
│  • Prevent duplicate facts                           │
└──────────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│  SQLite Memory Database                              │
│  • Facts (naive + discerned)                         │
│  • Sessions (with file hashes)                       │
│  • Embeddings (vector representations)               │
└──────────────────────────────────────────────────────┘
                      ↓
      ┌───────────────┴───────────────┐
      ↓                               ↓
┌──────────────┐              ┌──────────────┐
│ SQL Queries  │              │ RAG Retrieval│
│ (structured) │              │ (semantic)   │
└──────────────┘              └──────────────┘
```

## Key Features

### 1. Session Deduplication

**Problem Solved:** Prevents re-processing the same session files multiple times.

**How it works:**
- Computes SHA256 hash of session file
- Checks if hash exists in database before importing
- Skips duplicate sessions automatically

**Example:**
```python
memory = AgentMemory("my_memory.db")
parser = ClaudeCodeSessionParser(use_discerned=True)

session_data = parser.parse_session_file("session.jsonl")

# First import
memory.import_session(session_data, file_path="session.jsonl")
# → "✓ Imported session xyz with 38 facts"

# Second import (same file)
memory.import_session(session_data, file_path="session.jsonl")
# → "Session already imported (hash: a3186be7...), skipping"
```

### 2. Semantic Embeddings

**Problem Solved:** Find relevant memories by *meaning*, not just keywords.

**How it works:**
- Uses `llm embed` to generate vector embeddings for facts
- Stores embeddings in SQLite with pickled numpy arrays
- Enables cosine similarity search

**Supported Models:**
- `ada-002` - OpenAI text-embedding-ada-002 (1536 dimensions)
- Any model supported by `llm embed`

**Example:**
```python
memory = AgentMemory("my_memory.db")

# Generate embeddings for all facts
memory.embed_all_facts(model="ada-002")
# → "Generating embeddings for 28 facts..."
# → "✓ Generated 28 embeddings"
```

### 3. RAG Retrieval

**Problem Solved:** Retrieve relevant context from past sessions to augment LLM prompts.

**How it works:**
1. User asks a question
2. System embeds the question
3. Finds facts with highest cosine similarity
4. Returns top-k most relevant memories
5. Use memories to build augmented prompt

**Example:**
```python
# User asks a question
query = "How do I verify backup archives?"

# Find relevant memories
results = memory.find_similar_facts(
    query_text=query,
    top_k=5,
    model="ada-002",
    min_confidence=0.5
)

# Results (sorted by similarity):
# [1] 0.869 similarity: "description: verify that archive is not corrupted"
# [2] 0.861 similarity: "problem_description: Verify integrity of backup"
# [3] 0.854 similarity: "task_pattern: request->tools:Bash"
```

### 4. Hybrid Memory (Naive + Discerned)

**Strength:** Combines structured facts with semantic insights.

RAG retrieval searches **both** naive and discerned facts together:

**Naive facts provide:**
- Concrete commands: `executed_command: ssh root@192.168.20.4 ...`
- File paths: `mentions_path: /mnt/backups/`
- Connections: `connects_to_host: 192.168.20.4`

**Discerned facts provide:**
- Intent: `session_intent: verify server accessibility and backup integrity`
- Workflow: `workflow_pattern: connectivity_check_then_file_verification`
- Feedback: `user_feedback_type: approving`

**Together:** Answer "what worked before and why"

## Usage

### Setup

```bash
# 1. Install dependencies
pip install numpy

# 2. Configure llm with OpenAI API key
llm keys set openai

# 3. Test embedding
llm embed -m ada-002 -c "test"
```

### Import Sessions with Deduplication

```python
from agent_memory import AgentMemory
from session_parser import ClaudeCodeSessionParser

memory = AgentMemory("agent_memory.db")
parser = ClaudeCodeSessionParser(use_discerned=True)

# Import sessions (skips duplicates automatically)
for session_file in glob.glob("sessions/*.jsonl"):
    session_data = parser.parse_session_file(session_file)
    memory.import_session(session_data, file_path=session_file)
    # Only new sessions will be imported
```

### Generate Embeddings

```python
# Embed all facts (skips already-embedded facts)
memory.embed_all_facts(model="ada-002", batch_size=10)

# Embed specific fact
memory.embed_fact(fact_id=42, model="ada-002")
```

### RAG Query

```python
# Find relevant memories
memories = memory.find_similar_facts(
    query_text="How should I verify my unraid backups?",
    top_k=5,
    model="ada-002",
    min_confidence=0.5,
    extraction_method=None  # or "naive" or "discerned"
)

# Build augmented prompt
prompt = f"""You have access to these relevant memories:

{chr(10).join([
    f"- {m['predicate']}: {m['object']} (similarity: {m['similarity']:.2f})"
    for m in memories
])}

User question: How should I verify my unraid backups?

Provide a helpful answer using these memories."""

# Send to LLM with context
response = llm.generate(prompt)
```

## Demo

Run the complete RAG demo:

```bash
source ~/Environments/vestig/bin/activate
python rag_demo.py
```

**Demo shows:**
1. Session deduplication (import twice, second is skipped)
2. Embedding generation (28 facts embedded)
3. Semantic search (3 different queries)
4. RAG-augmented prompt construction

**Sample Output:**

```
📝 QUERY: "How do I verify a backup archive?"

  [1] Similarity: 0.869 | Method: naive | Confidence: 1.0
      description: verify that archive is not corrupted

  [2] Similarity: 0.861 | Method: discerned | Confidence: 0.85
      problem_description: Verify the integrity of the backup file

  [3] Similarity: 0.854 | Method: naive | Confidence: 0.5
      task_pattern: request->tools:Bash
```

## Performance

### Deduplication
- **Time:** <1ms per session (hash lookup)
- **Space:** 64 bytes per session (SHA256)
- **Benefit:** Prevents wasting time/money re-processing

### Embeddings
- **Time:** ~0.5-2s per fact (OpenAI API call)
- **Cost:** ~$0.0001 per fact (ada-002)
- **Space:** ~6KB per embedding (1536 floats)
- **For 1000 facts:**
  - Generation time: ~10-30 minutes
  - Cost: ~$0.10
  - Storage: ~6MB

### RAG Retrieval
- **Time:** <100ms for top-5 search across 1000 facts
- **Cost:** ~$0.0001 per query (embedding query text)
- **Accuracy:** 0.85-0.90 semantic similarity for relevant results

## Best Practices

### 1. Batch Import Sessions

```python
# Process all sessions at once
import glob

for session_file in glob.glob("~/.claude/sessions/*.jsonl"):
    session_data = parser.parse_session_file(session_file)
    memory.import_session(session_data, file_path=session_file)
    # Duplicates automatically skipped
```

### 2. Lazy Embedding Generation

```python
# Don't embed immediately - wait until you have many sessions
# Then batch generate embeddings
memory.embed_all_facts(model="ada-002")
```

### 3. Filter by Extraction Method

```python
# For structured queries, use naive facts only
naive_facts = memory.find_similar_facts(
    query_text="What SSH commands were used?",
    extraction_method="naive"
)

# For semantic queries, use discerned facts
semantic_facts = memory.find_similar_facts(
    query_text="What workflows got user approval?",
    extraction_method="discerned"
)

# Or search both (default)
all_facts = memory.find_similar_facts(
    query_text="How to verify backups?",
    extraction_method=None  # searches both
)
```

### 4. Confidence Filtering

```python
# Only retrieve high-confidence facts
facts = memory.find_similar_facts(
    query_text="...",
    min_confidence=0.8  # Only facts with confidence >= 0.8
)
```

## Agent Integration Example

```python
class MemoryAugmentedAgent:
    def __init__(self, memory_db: str):
        self.memory = AgentMemory(memory_db)

    def answer_question(self, question: str) -> str:
        # 1. Retrieve relevant memories
        memories = self.memory.find_similar_facts(
            query_text=question,
            top_k=5,
            min_confidence=0.7
        )

        # 2. Build context from memories
        context = []
        for mem in memories:
            context.append(
                f"• {mem['predicate']}: {mem['object']} "
                f"(from session {mem['source_session']})"
            )

        # 3. Augment prompt with memories
        prompt = f"""You are an AI assistant with access to memories from past sessions.

RELEVANT CONTEXT:
{chr(10).join(context)}

USER QUESTION:
{question}

Provide a helpful answer using the context above."""

        # 4. Generate response with LLM
        response = llm.generate(prompt)
        return response

# Usage
agent = MemoryAugmentedAgent("agent_memory.db")
answer = agent.answer_question("How do I verify my unraid backups?")
# → Leverages past sessions to provide contextual answer
```

## Troubleshooting

### "llm command not found"
```bash
pip install llm
```

### "No API key configured"
```bash
llm keys set openai
# Enter your OpenAI API key
```

### "Embedding generation failed"
Check `llm` is configured:
```bash
llm embed -m ada-002 -c "test"
```

### "No embeddings found"
Generate embeddings first:
```python
memory.embed_all_facts(model="ada-002")
```

### Slow RAG queries
**Problem:** Searching 10,000+ facts is slow (linear scan)

**Solutions:**
1. Use higher min_confidence to filter facts
2. Add FAISS/Annoy for approximate nearest neighbor search
3. Pre-filter by session metadata (project, date range)

## Future Enhancements

- [ ] FAISS/Annoy integration for faster similarity search
- [ ] Batch embedding API calls (reduce latency)
- [ ] Automatic re-embedding when facts change
- [ ] Temporal decay (older facts get lower weight)
- [ ] Cross-session pattern mining
- [ ] Multi-modal embeddings (code + text)
- [ ] Active learning (track which retrievals were useful)

## Summary

**Vestig RAG System provides:**

✅ **Deduplication** - Never re-process the same session twice
✅ **Dual Extraction** - Naive (structured) + Discerned (semantic) facts
✅ **Semantic Search** - Find relevant memories by meaning
✅ **RAG Ready** - Augment LLM prompts with retrieved context
✅ **Hybrid Memory** - Combines "what happened" with "why it happened"

**Use cases:**
- **Learning from past sessions:** "What worked last time?"
- **User preference detection:** "What does the user prefer?"
- **Workflow recommendation:** "What's the best approach for task X?"
- **Problem-solution matching:** "How did we solve similar problems?"
- **Context-aware assistance:** "Remember what we did with the unraid server?"

**The result:** Agents that get smarter over time by learning from experience.
