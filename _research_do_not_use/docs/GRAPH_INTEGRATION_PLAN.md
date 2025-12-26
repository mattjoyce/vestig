# SQLite-Graph Integration Plan

## Overview

Integrate [sqlite-graph](https://github.com/agentflare-ai/sqlite-graph) to use entities from naive extraction as proper graph nodes.

## Architecture

```
Session JSONL
     ↓
Naive Extraction → Entities (hosts, paths, tasks, actions, systems)
     ↓
┌────────────────────────────────────────┐
│  NODES (entities)                      │
│  • task_123 (type: task)               │
│  • action_456 (type: action)           │
│  • 192.168.20.4 (type: host)           │
│  • /mnt/backups/ (type: path)          │
│  • unraid (type: system)               │
│  • session_789 (type: session)         │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  EDGES (predicates)                    │
│  • task_123 --requested_in--> session  │
│  • action_456 --connects_to--> host    │
│  • action_456 --mentions--> path       │
│  • task_123 --targets--> system        │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  NODE ATTRIBUTES (from facts)          │
│  • description (text)                  │
│  • confidence (weight)                 │
│  • extraction_method (naive/discerned) │
│  • timestamp (temporal)                │
└────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────┐
│  EMBEDDINGS (semantic layer)           │
│  • Embed nodes (entities)              │
│  • Embed edges (relationships)         │
│  • Enable semantic graph traversal     │
└────────────────────────────────────────┘
```

## Entity Types from Naive Extraction

### Entities to Extract as Nodes

**From current naive extraction:**

1. **Tasks** - User requests
   - ID: `task_{uuid}`
   - Attributes: description, timestamp

2. **Actions** - Assistant actions
   - ID: `action_{uuid}`
   - Attributes: tool_used, timestamp

3. **Hosts** - SSH/network hosts
   - ID: IP or hostname (e.g., `192.168.20.4`)
   - Attributes: connection_count, last_accessed

4. **Paths** - File/directory paths
   - ID: Path string (e.g., `/mnt/backups/`)
   - Attributes: access_type (read/write)

5. **Systems** - Named systems
   - ID: System name (e.g., `unraid`)
   - Attributes: type (server/nas/etc)

6. **Commands** - Bash commands
   - ID: Command hash or ID
   - Attributes: command_text, execution_count

7. **Sessions** - Conversation sessions
   - ID: `session_{uuid}`
   - Attributes: project, start_time, intent

### Edge Types (Predicates)

**Task relationships:**
- `task --requested_in--> session`
- `task --targets--> system`
- `task --mentions--> path`
- `task --fulfilled_by--> action`

**Action relationships:**
- `action --used_tool--> tool`
- `action --executed_command--> command`
- `action --connects_to--> host`
- `action --accessed_file--> path`

**Discerned relationships:**
- `session --has_workflow--> workflow_pattern`
- `session --has_intent--> intent_description`
- `exchange --user_feedback--> feedback_type`
- `problem --solved_by--> solution`

## Implementation Steps

### 1. Install sqlite-graph

```bash
pip install sqlite-graph
```

### 2. Create Graph Schema

```python
from sqlite_graph import Graph, Node, Edge

class AgentMemoryGraph:
    def __init__(self, db_path: str):
        self.graph = Graph(db_path)
        self._initialize_schema()

    def _initialize_schema(self):
        # Define node types
        self.graph.create_node_type('task', {
            'description': 'TEXT',
            'timestamp': 'TEXT',
            'extraction_method': 'TEXT'
        })

        self.graph.create_node_type('action', {
            'tool': 'TEXT',
            'timestamp': 'TEXT'
        })

        self.graph.create_node_type('host', {
            'connection_count': 'INTEGER',
            'last_accessed': 'TEXT'
        })

        self.graph.create_node_type('path', {
            'access_type': 'TEXT'
        })

        self.graph.create_node_type('system', {
            'system_type': 'TEXT'
        })

        self.graph.create_node_type('session', {
            'project': 'TEXT',
            'start_time': 'TEXT',
            'intent': 'TEXT',
            'file_hash': 'TEXT'
        })

        # Define edge types
        self.graph.create_edge_type('requested_in')
        self.graph.create_edge_type('targets')
        self.graph.create_edge_type('mentions')
        self.graph.create_edge_type('connects_to')
        self.graph.create_edge_type('executed_command')
        self.graph.create_edge_type('fulfilled_by')
        self.graph.create_edge_type('has_workflow')
        self.graph.create_edge_type('user_feedback')
```

### 3. Update Session Parser

```python
class GraphSessionParser(ClaudeCodeSessionParser):
    def __init__(self, graph: AgentMemoryGraph, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph

    def parse_session_file(self, filepath: str):
        # Call parent to get facts
        result = super().parse_session_file(filepath)

        # Convert facts to graph nodes and edges
        self._build_graph_from_facts(result['facts'], result['metadata'])

        return result

    def _build_graph_from_facts(self, facts, metadata):
        # Create session node
        session_id = metadata['session_id']
        self.graph.add_node(
            id=session_id,
            type='session',
            project=metadata.get('project'),
            start_time=metadata.get('start_time')
        )

        # Process facts and create nodes/edges
        for fact in facts:
            if fact['extraction_method'] == 'naive':
                self._create_graph_entities(fact, session_id)

    def _create_graph_entities(self, fact, session_id):
        subject = fact['subject']
        predicate = fact['predicate']
        obj = fact['object']

        # Create nodes based on predicate type
        if predicate == 'connects_to_host':
            # Create action node
            self.graph.add_node(id=subject, type='action')
            # Create host node
            self.graph.add_node(id=obj, type='host')
            # Create edge
            self.graph.add_edge(
                from_id=subject,
                to_id=obj,
                type='connects_to',
                confidence=fact['confidence']
            )

        elif predicate == 'mentions_path':
            self.graph.add_node(id=subject, type='task')
            self.graph.add_node(id=obj, type='path')
            self.graph.add_edge(
                from_id=subject,
                to_id=obj,
                type='mentions'
            )

        elif predicate == 'targets_system':
            self.graph.add_node(id=subject, type='task')
            self.graph.add_node(id=obj, type='system')
            self.graph.add_edge(
                from_id=subject,
                to_id=obj,
                type='targets'
            )

        # ... handle other predicates
```

### 4. Graph Queries

```python
# Find all hosts connected from a session
hosts = graph.query("""
    MATCH (session:session)-[*]->(action:action)-[:connects_to]->(host:host)
    WHERE session.id = 'session_123'
    RETURN host
""")

# Find workflow patterns
workflows = graph.query("""
    MATCH (session:session)-[:has_workflow]->(pattern)
    WHERE session.project = 'unraid_admin'
    RETURN pattern, COUNT(*) as usage
    ORDER BY usage DESC
""")

# Find all tasks targeting a system
tasks = graph.query("""
    MATCH (task:task)-[:targets]->(system:system)
    WHERE system.id = 'unraid'
    RETURN task
""")

# Path finding: How did we get from task to host?
path = graph.query("""
    MATCH path = (task:task)-[*..5]->(host:host)
    WHERE task.id = 'task_123' AND host.id = '192.168.20.4'
    RETURN path
""")
```

### 5. Hybrid Querying (Graph + Semantic)

```python
class HybridMemory:
    def __init__(self, db_path: str):
        self.graph = AgentMemoryGraph(db_path)
        self.memory = AgentMemory(db_path)  # Keep for embeddings

    def find_relevant_context(self, query: str, method='hybrid'):
        results = {'graph': [], 'semantic': []}

        if method in ['hybrid', 'graph']:
            # Graph traversal (exact)
            # Extract entities from query
            entities = self._extract_entities(query)

            for entity in entities:
                # Find connected nodes
                connected = self.graph.query(f"""
                    MATCH (start)-[*..3]->(end)
                    WHERE start.id = '{entity}'
                    RETURN end
                """)
                results['graph'].extend(connected)

        if method in ['hybrid', 'semantic']:
            # Semantic search (fuzzy)
            similar = self.memory.find_similar_facts(query, top_k=5)
            results['semantic'] = similar

        return results
```

## Benefits of Graph Approach

### 1. Proper Entity Modeling
```python
# Before (string IDs):
subject = "action_123"

# After (graph nodes):
action = Node(id="action_123", type="action", tool="Bash", timestamp="...")
```

### 2. Graph Traversal
```python
# Find all sessions that used SSH to connect to unraid
graph.query("""
    MATCH (session:session)-[*]->(action:action)-[:connects_to]->(host:host)
    WHERE host.id = '192.168.20.4'
    RETURN session
""")
```

### 3. Path Finding
```python
# How did task X lead to connecting to host Y?
graph.query("""
    MATCH path = (task:task)-[*..10]->(host:host)
    WHERE task.description CONTAINS 'verify backup'
      AND host.id = '192.168.20.4'
    RETURN path
""")
```

### 4. Pattern Detection
```python
# What workflows involve this host?
graph.query("""
    MATCH (session:session)-[:has_workflow]->(workflow),
          (session)-[*]->(host:host {id: '192.168.20.4'})
    RETURN workflow, COUNT(*) as frequency
    ORDER BY frequency DESC
""")
```

## Migration Strategy

### Option 1: Full Migration
- Replace current facts table with graph
- Update all queries to use graph syntax

### Option 2: Hybrid (Recommended)
- Keep facts table for raw storage
- Build graph view on top
- Use graph for traversal, keep embeddings for semantic search

### Option 3: Parallel
- Run both systems side-by-side
- Graph for structural queries
- Facts + embeddings for semantic queries

## Next Steps

1. Install sqlite-graph
2. Create prototype with small session
3. Test graph queries vs SQL queries
4. Evaluate performance
5. Decide on migration strategy

## Considerations

**Pros:**
- ✅ Proper graph structure
- ✅ Native graph queries (more intuitive)
- ✅ Path finding built-in
- ✅ Better for relationship-heavy queries

**Cons:**
- ❓ Additional dependency (sqlite-graph)
- ❓ Migration effort
- ❓ Learning curve for graph queries

**Questions:**
- Does sqlite-graph support embeddings natively?
- Performance at scale (10k+ nodes)?
- Can we combine graph + semantic search efficiently?
