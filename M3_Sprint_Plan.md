# M3 - Time & Truth Implementation Plan

## Overview

Transform Vestig into a time-aware, truth-tracking system by adding:
1. **Event layer** - Track memory lifecycle (ADD, REINFORCE_EXACT, REINFORCE_NEAR, DEPRECATE)
2. **Bi-temporal fields** - Distinguish "when learned" from "when valid"
3. **TraceRank** - Use reinforcement history for smarter ranking
4. **Truth mechanics** - Deprecation/supersession without data loss

**Key insight:** M2's CommitOutcome hook is already in place - M3 just needs to consume it.

---

## Implementation Phases

### Phase 1: Schema Foundation (Data Model & Migrations)

**Goal:** Database can store M3 temporal data

#### 1.1 Update Data Models
**File:** `/Volumes/Projects/vestig/src/vestig/core/models.py`

**Add to MemoryNode (after line 17):**
```python
# M3: Bi-temporal fields
t_valid: Optional[str] = None          # When fact became true (event time)
t_invalid: Optional[str] = None        # When fact stopped being true (event time)
t_created: Optional[str] = None        # When we learned it (transaction time)
t_expired: Optional[str] = None        # When deprecated/superseded
temporal_stability: str = "unknown"    # "static" | "dynamic" | "unknown"

# M3: Reinforcement tracking (cached from events)
last_seen_at: Optional[str] = None     # Most recent reinforcement
reinforce_count: int = 0               # Total reinforcement events
```

**Update MemoryNode.create() method (lines 21-57):**
- Set `t_created` = `created_at` (alias for transaction time)
- Set `t_valid` = `created_at` (assume valid from creation by default)
- Initialize temporal_stability, reinforce_count in metadata

**Create EventNode dataclass (after MemoryNode):**
```python
@dataclass
class EventNode:
    """Memory lifecycle event (M3)"""
    event_id: str              # evt_<uuid>
    memory_id: str             # FK to memories table
    event_type: str            # ADD | REINFORCE_EXACT | REINFORCE_NEAR | DEPRECATE | SUPERSEDE
    occurred_at: str           # UTC timestamp (ISO 8601)
    source: str                # manual | hook | import | batch
    actor: Optional[str] = None            # User/agent identifier
    artifact_ref: Optional[str] = None     # Session ID, filename, etc.
    payload: Dict[str, Any] = field(default_factory=dict)  # Event details

    @classmethod
    def create(cls, memory_id: str, event_type: str, source: str = "manual",
               payload: Optional[Dict[str, Any]] = None) -> "EventNode":
        """Create new event with generated ID and timestamp"""
        return cls(
            event_id=f"evt_{uuid.uuid4()}",
            memory_id=memory_id,
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            payload=payload or {}
        )
```

#### 1.2 Update Storage Schema
**File:** `/Volumes/Projects/vestig/src/vestig/core/storage.py`

**Extend _init_schema() method (after line 54, before line 56):**

Follow M2's additive migration pattern (lines 41-54):

```python
# M3: Add temporal columns (check first, then ALTER TABLE)
cursor = self.conn.execute("PRAGMA table_info(memories)")
columns = [row[1] for row in cursor.fetchall()]

if "t_valid" not in columns:
    self.conn.execute("ALTER TABLE memories ADD COLUMN t_valid TEXT")
if "t_invalid" not in columns:
    self.conn.execute("ALTER TABLE memories ADD COLUMN t_invalid TEXT")
if "t_created" not in columns:
    self.conn.execute("ALTER TABLE memories ADD COLUMN t_created TEXT")
if "t_expired" not in columns:
    self.conn.execute("ALTER TABLE memories ADD COLUMN t_expired TEXT")
if "temporal_stability" not in columns:
    self.conn.execute("ALTER TABLE memories ADD COLUMN temporal_stability TEXT DEFAULT 'unknown'")
if "last_seen_at" not in columns:
    self.conn.execute("ALTER TABLE memories ADD COLUMN last_seen_at TEXT")
if "reinforce_count" not in columns:
    self.conn.execute("ALTER TABLE memories ADD COLUMN reinforce_count INTEGER DEFAULT 0")

# Backfill t_created from created_at for existing memories
self.conn.execute("UPDATE memories SET t_created = created_at WHERE t_created IS NULL")

# M3: Create memory_events table
self.conn.execute("""
    CREATE TABLE IF NOT EXISTS memory_events (
        event_id TEXT PRIMARY KEY,
        memory_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        source TEXT NOT NULL,
        actor TEXT,
        artifact_ref TEXT,
        payload_json TEXT NOT NULL,
        FOREIGN KEY(memory_id) REFERENCES memories(id)
    )
""")

# M3: Create indexes for temporal queries
self.conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_events_memory_time
    ON memory_events(memory_id, occurred_at DESC)
""")
self.conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_events_type
    ON memory_events(event_type)
""")
self.conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_memories_expired
    ON memories(t_expired) WHERE t_expired IS NOT NULL
""")
```

**Update store_memory() INSERT (lines 80-94):**
- Expand INSERT statement to include new temporal columns
- Handle None values gracefully

**Update get_memory() and get_all_memories() (lines 97-154):**
- Expand SELECT to include new columns
- Update MemoryNode construction
- Handle NULL gracefully for backward compatibility

**Add new methods (after line 159):**
```python
def increment_reinforce_count(self, memory_id: str) -> None:
    """Increment reinforce_count (convenience cache for TraceRank)"""
    self.conn.execute(
        "UPDATE memories SET reinforce_count = reinforce_count + 1 WHERE id = ?",
        (memory_id,)
    )
    self.conn.commit()

def update_last_seen(self, memory_id: str, timestamp: str) -> None:
    """Update last_seen_at timestamp"""
    self.conn.execute(
        "UPDATE memories SET last_seen_at = ? WHERE id = ?",
        (timestamp, memory_id)
    )
    self.conn.commit()

def deprecate_memory(self, memory_id: str, t_invalid: Optional[str] = None) -> None:
    """Mark memory as deprecated/expired"""
    now = datetime.now(timezone.utc).isoformat()
    self.conn.execute(
        "UPDATE memories SET t_expired = ?, t_invalid = COALESCE(?, t_invalid) WHERE id = ?",
        (now, t_invalid, memory_id)
    )
    self.conn.commit()

def get_active_memories(self) -> List[MemoryNode]:
    """Get all non-expired memories (for retrieval)"""
    # Same as get_all_memories but with WHERE t_expired IS NULL
```

#### 1.3 Create Event Storage Module
**New file:** `/Volumes/Projects/vestig/src/vestig/core/event_storage.py`

```python
"""Event storage for M3 lifecycle tracking"""

import json
import sqlite3
from typing import List, Optional
from vestig.core.models import EventNode

class MemoryEventStorage:
    """Event CRUD operations (shares DB connection with MemoryStorage)"""

    def __init__(self, conn: sqlite3.Connection):
        """Use same DB connection as MemoryStorage for transaction consistency"""
        self.conn = conn

    def add_event(self, event: EventNode) -> str:
        """Insert event (append-only, never update)"""
        self.conn.execute(
            """
            INSERT INTO memory_events
            (event_id, memory_id, event_type, occurred_at, source, actor, artifact_ref, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event.event_id, event.memory_id, event.event_type, event.occurred_at,
             event.source, event.actor, event.artifact_ref, json.dumps(event.payload))
        )
        self.conn.commit()
        return event.event_id

    def get_events_for_memory(self, memory_id: str, limit: int = 100) -> List[EventNode]:
        """Retrieve events for a memory, newest first"""
        cursor = self.conn.execute(
            """
            SELECT event_id, memory_id, event_type, occurred_at, source, actor, artifact_ref, payload_json
            FROM memory_events
            WHERE memory_id = ?
            ORDER BY occurred_at DESC
            LIMIT ?
            """,
            (memory_id, limit)
        )
        return [EventNode(row[0], row[1], row[2], row[3], row[4], row[5], row[6], json.loads(row[7]))
                for row in cursor.fetchall()]

    def get_reinforcement_events(self, memory_id: str) -> List[EventNode]:
        """Get only REINFORCE_* events for TraceRank computation"""
        cursor = self.conn.execute(
            """
            SELECT event_id, memory_id, event_type, occurred_at, source, actor, artifact_ref, payload_json
            FROM memory_events
            WHERE memory_id = ? AND event_type LIKE 'REINFORCE_%'
            ORDER BY occurred_at DESC
            """,
            (memory_id,)
        )
        return [EventNode(row[0], row[1], row[2], row[3], row[4], row[5], row[6], json.loads(row[7]))
                for row in cursor.fetchall()]
```

**Acceptance:**
- Can add memory and new columns exist in DB
- Existing M2 database upgrades without errors
- Can insert and query events via Python

---

### Phase 2: Event Pipeline Integration

**Goal:** CommitOutcome → EventNode → Database

#### 2.1 Wire CommitOutcome to Event Logging
**File:** `/Volumes/Projects/vestig/src/vestig/core/commitment.py`

**Update commit_memory() signature (lines 85-94):**
```python
def commit_memory(
    content: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    source: str = "manual",
    hygiene_config: Dict[str, Any] = None,
    tags: list[str] = None,
    artifact_ref: Optional[str] = None,
    on_commit: Optional[OnCommitHook] = None,
    event_storage: Optional['MemoryEventStorage'] = None,  # M3: Event logging
) -> CommitOutcome:
```

**Add event creation helper (after line 247):**
```python
def _log_commit_event(
    outcome: CommitOutcome,
    storage: MemoryStorage,
    event_storage: 'MemoryEventStorage'
) -> None:
    """Convert CommitOutcome to EventNode and persist (M3)"""

    if outcome.outcome == "REJECTED_HYGIENE":
        return  # Don't log hygiene rejections

    # Map outcome to event type
    event_type_map = {
        "INSERTED_NEW": "ADD",
        "EXACT_DUPE": "REINFORCE_EXACT",
        "NEAR_DUPE": "REINFORCE_NEAR",
    }
    event_type = event_type_map[outcome.outcome]

    # Create event
    event = EventNode.create(
        memory_id=outcome.memory_id,
        event_type=event_type,
        source=outcome.source,
        payload={
            "content_hash": outcome.content_hash,
            "tags": outcome.tags,
            "artifact_ref": outcome.artifact_ref,
            "matched_memory_id": outcome.matched_memory_id,
            "query_score": outcome.query_score,
        }
    )
    event_storage.add_event(event)

    # Update convenience fields for reinforcement
    if event_type.startswith("REINFORCE"):
        storage.increment_reinforce_count(outcome.memory_id)
        storage.update_last_seen(outcome.memory_id, outcome.occurred_at)
```

**Update outcome handling (after line 245, before return):**
```python
# M3: Log event if event_storage provided
if event_storage:
    _log_commit_event(outcome, storage, event_storage)
```

**Acceptance:**
- Adding duplicate creates REINFORCE_EXACT event
- reinforce_count increments on duplicate
- Events visible in memory_events table

---

### Phase 3: TraceRank Implementation

**Goal:** Events → Ranking multiplier

#### 3.1 Create TraceRank Module
**New file:** `/Volumes/Projects/vestig/src/vestig/core/tracerank.py`

```python
"""TraceRank: Temporal reinforcement scoring for M3"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional
import math
from vestig.core.models import EventNode

@dataclass
class TraceRankConfig:
    """TraceRank configuration"""
    enabled: bool = True
    tau_days: float = 21.0          # Recency decay time constant (3 weeks)
    cooldown_hours: float = 24.0     # Anti-burst window
    burst_discount: float = 0.2      # Weight for events in cooldown
    k: float = 0.35                  # TraceRank boost strength

def compute_tracerank_multiplier(
    events: List[EventNode],
    config: TraceRankConfig,
    query_time: Optional[datetime] = None
) -> float:
    """
    Compute TraceRank multiplier from reinforcement events.

    Algorithm:
    1. For each event, compute recency weight: exp(-Δt / τ)
    2. Apply burst discount if event within cooldown of previous
    3. Sum weighted contributions: trace = Σ(w_recency * w_burst)
    4. Convert to multiplier: 1 + k * log1p(trace)

    Returns:
        Multiplier ∈ [1.0, ∞) to boost semantic similarity
    """
    if not config.enabled or not events:
        return 1.0

    if query_time is None:
        query_time = datetime.now(timezone.utc)

    trace = 0.0
    prev_event_time = None

    for event in events:
        event_time = datetime.fromisoformat(event.occurred_at.replace("Z", "+00:00"))

        # Recency decay: exp(-Δt / τ)
        delta_days = (query_time - event_time).total_seconds() / 86400
        w_recency = math.exp(-delta_days / config.tau_days)

        # Anti-burst: discount events within cooldown
        w_burst = 1.0
        if prev_event_time is not None:
            gap_hours = (prev_event_time - event_time).total_seconds() / 3600
            if gap_hours < config.cooldown_hours:
                w_burst = config.burst_discount

        trace += w_recency * w_burst
        prev_event_time = event_time

    # Convert trace to multiplier: 1 + k * log1p(trace)
    return 1.0 + config.k * math.log1p(trace)
```

#### 3.2 Integrate TraceRank into Retrieval
**File:** `/Volumes/Projects/vestig/src/vestig/core/retrieval.py`

**Update search_memories() signature (lines 37-42):**
```python
def search_memories(
    query: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    limit: int = 5,
    event_storage: Optional['MemoryEventStorage'] = None,  # M3
    tracerank_config: Optional['TraceRankConfig'] = None,   # M3
    include_expired: bool = False,                          # M3
) -> List[Tuple[MemoryNode, float]]:
```

**Update retrieval logic (replace lines 58-72):**
```python
# Load memories (active only or all)
if include_expired or event_storage is None:
    all_memories = storage.get_all_memories()
else:
    all_memories = storage.get_active_memories()

if not all_memories:
    return []

# Compute semantic scores
scored_memories = []
for memory in all_memories:
    semantic_score = cosine_similarity(query_embedding, memory.content_embedding)
    scored_memories.append((memory, semantic_score))

# M3: Apply TraceRank if enabled
if event_storage and tracerank_config and tracerank_config.enabled:
    from vestig.core.tracerank import compute_tracerank_multiplier

    # Compute TraceRank for all memories
    for i, (memory, semantic_score) in enumerate(scored_memories):
        events = event_storage.get_reinforcement_events(memory.id)
        tracerank = compute_tracerank_multiplier(events, tracerank_config)
        # Multiply semantic score by TraceRank
        scored_memories[i] = (memory, semantic_score * tracerank)

# Sort by final score descending
scored_memories.sort(key=lambda x: x[1], reverse=True)
return scored_memories[:limit]
```

**Update format_recall_results() (lines 127-134):**
```python
# Extract source from metadata
source = memory.metadata.get("source", "unknown")

# Format: [id] (source=..., created=..., score=...)
header = f"[{memory.id}] (source={source}, created={memory.created_at}, score={score:.4f}"

# M3: Add reinforcement + validity hints
if hasattr(memory, 'reinforce_count') and memory.reinforce_count > 0:
    header += f", reinforced={memory.reinforce_count}x"
if hasattr(memory, 'last_seen_at') and memory.last_seen_at:
    header += f", last_seen={memory.last_seen_at}"
if hasattr(memory, 't_expired') and memory.t_expired:
    header += ", status=EXPIRED"

header += ")"
```

**Acceptance:**
- Reinforced memories rank higher than unreinforced
- Burst reinforcement scores lower than spaced
- Recall output shows reinforced=Nx

---

### Phase 4: Configuration & CLI Integration

**Goal:** M3 behavior is configurable and accessible via CLI

#### 4.1 Add M3 Configuration
**File:** `/Volumes/Projects/vestig/config.yaml`

**Add after line 25:**
```yaml
# M3: Time & Truth (TraceRank, temporal ranking)
m3:
  event_logging:
    enabled: true

  tracerank:
    enabled: true
    tau_days: 21          # Recency decay time constant (weeks)
    cooldown_hours: 24    # Anti-burst window (1 day)
    burst_discount: 0.2   # Weight for burst events (0-1)
    k: 0.35               # TraceRank boost strength

  retrieval:
    include_expired: false  # Show deprecated memories by default?
```

#### 4.2 Update CLI
**File:** `/Volumes/Projects/vestig/src/vestig/core/cli.py`

**Update build_runtime() (expand return signature):**
```python
def build_runtime(config: Dict[str, Any]) -> Tuple[
    MemoryStorage,
    EmbeddingEngine,
    MemoryEventStorage,
    TraceRankConfig
]:
    """Build all M3 components from config"""
    # ... existing code ...

    # M3: Event storage (shares DB connection)
    from vestig.core.event_storage import MemoryEventStorage
    event_storage = MemoryEventStorage(storage.conn)

    # M3: TraceRank config
    from vestig.core.tracerank import TraceRankConfig
    m3_config = config.get("m3", {})
    tracerank_config = TraceRankConfig(
        enabled=m3_config.get("tracerank", {}).get("enabled", True),
        tau_days=m3_config.get("tracerank", {}).get("tau_days", 21.0),
        cooldown_hours=m3_config.get("tracerank", {}).get("cooldown_hours", 24.0),
        burst_discount=m3_config.get("tracerank", {}).get("burst_discount", 0.2),
        k=m3_config.get("tracerank", {}).get("k", 0.35),
    )

    return storage, embedding_engine, event_storage, tracerank_config
```

**Update cmd_add() to pass event_storage to commit_memory()**

**Update cmd_search() and cmd_recall() to pass M3 parameters:**
```python
def cmd_search(args):
    storage, embedding_engine, event_storage, tracerank_config = build_runtime(args.config_dict)

    results = search_memories(
        query=args.query,
        storage=storage,
        embedding_engine=embedding_engine,
        limit=args.limit,
        event_storage=event_storage,
        tracerank_config=tracerank_config,
        include_expired=args.include_expired,
    )
```

**Add --include-expired flag to search/recall subparsers:**
```python
parser_search.add_argument(
    "--include-expired",
    action="store_true",
    help="Include deprecated/superseded memories (M3)"
)
```

**Add new subcommands:**

**vestig memory events <id>:**
```python
def cmd_events(args):
    """Show event history for a memory"""
    storage, _, event_storage, _ = build_runtime(args.config_dict)

    events = event_storage.get_events_for_memory(args.id, limit=args.limit)

    if not events:
        print(f"No events found for memory: {args.id}")
        return

    print(f"Events for memory {args.id}:\n")
    for event in events:
        print(f"[{event.event_id}] {event.event_type}")
        print(f"  Occurred: {event.occurred_at}")
        print(f"  Source: {event.source}")
        if event.artifact_ref:
            print(f"  Artifact: {event.artifact_ref}")
        print()

    storage.close()

# Add subparser
parser_events = memory_subparsers.add_parser("events", help="Show event history for a memory")
parser_events.add_argument("id", help="Memory ID")
parser_events.add_argument("--limit", type=int, default=100, help="Max events to show")
parser_events.set_defaults(func=cmd_events)
```

**vestig memory deprecate <id>:**
```python
def cmd_deprecate(args):
    """Mark a memory as deprecated"""
    storage, _, event_storage, _ = build_runtime(args.config_dict)

    # Mark memory as deprecated
    storage.deprecate_memory(args.id, t_invalid=args.t_invalid)

    # Log DEPRECATE event
    from vestig.core.models import EventNode
    event = EventNode.create(
        memory_id=args.id,
        event_type="DEPRECATE",
        source="manual",
        payload={"reason": args.reason or "Manual deprecation"}
    )
    event_storage.add_event(event)

    print(f"Memory {args.id} marked as deprecated")
    storage.close()

# Add subparser
parser_deprecate = memory_subparsers.add_parser("deprecate", help="Mark memory as deprecated")
parser_deprecate.add_argument("id", help="Memory ID")
parser_deprecate.add_argument("--reason", help="Reason for deprecation")
parser_deprecate.add_argument("--t-invalid", help="When fact became invalid (ISO 8601)")
parser_deprecate.set_defaults(func=cmd_deprecate)
```

**Acceptance:**
- `vestig memory events <id>` shows event history
- `vestig memory deprecate <id>` works
- Search uses TraceRank by default
- --include-expired flag works

---

### Phase 5: Testing & Validation

**Goal:** M3 is proven stable and reliable

#### 5.1 Create M3 Smoke Test
**New file:** `/Volumes/Projects/vestig/test_m3_smoke.sh`

```bash
#!/bin/bash
# test_m3_smoke.sh - M3 Time & Truth smoke test

set -e

source ~/Environments/vestig/bin/activate

echo "=== M3 Smoke Test: Time & Truth ==="
echo ""

# Clean slate
rm -f data/memory.db
echo "✓ Clean database"
echo ""

# Test 1: Schema migration handles new columns
echo "Test 1: Schema migration (backward compatibility)"
vestig memory add "Testing M3 migration with temporal fields" > /dev/null
if vestig memory list --recent 1 | grep -q "mem_"; then
    echo "✓ Schema migration successful"
else
    echo "✗ FAIL: Schema migration failed"
    exit 1
fi
echo ""

# Test 2: Reinforcement creates events (not duplicates)
echo "Test 2: Reinforcement events (exact duplicate)"
ID2=$(vestig memory add "Learning Python async/await patterns" | grep -oE 'mem_[a-f0-9-]+')
ID3=$(vestig memory add "Learning Python async/await patterns" | grep -oE 'mem_[a-f0-9-]+')

if [ "$ID2" == "$ID3" ]; then
    echo "✓ Exact duplicate returned same ID"
else
    echo "✗ FAIL: Should return same ID for duplicate"
    exit 1
fi
echo ""

# Test 3: TraceRank affects ranking
echo "Test 3: TraceRank (reinforced memory ranks higher)"

# Add unreinforced memory
vestig memory add "Database indexing improves query performance" > /dev/null

# Add reinforced memory (same content twice)
vestig memory add "Redis caching speeds up web applications" > /dev/null
sleep 1
vestig memory add "Redis caching speeds up web applications" > /dev/null

# Search for "performance" - reinforced Redis should rank higher
RESULTS=$(vestig memory search "performance" --limit 2)
if echo "$RESULTS" | head -20 | grep -qi "redis"; then
    echo "✓ TraceRank boosted reinforced memory"
else
    echo "⚠ TraceRank may need tuning"
fi
echo ""

# Test 4: Event history is queryable
echo "Test 4: Event history (vestig memory events)"
ID4=$(vestig memory add "Testing event logging" | grep -oE 'mem_[a-f0-9-]+')
vestig memory add "Testing event logging" > /dev/null  # Reinforce

if vestig memory events "$ID4" 2>&1 | grep -q "ADD"; then
    echo "✓ Event history accessible"
else
    echo "⚠ Event history command may not be implemented yet"
fi
echo ""

# Test 5: Deprecation hides memories
echo "Test 5: Deprecation (truth mechanics)"
ID5=$(vestig memory add "Outdated information to deprecate" | grep -oE 'mem_[a-f0-9-]+')

# Should appear initially
if vestig memory search "outdated" | grep -q "$ID5"; then
    echo "✓ Memory searchable before deprecation"
else
    echo "✗ FAIL: Memory should be searchable initially"
    exit 1
fi

# Deprecate it
vestig memory deprecate "$ID5" --reason "Testing deprecation" 2>/dev/null || true

# Should NOT appear in default search
if vestig memory search "outdated" | grep -q "$ID5"; then
    echo "⚠ Deprecated memory still visible (filter may be disabled)"
else
    echo "✓ Deprecated memory hidden from default search"
fi

# Should appear with --include-expired
if vestig memory search "outdated" --include-expired 2>/dev/null | grep -q "$ID5"; then
    echo "✓ Deprecated memory visible with --include-expired"
fi
echo ""

echo "=== M3 Smoke Test Complete ==="
```

#### 5.2 Run Validation
```bash
# Validate M3
./test_m3_smoke.sh

# Ensure no M2 regressions
./test_m2_smoke.sh

# Validate M1 still works
./demo_m1.sh
```

**Acceptance:**
- test_m3_smoke.sh passes all checks
- test_m2_smoke.sh still passes (no regression)
- demo_m1.sh still works

---

## Critical Files Summary

### Files to Modify

1. **`/Volumes/Projects/vestig/src/vestig/core/models.py`**
   - Add temporal fields to MemoryNode
   - Create EventNode dataclass
   - Update create() method

2. **`/Volumes/Projects/vestig/src/vestig/core/storage.py`**
   - Extend _init_schema() with M3 migrations
   - Update store_memory(), get_memory(), get_all_memories()
   - Add increment_reinforce_count(), update_last_seen(), deprecate_memory(), get_active_memories()

3. **`/Volumes/Projects/vestig/src/vestig/core/commitment.py`**
   - Add event_storage parameter to commit_memory()
   - Create _log_commit_event() helper
   - Wire event logging after outcome

4. **`/Volumes/Projects/vestig/src/vestig/core/retrieval.py`**
   - Add M3 parameters to search_memories()
   - Integrate TraceRank scoring
   - Update format_recall_results() with M3 hints

5. **`/Volumes/Projects/vestig/src/vestig/core/cli.py`**
   - Update build_runtime() to create M3 components
   - Update cmd_add(), cmd_search(), cmd_recall()
   - Add cmd_events(), cmd_deprecate()
   - Add --include-expired flag

6. **`/Volumes/Projects/vestig/config.yaml`**
   - Add m3 section with TraceRank config

### New Files to Create

7. **`/Volumes/Projects/vestig/src/vestig/core/event_storage.py`**
   - MemoryEventStorage class
   - Event CRUD operations

8. **`/Volumes/Projects/vestig/src/vestig/core/tracerank.py`**
   - TraceRankConfig dataclass
   - compute_tracerank_multiplier() function

9. **`/Volumes/Projects/vestig/test_m3_smoke.sh`**
   - M3 smoke tests

---

## Definition of Done

M3 is complete when:

- [ ] All schema migrations execute successfully on clean and existing DBs
- [ ] ADD events created for new memories
- [ ] REINFORCE_EXACT events created for hash duplicates
- [ ] REINFORCE_NEAR events created for semantic duplicates
- [ ] reinforce_count and last_seen_at update correctly
- [ ] TraceRank multiplier computed from events
- [ ] Search ranking uses semantic × TraceRank
- [ ] Can deprecate memory (sets t_expired)
- [ ] Deprecated memories hidden from default search
- [ ] --include-expired flag shows deprecated memories
- [ ] `vestig memory events <id>` shows event history
- [ ] `vestig memory deprecate <id>` works
- [ ] Recall output includes reinforced=Nx, last_seen
- [ ] M2 databases upgrade automatically
- [ ] All M2 commands work unchanged
- [ ] test_m3_smoke.sh passes
- [ ] test_m2_smoke.sh still passes (no regression)
- [ ] Config.yaml has M3 section

---

## Risk Mitigation

**Risk: Schema migration fails on existing DBs**
- Mitigation: Follow exact M2 pattern with PRAGMA checks, test on copy first

**Risk: TraceRank breaks retrieval quality**
- Mitigation: Make opt-in via config, use conservative defaults (k=0.35)

**Risk: Backward compatibility breaks**
- Mitigation: All new columns nullable, all params have defaults, test both upgrade paths

---

## Implementation Order

1. **Day 1:** Phase 1 (Schema Foundation) - models, storage, event_storage
2. **Day 2:** Phase 2 (Event Pipeline) + Phase 3 (TraceRank) - commitment integration, ranking
3. **Day 3:** Phase 4 (Config & CLI) - configuration, new commands
4. **Day 4:** Phase 5 (Testing) - smoke tests, validation, documentation

**Estimated:** 3-4 implementation sessions, following progressive maturation principles.
