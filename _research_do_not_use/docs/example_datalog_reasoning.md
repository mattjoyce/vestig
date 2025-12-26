# Datalog Reasoning with Naive + Discerned Facts

## Example: Reasoning Across Both Extraction Types

When you export facts to Datalog format, BOTH naive and discerned facts are included, enabling powerful cross-reasoning:

```python
from pyDatalog import pyDatalog

# Define terms (predicates from both naive and discerned)
pyDatalog.create_terms('connects_to_host, used_tool, executed_command')  # Naive
pyDatalog.create_terms('session_intent, workflow_pattern, user_feedback_type')  # Discerned
pyDatalog.create_terms('assistant_tone, knowledge_type')  # Discerned
pyDatalog.create_terms('Session, Host, Tool, Intent, Workflow, successful_session')

# Load facts (both naive and discerned from database)
# Naive facts:
+connects_to_host('action_123', '192.168.20.4')
+used_tool('action_123', 'Bash')
+executed_command('action_123', 'ssh root@192.168.20.4 ...')

# Discerned facts:
+session_intent('session_abc', 'verify server and backup')
+workflow_pattern('session_abc', 'verification_pattern')
+user_feedback_type('exchange_789', 'approving')
+assistant_tone('exchange_789', 'direct')

# REASONING RULES: Combine naive + discerned for insights

# Rule 1: A session was successful if:
#   - User gave approving feedback (discerned)
#   - AND assistant used tools (naive)
successful_session(Session) <= (
    user_feedback_type(_, 'approving') &
    session_intent(Session, _) &
    used_tool(_, 'Bash')
)

# Rule 2: Recommended workflow for host connections
# If sessions targeting a host had direct tone and approval,
# then that workflow is recommended
recommended_workflow(Host, Workflow) <= (
    connects_to_host(Action, Host) &
    workflow_pattern(Session, Workflow) &
    user_feedback_type(_, 'approving') &
    assistant_tone(_, 'direct')
)

# Rule 3: Effective troubleshooting sessions
# Sessions that are troubleshooting type + got approval + used SSH
effective_troubleshooting(Session) <= (
    knowledge_type(Session, 'troubleshooting') &
    user_feedback_type(_, 'approving') &
    connects_to_host(_, Host)
)

# QUERIES:
# Which sessions were successful?
successful_session(X)?
# → session_abc

# What workflow is recommended for 192.168.20.4?
recommended_workflow('192.168.20.4', W)?
# → verification_pattern

# Which sessions demonstrate effective troubleshooting?
effective_troubleshooting(S)?
# → session_abc
```

## Power of Cross-Reasoning

### Example 1: Learning User Preferences
```python
# Naive: "User used sshpass command"
+executed_command('action_1', 'sshpass -p pass ssh root@host')

# Discerned: "User approved the result"
+user_feedback_type('exchange_1', 'approving')

# Rule: If user approved a command pattern, it's a preference
user_prefers_command(Cmd) <= (
    executed_command(Action, Cmd) &
    user_feedback_type(Exchange, 'approving') &
    same_session(Action, Exchange)
)

# Query: What commands does user prefer?
user_prefers_command(C)?
# → 'sshpass -p pass ssh root@host'
```

### Example 2: Workflow Optimization
```python
# Naive: Track what tools were used
+used_tool('action_1', 'Bash')
+used_tool('action_2', 'Read')
+connects_to_host('action_1', '192.168.20.4')

# Discerned: Track if it was efficient
+assistant_tone('exchange_1', 'direct')  # Not overcomplicated
+workflow_pattern('session_1', 'verification_pattern')

# Rule: Efficient workflows are those with direct tone + approval
efficient_workflow(Workflow) <= (
    workflow_pattern(Session, Workflow) &
    assistant_tone(_, 'direct') &
    user_feedback_type(_, 'approving')
)

# Query: What are efficient workflows?
efficient_workflow(W)?
# → verification_pattern
```

### Example 3: Error Pattern Detection
```python
# Naive: Commands that failed
+executed_command('action_5', 'tar -xf broken.tar.gz')
+operation_type('action_5', 'archive_manipulation')

# Discerned: User gave corrective feedback
+user_feedback_type('exchange_5', 'corrective')
+problem_description('problem_5', 'Archive was corrupted')

# Rule: Problematic operations = operation + corrective feedback
problematic_operation(OpType) <= (
    operation_type(Action, OpType) &
    user_feedback_type(Exchange, 'corrective') &
    same_session(Action, Exchange)
)

# Query: What operations tend to have problems?
problematic_operation(Op)?
# → archive_manipulation
```

## Benefits of Combining Both

| Naive Facts | Discerned Facts | Combined Reasoning |
|------------|-----------------|-------------------|
| **What** happened | **Why** it happened | **Learn** patterns |
| Used SSH | User approved | → Preferred method |
| Executed tar command | Corrective feedback | → Avoid this approach |
| Connected to host | Verification workflow | → Recommended workflow |
| Used 3 tools | Direct tone | → Efficient pattern |

## Real-World Use Case

**Agent's internal reasoning when user asks:**
> "How should I verify backups on my unraid server?"

**Datalog query:**
```python
# Find all sessions that:
# 1. Connected to unraid (naive)
# 2. Had verification workflow (discerned)
# 3. Got user approval (discerned)
recommended_approach(Commands, Workflow) <= (
    targets_system(Task, 'unraid') &
    workflow_pattern(Session, Workflow) &
    user_feedback_type(_, 'approving') &
    executed_command(Action, Commands) &
    same_session(Task, Action)
)
```

**Agent response:**
"Based on past successful sessions, I recommend the `verification_pattern` workflow:
1. SSH to server (you've used `192.168.20.4` before)
2. List files in `/mnt/backups/`
3. Test archive with `tar -tzf` flag

This approach got your approval last time."

---

## The Value Proposition

**Naive alone:** "You connected to 192.168.20.4 three times"
**Discerned alone:** "You approve direct responses"
**Combined:** "When you verify backups on 192.168.20.4 using direct approaches, you're satisfied - let's do that again"

This is the **emergence** you mentioned - workflows and preferences emerge from combining structural facts (naive) with semantic understanding (discerned).
