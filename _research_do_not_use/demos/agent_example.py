"""
Practical Agent Example: Using Memory to Solve Tasks

This demonstrates how an LLM agent uses the memory system to:
1. Recall past solutions
2. Learn from patterns
3. Apply context to new problems
"""

from agent_memory import AgentMemory
import json

class ClaudeAgent:
    """LLM Agent with memory-augmented reasoning"""
    
    def __init__(self, memory_path: str = "agent_memory.db"):
        self.memory = AgentMemory(memory_path)
    
    def solve_task(self, task_description: str) -> dict:
        """
        Solve a task using memory-augmented context
        
        Returns enriched context for LLM prompt
        """
        # Step 1: Extract key entities from task
        entities = self._extract_entities(task_description)
        
        # Step 2: Query memory for relevant context
        context = {
            'task': task_description,
            'relevant_facts': [],
            'similar_sessions': [],
            'applicable_patterns': [],
            'known_solutions': []
        }
        
        # Find facts about mentioned entities
        for entity in entities:
            facts = self.memory.query_facts(subject=entity)
            context['relevant_facts'].extend(facts)
        
        # Find similar past work
        if 'unraid' in task_description.lower():
            sessions = self.memory.find_related_sessions({'project': 'unraid_admin'})
            context['similar_sessions'] = sessions
        
        # Look for relevant patterns
        patterns = self.memory.find_patterns()
        context['applicable_patterns'] = patterns
        
        # Find past solutions
        solutions = self.memory.get_solutions()
        context['known_solutions'] = solutions
        
        # Step 3: Build enriched prompt
        enriched_prompt = self._build_prompt(context)
        
        return {
            'context': context,
            'prompt': enriched_prompt,
            'confidence': self._calculate_confidence(context)
        }
    
    def _extract_entities(self, text: str) -> list:
        """Simple entity extraction (would use NER in production)"""
        entities = []
        
        # Simple keyword matching
        keywords = {
            'unraid': 'system',
            'server': 'system',
            'ssh': 'tool',
            'docker': 'tool',
            'zip': 'filetype',
            'archive': 'operation'
        }
        
        text_lower = text.lower()
        for keyword, entity_type in keywords.items():
            if keyword in text_lower:
                entities.append(keyword)
        
        return entities
    
    def _build_prompt(self, context: dict) -> str:
        """Build enriched LLM prompt with memory context"""
        
        prompt = f"""Task: {context['task']}

## Relevant Past Experience

"""
        # Add similar sessions
        if context['similar_sessions']:
            prompt += "### Similar Past Sessions:\n"
            for session in context['similar_sessions'][:3]:
                prompt += f"- {session['summary']}\n"
            prompt += "\n"
        
        # Add known solutions
        if context['known_solutions']:
            prompt += "### Solutions You've Used Before:\n"
            for sol in context['known_solutions'][:3]:
                prompt += f"- {sol['object'][:150]}...\n"
            prompt += "\n"
        
        # Add relevant facts
        if context['relevant_facts']:
            prompt += "### What You Know:\n"
            fact_groups = {}
            for fact in context['relevant_facts']:
                pred = fact['predicate']
                if pred not in fact_groups:
                    fact_groups[pred] = []
                fact_groups[pred].append(fact['object'])
            
            for pred, objects in list(fact_groups.items())[:5]:
                prompt += f"- {pred}: {', '.join(objects[:3])}\n"
            prompt += "\n"
        
        prompt += """## Your Task

Based on your past experience above, please solve the current task.
If you've done something similar before, build on that knowledge.
"""
        
        return prompt
    
    def _calculate_confidence(self, context: dict) -> float:
        """Calculate confidence based on available context"""
        score = 0.5  # Base confidence
        
        if context['similar_sessions']:
            score += 0.2
        
        if context['known_solutions']:
            score += 0.2
        
        if context['relevant_facts']:
            score += 0.1
        
        return min(1.0, score)


# Example usage scenarios

def scenario_1_new_unraid_task():
    """Agent encounters a new Unraid task"""
    
    print("=" * 80)
    print("SCENARIO 1: New Unraid Task")
    print("=" * 80)
    
    agent = ClaudeAgent("example_agent_memory.db")
    
    new_task = """
    I need to check if my Unraid server has enough disk space 
    for a new Docker container. Can you SSH in and check?
    """
    
    result = agent.solve_task(new_task)
    
    print(f"\nTask: {new_task.strip()}\n")
    print(f"Confidence: {result['confidence']:.2f}\n")
    print("=" * 80)
    print(result['prompt'])
    print("=" * 80)


def scenario_2_known_pattern():
    """Agent recognizes a pattern it's seen before"""
    
    print("\n\n" + "=" * 80)
    print("SCENARIO 2: Recognized Pattern (Archive Verification)")
    print("=" * 80)
    
    agent = ClaudeAgent("example_agent_memory.db")
    
    task = """
    Can you verify the integrity of some tar.gz files I have 
    on my Unraid server at /mnt/user/backups/?
    """
    
    result = agent.solve_task(task)
    
    print(f"\nTask: {task.strip()}\n")
    print(f"Confidence: {result['confidence']:.2f}\n")
    
    # Show that agent found relevant past work
    print("\nAgent's Memory Recall:")
    print(f"- Found {len(result['context']['similar_sessions'])} similar sessions")
    print(f"- Found {len(result['context']['relevant_facts'])} relevant facts")
    
    # Show specific relevant facts
    print("\nRelevant Facts Retrieved:")
    for fact in result['context']['relevant_facts'][:5]:
        print(f"  {fact['predicate']}: {fact['object'][:80]}...")


def scenario_3_learning_from_patterns():
    """Show what the agent has learned"""
    
    print("\n\n" + "=" * 80)
    print("SCENARIO 3: What Has the Agent Learned?")
    print("=" * 80)
    
    memory = AgentMemory("example_agent_memory.db")
    
    print("\n### Tools I Know How to Use:")
    tools = memory.query_facts(predicate='used_tool')
    tool_names = set(f['object'] for f in tools)
    for tool in tool_names:
        count = len([f for f in tools if f['object'] == tool])
        print(f"  - {tool}: used {count} times")
    
    print("\n### Systems I've Worked With:")
    systems = memory.query_facts(predicate='targets_system')
    for system in set(f['object'] for f in systems):
        print(f"  - {system}")
    
    print("\n### Hosts I've Connected To:")
    hosts = memory.query_facts(predicate='connects_to_host')
    for host in set(f['object'] for f in hosts):
        print(f"  - {host}")
    
    print("\n### Operation Types I've Performed:")
    ops = memory.query_facts(predicate='operation_type')
    for op in set(f['object'] for f in ops):
        count = len([f for f in ops if f['object'] == op])
        print(f"  - {op}: {count} times")
    
    print("\n### Sessions Analyzed:")
    stats = memory.stats()
    print(f"  - Total sessions: {stats['total_sessions']}")
    print(f"  - Total facts: {stats['total_facts']}")
    print(f"  - Unique entities: {stats['unique_entities']}")


def scenario_4_datalog_reasoning():
    """Example of Datalog-style reasoning queries"""
    
    print("\n\n" + "=" * 80)
    print("SCENARIO 4: Advanced Reasoning with Datalog")
    print("=" * 80)
    
    print("""
With PyDatalog, the agent could answer questions like:

Q: "What tasks require connecting to host 192.168.20.4?"

Datalog query:
    requires_host(Task, Host) <= (
        requested_in_session(Task, Session) &
        connects_to_host(Action, Host) &
        requested_in_session(Action, Session)
    )
    
    ?requires_host(Task, '192.168.20.4')

Result: task_98b33414, task_b4bffcea, task_ae4eb2a2, task_ae20d336


Q: "What tools are typically used together?"

Datalog query:
    co_occurs(Tool1, Tool2) <= (
        used_tool(Action1, Tool1) &
        used_tool(Action2, Tool2) &
        same_session(Action1, Action2) &
        (Tool1 != Tool2)
    )
    
    ?co_occurs('Bash', X)

Result: Often used with: Read, TodoWrite, Write


Q: "Can I solve this new task based on past experience?"

Datalog query:
    can_solve(NewTask, Method) <= (
        targets_system(NewTask, System) &
        targets_system(OldTask, System) &
        provides_solution(OldTask, Method)
    )
    
    can_solve('verify_archives_on_unraid', Method)?

Result: Yes, I've done archive verification on Unraid before!
""")


def scenario_5_multi_session_learning():
    """Show how memory grows with multiple sessions"""
    
    print("\n\n" + "=" * 80)
    print("SCENARIO 5: Multi-Session Learning (Conceptual)")
    print("=" * 80)
    
    print("""
As you import more Claude Code sessions, the agent learns:

Session 1 (unraid_admin): 
  - Learned: SSH to 192.168.20.4
  - Learned: Use sshpass for automation
  - Learned: Archive verification with unzip -t

Session 2 (biophonyai):
  - Learned: Python environment setup
  - Learned: AudioMoth data processing
  - Learned: ML model training workflow

Session 3 (calvary_ai_governance):
  - Learned: Healthcare AI risk assessment
  - Learned: Policy documentation workflow
  - Learned: Stakeholder communication patterns

Now when you ask: "Help me set up a Python environment for audio processing"

Agent thinks:
  1. Check biophonyai sessions (Python + audio)
  2. Recall successful environment setup patterns
  3. Apply that workflow to new task
  
The agent builds a "personal knowledge graph" of YOUR workflows,
preferences, and solutions!
""")


if __name__ == '__main__':
    # Run all scenarios
    scenario_1_new_unraid_task()
    scenario_2_known_pattern()
    scenario_3_learning_from_patterns()
    scenario_4_datalog_reasoning()
    scenario_5_multi_session_learning()
    
    print("\n\n" + "=" * 80)
    print("Key Takeaway:")
    print("=" * 80)
    print("""
The agent memory system enables:

✓ Learning from past sessions
✓ Recognizing patterns in your work
✓ Recalling successful solutions
✓ Building context-aware responses
✓ Improving over time through accumulation of experience

This is the foundation for a truly personalized AI assistant that
gets better at helping YOU specifically!
""")
