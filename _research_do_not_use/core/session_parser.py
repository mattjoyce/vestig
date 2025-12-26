"""
Claude Code Session Parser
Extracts entities, predicates, and facts from Claude Code session JSONL files
for building an agent memory graph database.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

@dataclass
class Fact:
    """Represents a subject-predicate-object triple with metadata"""
    subject: str
    predicate: str
    object: str
    value_type: str  # string|number|date|reference|boolean
    confidence: float  # 0.0-1.0
    source_session: str
    timestamp: str
    context: str = ""  # Additional context for the fact
    extraction_method: str = "naive"  # naive|discerned - how the fact was extracted

    def to_dict(self):
        return asdict(self)


class ClaudeCodeSessionParser:
    """Parse Claude Code session files and extract structured facts"""

    def __init__(self, use_discerned: bool = False, llm_model: str = "claude-3-5-haiku-20241022"):
        """
        Initialize parser

        Args:
            use_discerned: Enable LLM-based discerned extraction (slower, more insightful)
            llm_model: LLM model to use for discerned extraction
        """
        self.facts: List[Fact] = []
        self.session_metadata: Dict[str, Any] = {}
        self.use_discerned = use_discerned
        self.llm_model = llm_model
        
    def parse_session_file(self, filepath: str) -> Dict[str, Any]:
        """Parse a JSONL session file and extract all facts"""
        with open(filepath, 'r') as f:
            messages = [json.loads(line) for line in f.readlines()]

        # Extract session metadata
        self._extract_session_metadata(messages)

        # Phase 1: Naive extraction (fast, structured)
        self._extract_facts_from_messages(messages)
        naive_count = len(self.facts)

        # Phase 2: Discerned extraction (slow, semantic)
        if self.use_discerned:
            try:
                from discerned_extractor import DiscernedExtractor

                print(f"Running discerned extraction with {self.llm_model}...")
                extractor = DiscernedExtractor(model=self.llm_model)
                discerned_facts = extractor.extract_session_insights(
                    messages,
                    self.session_metadata.get('session_id', 'unknown')
                )
                self.facts.extend(discerned_facts)
                print(f"Naive: {naive_count} facts, Discerned: {len(discerned_facts)} facts")
            except ImportError as e:
                print(f"Warning: Could not import discerned_extractor: {e}")
            except Exception as e:
                print(f"Warning: Discerned extraction failed: {e}")

        return {
            'metadata': self.session_metadata,
            'facts': [f.to_dict() for f in self.facts],
            'summary': self._generate_session_summary()
        }
    
    def _extract_session_metadata(self, messages: List[Dict]) -> None:
        """Extract high-level session information"""
        # Find first user/assistant message for metadata
        for msg in messages:
            if msg.get('sessionId'):
                self.session_metadata = {
                    'session_id': msg.get('sessionId'),
                    'cwd': msg.get('cwd'),
                    'version': msg.get('version'),
                    'git_branch': msg.get('gitBranch', ''),
                    'start_time': None,
                    'end_time': None,
                    'total_messages': 0,
                    'tools_used': set(),
                    'project': self._infer_project(msg.get('cwd', ''))
                }
                break
        
        # Count messages and track time range
        user_msgs = []
        assistant_msgs = []
        
        for msg in messages:
            msg_type = msg.get('type')
            timestamp = msg.get('timestamp')
            
            if msg_type == 'user':
                user_msgs.append(msg)
            elif msg_type == 'assistant':
                assistant_msgs.append(msg)
                
            # Track tools used
            content = msg.get('message', {}).get('content', [])
            if isinstance(content, list):
                for block in content:
                    if block.get('type') == 'tool_use':
                        self.session_metadata['tools_used'].add(block.get('name'))
        
        # Set time range
        timestamps = [m.get('timestamp') for m in messages if m.get('timestamp')]
        if timestamps:
            self.session_metadata['start_time'] = min(timestamps)
            self.session_metadata['end_time'] = max(timestamps)
        
        self.session_metadata['total_messages'] = len(user_msgs) + len(assistant_msgs)
        self.session_metadata['user_messages'] = len(user_msgs)
        self.session_metadata['assistant_messages'] = len(assistant_msgs)
        self.session_metadata['tools_used'] = list(self.session_metadata['tools_used'])
    
    def _infer_project(self, cwd: str) -> str:
        """Infer project name from working directory"""
        if not cwd:
            return "unknown"
        parts = cwd.rstrip('/').split('/')
        return parts[-1] if parts else "unknown"
    
    def _extract_facts_from_messages(self, messages: List[Dict]) -> None:
        """Extract facts from all messages in the session"""
        session_id = self.session_metadata.get('session_id', 'unknown')
        
        # Track conversation flow for relationship extraction
        conversation_pairs = []
        
        for i, msg in enumerate(messages):
            msg_type = msg.get('type')
            timestamp = msg.get('timestamp', '')
            
            if msg_type == 'user':
                # Extract user tasks/requests
                self._extract_user_intent(msg, session_id, timestamp)
                
            elif msg_type == 'assistant':
                # Extract tool usage, discoveries, solutions
                self._extract_assistant_actions(msg, session_id, timestamp)
                
                # Track conversation pairs for context
                if i > 0 and messages[i-1].get('type') == 'user':
                    conversation_pairs.append((messages[i-1], msg))
        
        # Extract patterns and relationships from conversation flow
        self._extract_conversation_patterns(conversation_pairs, session_id)
    
    def _extract_user_intent(self, msg: Dict, session_id: str, timestamp: str) -> None:
        """Extract facts from user messages (tasks, requests, problems)"""
        content = msg.get('message', {}).get('content', '')
        
        if not isinstance(content, str):
            return
        
        # Create a task fact
        task_id = f"task_{msg.get('uuid', 'unknown')[:8]}"
        
        self.facts.append(Fact(
            subject=task_id,
            predicate="requested_in_session",
            object=session_id,
            value_type="reference",
            confidence=1.0,
            source_session=session_id,
            timestamp=timestamp,
            context=content[:200]
        ))
        
        self.facts.append(Fact(
            subject=task_id,
            predicate="description",
            object=content,
            value_type="string",
            confidence=1.0,
            source_session=session_id,
            timestamp=timestamp
        ))
        
        # Extract entities mentioned (simple pattern matching)
        self._extract_mentioned_entities(content, task_id, session_id, timestamp)
    
    def _extract_mentioned_entities(self, text: str, subject: str, 
                                    session_id: str, timestamp: str) -> None:
        """Extract technical entities mentioned in text"""
        # File paths
        file_paths = re.findall(r'/[\w/\-_.]+', text)
        for path in file_paths[:5]:  # Limit to avoid noise
            self.facts.append(Fact(
                subject=subject,
                predicate="mentions_path",
                object=path,
                value_type="string",
                confidence=0.8,
                source_session=session_id,
                timestamp=timestamp
            ))
        
        # Commands (simple detection)
        if any(cmd in text.lower() for cmd in ['ssh', 'scp', 'rsync', 'git', 'docker']):
            # Extract command type
            for cmd in ['ssh', 'scp', 'rsync', 'git', 'docker', 'unzip', 'ls', 'cd']:
                if cmd in text.lower():
                    self.facts.append(Fact(
                        subject=subject,
                        predicate="involves_command",
                        object=cmd,
                        value_type="string",
                        confidence=0.9,
                        source_session=session_id,
                        timestamp=timestamp
                    ))
        
        # System/server names
        servers = re.findall(r'(?:unraid|server|nas|host)', text.lower())
        for server in set(servers):
            self.facts.append(Fact(
                subject=subject,
                predicate="targets_system",
                object=server,
                value_type="string",
                confidence=0.7,
                source_session=session_id,
                timestamp=timestamp
            ))
    
    def _extract_assistant_actions(self, msg: Dict, session_id: str, timestamp: str) -> None:
        """Extract facts from assistant messages (tools used, discoveries, solutions)"""
        content = msg.get('message', {}).get('content', [])
        
        if not isinstance(content, list):
            return
        
        action_id = f"action_{msg.get('uuid', 'unknown')[:8]}"
        
        for block in content:
            block_type = block.get('type')
            
            if block_type == 'tool_use':
                # Extract tool usage
                tool_name = block.get('name')
                tool_input = block.get('input', {})
                
                self.facts.append(Fact(
                    subject=action_id,
                    predicate="used_tool",
                    object=tool_name,
                    value_type="string",
                    confidence=1.0,
                    source_session=session_id,
                    timestamp=timestamp
                ))
                
                # Extract specific tool details
                if tool_name == 'Bash':
                    command = tool_input.get('command', '')
                    self.facts.append(Fact(
                        subject=action_id,
                        predicate="executed_command",
                        object=command[:500],  # Truncate long commands
                        value_type="string",
                        confidence=1.0,
                        source_session=session_id,
                        timestamp=timestamp
                    ))
                    
                    # Analyze command for insights
                    self._analyze_bash_command(command, action_id, session_id, timestamp)
                
                elif tool_name in ['Read', 'Write']:
                    filepath = tool_input.get('path', '')
                    if filepath:
                        self.facts.append(Fact(
                            subject=action_id,
                            predicate="accessed_file",
                            object=filepath,
                            value_type="string",
                            confidence=1.0,
                            source_session=session_id,
                            timestamp=timestamp
                        ))
            
            elif block_type == 'text':
                # Analyze text for discoveries, solutions, issues
                text = block.get('text', '')
                self._extract_insights_from_text(text, action_id, session_id, timestamp)
    
    def _analyze_bash_command(self, command: str, subject: str, 
                              session_id: str, timestamp: str) -> None:
        """Analyze bash commands for patterns and insights"""
        # Detect SSH usage
        if 'ssh' in command and '@' in command:
            # Extract target host
            match = re.search(r'@([\w\.\-]+)', command)
            if match:
                host = match.group(1)
                self.facts.append(Fact(
                    subject=subject,
                    predicate="connects_to_host",
                    object=host,
                    value_type="string",
                    confidence=0.9,
                    source_session=session_id,
                    timestamp=timestamp
                ))
        
        # Detect file operations
        if any(op in command for op in ['unzip', 'zip', 'tar', 'gzip']):
            self.facts.append(Fact(
                subject=subject,
                predicate="operation_type",
                object="archive_manipulation",
                value_type="string",
                confidence=0.8,
                source_session=session_id,
                timestamp=timestamp
            ))
        
        # Detect verification operations
        if any(verify in command for verify in ['-t', 'test', 'verify', 'check']):
            self.facts.append(Fact(
                subject=subject,
                predicate="operation_type",
                object="verification",
                value_type="string",
                confidence=0.8,
                source_session=session_id,
                timestamp=timestamp
            ))
    
    def _extract_insights_from_text(self, text: str, subject: str,
                                   session_id: str, timestamp: str) -> None:
        """Extract discoveries, solutions, and patterns from assistant text"""
        text_lower = text.lower()
        
        # Detect problem/solution patterns
        if any(word in text_lower for word in ['error', 'issue', 'problem', 'failed']):
            self.facts.append(Fact(
                subject=subject,
                predicate="identifies_issue",
                object=text[:300],
                value_type="string",
                confidence=0.7,
                source_session=session_id,
                timestamp=timestamp,
                context="Issue identification"
            ))
        
        if any(word in text_lower for word in ['solution', 'fix', 'resolved', 'works']):
            self.facts.append(Fact(
                subject=subject,
                predicate="provides_solution",
                object=text[:300],
                value_type="string",
                confidence=0.7,
                source_session=session_id,
                timestamp=timestamp,
                context="Solution description"
            ))
        
        # Detect discoveries/findings
        if any(word in text_lower for word in ['found', 'discovered', 'detected', 'shows that']):
            self.facts.append(Fact(
                subject=subject,
                predicate="discovery",
                object=text[:300],
                value_type="string",
                confidence=0.6,
                source_session=session_id,
                timestamp=timestamp,
                context="Discovery or finding"
            ))
    
    def _extract_conversation_patterns(self, pairs: List[Tuple[Dict, Dict]], 
                                      session_id: str) -> None:
        """Extract patterns from user-assistant conversation pairs"""
        # This could detect common workflows, debugging patterns, etc.
        # For now, we'll just track successful task completions
        
        for user_msg, assistant_msg in pairs:
            user_content = user_msg.get('message', {}).get('content', '')
            assistant_content = assistant_msg.get('message', {}).get('content', [])
            
            # Check if assistant used tools (indicates action taken)
            if isinstance(assistant_content, list):
                tools_used = [b.get('name') for b in assistant_content 
                             if b.get('type') == 'tool_use']
                
                if tools_used:
                    pattern_id = f"pattern_{len(self.facts)}"
                    # Convert user_content to string if it's not already
                    context_str = user_content if isinstance(user_content, str) else str(user_content)
                    self.facts.append(Fact(
                        subject=pattern_id,
                        predicate="task_pattern",
                        object=f"request->tools:{','.join(tools_used)}",
                        value_type="string",
                        confidence=0.5,
                        source_session=session_id,
                        timestamp=assistant_msg.get('timestamp', ''),
                        context=context_str[:100]
                    ))
    
    def _generate_session_summary(self) -> str:
        """Generate a natural language summary of the session"""
        meta = self.session_metadata
        
        summary_parts = [
            f"Session in project '{meta.get('project')}' with {meta.get('total_messages')} messages.",
            f"Tools used: {', '.join(meta.get('tools_used', []))}.",
            f"Extracted {len(self.facts)} facts."
        ]
        
        return ' '.join(summary_parts)


def main():
    """Example usage"""
    import sys
    import argparse

    parser_cli = argparse.ArgumentParser(description='Parse Claude Code session files')
    parser_cli.add_argument('session_file', help='Path to session JSONL file')
    parser_cli.add_argument('--discerned', action='store_true',
                           help='Enable LLM-based discerned extraction (slower, more insights)')
    parser_cli.add_argument('--model', default='claude-3-5-haiku-20241022',
                           help='LLM model for discerned extraction')

    args = parser_cli.parse_args()

    parser = ClaudeCodeSessionParser(
        use_discerned=args.discerned,
        llm_model=args.model
    )
    result = parser.parse_session_file(args.session_file)

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
