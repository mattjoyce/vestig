"""
LLM-Based Discerned Fact Extractor
Uses Simon Willison's llm Python API to extract semantic insights from Claude Code sessions
"""

import json
from typing import List, Dict, Any
from session_parser import Fact
import llm


class DiscernedExtractor:
    """Extract high-level insights using LLM analysis"""

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        """
        Initialize discerned extractor

        Args:
            model: LLM model to use (default: claude-3-5-haiku for speed/cost)
        """
        self.model = model

    def extract_session_insights(self, messages: List[Dict], session_id: str) -> List[Fact]:
        """
        Extract discerned facts from entire session

        Focus areas:
        1. Facts & insights (high-level understanding)
        2. Feedback dynamics (user→LLM, LLM→user)
        3. Pattern detection (recurring behaviors)
        4. Workflow emergence (how tasks flow)
        """
        facts = []

        # Group messages into conversation pairs
        conversation_pairs = self._extract_conversation_pairs(messages)

        # Extract feedback dynamics from exchanges
        facts.extend(self._extract_feedback_dynamics(conversation_pairs, session_id))

        # Extract session-level patterns and workflows
        facts.extend(self._extract_session_level_insights(messages, session_id))

        return facts

    def _extract_conversation_pairs(self, messages: List[Dict]) -> List[Dict[str, Any]]:
        """Group user-assistant message pairs with context"""
        pairs = []

        for i, msg in enumerate(messages):
            if msg.get('type') == 'user' and i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.get('type') == 'assistant':
                    pairs.append({
                        'user_msg': msg,
                        'assistant_msg': next_msg,
                        'pair_index': len(pairs),
                        'timestamp': msg.get('timestamp', '')
                    })

        return pairs

    def _extract_feedback_dynamics(self, pairs: List[Dict], session_id: str) -> List[Fact]:
        """
        Analyze user→LLM and LLM→user feedback dynamics

        User→LLM:
        - Corrective: "No, that's wrong", "Actually..."
        - Clarifying: "I meant...", "To be clear..."
        - Approving: "Perfect!", "That worked"
        - Rejecting: "Stop", "That's not what I asked"

        LLM→User:
        - Enthusiastic: Overcomplicated, unnecessary features
        - Cautious: Too many questions, hedging
        - Corrective: Fixing misconceptions
        - Pedagogical: Explaining vs doing
        """
        facts = []

        # Sample pairs for analysis (don't analyze every single one - too expensive)
        sample_pairs = self._sample_significant_pairs(pairs, max_samples=5)

        for pair in sample_pairs:
            user_content = self._extract_text_content(pair['user_msg'])
            assistant_content = self._extract_text_content(pair['assistant_msg'])

            if not user_content or not assistant_content:
                continue

            # Build prompt for LLM analysis
            prompt = self._build_feedback_analysis_prompt(user_content, assistant_content)

            # Call LLM
            analysis = self._call_llm(prompt)

            # Parse response and create facts
            facts.extend(self._parse_feedback_analysis(
                analysis,
                pair['user_msg'].get('uuid', '')[:8],
                session_id,
                pair['timestamp']
            ))

        return facts

    def _extract_session_level_insights(self, messages: List[Dict], session_id: str) -> List[Fact]:
        """
        Extract high-level session insights:
        - Overall user intent/goal
        - Workflow patterns that emerged
        - Problem-solution relationships
        - User preferences detected
        """
        facts = []

        # Build session summary
        session_summary = self._build_session_summary(messages)

        # Analyze for patterns and workflows
        prompt = self._build_session_analysis_prompt(session_summary)
        analysis = self._call_llm(prompt)

        # Parse and create facts
        facts.extend(self._parse_session_analysis(analysis, session_id, messages))

        return facts

    def _sample_significant_pairs(self, pairs: List[Dict], max_samples: int = 5) -> List[Dict]:
        """
        Sample the most significant conversation pairs for analysis

        Prioritize:
        - First pair (sets context)
        - Pairs with long user messages (detailed requests)
        - Pairs with corrections (user pushback)
        - Last pair (resolution)
        """
        if len(pairs) <= max_samples:
            return pairs

        significant = []

        # Always include first
        if pairs:
            significant.append(pairs[0])

        # Always include last
        if len(pairs) > 1:
            significant.append(pairs[-1])

        # Sample from middle based on message length (proxy for significance)
        middle_pairs = sorted(
            pairs[1:-1],
            key=lambda p: len(self._extract_text_content(p['user_msg'])),
            reverse=True
        )

        remaining_slots = max_samples - len(significant)
        significant.extend(middle_pairs[:remaining_slots])

        return sorted(significant, key=lambda p: p['pair_index'])

    def _extract_text_content(self, msg: Dict) -> str:
        """Extract text content from message"""
        content = msg.get('message', {}).get('content', '')

        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Extract text blocks
            texts = [block.get('text', '') for block in content if block.get('type') == 'text']
            return ' '.join(texts)

        return ''

    def _build_feedback_analysis_prompt(self, user_msg: str, assistant_msg: str) -> str:
        """Build prompt for analyzing feedback dynamics"""
        return f"""Analyze this conversation exchange between a user and an AI assistant.

USER MESSAGE:
{user_msg[:500]}

ASSISTANT RESPONSE:
{assistant_msg[:500]}

Identify the feedback dynamics:

1. USER→ASSISTANT feedback type (pick one):
   - corrective: User corrects or disagrees with assistant
   - clarifying: User clarifies their request
   - approving: User approves or confirms
   - rejecting: User rejects approach
   - requesting: Initial request (no feedback yet)

2. ASSISTANT→USER tone (pick one):
   - enthusiastic: Overcomplicated or adding unnecessary features
   - cautious: Hedging, asking many questions
   - corrective: Fixing user misconceptions
   - pedagogical: Explaining concepts
   - direct: Just executing the task

Return JSON only:
{{"user_feedback": "type", "assistant_tone": "tone", "reasoning": "brief explanation"}}"""

    def _build_session_analysis_prompt(self, session_summary: str) -> str:
        """Build prompt for session-level analysis"""
        return f"""Analyze this Claude Code session and extract key insights.

SESSION SUMMARY:
{session_summary}

Extract:

1. OVERALL INTENT: What was the user trying to accomplish? (1 sentence)

2. WORKFLOW PATTERN: What workflow emerged? (e.g., "verify_then_cleanup", "debug_cycle", "exploration_pattern")

3. USER PREFERENCES: Any preferences detected? (e.g., "prefers tool X", "always checks Y first")

4. PROBLEM-SOLUTION PAIRS: Were there problems that led to solutions?

5. KNOWLEDGE TYPE: What kind of knowledge was this? (procedural, declarative, troubleshooting, learning)

Return JSON only:
{{
  "intent": "...",
  "workflow_pattern": "...",
  "user_preferences": ["..."],
  "problem_solution_pairs": [{{"problem": "...", "solution": "..."}}],
  "knowledge_type": "..."
}}"""

    def _build_session_summary(self, messages: List[Dict]) -> str:
        """Build concise session summary for analysis"""
        summary_parts = []

        for msg in messages[:20]:  # Limit to first 20 messages for cost control
            msg_type = msg.get('type')
            content = self._extract_text_content(msg)

            if content:
                prefix = "USER:" if msg_type == 'user' else "ASSISTANT:"
                summary_parts.append(f"{prefix} {content[:200]}")

        return '\n'.join(summary_parts)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM using Simon Willison's llm Python API"""
        try:
            # Get the model
            model = llm.get_model(self.model)

            # Generate response
            response = model.prompt(prompt)

            # Extract text from response
            if hasattr(response, 'text'):
                return response.text()
            return str(response)

        except Exception as e:
            print(f"LLM call error: {e}")
            return "{}"

    def _parse_feedback_analysis(self, analysis: str, msg_id: str,
                                 session_id: str, timestamp: str) -> List[Fact]:
        """Parse LLM feedback analysis into facts"""
        facts = []

        try:
            # Extract JSON from response (LLM might include extra text)
            json_start = analysis.find('{')
            json_end = analysis.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(analysis[json_start:json_end])
            else:
                return facts

            exchange_id = f"exchange_{msg_id}"

            # User feedback fact
            if 'user_feedback' in data and data['user_feedback'] != 'requesting':
                facts.append(Fact(
                    subject=exchange_id,
                    predicate="user_feedback_type",
                    object=data['user_feedback'],
                    value_type="string",
                    confidence=0.85,
                    source_session=session_id,
                    timestamp=timestamp,
                    context=data.get('reasoning', ''),
                    extraction_method="discerned"
                ))

            # Assistant tone fact
            if 'assistant_tone' in data:
                facts.append(Fact(
                    subject=exchange_id,
                    predicate="assistant_tone",
                    object=data['assistant_tone'],
                    value_type="string",
                    confidence=0.85,
                    source_session=session_id,
                    timestamp=timestamp,
                    context=data.get('reasoning', ''),
                    extraction_method="discerned"
                ))

        except json.JSONDecodeError as e:
            print(f"Failed to parse feedback analysis: {e}")

        return facts

    def _parse_session_analysis(self, analysis: str, session_id: str,
                               messages: List[Dict]) -> List[Fact]:
        """Parse session-level analysis into facts"""
        facts = []

        try:
            # Extract JSON
            json_start = analysis.find('{')
            json_end = analysis.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(analysis[json_start:json_end])
            else:
                return facts

            timestamp = messages[0].get('timestamp', '') if messages else ''

            # Overall intent
            if data.get('intent'):
                facts.append(Fact(
                    subject=session_id,
                    predicate="session_intent",
                    object=data['intent'],
                    value_type="string",
                    confidence=0.9,
                    source_session=session_id,
                    timestamp=timestamp,
                    extraction_method="discerned"
                ))

            # Workflow pattern
            if data.get('workflow_pattern'):
                facts.append(Fact(
                    subject=session_id,
                    predicate="workflow_pattern",
                    object=data['workflow_pattern'],
                    value_type="string",
                    confidence=0.85,
                    source_session=session_id,
                    timestamp=timestamp,
                    extraction_method="discerned"
                ))

            # User preferences
            for pref in data.get('user_preferences', []):
                if pref:
                    facts.append(Fact(
                        subject=session_id,
                        predicate="user_preference",
                        object=pref,
                        value_type="string",
                        confidence=0.8,
                        source_session=session_id,
                        timestamp=timestamp,
                        extraction_method="discerned"
                    ))

            # Problem-solution pairs
            for pair in data.get('problem_solution_pairs', []):
                if pair.get('problem') and pair.get('solution'):
                    problem_id = f"problem_{len(facts)}"

                    facts.append(Fact(
                        subject=problem_id,
                        predicate="problem_description",
                        object=pair['problem'],
                        value_type="string",
                        confidence=0.85,
                        source_session=session_id,
                        timestamp=timestamp,
                        extraction_method="discerned"
                    ))

                    facts.append(Fact(
                        subject=problem_id,
                        predicate="solution_found",
                        object=pair['solution'],
                        value_type="string",
                        confidence=0.85,
                        source_session=session_id,
                        timestamp=timestamp,
                        extraction_method="discerned"
                    ))

            # Knowledge type
            if data.get('knowledge_type'):
                facts.append(Fact(
                    subject=session_id,
                    predicate="knowledge_type",
                    object=data['knowledge_type'],
                    value_type="string",
                    confidence=0.85,
                    source_session=session_id,
                    timestamp=timestamp,
                    extraction_method="discerned"
                ))

        except json.JSONDecodeError as e:
            print(f"Failed to parse session analysis: {e}")

        return facts


def main():
    """Test the discerned extractor"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python discerned_extractor.py <session_file.jsonl>")
        sys.exit(1)

    # Load session
    with open(sys.argv[1], 'r') as f:
        messages = [json.loads(line) for line in f.readlines()]

    # Extract session ID
    session_id = 'test_session'
    for msg in messages:
        if msg.get('sessionId'):
            session_id = msg['sessionId']
            break

    # Run discerned extraction
    extractor = DiscernedExtractor()
    facts = extractor.extract_session_insights(messages, session_id)

    print(f"\n=== Discerned Extraction Results ===")
    print(f"Extracted {len(facts)} discerned facts\n")

    # Group by predicate
    by_predicate = {}
    for fact in facts:
        pred = fact.predicate
        if pred not in by_predicate:
            by_predicate[pred] = []
        by_predicate[pred].append(fact)

    for predicate, fact_list in by_predicate.items():
        print(f"\n{predicate.upper()} ({len(fact_list)} facts):")
        for fact in fact_list:
            print(f"  - {fact.object[:100]}")
            if fact.context:
                print(f"    Context: {fact.context[:100]}")


if __name__ == '__main__':
    main()
