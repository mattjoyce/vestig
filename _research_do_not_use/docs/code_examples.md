# Code Examples: How Predicates Are Determined

## 1. Direct Tool Detection (Confidence: 1.0)

**Location**: session_parser.py:226-238

```python
if block_type == 'tool_use':
    # Extract tool usage
    tool_name = block.get('name')
    tool_input = block.get('input', {})
    
    # ALWAYS CREATE THIS FACT - it's a direct observation
    self.facts.append(Fact(
        subject=action_id,
        predicate="used_tool",        # ← PREDICATE: hard-coded
        object=tool_name,              # ← OBJECT: from data
        value_type="string",
        confidence=1.0,                # ← 100% certain
        source_session=session_id,
        timestamp=timestamp
    ))
```

**Why it qualifies**: Tool calls are explicit in the JSONL structure.
No inference needed - it's a direct observation.

---

## 2. SSH Host Extraction (Confidence: 0.9)

**Location**: session_parser.py:278-292

```python
def _analyze_bash_command(self, command: str, subject: str, 
                          session_id: str, timestamp: str) -> None:
    # Detect SSH usage
    if 'ssh' in command and '@' in command:
        # Extract target host
        match = re.search(r'@([\w\.\-]+)', command)
        if match:
            host = match.group(1)
            
            # CREATE FACT - inferred from pattern
            self.facts.append(Fact(
                subject=subject,
                predicate="connects_to_host",  # ← PREDICATE: inferred meaning
                object=host,                    # ← OBJECT: extracted via regex
                value_type="string",
                confidence=0.9,                 # ← 90% - regex could fail
                source_session=session_id,
                timestamp=timestamp
            ))
```

**Why it qualifies**: 
- Clear pattern: `ssh user@HOST`
- Regex reliably extracts host
- 0.9 confidence because edge cases exist (e.g., SSH in a comment)

---

## 3. Operation Type Detection (Confidence: 0.8)

**Location**: session_parser.py:294-316

```python
# Detect file operations
if any(op in command for op in ['unzip', 'zip', 'tar', 'gzip']):
    self.facts.append(Fact(
        subject=subject,
        predicate="operation_type",        # ← PREDICATE: categorization
        object="archive_manipulation",     # ← OBJECT: semantic label
        value_type="string",
        confidence=0.8,                    # ← 80% - command might be in string
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
        confidence=0.8,                    # ← Same confidence level
        source_session=session_id,
        timestamp=timestamp
    ))
```

**Why it qualifies**:
- Identifies semantic meaning (archive vs verification)
- Keyword-based detection
- 0.8 confidence because keywords could appear in unrelated context

---

## 4. Discovery Detection (Confidence: 0.6)

**Location**: session_parser.py:348-359

```python
def _extract_insights_from_text(self, text: str, subject: str,
                               session_id: str, timestamp: str) -> None:
    text_lower = text.lower()
    
    # Detect discoveries/findings
    if any(word in text_lower for word in ['found', 'discovered', 'detected', 'shows that']):
        self.facts.append(Fact(
            subject=subject,
            predicate="discovery",             # ← PREDICATE: semantic extraction
            object=text[:300],                 # ← OBJECT: actual text snippet
            value_type="string",
            confidence=0.6,                    # ← 60% - casual language possible
            source_session=session_id,
            timestamp=timestamp,
            context="Discovery or finding"
        ))
```

**Why it qualifies**:
- Extracts potential insights from natural language
- Lower confidence (0.6) because "found" might be casual ("I found myself...")
- Stores actual text for human review
- Useful despite uncertainty - can be filtered by confidence threshold later

---

## 5. What Does NOT Get Stored

**Examples from the code:**

```python
# NOT stored: Generic articles, prepositions
if word in ['the', 'a', 'an', 'and', 'or', 'but']:
    continue  # Skip

# NOT stored: Empty or whitespace-only content
if not content.strip():
    return  # Don't create facts

# NOT stored: Duplicate facts
# Database has UNIQUE constraint:
# UNIQUE(subject, predicate, object, source_session)
```

**Rationale**:
- Generic words: No semantic value
- Empty content: Nothing to learn
- Duplicates: Already captured, wastes space

