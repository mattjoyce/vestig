"""Document ingestion with LLM-based memory extraction"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from vestig.core.entity_extraction import (
    load_prompts,
    substitute_tokens,
    call_llm,
)
from vestig.core.commitment import commit_memory
from vestig.core.embeddings import EmbeddingEngine
from vestig.core.storage import MemoryStorage
from vestig.core.event_storage import MemoryEventStorage


@dataclass
class ExtractedMemory:
    """Memory extracted from document by LLM"""

    content: str
    confidence: float
    rationale: str


@dataclass
class IngestionResult:
    """Result of document ingestion"""

    document_path: str
    chunks_processed: int
    memories_extracted: int
    memories_committed: int
    memories_deduplicated: int
    entities_created: int
    errors: List[str]


def chunk_text_by_chars(text: str, chunk_size: int = 20000, overlap: int = 500) -> List[str]:
    """
    Chunk text by character count with overlap.

    Simple chunking strategy that tries to break at paragraph boundaries.
    Uses character count as proxy for tokens (roughly 1 token = 4 chars).

    Args:
        text: Text to chunk
        chunk_size: Max characters per chunk (default 20k chars ≈ 5k tokens)
        overlap: Character overlap between chunks (for context continuity)

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Find end of chunk
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break

        # Try to break at paragraph boundary (double newline)
        break_point = text.rfind("\n\n", start, end)
        if break_point == -1:
            # Try single newline
            break_point = text.rfind("\n", start, end)
        if break_point == -1:
            # Try space
            break_point = text.rfind(" ", start, end)
        if break_point == -1:
            # Hard break
            break_point = end

        chunks.append(text[start:break_point])

        # Next chunk starts with overlap
        start = max(start + 1, break_point - overlap)

    return chunks


def extract_memories_from_chunk(
    chunk: str,
    model: str = "claude-sonnet-4.5",
    min_confidence: float = 0.6,
) -> List[ExtractedMemory]:
    """
    Extract memories from a text chunk using LLM.

    Args:
        chunk: Text chunk to extract from
        model: LLM model to use
        min_confidence: Minimum confidence threshold

    Returns:
        List of ExtractedMemory objects

    Raises:
        ValueError: If LLM returns invalid JSON
    """
    # Load and substitute prompt
    prompts = load_prompts()
    template = prompts.get("extract_memories_from_session")

    if not template:
        raise ValueError(
            "'extract_memories_from_session' prompt not found in prompts.yaml"
        )

    prompt = substitute_tokens(template, content=chunk)

    # Call LLM
    try:
        response = call_llm(prompt, model=model)
    except Exception as e:
        raise ValueError(f"LLM call failed: {e}")

    # Parse JSON response
    try:
        result = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from LLM: {e}\nResponse: {response[:500]}")

    # Validate structure
    if "memories" not in result:
        raise ValueError("Missing 'memories' key in extraction result")

    if not isinstance(result["memories"], list):
        raise ValueError("'memories' must be a list")

    # Parse and validate memories
    memories = []
    for i, mem_dict in enumerate(result["memories"]):
        # Validate required fields
        if "content" not in mem_dict:
            print(f"Warning: Memory {i} missing 'content' field, skipping")
            continue
        if "confidence" not in mem_dict:
            print(f"Warning: Memory {i} missing 'confidence' field, skipping")
            continue

        content = str(mem_dict["content"]).strip()
        rationale = str(mem_dict.get("rationale", "")).strip()

        # Validate confidence
        try:
            confidence = float(mem_dict["confidence"])
        except (ValueError, TypeError):
            print(f"Warning: Memory {i} has invalid confidence, skipping")
            continue

        if not (0.0 <= confidence <= 1.0):
            print(f"Warning: Memory {i} confidence out of range, skipping")
            continue

        # Apply confidence threshold
        if confidence < min_confidence:
            continue

        # Skip empty content
        if not content or len(content) < 10:
            continue

        memories.append(
            ExtractedMemory(
                content=content,
                confidence=confidence,
                rationale=rationale,
            )
        )

    return memories


def ingest_document(
    document_path: str,
    storage: MemoryStorage,
    embedding_engine: EmbeddingEngine,
    event_storage: Optional[MemoryEventStorage] = None,
    m4_config: Optional[Dict[str, Any]] = None,
    chunk_size: int = 20000,
    chunk_overlap: int = 500,
    extraction_model: str = "claude-sonnet-4.5",
    min_confidence: float = 0.6,
    source: str = "document_ingest",
) -> IngestionResult:
    """
    Ingest document by extracting memories with LLM and committing them.

    Args:
        document_path: Path to document file
        storage: Storage instance
        embedding_engine: Embedding engine
        event_storage: Optional event storage
        m4_config: Optional M4 config (for entity extraction)
        chunk_size: Characters per chunk (default 20k ≈ 5k tokens)
        chunk_overlap: Character overlap between chunks
        extraction_model: LLM model for memory extraction
        min_confidence: Minimum confidence for extracted memories
        source: Source tag for committed memories

    Returns:
        IngestionResult with statistics

    Raises:
        FileNotFoundError: If document doesn't exist
        ValueError: If extraction fails
    """
    # Read document
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    text = path.read_text(encoding="utf-8")

    # Chunk text
    chunks = chunk_text_by_chars(text, chunk_size=chunk_size, overlap=chunk_overlap)

    print(f"Processing {path.name}: {len(text):,} chars → {len(chunks)} chunks")

    # Track results
    memories_extracted = 0
    memories_committed = 0
    memories_deduplicated = 0
    errors = []

    # Track entities before and after
    entities_before = len(storage.get_all_entities())

    # Process each chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}/{len(chunks)}: ", end="", flush=True)

        try:
            # Extract memories from chunk
            extracted = extract_memories_from_chunk(
                chunk,
                model=extraction_model,
                min_confidence=min_confidence,
            )

            print(f"{len(extracted)} memories extracted", flush=True)
            memories_extracted += len(extracted)

            # Commit each memory
            for memory in extracted:
                try:
                    outcome = commit_memory(
                        content=memory.content,
                        storage=storage,
                        embedding_engine=embedding_engine,
                        source=source,
                        event_storage=event_storage,
                        m4_config=m4_config,
                        artifact_ref=path.name,
                    )

                    if outcome.outcome == "INSERTED_NEW":
                        memories_committed += 1
                    else:
                        memories_deduplicated += 1

                except Exception as e:
                    error_msg = f"Failed to commit memory: {e}"
                    errors.append(error_msg)
                    print(f"    Error: {error_msg}")

        except Exception as e:
            error_msg = f"Chunk {i} extraction failed: {e}"
            errors.append(error_msg)
            print(f"Error: {error_msg}")

    # Track entities created
    entities_after = len(storage.get_all_entities())
    entities_created = entities_after - entities_before

    return IngestionResult(
        document_path=document_path,
        chunks_processed=len(chunks),
        memories_extracted=memories_extracted,
        memories_committed=memories_committed,
        memories_deduplicated=memories_deduplicated,
        entities_created=entities_created,
        errors=errors,
    )
