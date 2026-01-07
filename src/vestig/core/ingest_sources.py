"""Normalization helpers for ingesting different document formats."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vestig.core.ingestion import TemporalHints

SUPPORTED_FORMATS = {"auto", "plain", "claude-session"}


def detect_format(path: Path | None, raw_text: str) -> str:
    if path and path.suffix.lower() == ".jsonl":
        if _looks_like_claude_session(raw_text):
            return "claude-session"
    return "plain"


def normalize_document_text(
    raw_text: str,
    source_format: str = "auto",
    format_config: dict[str, Any] | None = None,
    path: Path | None = None,
) -> tuple[str, str, TemporalHints]:
    """
    Normalize document text and extract temporal hints.

    Returns:
        Tuple of (normalized_text, resolved_format, temporal_hints)
    """
    # Import here to avoid circular dependency
    from vestig.core.ingestion import TemporalHints

    if source_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unknown ingest format: {source_format}")

    resolved_format = source_format
    if source_format == "auto":
        resolved_format = detect_format(path, raw_text)

    if resolved_format == "plain":
        # Plain text: Use file mtime if available, else now
        if path:
            temporal_hints = TemporalHints.from_file_mtime(path)
        else:
            temporal_hints = TemporalHints.from_now()
        return raw_text, resolved_format, temporal_hints

    if resolved_format == "claude-session":
        session_config = format_config or {}
        normalized_text, temporal_hints = extract_claude_session_text_with_temporal(
            raw_text, session_config
        )
        return normalized_text, resolved_format, temporal_hints

    raise ValueError(f"Unhandled ingest format: {resolved_format}")


def _looks_like_claude_session(raw_text: str) -> bool:
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(event, dict):
            return False
        event_type = event.get("type")
        if event_type in {"user", "assistant"}:
            return True
        message = event.get("message")
        if isinstance(message, dict) and message.get("role") in {"user", "assistant"}:
            return True
    return False


def _normalize_timestamp(timestamp: str | None) -> str | None:
    if not timestamp or not isinstance(timestamp, str):
        return None
    try:
        normalized = timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def _earliest_timestamp(timestamps: list[str]) -> str | None:
    if not timestamps:
        return None
    parsed = []
    for ts in timestamps:
        try:
            parsed.append(datetime.fromisoformat(ts))
        except ValueError:
            continue
    if not parsed:
        return None
    return min(parsed).isoformat()


def _extract_text_from_block(block: Any) -> str:
    if isinstance(block, str):
        return block.strip()
    if isinstance(block, dict):
        text = block.get("text", "")
        if isinstance(text, str):
            return text.strip()
    return ""


def extract_claude_session_text(raw_text: str, config: dict[str, Any]) -> str:
    """Extract text from claude-session JSONL (backward compatible, no temporal)."""
    include_roles = set(config.get("include_roles", ["user", "assistant"]))
    include_message_types = set(config.get("include_message_types", ["text"]))
    drop_thinking = config.get("drop_thinking", True)
    drop_tool_use = config.get("drop_tool_use", True)

    blocks: list[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        message = event.get("message") if isinstance(event, dict) else None
        role = None
        content = None

        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        elif event_type in {"user", "assistant"}:
            role = event_type
            content = event.get("content")

        if role not in include_roles:
            continue

        filtered_blocks = _filter_message_blocks(
            role,
            content,
            include_message_types=include_message_types,
            drop_thinking=drop_thinking,
            drop_tool_use=drop_tool_use,
        )
        blocks.extend(filtered_blocks)

    return "\n\n".join(blocks)


def extract_claude_session_text_with_temporal(
    raw_text: str, config: dict[str, Any]
) -> tuple[str, TemporalHints]:
    """
    Extract text from claude-session JSONL with temporal metadata.

    Returns:
        Tuple of (normalized_text, temporal_hints)
    """
    from vestig.core.ingestion import TemporalHints

    include_roles = set(config.get("include_roles", ["user", "assistant"]))
    include_message_types = set(config.get("include_message_types", ["text"]))
    drop_thinking = config.get("drop_thinking", True)
    drop_tool_use = config.get("drop_tool_use", True)

    blocks: list[str] = []
    timestamps: list[str] = []  # Collect all event timestamps

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        # Extract timestamp from event (if present)
        event_timestamp = _normalize_timestamp(event.get("timestamp"))
        if event_timestamp:
            timestamps.append(event_timestamp)

        event_type = event.get("type")
        message = event.get("message") if isinstance(event, dict) else None
        role = None
        content = None

        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        elif event_type in {"user", "assistant"}:
            role = event_type
            content = event.get("content")

        if role not in include_roles:
            continue

        filtered_blocks = _filter_message_blocks(
            role,
            content,
            include_message_types=include_message_types,
            drop_thinking=drop_thinking,
            drop_tool_use=drop_tool_use,
        )
        blocks.extend(filtered_blocks)

    normalized_text = "\n\n".join(blocks)

    # Extract temporal hints from earliest timestamp
    if timestamps:
        earliest = _earliest_timestamp(timestamps) or timestamps[0]
        temporal_hints = TemporalHints.from_timestamp(
            timestamp=earliest,
            evidence=f"Extracted from claude-session JSONL (earliest of {len(timestamps)} events)",
        )
    else:
        # No timestamps found - fallback to now
        temporal_hints = TemporalHints.from_now()

    return normalized_text, temporal_hints


def extract_claude_session_chunks(
    raw_text: str,
    config: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, TemporalHints]]:
    """
    Extract chunks from claude-session JSONL with per-chunk temporal hints.
    """
    from vestig.core.ingestion import TemporalHints

    include_roles = set(config.get("include_roles", ["user", "assistant"]))
    include_message_types = set(config.get("include_message_types", ["text"]))
    drop_thinking = config.get("drop_thinking", True)
    drop_tool_use = config.get("drop_tool_use", True)

    blocks: list[tuple[str, str | None]] = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        event_timestamp = _normalize_timestamp(event.get("timestamp"))

        event_type = event.get("type")
        message = event.get("message") if isinstance(event, dict) else None
        role = None
        content = None

        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        elif event_type in {"user", "assistant"}:
            role = event_type
            content = event.get("content")

        if role not in include_roles:
            continue

        filtered_blocks = _filter_message_blocks(
            role,
            content,
            include_message_types=include_message_types,
            drop_thinking=drop_thinking,
            drop_tool_use=drop_tool_use,
        )
        for block in filtered_blocks:
            blocks.append((block, event_timestamp))

    chunked = _chunk_blocks(blocks, chunk_size, chunk_overlap)
    chunks_with_hints: list[tuple[str, TemporalHints]] = []
    for chunk_text, timestamps in chunked:
        earliest = _earliest_timestamp(timestamps)
        if earliest:
            hints = TemporalHints.from_timestamp(
                timestamp=earliest,
                evidence=f"Extracted from claude-session JSONL chunk ({len(timestamps)} events)",
            )
        else:
            hints = TemporalHints.from_now()
        chunks_with_hints.append((chunk_text, hints))

    return chunks_with_hints


def _chunk_blocks(
    blocks: list[tuple[str, str | None]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, list[str]]]:
    sep = "\n\n"
    sep_len = len(sep)
    chunks: list[tuple[str, list[str]]] = []
    current: list[tuple[str, str | None]] = []
    current_len = 0

    def packed_length(parts: list[tuple[str, str | None]]) -> int:
        if not parts:
            return 0
        total = sum(len(text) for text, _ in parts)
        total += sep_len * (len(parts) - 1)
        return total

    def pack(parts: list[tuple[str, str | None]]) -> tuple[str, list[str]]:
        text = sep.join(text for text, _ in parts)
        timestamps = [ts for _, ts in parts if ts]
        return text, timestamps

    def overlap_blocks(parts: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
        if chunk_overlap <= 0:
            return []
        overlap_parts: list[tuple[str, str | None]] = []
        overlap_len = 0
        for text, ts in reversed(parts):
            add_len = len(text) + (sep_len if overlap_parts else 0)
            overlap_len += add_len
            overlap_parts.append((text, ts))
            if overlap_len >= chunk_overlap:
                break
        overlap_parts.reverse()
        return overlap_parts

    for text, ts in blocks:
        add_len = len(text) + (sep_len if current else 0)
        if current and current_len + add_len > chunk_size:
            chunks.append(pack(current))
            current = overlap_blocks(current)
            current_len = packed_length(current)

        if not current and len(text) > chunk_size:
            chunks.append((text, [ts] if ts else []))
            continue

        add_len = len(text) + (sep_len if current else 0)
        current.append((text, ts))
        current_len += add_len

    if current:
        chunks.append(pack(current))

    return chunks


def _filter_message_blocks(
    role: str,
    content: Any,
    *,
    include_message_types: set[str],
    drop_thinking: bool,
    drop_tool_use: bool,
) -> list[str]:
    if isinstance(content, str):
        return [f"{role}: {content.strip()}"] if content.strip() else []

    items: list[Any]
    if isinstance(content, list):
        items = content
    elif isinstance(content, dict):
        items = [content]
    else:
        return []

    blocks: list[str] = []
    for item in items:
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "thinking" and drop_thinking:
                continue
            if item_type in {"tool_use", "tool_result"} and drop_tool_use:
                continue
            if item_type and item_type not in include_message_types:
                continue
        text = _extract_text_from_block(item)
        if text:
            blocks.append(f"{role}: {text}")

    return blocks
