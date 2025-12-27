"""Normalization helpers for ingesting different document formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
) -> tuple[str, str]:
    if source_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unknown ingest format: {source_format}")

    resolved_format = source_format
    if source_format == "auto":
        resolved_format = detect_format(path, raw_text)

    if resolved_format == "plain":
        return raw_text, resolved_format

    if resolved_format == "claude-session":
        session_config = format_config or {}
        return extract_claude_session_text(raw_text, session_config), resolved_format

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


def _extract_text_from_block(block: Any) -> str:
    if isinstance(block, str):
        return block.strip()
    if isinstance(block, dict):
        text = block.get("text", "")
        if isinstance(text, str):
            return text.strip()
    return ""


def extract_claude_session_text(raw_text: str, config: dict[str, Any]) -> str:
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
