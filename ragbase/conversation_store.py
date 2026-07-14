

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


def save_conversation(
    history_dir: Path,
    session_id: str,
    messages: Sequence[Mapping[str, str]],
    source_names: Sequence[str],
) -> Path:
    
    history_dir.mkdir(parents=True, exist_ok=True)
    safe_session_id = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)
    path = history_dir / f"{safe_session_id}.json"
    now = datetime.now().astimezone().isoformat()
    existing = load_conversation(path) if path.exists() else {}
    payload = {
        "schema_version": 1,
        "session_id": session_id,
        "title": _conversation_title(messages, source_names),
        "created_at": existing.get("created_at") or existing.get("saved_at") or now,
        "updated_at": now,
        "saved_at": now,
        "source_names": list(source_names),
        "messages": [dict(message) for message in messages],
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def load_conversation(path: Path) -> dict:
    
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("session_id"):
        raise ValueError(f"invalid conversation file: {path}")
    return payload


def list_conversations(history_dir: Path) -> list[dict]:
    
    if not history_dir.exists():
        return []

    by_session: dict[str, dict] = {}
    for path in history_dir.glob("*.json"):
        try:
            payload = load_conversation(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        session_id = str(payload["session_id"])
        updated_at = str(
            payload.get("updated_at")
            or payload.get("saved_at")
            or datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
        )
        summary = {
            "session_id": session_id,
            "title": payload.get("title") or _conversation_title(
                payload.get("messages") or [],
                payload.get("source_names") or [],
            ),
            "updated_at": updated_at,
            "source_names": list(payload.get("source_names") or []),
            "message_count": len(payload.get("messages") or []),
            "path": path,
        }
        previous = by_session.get(session_id)
        if previous is None or summary["updated_at"] > previous["updated_at"]:
            by_session[session_id] = summary
    return sorted(by_session.values(), key=lambda item: item["updated_at"], reverse=True)


def delete_conversation(history_dir: Path, session_id: str) -> int:
    
    deleted = 0
    if not history_dir.exists():
        return deleted
    for path in history_dir.glob("*.json"):
        try:
            payload = load_conversation(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if str(payload.get("session_id")) == session_id:
            path.unlink()
            deleted += 1
    return deleted


def _conversation_title(
    messages: Sequence[Mapping[str, str]],
    source_names: Sequence[str],
) -> str:
    
    for message in messages:
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"]).strip().replace("\n", " ")[:60]
    if source_names:
        return str(source_names[0])[:60]
    return "新对话"
