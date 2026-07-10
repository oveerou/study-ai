from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def close_vector_store(vector_store: Any | None) -> None:
    if vector_store is None:
        return
    client = getattr(vector_store, "client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def archive_conversation(
    history_dir: Path,
    session_id: str,
    messages: Sequence[Mapping[str, str]],
    source_names: Sequence[str],
) -> Path | None:
    has_user_message = any(message.get("role") == "user" for message in messages)
    if not source_names and not has_user_message:
        return None

    saved_at = datetime.now().astimezone()
    payload = {
        "session_id": session_id,
        "saved_at": saved_at.isoformat(),
        "source_names": list(source_names),
        "messages": [dict(message) for message in messages],
    }
    history_dir.mkdir(parents=True, exist_ok=True)
    archive_path = history_dir / f"{saved_at:%Y%m%d-%H%M%S-%f}-{session_id}.json"
    archive_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return archive_path
